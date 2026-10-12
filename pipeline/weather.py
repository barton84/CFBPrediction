"""Game-day weather from Open-Meteo (free, no key).

forecast_games(): kickoff-window forecast for this week's outdoor games.
backfill_and_fit(): pulls historical weather for 2022-2025 outdoor FBS games and refits
how wind, rain, snow and cold move the total, then writes the fitted values into
data/model.json so the site and the daily job use them.
"""
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

FORECAST = 'https://api.open-meteo.com/v1/forecast'
ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
HOURLY = 'temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_gusts_10m'


def _call(base, lats, lons, start, end, tries=2, timeout=30):
    q = {'latitude': ','.join(f'{x:.4f}' for x in lats), 'longitude': ','.join(f'{x:.4f}' for x in lons),
         'hourly': HOURLY, 'temperature_unit': 'fahrenheit', 'wind_speed_unit': 'mph', 'precipitation_unit': 'inch',
         'timezone': 'GMT', 'start_date': start, 'end_date': end}
    url = base + '?' + urllib.parse.urlencode(q)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'cfb-prediction-model'}), timeout=timeout) as r:
                js = json.loads(r.read().decode())
            return js if isinstance(js, list) else [js]
        except Exception as e:
            print('  weather retry', i + 1, e, flush=True)
            time.sleep(10 * (i + 1))
    return None


def reachable(base):
    try:
        url = base + '?latitude=39&longitude=-95&hourly=temperature_2m'
        if base == ARCHIVE:
            url += '&start_date=2024-10-05&end_date=2024-10-05'
        urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'cfb-prediction-model'}), timeout=15).read()
        return True
    except Exception:
        return False


def _window(js, kickoff_utc):
    """Average over the 3.5 hours from kickoff; precipitation summed."""
    h = js.get('hourly', {})
    if not h:
        return None
    t = pd.to_datetime(h['time'], utc=True)
    k = pd.Timestamp(kickoff_utc)
    if k.tzinfo is None:
        k = k.tz_localize('UTC')
    m = (t >= k.floor('h')) & (t < k + pd.Timedelta(hours=3.5))
    if not m.any():
        return None
    def arr(name):
        return np.array(h.get(name, [np.nan] * len(t)), dtype=float)[np.asarray(m)]
    return {'temp_f': float(np.nanmean(arr('temperature_2m'))), 'wind_mph': float(np.nanmean(arr('wind_speed_10m'))),
            'gust_mph': float(np.nanmax(arr('wind_gusts_10m'))), 'precip_in': float(np.nansum(arr('precipitation'))),
            'snow_in': float(np.nansum(arr('snowfall')))}


def fetch_for(games, base=FORECAST, batch=20, budget_s=None, max_fail=3, on_progress=None):
    """games: DataFrame with game_id, g_lat, g_lon, start_date (UTC ISO). Returns {game_id: weather}.
    Paced to stay well under Open-Meteo's free limits (about 400 locations a minute), stops after a
    few failures in a row or when the time budget runs out, and returns whatever it got."""
    out = {}
    if not reachable(base):
        print('weather service not reachable; skipping')
        return out
    t0 = time.time()
    fails = 0
    g = games.dropna(subset=['g_lat', 'g_lon']).copy()
    g['ts'] = pd.to_datetime(g.start_date, utc=True, format='ISO8601')
    g['day'] = g.ts.dt.strftime('%Y-%m-%d')
    for day, x in g.groupby('day'):
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        for i in range(0, len(x), batch):
            if budget_s and time.time() - t0 > budget_s:
                print('weather: time budget used, stopping for now', flush=True)
                return out
            part = x.iloc[i:i + batch]
            js = _call(base, part.g_lat.values, part.g_lon.values, day, end)
            if not js:
                fails += 1
                if fails >= max_fail:
                    print('weather: service is refusing requests, stopping for now', flush=True)
                    return out
                time.sleep(60)
                continue
            fails = 0
            for (gid, ts), w in zip(zip(part.game_id, part.ts), js):
                r = _window(w, ts)
                if r:
                    out[int(gid)] = r
            if on_progress:
                on_progress(out)
            time.sleep(max(0.5, len(part) * 60 / 400))
    return out


def adjustment(w, coefs):
    """Points the weather moves the total."""
    if not w:
        return 0.0, {}
    parts = {
        'Wind': coefs['wind_per_mph_over_10'] * max(0.0, (w.get('wind_mph') or 0) - 10),
        'Rain': coefs['rain'] if (w.get('precip_in') or 0) >= 0.1 and (w.get('snow_in') or 0) < 0.1 else 0.0,
        'Snow': coefs['snow'] if (w.get('snow_in') or 0) >= 0.1 else 0.0,
        'Cold': coefs['cold_below_32'] if (w.get('temp_f') is not None and w['temp_f'] < 32) else 0.0,
    }
    return float(sum(parts.values())), parts


TARGETS = 'weather_targets.json'
HISTORY = 'weather_history.json'


def save_targets(work, state_dir, data_dir, seasons=(2022, 2023, 2024, 2025)):
    """Called by retrain: the outdoor games to collect weather for, with the no-weather total
    model's miss on each, so the fit later needs nothing but this small file."""
    import train
    model = json.load(open(os.path.join(data_dir, 'model.json')))
    d = train.load_frame()
    d = d[d.season.isin(seasons) & d.total.notna() & (~d.dome.fillna(False).astype(bool)) & d.g_lat.notna()]
    f = model['total']['features']
    X = np.c_[d[f].fillna(d[f].mean()).values, np.ones(len(d))]
    d['resid'] = d.total - (X @ np.array(model['total']['coef']) + model['total']['median_offset'])
    rows = [{'game_id': int(r.game_id), 'lat': round(float(r.g_lat), 4), 'lon': round(float(r.g_lon), 4),
             'start': r.start_date, 'resid': round(float(r.resid), 2), 'season': int(r.season)} for r in d.itertuples()]
    json.dump(rows, open(os.path.join(state_dir, TARGETS), 'w'))
    print('weather targets saved:', len(rows), 'games')


def backfill(state_dir, data_dir, budget_s=240):
    """Collect a slice of historical weather (saved as it goes) and refit once coverage is high.
    Runs inside both workflows; a few runs finish the whole backfill."""
    tp = os.path.join(state_dir, TARGETS)
    if not os.path.exists(tp):
        print('no weather targets yet; run the retrain workflow first')
        return
    targets = pd.DataFrame(json.load(open(tp)))
    hp = os.path.join(state_dir, HISTORY)
    hist = json.load(open(hp)) if os.path.exists(hp) else {}
    todo = targets[~targets.game_id.astype(str).isin(hist.keys())]
    print(f'weather history: {len(targets) - len(todo)} of {len(targets)} games collected', flush=True)
    if len(todo):
        def save(got):
            hist.update({str(k): v for k, v in got.items()})
            json.dump(hist, open(hp, 'w'))
        g = todo.rename(columns={'lat': 'g_lat', 'lon': 'g_lon', 'start': 'start_date'})
        save(fetch_for(g[['game_id', 'g_lat', 'g_lon', 'start_date']], base=ARCHIVE, budget_s=budget_s, on_progress=save))
        print(f'weather history now: {sum(1 for k in targets.game_id if str(k) in hist)} of {len(targets)}', flush=True)
    have = targets[targets.game_id.astype(str).isin(hist.keys())]
    if len(have) < 0.9 * len(targets):
        print('not enough weather yet to refit; keeping current values')
        return
    fit(have, hist, data_dir)


def fit(have, hist, data_dir):
    w = pd.DataFrame([hist[str(i)] for i in have.game_id])
    resid = have.resid.values
    F = pd.DataFrame({'wind': (w.wind_mph - 10).clip(lower=0), 'rain': ((w.precip_in >= 0.1) & (w.snow_in < 0.1)).astype(float),
                      'snow': (w.snow_in >= 0.1).astype(float), 'cold': (w.temp_f < 32).astype(float)})
    ok = (F.notna().all(axis=1) & ~np.isnan(resid)).values
    names = ['wind_per_mph_over_10', 'rain', 'snow', 'cold_below_32']
    cols = ['wind', 'rain', 'snow', 'cold']
    # only fit a term with enough games behind it; rare ones (snow) keep the starting value
    use = [c for c in cols if (F[c][ok] > 0).sum() >= 15]
    A = np.c_[F.loc[ok, use].values, np.ones(ok.sum())]
    b = np.linalg.lstsq(A, resid[ok], rcond=None)[0]
    res = resid[ok] - A @ b
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * res.var())
    seasons = sorted(have.season.unique())
    mp = os.path.join(data_dir, 'model.json')
    model = json.load(open(mp))
    old = model['total']['weather']
    vals, ts = {}, {}
    for n, c in zip(names, cols):
        if c in use:
            i = use.index(c)
            vals[n] = round(float(b[i]), 3)
            ts[n] = round(float(b[i] / se[i]), 2) if se[i] > 0 else None
        else:
            vals[n] = old.get(n)
            ts[n] = None
    model['total']['weather'] = {'fitted': True, **vals, 'n_games': int(ok.sum()), 't_stats': ts,
                                 'counts': {'wind_over_15': int((w.wind_mph > 15).sum()), 'rain': int(F.rain.sum()),
                                            'snow': int(F.snow.sum()), 'cold': int(F.cold.sum())},
                                 'not_fitted_terms': [n for n, c in zip(names, cols) if c not in use],
                                 'note': f'Fitted on {int(ok.sum())} outdoor FBS games, {seasons[0]}-{seasons[-1]}, Open-Meteo historical weather.'
                                         + (' Too few snow games to fit; snow keeps the research value.' if 'snow' not in use else '')}
    json.dump(model, open(mp, 'w'), indent=1)
    print('weather fitted:', model['total']['weather'])


if __name__ == '__main__':
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    budget = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == '--backfill' else 240
    backfill(os.path.join(root, 'state'), os.path.join(root, 'data'), budget_s=budget)
