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


def _call(base, lats, lons, start, end):
    q = {'latitude': ','.join(f'{x:.4f}' for x in lats), 'longitude': ','.join(f'{x:.4f}' for x in lons),
         'hourly': HOURLY, 'temperature_unit': 'fahrenheit', 'wind_speed_unit': 'mph', 'precipitation_unit': 'inch',
         'timezone': 'GMT', 'start_date': start, 'end_date': end}
    url = base + '?' + urllib.parse.urlencode(q)
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'cfb-prediction-model'}), timeout=60) as r:
                js = json.loads(r.read().decode())
            return js if isinstance(js, list) else [js]
        except Exception as e:
            print('  weather retry', i + 1, e, flush=True)
            time.sleep(3 * (i + 1))
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


def fetch_for(games, base=FORECAST, batch=50):
    """games: DataFrame with game_id, g_lat, g_lon, start_date (UTC ISO). Returns {game_id: weather}."""
    out = {}
    if not reachable(base):
        print('weather service not reachable; skipping')
        return out
    g = games.dropna(subset=['g_lat', 'g_lon']).copy()
    g['ts'] = pd.to_datetime(g.start_date, utc=True, format='ISO8601')
    g['day'] = g.ts.dt.strftime('%Y-%m-%d')
    for day, x in g.groupby('day'):
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        for i in range(0, len(x), batch):
            part = x.iloc[i:i + batch]
            js = _call(base, part.g_lat.values, part.g_lon.values, day, end)
            if not js:
                continue
            for (gid, ts), w in zip(zip(part.game_id, part.ts), js):
                r = _window(w, ts)
                if r:
                    out[int(gid)] = r
            time.sleep(0.3)
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


def backfill_and_fit(work, raw, data_dir, seasons=(2022, 2023, 2024, 2025)):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
    cache = os.path.join(os.path.dirname(data_dir), 'state', 'weather_history.json')
    hist = json.load(open(cache)) if os.path.exists(cache) else {}
    g = pd.read_parquet(f'{work}/games.parquet')
    s = pd.read_parquet(f'{work}/situ.parquet')
    g = g.merge(s, on='game_id')
    g = g[g.season.isin(seasons) & (g.home_division == 'fbs') & (g.away_division == 'fbs') & (~g.dome) & g.g_lat.notna()]
    todo = g[~g.game_id.astype(str).isin(hist.keys())]
    print('weather backfill: need', len(todo), 'games', flush=True)
    if len(todo) and not reachable(ARCHIVE):
        raise RuntimeError('Open-Meteo archive not reachable from here')
    if len(todo):
        got = fetch_for(todo[['game_id', 'g_lat', 'g_lon', 'start_date']], base=ARCHIVE)
        hist.update({str(k): v for k, v in got.items()})
        json.dump(hist, open(cache, 'w'))
    # residual of the no-weather total model on these games
    model = json.load(open(os.path.join(data_dir, 'model.json')))
    import train
    d = train.load_frame()
    d = d[d.game_id.isin([int(k) for k in hist]) & d.total.notna()]
    co = np.array(model['total']['coef'])
    X = np.c_[d[model['total']['features']].fillna(d[model['total']['features']].mean()).values, np.ones(len(d))]
    d['resid'] = d.total - (X @ co + model['total']['median_offset'])
    w = pd.DataFrame([hist[str(i)] for i in d.game_id])
    d = d.reset_index(drop=True)
    F = pd.DataFrame({'wind': (w.wind_mph - 10).clip(lower=0), 'rain': ((w.precip_in >= 0.1) & (w.snow_in < 0.1)).astype(float),
                      'snow': (w.snow_in >= 0.1).astype(float), 'cold': (w.temp_f < 32).astype(float)})
    ok = F.notna().all(axis=1) & d.resid.notna()
    A = np.c_[F[ok].values, np.ones(ok.sum())]
    b = np.linalg.lstsq(A, d.resid[ok].values, rcond=None)[0]
    res = d.resid[ok].values - A @ b
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * res.var())
    names = ['wind_per_mph_over_10', 'rain', 'snow', 'cold_below_32']
    fitted = {n: round(float(v), 3) for n, v in zip(names, b[:4])}
    model['total']['weather'] = {'fitted': True, **fitted, 'n_games': int(ok.sum()),
                                 't_stats': {n: round(float(v / e), 2) for n, v, e in zip(names, b[:4], se[:4])},
                                 'counts': {'wind_over_15': int((w.wind_mph > 15).sum()), 'rain': int(F.rain.sum()),
                                            'snow': int(F.snow.sum()), 'cold': int(F.cold.sum())},
                                 'note': f'Fitted on {int(ok.sum())} outdoor FBS games, {seasons[0]}-{seasons[-1]}, Open-Meteo historical weather.'}
    json.dump(model, open(os.path.join(data_dir, 'model.json'), 'w'), indent=1)
    print('weather fitted:', model['total']['weather'])
