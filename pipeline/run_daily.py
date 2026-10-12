"""Daily job: refresh this season's data, rebuild ratings, price every game, pull
DraftKings lines and weather, and write the JSON files the site reads.

    python pipeline/run_daily.py            # normal run
    python pipeline/run_daily.py --offline  # skip downloads and APIs (uses cached files)
"""
import datetime as dt
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'model'))
sys.path.insert(0, HERE)
from util import ROOT, current_season, dump             # noqa: E402
from fetch import fetch_seasons                         # noqa: E402
from data_build import build_all                        # noqa: E402
from features import build                              # noqa: E402
from core import METRICS, load                          # noqa: E402
from evaluate import frame                              # noqa: E402
from stack import bucket                                # noqa: E402
import situational                                      # noqa: E402
import odds as oddsmod                                  # noqa: E402
import weather as wx                                    # noqa: E402

RAW = os.path.join(ROOT, '.cache', 'raw')
WORK = os.path.join(ROOT, '.cache', 'work_daily')
STATE = os.path.join(ROOT, 'state')
DATA = os.path.join(ROOT, 'data')

LABELS = {'x_sr': 'Success rate edge', 'x_ypp': 'Yards per play edge', 'x_fpi': 'ESPN FPI projection', 'loc': 'Home field',
          'fluck_diff': 'Fumble luck (regressed)', 'x_epa': 'EPA per play edge', 'x_pts': 'Points edge'}
TLABELS = {'xp_sum': 'Adjusted scoring (both teams)', 'x_pace': 'Pace (plays per game)', 'x_tot_eff': 'Offense vs defense EPA',
           'dome_f': 'Dome', 'env': 'Scoring environment this season'}


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def load_prior(season):
    path = f'{STATE}/prior_{season}.json'
    js = json.load(open(path))
    prior = {m: pd.DataFrame(js['prior'][m], columns=['team_id', 'o', 'd']).astype({'team_id': 'int64'}) for m in METRICS}
    return prior, js['params'], js['env_last']


def team_table(raw):
    parts = []
    for y in [2026, 2024, 2019]:
        f = f'{raw}/team_info_{y}.parquet'
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
    t = pd.concat(parts).drop_duplicates('team_id')
    t['team_id'] = pd.to_numeric(t.team_id).astype('int64')
    return t


def spread_pred(d, model):
    sp = model['spread']
    use_fpi = d.x_fpi.notna().values
    pred = np.full(len(d), np.nan)
    comps = [None] * len(d)
    bk = bucket(d.wk.values)
    for i in range(len(d)):
        feats = sp['features'] if use_fpi[i] else sp['fallback_features']
        co = np.array((sp['coef'] if use_fpi[i] else sp['fallback_coef'])[str(int(bk[i]))])
        x = np.array([0.0 if pd.isna(d.iloc[i][f]) else float(d.iloc[i][f]) for f in feats])
        parts = co[:-1] * x
        pred[i] = parts.sum() + co[-1]
        c = [{'label': LABELS.get(f, f), 'pts': round(float(v), 2)} for f, v in zip(feats, parts)]
        c.append({'label': 'Baseline', 'pts': round(float(co[-1]), 2)})
        comps[i] = c
    return pred, comps, use_fpi


def total_pred(d, model):
    tm = model['total']
    co = np.array(tm['coef'])
    X = d[tm['features']].astype(float)
    X = X.fillna(X.mean())
    parts = X.values * co[:-1]
    pred = parts.sum(axis=1) + co[-1] + tm['median_offset']
    comps = [[{'label': TLABELS.get(f, f), 'pts': round(float(v), 2)} for f, v in zip(tm['features'], row)] +
             [{'label': 'Baseline', 'pts': round(float(co[-1] + tm['median_offset']), 2)}] for row in parts]
    return pred, comps


def cover_p(edge, slope, cap=0.56):
    p = 1 / (1 + np.exp(-slope * np.abs(edge)))
    return np.minimum(p, cap)


def main(offline=False):
    season = current_season()
    t0 = now_utc()
    if not offline:
        fetch_seasons(RAW, [season])
        try:
            import cfbd
            if not os.path.exists(f'{STATE}/venues.json') or (dt.datetime.now().weekday() == 0):
                cfbd.refresh_venues(STATE)
        except Exception as e:
            print('venues refresh skipped:', e)
    build_all(RAW, WORK, [season])
    prior, params, env_last = load_prior(season)
    g, F, TR, _ = build(WORK, params, seasons=[season], verbose=False, initial_prior=prior)
    F.to_parquet(f'{WORK}/feats.parquet')
    gl, _ = load(WORK)
    situ = situational.build(gl, RAW, WORK, venues_json=f'{STATE}/venues.json')
    model = json.load(open(f'{DATA}/model.json'))

    d = frame(WORK, F).merge(situ, on='game_id', how='left')
    d['xp_sum'] = d.xp_home + d.xp_away
    d['dome_f'] = d.dome.fillna(False).astype(float)
    fbs_done = d[d.fbs_both & d.total.notna()]
    env = {}
    for wk in range(1, 22):
        pr = fbs_done[fbs_done.wk < wk]
        env[wk] = pr.total.mean() if len(pr) >= 80 else env_last
    d['env'] = d.wk.map(env).fillna(env_last)
    d = d[d.fbs_both].copy()
    d['pred'], d['comps'], d['uses_fpi'] = spread_pred(d, model)
    d['pt'], d['tcomps'] = total_pred(d, model)

    ti = team_table(RAW)
    tinfo = ti.set_index('team_id')
    sched = gl[gl.season == season].set_index('game_id')
    games_all = pd.read_parquet(f'{WORK}/games.parquet').set_index('game_id')

    # current week: the earliest week that still has FBS games to play
    upcoming = d[(d.completed != True)]
    week = int(upcoming.wk.min()) if len(upcoming) else int(d.wk.max())

    # ---------- DraftKings lines ----------
    lines_path = f'{STATE}/lines_{season}.json'
    store = json.load(open(lines_path)) if os.path.exists(lines_path) else {}
    unmatched, credits = [], None
    if not offline:
        ev, credits = oddsmod.fetch()
        if ev is not None:
            ln = oddsmod.parse(ev)
            sch = gl[(gl.season == season)][['game_id', 'home_id', 'away_id', 'start_date']]
            m, unmatched = oddsmod.match(ln, sch, ti)
            stamp = t0.isoformat(timespec='minutes')
            for _, r in m.iterrows():
                gid = str(int(r.game_id))
                if gid not in d.game_id.astype(str).values:
                    continue
                row = d[d.game_id == int(gid)].iloc[0]
                kickoff = pd.Timestamp(row.start_date)
                if kickoff.tzinfo is None:
                    kickoff = kickoff.tz_localize('UTC')
                if kickoff <= pd.Timestamp(t0):
                    continue                     # never overwrite a line after kickoff
                snap = {'spread': r.dk_home_line, 'total': r.dk_total, 'ts': stamp, 'home_price': r.dk_home_price,
                        'away_price': r.dk_away_price, 'over_price': r.dk_over_price, 'under_price': r.dk_under_price,
                        'model_margin': round(float(row.pred), 2), 'model_total': round(float(row.pt), 2)}
                ent = store.setdefault(gid, {})
                if 'first' not in ent and r.dk_home_line is not None:
                    ent['first'] = snap
                ent['last'] = snap
            dump(store, lines_path)

    # ---------- weather for this week's outdoor games ----------
    wk_games = d[(d.wk == week)]
    weather = {}
    if not offline:
        todo = wk_games[(~wk_games.dome) & wk_games.g_lat.notna() & (wk_games.completed != True)]
        start = pd.to_datetime(todo.start_date, utc=True, format='ISO8601')
        todo = todo[(start - pd.Timestamp(t0)) < pd.Timedelta(days=15)]
        if len(todo):
            weather = wx.fetch_for(todo[['game_id', 'g_lat', 'g_lon', 'start_date']])
    wpath = f'{STATE}/weather_{season}.json'
    wstore = json.load(open(wpath)) if os.path.exists(wpath) else {}
    wstore.update({str(k): {**v, 'ts': t0.isoformat(timespec='minutes')} for k, v in weather.items()})
    dump(wstore, wpath)

    # ---------- QB info ----------
    st = pd.read_parquet(f'{WORK}/starters.parquet').merge(gl[['game_id', 'start_date']], on='game_id')
    st['t'] = pd.to_datetime(st.start_date, utc=True, format='ISO8601')
    st = st.sort_values('t')
    qb = {}
    for team, x in st.groupby('team_id'):
        tot = x.groupby('qb_name').qb_db.sum().sort_values(ascending=False)
        last = x.iloc[-1]
        qb[int(team)] = {'primary': tot.index[0], 'last': last.qb_name, 'changed': bool(last.qb_name != tot.index[0]),
                         'primary_db': int(tot.iloc[0])}

    def team_obj(tid, rank):
        t = tinfo.loc[tid] if tid in tinfo.index else None
        return {'id': int(tid), 'school': (t.school if t is not None else str(tid)),
                'abbr': (t.abbreviation if t is not None else ''), 'mascot': (t.mascot if t is not None else ''),
                'color': (t.color if t is not None else None), 'alt': (t.alt_color if t is not None else None),
                'logo': (t.logo if t is not None else None), 'rank': None if pd.isna(rank) else int(rank),
                'qb': qb.get(int(tid))}

    def game_obj(r):
        gid = str(int(r.game_id))
        s = sched.loc[int(r.game_id)] if int(r.game_id) in sched.index else None
        ent = store.get(gid, {})
        last, first = ent.get('last'), ent.get('first')
        if last and last.get('spread') is not None:
            line = {'spread': last['spread'], 'total': last.get('total'), 'source': 'DK', 'ts': last['ts'],
                    'home_price': last.get('home_price'), 'away_price': last.get('away_price'),
                    'over_price': last.get('over_price'), 'under_price': last.get('under_price')}
        elif not pd.isna(r.home_line):
            line = {'spread': float(r.home_line), 'total': None if pd.isna(r.over_under) else float(r.over_under),
                    'source': 'ESPN', 'ts': None}
        else:
            line = None
        w = wstore.get(gid)
        o = {'id': int(r.game_id), 'wk': int(r.wk), 'kickoff': r.start_date,
             'tbd': bool(s.start_time_tbd) if s is not None and not pd.isna(s.start_time_tbd) else False,
             'venue': r.venue, 'neutral': bool(r.neutral_site), 'conf_game': bool(r.conference_game) if not pd.isna(r.conference_game) else False,
             'dome': bool(r.dome), 'home': team_obj(r.home_id, s.home_rank if s is not None else np.nan),
             'away': team_obj(r.away_id, s.away_rank if s is not None else np.nan),
             'line': line, 'first': first, 'fpi': None if pd.isna(r.x_fpi) else round(float(r.x_fpi), 1),
             'model': {'margin': round(float(r.pred), 2), 'total': round(float(r.pt), 2), 'comps': r.comps,
                       'tcomps': r.tcomps, 'uses_fpi': bool(r.uses_fpi)},
             'weather': w,
             'situ': {'travel_home_km': None if pd.isna(r.trav_h) else round(float(r.trav_h)),
                      'travel_away_km': None if pd.isna(r.trav_a) else round(float(r.trav_a)),
                      'rest_home': None if pd.isna(r.rest_h) else round(float(r.rest_h), 1),
                      'rest_away': None if pd.isna(r.rest_a) else round(float(r.rest_a), 1),
                      'elev_m': None if pd.isna(r.g_elev) else round(float(r.g_elev)),
                      'ko_et_hour': None if pd.isna(r.ko_et_hour) else round(float(r.ko_et_hour), 2)},
             'final': bool(r.completed == True)}
        if r.completed == True:
            o['score'] = {'home': int(r.home_points), 'away': int(r.away_points)}
        return o

    week_games = [game_obj(r) for _, r in d[d.wk == week].sort_values('start_date').iterrows()]
    dump({'season': season, 'week': week, 'games': week_games}, f'{DATA}/week.json')

    # ---------- results: every completed game this season ----------
    sp_slope = model['spread']['cover_slope']
    res = []
    for _, r in d[(d.completed == True)].sort_values(['wk', 'start_date']).iterrows():
        o = game_obj(r)
        ent = store.get(str(int(r.game_id)), {})
        if ent.get('last') and ent['last'].get('spread') is not None:
            gl_line, src = ent['last']['spread'], 'DK game-day'
            gl_total = ent['last'].get('total')
            model_m = ent['last'].get('model_margin', r.pred)
            model_t = ent['last'].get('model_total', r.pt)
        elif not pd.isna(r.home_line):
            gl_line, src = float(r.home_line), 'ESPN close'
            gl_total = None if pd.isna(r.over_under) else float(r.over_under)
            model_m, model_t = float(r.pred), float(r.pt)
        else:
            continue
        margin = r.home_points - r.away_points
        edge = model_m + gl_line
        cover = margin + gl_line
        rec = {'id': int(r.game_id), 'wk': int(r.wk), 'home': o['home']['school'], 'away': o['away']['school'],
               'home_abbr': o['home']['abbr'], 'away_abbr': o['away']['abbr'], 'neutral': o['neutral'],
               'line': gl_line, 'line_src': src, 'model_margin': round(float(model_m), 1), 'edge': round(float(edge), 1),
               'score_home': int(r.home_points), 'score_away': int(r.away_points),
               'result': None if cover == 0 or edge == 0 else bool(np.sign(edge) == np.sign(cover)), 'push': bool(cover == 0)}
        if gl_total is not None:
            te = model_t - gl_total
            tr = (r.home_points + r.away_points) - gl_total
            rec.update({'total_line': gl_total, 'model_total': round(float(model_t), 1), 'total_edge': round(float(te), 1),
                        'total_result': None if tr == 0 or te == 0 else bool(np.sign(te) == np.sign(tr))})
        if ent.get('first') and ent['first'].get('spread') is not None:
            f = ent['first']
            rec['posted'] = {'spread': f['spread'], 'ts': f['ts'], 'model_margin': f.get('model_margin')}
        res.append(rec)
    dump({'season': season, 'games': res}, f'{DATA}/results.json')

    # ---------- team ratings for the current week ----------
    rt = TR[(TR.season == season) & (TR.wk == week)].copy()
    fbs_ids = set(gl[(gl.season == season) & (gl.home_division == 'fbs')].home_id)
    rt = rt[rt.team_id.isin(fbs_ids)]
    sp = model['spread']
    fb = sp['fallback_features']
    co = np.array(sp['fallback_coef'][str(int(bucket(np.array([week]))[0]))])
    means = {m: (rt[f'{m}_o'].mean(), rt[f'{m}_d'].mean()) for m in METRICS}
    power = np.zeros(len(rt))
    for f, c in zip(fb, co[:-1]):
        if f.startswith('x_'):
            m = f[2:]
            power += c * ((rt[f'{m}_o'] - rt[f'{m}_d']) - (means[m][0] - means[m][1])).values
    rt['power'] = power
    tgs = pd.read_parquet(f'{WORK}/team_games.parquet')
    played = gl[(gl.season == season) & (gl.completed == True)]
    long = pd.concat([played[['home_id', 'home_points', 'away_points']].rename(columns={'home_id': 't', 'home_points': 'pf', 'away_points': 'pa'}),
                      played[['away_id', 'away_points', 'home_points']].rename(columns={'away_id': 't', 'away_points': 'pf', 'home_points': 'pa'})])
    agg = long.groupby('t').agg(pf=('pf', 'mean'), pa=('pa', 'mean'), w=('pf', lambda s: 0))
    wl = long.assign(win=long.pf > long.pa).groupby('t').win.agg(['sum', 'count'])
    raw_sr = tgs.groupby('team_id').apply(lambda x: np.average(x.sr, weights=x.plays))
    raw_sr_d = tgs.groupby('opp_id').apply(lambda x: np.average(x.sr, weights=x.plays))
    teams = []
    for _, r in rt.sort_values('power', ascending=False).iterrows():
        t = int(r.team_id)
        info = tinfo.loc[t] if t in tinfo.index else None
        teams.append({'id': t, 'school': info.school if info is not None else str(t), 'abbr': info.abbreviation if info is not None else '',
                      'conf': next(iter(gl[(gl.season == season) & (gl.home_id == t)].home_conference.dropna().tolist() or
                                        gl[(gl.season == season) & (gl.away_id == t)].away_conference.dropna().tolist() or ['']), ''),
                      'logo': info.logo if info is not None else None, 'color': info.color if info is not None else None,
                      'gp': int(r.gp), 'record': f"{int(wl.loc[t, 'sum'])}-{int(wl.loc[t, 'count'] - wl.loc[t, 'sum'])}" if t in wl.index else '0-0',
                      'power': round(float(r.power), 1),
                      'off_sr': round(float(r.sr_mu + r.sr_o) * 100, 1), 'def_sr': round(float(r.sr_mu + r.sr_d) * 100, 1),
                      'off_epa': round(float(r.epa_mu + r.epa_o), 3), 'def_epa': round(float(r.epa_mu + r.epa_d), 3),
                      'off_ypp': round(float(r.ypp_mu + r.ypp_o), 2), 'def_ypp': round(float(r.ypp_mu + r.ypp_d), 2),
                      'raw_sr': None if t not in raw_sr.index else round(float(raw_sr[t]) * 100, 1),
                      'raw_def_sr': None if t not in raw_sr_d.index else round(float(raw_sr_d[t]) * 100, 1),
                      'ppg': None if t not in agg.index else round(float(agg.loc[t, 'pf']), 1),
                      'opp_ppg': None if t not in agg.index else round(float(agg.loc[t, 'pa']), 1),
                      'pace': round(float(r.pace_mu / 2 + r.pace_o), 1),
                      'qb': qb.get(t, {}).get('primary'), 'qb_changed': qb.get(t, {}).get('changed', False),
                      'qb_last': qb.get(t, {}).get('last'),
                      'r': {f'{m}_{side}': round(float(r[f'{m}_{side}']), 4) for m in METRICS for side in ['o', 'd']}})
    mus = {m: round(float(rt[f'{m}_mu'].iloc[0]), 4) for m in METRICS} if len(rt) else {}
    dump({'season': season, 'week': week, 'through': week - 1, 'teams': teams, 'mu': mus,
          'env': round(float(env.get(week, env_last)), 2)}, f'{DATA}/ratings.json')

    # ---------- summary for the header ----------
    graded = [x for x in res if x['result'] is not None]
    def rec(xs, key='result'):
        w = sum(1 for x in xs if x[key]); l = sum(1 for x in xs if x[key] is False)
        return {'w': w, 'l': l}
    edges = [(g['model']['margin'] + g['line']['spread']) for g in week_games if g['line'] and not g['final']]
    site = {'season': season, 'week': week, 'ratings_through': week - 1, 'built': t0.isoformat(timespec='minutes'),
            'flag5': int(sum(1 for e in edges if 5 <= abs(e) < 7)), 'flag7': int(sum(1 for e in edges if abs(e) >= 7)),
            'ats': rec(graded), 'ats5': rec([x for x in graded if abs(x['edge']) >= 5]),
            'ats7': rec([x for x in graded if abs(x['edge']) >= 7]),
            'ou': rec([x for x in res if x.get('total_result') is not None], 'total_result'),
            'dk_lines': int(sum(1 for g in week_games if g['line'] and g['line']['source'] == 'DK')),
            'games_this_week': len(week_games), 'weather_loaded': int(sum(1 for g in week_games if g['weather'])),
            'unmatched_odds': unmatched, 'odds_credits': credits, 'offline': offline,
            'data_through': str(gl[(gl.season == season) & (gl.completed == True)].start_date.max())}
    dump(site, f'{DATA}/site.json')
    print(json.dumps({k: v for k, v in site.items() if k != 'unmatched_odds'}, default=str))
    if unmatched:
        print('unmatched odds events:', unmatched[:20])


if __name__ == '__main__':
    main(offline='--offline' in sys.argv)
