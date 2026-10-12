"""Full rebuild: download 2019 to now, rebuild ratings with fitted priors, rerun the
leave-one-season-out backtest, refit the model and save the preseason prior for the
current season. Run by the 'retrain' GitHub workflow (manual) and once each August.

    python pipeline/retrain.py
"""
import copy
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'model'))
sys.path.insert(0, HERE)
from fetch import fetch_seasons                     # noqa: E402
from data_build import build_all, extra_table       # noqa: E402
from features import build, DEFAULT                 # noqa: E402
from core import METRICS, make_prior                # noqa: E402
import situational                                  # noqa: E402
from core import load                               # noqa: E402
from util import current_season, ROOT               # noqa: E402

RAW = os.path.join(ROOT, '.cache', 'raw')
WORK = os.path.join(ROOT, '.cache', 'work')
STATE = os.path.join(ROOT, 'state')
DATA = os.path.join(ROOT, 'data')


def fit_prior_coefs(finals, extra, fbs_ids):
    rows = []
    for y in sorted(finals.season.unique()):
        cur, prev = finals[finals.season == y], finals[finals.season == y - 1]
        if prev.empty or y < 2021:
            continue
        m = cur.merge(prev, on='team_id', suffixes=('', '_prev')).merge(
            extra[extra.season == y].drop(columns='season'), on='team_id', how='left')
        rows.append(m[m.team_id.isin(fbs_ids)])
    D = pd.concat(rows)
    D['ret_off'] = D.ret_off.fillna(D.ret_off.mean()) - 0.38
    D['qb_ret'] = D.qb_ret.astype(float).fillna(0.4) - 0.4
    D['portal'] = D.net_transfer_talent.fillna(0) / 500.0
    coefs = {}
    for m in METRICS:
        coefs[m] = {}
        for side in ['o', 'd']:
            y = D[f'{m}_{side}'].values - D.groupby('season')[f'{m}_{side}'].transform('mean').values
            X = np.c_[D[f'{m}_{side}_prev'].values, D.ret_off.values, D.qb_ret.values, D.portal.values]
            coefs[m][side] = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    return coefs


def main(skip_fetch=False):
    season = current_season()
    seasons = list(range(2019, season + 1))
    if not skip_fetch:
        fetch_seasons(RAW, seasons)
    build_all(RAW, WORK, seasons)
    ex = extra_table(RAW, WORK, list(range(2020, season + 2)))

    p = copy.deepcopy(DEFAULT)
    p['decay'] = 0.98
    p['lam'] = {k: 2.0 for k in p['lam']}
    p['rho'] = {k: 0.95 for k in p['rho']}
    g, F, TR, FIN = build(WORK, p, seasons=seasons, verbose=True)
    fbs = set(g[g.home_division == 'fbs'].home_id)
    p['prior_coef'] = fit_prior_coefs(FIN, ex, fbs)
    g, F, TR, FIN = build(WORK, p, seasons=seasons, verbose=True, extra=ex)
    F.to_parquet(f'{WORK}/feats.parquet')
    FIN.to_parquet(f'{WORK}/finals.parquet')

    gl, _ = load(WORK)
    try:
        import cfbd
        cfbd.refresh_venues(STATE)
    except Exception as e:
        print('venues refresh skipped:', e)
    situational.build(gl, RAW, WORK, venues_json=f'{STATE}/venues.json').to_parquet(f'{WORK}/situ.parquet')

    import train
    train.WORK, train.OUT = WORK, DATA
    train.main()
    import feature_tests
    feature_tests.main(f'{DATA}/tests.json')

    # preseason prior for the current season (and next, once a season is complete)
    os.makedirs(STATE, exist_ok=True)
    done = g[g.completed == True]
    for target in [season]:
        last = FIN[FIN.season == target - 1]
        if last.empty:
            continue
        pr = make_prior(last.drop(columns='season'), target, p, ex)
        env_last = float(done[(done.season == target - 1) & (done.home_division == 'fbs') & (done.away_division == 'fbs')]
                         .eval('home_points + away_points').mean())
        out = {'season': target, 'params': {k: v for k, v in p.items()}, 'env_last': env_last,
               'prior': {m: pr[m].round(5).values.tolist() for m in METRICS}}
        json.dump(out, open(f'{STATE}/prior_{target}.json', 'w'))
        print('saved prior for', target)

    # weather: save the list of games to collect; the separate weather step fetches them in
    # paced slices (see weather.py) so a slow or throttled weather service never blocks this job
    try:
        import weather
        weather.save_targets(WORK, STATE, DATA)
    except Exception as e:
        print('weather targets skipped:', e)


if __name__ == '__main__':
    main(skip_fetch='--skip-fetch' in sys.argv)
