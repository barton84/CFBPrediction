"""Reproducible record of what was tested: each candidate input is added to (or removed
from) the final spread model and rescored leave-one-season-out on 2022-2025. Also checks
whether situational factors predict the result *against the closing line*.
Writes data/tests.json for the Method & Testing tab."""
import json
import numpy as np
import pandas as pd
import train
from evaluate import ats

CANDIDATES = [
    ('x_epa', 'EPA per play (opponent-adjusted)'),
    ('x_pts', 'Points scored and allowed (opponent-adjusted)'),
    ('x_mkt', 'Market power rating from earlier closing lines'),
    ('trav_diff', 'Travel distance'),
    ('tz_abs_a', 'Time zones crossed'),
    ('alt_edge', 'Altitude (visitor from low elevation)'),
    ('body_clock', 'West-coast team in an early eastern kickoff'),
    ('rest_diff', 'Rest days'),
    ('bye_edge', 'Coming off a bye'),
    ('short_edge', 'Short week'),
    ('ko_night_loc', 'Night game at home'),
    ('weekday', 'Weekday game'),
    ('tom_diff', 'Turnover margin to date'),
    ('qbprev_edge', 'QB change known before kickoff (last start was not the main QB)'),
    ('qbchg_edge', 'QB change, actual starter (needs late news)'),
]
REMOVE = [('x_fpi', 'ESPN FPI'), ('x_sr', 'Success rate'), ('x_ypp', 'Yards per play'), ('fluck_diff', 'Fumble luck')]


def score(L, feats):
    out = []
    for s in train.TEST_SPREAD:
        tr, te = L[L.season != s], L[L.season == s].copy()
        te['pred'] = train.pred_spread(te, feats, train.fit_spread(tr, feats))
        out.append(te)
    r = pd.concat(out)
    a = ats(r, 'pred')
    return {'mae': round(float((r.margin - r.pred).abs().mean()), 3), 'ats': round(float(a.win.mean()), 4),
            'ats5': round(float(a[a.abs_edge >= 5].win.mean()), 4), 'n5': int((a.abs_edge >= 5).sum()),
            'ats7': round(float(a[a.abs_edge >= 7].win.mean()), 4), 'n7': int((a.abs_edge >= 7).sum())}


def main(out_path):
    d = train.load_frame()
    d['ko_night_loc'] = d.ko_night * d['loc']
    L = d[d.margin.notna() & d.home_line.notna() & d.home_line.abs().lt(70) & (d.season <= 2025) & d.x_fpi.notna()].copy()
    base = train.SPREAD_F
    res = {'final': {'features': base, **score(L, base)}, 'added': [], 'removed': []}
    for f, label in CANDIDATES:
        res['added'].append({'feature': f, 'label': label, **score(L, base + [f])})
        print('tested', f, flush=True)
    for f, label in REMOVE:
        res['removed'].append({'feature': f, 'label': label, **score(L, [x for x in base if x != f])})
    # does each situational factor predict the cover margin vs the closing line?
    L['cover'] = L.margin + L.home_line
    situ = [c for c, _ in CANDIDATES if c not in ('x_epa', 'x_pts', 'x_mkt')]
    X = np.c_[L[situ].fillna(0).values, np.ones(len(L))]
    b = np.linalg.lstsq(X, L.cover.values, rcond=None)[0]
    e = L.cover.values - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * e.var())
    res['vs_line'] = [{'feature': f, 'label': dict(CANDIDATES)[f], 'pts': round(float(c), 2), 't': round(float(c / s), 1)}
                      for f, c, s in zip(situ, b, se)]
    res['n_games'] = int(len(L))
    json.dump(res, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    import sys
    train.WORK = sys.argv[1]
    main(sys.argv[2])
