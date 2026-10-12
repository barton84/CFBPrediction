"""Fit the final spread and total models, run the leave-one-season-out backtest,
and write model.json + backtest.json for the site and the daily job."""
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from evaluate import frame, ats
from stack import bucket

WORK = sys.argv[1] if len(sys.argv) > 1 else '../work'
OUT = sys.argv[2] if len(sys.argv) > 2 else '../work/out'

SPREAD_F = ['x_sr', 'x_ypp', 'x_fpi', 'loc', 'fluck_diff']
SPREAD_FALLBACK = ['x_sr', 'x_ypp', 'x_epa', 'x_pts', 'loc', 'fluck_diff']   # when ESPN FPI is missing
TOTAL_F = ['xp_sum', 'x_pace', 'x_tot_eff', 'dome_f', 'env']
TEST_SPREAD = [2022, 2023, 2024, 2025]
TEST_TOTAL = [2023, 2024, 2025]          # closing totals are only in the data from 2023


def load_frame():
    d = frame(WORK)
    s = pd.read_parquet(f'{WORK}/situ.parquet')
    d = d.merge(s, on='game_id', how='left')
    d = d[d.fbs_both].copy()
    d['xp_sum'] = d.xp_home + d.xp_away
    d['dome_f'] = d.dome.fillna(False).astype(float)
    # scoring environment: average FBS total in earlier weeks of this season (80+ games), else last season's
    env = {}
    done = d[d.total.notna()]
    for y, x in done.groupby('season'):
        for wk in range(1, 22):
            pr = x[x.wk < wk]
            env[(y, wk)] = pr.total.mean() if len(pr) >= 80 else np.nan
    last = done.groupby('season').total.mean().to_dict()
    d['env'] = [env.get((y, w), np.nan) for y, w in zip(d.season, d.wk)]
    d['env'] = d.env.fillna(d.season.map(lambda y: last.get(y - 1, 53.0)))
    return d


def ols(X, y, ridge=1.0):
    X = np.c_[X, np.ones(len(X))]
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[-1, -1] -= ridge
    return np.linalg.solve(A, X.T @ y)


def apply(X, co):
    return np.c_[X, np.ones(len(X))] @ co


def fit_spread(tr, feats):
    out = {}
    for b in [0, 1, 2]:
        x = tr[bucket(tr.wk) == b].dropna(subset=feats)
        out[b] = ols(x[feats].values, x.margin.values)
    return out


def pred_spread(te, feats, cos):
    p = np.full(len(te), np.nan)
    bk = bucket(te.wk)
    X = te[feats].fillna(0).values
    for b, co in cos.items():
        m = bk == b
        p[m] = apply(X[m], co)
    return p


def fit_total(tr):
    x = tr.dropna(subset=TOTAL_F + ['total'])
    co = ols(x[TOTAL_F].values, x.total.values, ridge=0.1)
    med = float(np.median(x.total.values - apply(x[TOTAL_F].values, co)))
    return co, med


def logit_slope(edge, win):
    f = lambda p: -np.sum(win * np.log(1 / (1 + np.exp(-p[0] * edge))) + (1 - win) * np.log(1 - 1 / (1 + np.exp(-p[0] * edge))))
    return float(minimize(f, [0.02]).x[0])


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    d = load_frame()
    done = d[d.margin.notna() & (d.season <= 2025)]
    lined = done[done.home_line.notna() & done.home_line.abs().lt(70)]

    # ---------- spread backtest (leave one season out) ----------
    bt = []
    for s in TEST_SPREAD:
        tr = lined[lined.season != s]
        te = lined[lined.season == s].copy()
        te['pred'] = pred_spread(te, SPREAD_F, fit_spread(tr, SPREAD_F))
        miss = te.x_fpi.isna()
        if miss.any():
            te.loc[miss, 'pred'] = pred_spread(te[miss], SPREAD_FALLBACK, fit_spread(tr, SPREAD_FALLBACK))
        bt.append(te)
    bt = pd.concat(bt)
    a = ats(bt, 'pred')
    # calibration of cover chance, each season scored with a slope fit on the others
    slopes = {s: logit_slope(a[a.season != s].abs_edge.values, a[a.season != s].win.values.astype(float)) for s in TEST_SPREAD}
    a['p_cover'] = [1 / (1 + np.exp(-slopes[s] * e)) for s, e in zip(a.season, a.abs_edge)]

    # ---------- totals backtest ----------
    tdone = d[d.total.notna() & (d.season.between(2020, 2025))]
    tb = []
    for s in TEST_TOTAL:
        co, med = fit_total(tdone[tdone.season != s])
        te = tdone[tdone.season == s].copy()
        te['pt'] = apply(te[TOTAL_F].fillna(te[TOTAL_F].mean()).values, co) + med
        tb.append(te)
    tb = pd.concat(tb)
    tb = tb[tb.over_under.between(25, 95)].copy()
    tb['edge'] = tb.pt - tb.over_under
    tb['res'] = tb.total - tb.over_under
    tb = tb[tb.res != 0]
    tb['win'] = np.sign(tb.edge) == np.sign(tb.res)
    tb['abs_edge'] = tb.edge.abs()
    tslopes = {s: logit_slope(tb[tb.season != s].abs_edge.values, tb[tb.season != s].win.values.astype(float)) for s in TEST_TOTAL}
    tb['p_hit'] = [1 / (1 + np.exp(-tslopes[s] * e)) for s, e in zip(tb.season, tb.abs_edge)]

    # ---------- final fits on every completed season ----------
    spread_cos = fit_spread(lined, SPREAD_F)
    fallback_cos = fit_spread(lined, SPREAD_FALLBACK)
    # QB change worth: refit with the realized-starter flag to size the manual override
    qb_feats = SPREAD_F + ['qbchg_edge']
    qx = lined.dropna(subset=qb_feats)
    qb_co = ols(qx[qb_feats].values, qx.margin.values)
    qb_points = float(qb_co[len(SPREAD_F)])
    tot_co, tot_med = fit_total(tdone)
    tq = tdone.dropna(subset=TOTAL_F + ['qbchg_h', 'qbchg_a'])
    tq_co = ols(np.c_[tq[TOTAL_F].values, tq.qbchg_h.values + tq.qbchg_a.values], tq.total.values, ridge=0.1)
    qb_total_points = float(tq_co[len(TOTAL_F)])
    spread_slope = logit_slope(a.abs_edge.values, a.win.values.astype(float))
    total_slope = logit_slope(tb.abs_edge.values, tb.win.values.astype(float))

    # residual distribution of margin around the model, for the key-number ladder
    resid = (lined.margin - pred_spread(lined, SPREAD_F, spread_cos)).round().clip(-60, 60)
    margins = done.margin.abs()
    key = {int(k): float((margins == k).mean()) for k in range(1, 29)}
    # how often a game lands exactly on an integer line, among games priced near that number
    push = {}
    for k in range(0, 41):
        near = lined[(lined.home_line.abs() - k).abs() <= 1.5]
        if len(near) >= 40:
            fav_margin = np.where(near.home_line <= 0, near.margin, -near.margin)
            push[k] = round(float((fav_margin == k).mean()), 4)
    # chance the favorite covers at each line, used for the alt-line ladder
    

    model = {
        'spread': {'features': SPREAD_F, 'coef': {str(b): c.tolist() for b, c in spread_cos.items()},
                   'fallback_features': SPREAD_FALLBACK, 'fallback_coef': {str(b): c.tolist() for b, c in fallback_cos.items()},
                   'buckets': 'weeks 1-3, 4-6, 7+ (bowls use 7+)', 'cover_slope': spread_slope, 'qb_change_points': qb_points,
                   'resid_sd': float(resid.std())},
        'total': {'features': TOTAL_F, 'coef': tot_co.tolist(), 'median_offset': tot_med, 'hit_slope': total_slope,
                  'qb_change_points': qb_total_points,
                  'weather': {'fitted': False, 'wind_per_mph_over_10': -0.25, 'rain': -1.0, 'snow': -2.5, 'cold_below_32': -1.0,
                              'note': 'Starting values from published research. Refit automatically once the weather backfill has run.'}},
        'key_margins': key,
        'push_rate': push,
        'trained_on': 'FBS vs FBS games 2019-2025 (spreads), 2020-2025 (totals)',
    }
    json.dump(model, open(f'{OUT}/model.json', 'w'), indent=1)

    # ---------- backtest summary for the site ----------
    def rec(x):
        return {'n': int(len(x)), 'w': int(x.win.sum()), 'l': int((~x.win).sum()), 'pct': round(float(x.win.mean()), 4) if len(x) else None}
    summary = {'spread': {'all': rec(a)}, 'total': {'all': rec(tb)}}
    for t in [3, 5, 7]:
        summary['spread'][f'{t}+'] = rec(a[a.abs_edge >= t])
        summary['total'][f'{t}+'] = rec(tb[tb.abs_edge >= t])
    summary['spread']['by_season'] = {int(s): {'all': rec(x), '5+': rec(x[x.abs_edge >= 5]), '7+': rec(x[x.abs_edge >= 7])} for s, x in a.groupby('season')}
    summary['total']['by_season'] = {int(s): {'all': rec(x), '5+': rec(x[x.abs_edge >= 5]), '7+': rec(x[x.abs_edge >= 7])} for s, x in tb.groupby('season')}
    wb = pd.cut(a.wk, [0, 3, 6, 9, 13, 30], labels=['Weeks 1-3', 'Weeks 4-6', 'Weeks 7-9', 'Weeks 10-13', 'Conf. champs & bowls'])
    summary['spread']['by_week'] = {str(k): {'all': rec(x), '5+': rec(x[x.abs_edge >= 5])} for k, x in a.groupby(wb, observed=True)}
    summary['mae'] = {'spread_model': round(float((bt.margin - bt.pred).abs().mean()), 2),
                      'spread_line': round(float((bt.margin + bt.home_line).abs().mean()), 2),
                      'total_model': round(float((tb.total - tb.pt).abs().mean()), 2),
                      'total_line': round(float((tb.total - tb.over_under).abs().mean()), 2)}

    def calib(x, pcol):
        bins = [0, 0.51, 0.52, 0.53, 0.55, 1]
        lab = ['Under 51%', '51 to 52%', '52 to 53%', '53 to 55%', '55% and up']
        c = pd.cut(x[pcol], bins, labels=lab)
        return [{'bucket': str(k), 'n': int(len(v)), 'shown': round(float(v[pcol].mean()), 3), 'actual': round(float(v.win.mean()), 3)}
                for k, v in x.groupby(c, observed=True)]
    summary['calibration'] = {'spread': calib(a, 'p_cover'), 'total': calib(tb, 'p_hit')}
    summary['coverage'] = {'spread_games': int(len(a)), 'total_games': int(len(tb)),
                           'spread_seasons': TEST_SPREAD, 'total_seasons': TEST_TOTAL}
    json.dump(summary, open(f'{OUT}/backtest.json', 'w'), indent=1)

    # per-game backtest rows (compact) so the site can show any past week
    keep = a[['season', 'wk', 'home_team', 'away_team', 'home_line', 'pred', 'margin', 'edge', 'abs_edge', 'win']].copy()
    keep['pred'] = keep.pred.round(1)
    keep['edge'] = keep.edge.round(1)
    keep.to_json(f'{OUT}/backtest_games.json', orient='records')
    print(json.dumps(summary['spread'], indent=0)[:1500])
    print(json.dumps(summary['total'], indent=0)[:800])
    print(summary['mae'], 'qb pts', qb_points, 'qb total', qb_total_points, 'slopes', spread_slope, total_slope)


if __name__ == '__main__':
    main()
