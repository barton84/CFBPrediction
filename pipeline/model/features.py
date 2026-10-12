"""Walk every season week by week, build pregame ratings and per-game features."""
import numpy as np
import pandas as pd
from core import load, SeasonRater, METRICS, final_ratings, make_prior

DEFAULT = {
    'decay': 0.97,
    'lam': {'epa': 3.0, 'sr': 3.0, 'ypp': 3.0, 'pts': 3.0, 'pace': 4.0},
    'rho': {'epa': 0.6, 'sr': 0.6, 'ypp': 0.6, 'pts': 0.6, 'pace': 0.6},
    'league_mu': {'epa': 0.05, 'sr': 0.42, 'ypp': 5.8, 'pts': 28.0, 'pace': 140},
    'fcs_prior': {'epa': {'o': -0.20, 'd': 0.20}, 'sr': {'o': -0.06, 'd': 0.06}, 'ypp': {'o': -1.3, 'd': 1.3},
                  'pts': {'o': -12.0, 'd': 12.0}, 'pace': {'o': 0.0, 'd': 0.0}},
    'mkt_lam': 2.0, 'mkt_rho': 0.65, 'mkt_decay': 0.95,
}


def market_ratings(lines, teams, prior, lam, decay, wk):
    """Ridge fit of team power from posted spreads: -home_line = R_h - R_a + H*loc."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    d = lines[lines.wk < wk]
    pr = np.array([prior.get(t, -18.0) for t in teams])
    if len(d) < 10:
        return dict(zip(teams, pr)), 2.5
    rows = len(d)
    X = np.zeros((rows, n + 1))
    hi = d.home_id.map(idx).values
    ai = d.away_id.map(idx).values
    X[np.arange(rows), hi] = 1
    X[np.arange(rows), ai] = -1
    X[:, n] = (~d.neutral_site.values).astype(float)
    y = -d.home_line.values
    w = decay ** (wk - 1 - d.wk.values)
    A = X.T @ (X * w[:, None])
    b = X.T @ (w * y)
    P = np.full(n + 1, lam)
    P[n] = 1e-3
    A[np.diag_indices(n + 1)] += P
    b[:n] += lam * pr
    b[n] += 1e-3 * 2.5
    s = np.linalg.solve(A, b)
    return dict(zip(teams, s[:n])), s[n]


def build(work, params=DEFAULT, seasons=range(2019, 2027), verbose=True, extra=None, initial_prior=None, initial_mkt_prior=None):
    g, tg = load(work)
    finals = []
    prior = initial_prior or {}
    mkt_prior = initial_mkt_prior or {}
    feats = []
    team_ratings = []
    for y in seasons:
        gy = g[g.season == y]
        r = SeasonRater(g, tg, y, prior, params)
        lines = gy[gy.home_line.notna()][['wk', 'home_id', 'away_id', 'neutral_site', 'home_line']]
        for wk in sorted(gy.wk.unique()):
            rt = r.ratings(wk).set_index('team_id')
            mk, mk_h = market_ratings(lines, r.teams, mkt_prior, params['mkt_lam'], params['mkt_decay'], wk)
            rt['mkt'] = pd.Series(mk)
            games = gy[gy.wk == wk][['game_id', 'home_id', 'away_id']]
            games = games[games.home_id.isin(rt.index) & games.away_id.isin(rt.index)]
            if len(games):
                H = rt.loc[games.home_id.values].reset_index(drop=True)
                A = rt.loc[games.away_id.values].reset_index(drop=True)
                f = pd.DataFrame({'game_id': games.game_id.values, 'season': y, 'wk': wk,
                                  'home_id': games.home_id.values, 'away_id': games.away_id.values,
                                  'gp_h': H.gp.values, 'gp_a': A.gp.values, 'mkt_h': H.mkt.values, 'mkt_a': A.mkt.values})
                for m in METRICS:
                    f[f'{m}_ho'], f[f'{m}_hd'] = H[f'{m}_o'].values, H[f'{m}_d'].values
                    f[f'{m}_ao'], f[f'{m}_ad'] = A[f'{m}_o'].values, A[f'{m}_d'].values
                    f[f'{m}_mu'] = H[f'{m}_mu'].values
                feats.append(f)
            # keep a snapshot of ratings for the site
            snap = rt.reset_index()
            snap['season'], snap['wk'] = y, wk
            team_ratings.append(snap)
        fin = final_ratings(g, tg, y, prior, params)
        fin['season'] = y
        finals.append(fin)
        prior = make_prior(fin, y + 1, params, extra)
        # market prior for next season = regressed end-of-season market rating
        endwk = int(gy.wk.max()) + 1
        mk_end, _ = market_ratings(lines, r.teams, mkt_prior, params['mkt_lam'], params['mkt_decay'], endwk)
        mean_fbs = np.mean([v for t, v in mk_end.items() if t in set(gy[gy.home_division == 'fbs'].home_id)])
        mkt_prior = {t: params['mkt_rho'] * (v - mean_fbs) for t, v in mk_end.items()}
        if verbose:
            print('season', y, 'done', flush=True)
    F = pd.concat(feats, ignore_index=True)
    TR = pd.concat(team_ratings, ignore_index=True)
    return g, F, TR, pd.concat(finals, ignore_index=True)
