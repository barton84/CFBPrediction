"""Core of the CFB spread/total model.

Ratings are opponent-adjusted with weighted ridge regression that shrinks each team
toward a preseason prior. They are recomputed before every week using only games
already played, so every backtest number is something you could have known pregame.
"""
import numpy as np
import pandas as pd

METRICS = ['epa', 'sr', 'ypp', 'pts', 'pace']   # pace = plays per game (both teams)


def load(work):
    g = pd.read_parquet(f'{work}/games.parquet')
    g = g[g.season_type.isin(['regular', 'postseason'])].copy()
    regmax = g[g.season_type == 'regular'].groupby('season').week.max()
    g['wk'] = np.where(g.season_type == 'postseason', g.season.map(regmax) + 1, g.week).astype(int)
    g['neutral_site'] = g.neutral_site.fillna(False).astype(bool)
    g['margin'] = g.home_points - g.away_points
    g['total'] = g.home_points + g.away_points
    g['fbs_both'] = (g.home_division == 'fbs') & (g.away_division == 'fbs')
    tg = pd.read_parquet(f'{work}/team_games.parquet')
    tg = tg.merge(g[['game_id', 'wk', 'home_id', 'away_id', 'neutral_site', 'home_points', 'away_points']], on='game_id')
    tg['loc'] = np.where(tg.neutral_site, 0, np.where(tg.team_id == tg.home_id, 1, -1))
    tg['pts'] = np.where(tg.team_id == tg.home_id, tg.home_points, tg.away_points)
    # pace: total scrimmage plays in game, assigned to both teams
    tot = tg.groupby('game_id').plays_all.sum().rename('pace')
    tg = tg.merge(tot, on='game_id')
    return g, tg


def ridge_ratings(obs, teams, prior_o, prior_d, lam, w):
    """obs columns: o (team idx), d (opp idx), y, loc. Solve y = mu + O[o] + D[d] + h*loc.
    Ridge pulls O, D toward priors (deviations from league mean)."""
    n = len(teams)
    k = 2 * n + 2
    # sparse normal equations: each row has 1 at o, 1 at n+d, 1 at mu, loc at h
    o = obs.o.values
    dd = n + obs.d.values
    mu = np.full(len(obs), 2 * n)
    hh = np.full(len(obs), 2 * n + 1)
    loc = obs['loc'].values.astype(float)
    y = obs.y.values
    cols = [o, dd, mu, hh]
    vals = [np.ones(len(obs)), np.ones(len(obs)), np.ones(len(obs)), loc]
    A = np.zeros((k, k))
    b = np.zeros(k)
    for ci, vi in zip(cols, vals):
        np.add.at(b, ci, w * vi * y)
        for cj, vj in zip(cols, vals):
            np.add.at(A, (ci, cj), w * vi * vj)
    P = np.zeros(k)
    P[:2 * n] = lam
    P[2 * n:] = 1e-6
    pri = np.concatenate([prior_o, prior_d, [0, 0]])
    A[np.diag_indices(k)] += P
    b += P * pri
    sol = np.linalg.solve(A, b)
    return sol[:n], sol[n:2 * n], sol[2 * n], sol[2 * n + 1]


class SeasonRater:
    """Produces pre-week ratings for one season."""

    def __init__(self, g, tg, season, prior, params):
        self.g = g[g.season == season]
        self.tg = tg[tg.season == season]
        self.season = season
        self.prior = prior            # dict metric -> DataFrame(team_id, o, d) deviations
        self.p = params
        teams = set(self.g.home_id) | set(self.g.away_id)
        self.teams = np.array(sorted(teams))
        self.idx = {t: i for i, t in enumerate(self.teams)}

    def prior_vec(self, m, side):
        pr = self.prior.get(m)
        v = np.zeros(len(self.teams))
        if pr is None:
            return v
        s = pr.set_index('team_id')[side]
        for t, i in self.idx.items():
            v[i] = s.get(t, self.p['fcs_prior'][m][side])
        return v

    def ratings(self, wk):
        p = self.p
        tg = self.tg[self.tg.wk < wk]
        out = pd.DataFrame({'team_id': self.teams})
        out['gp'] = out.team_id.map(tg.groupby('team_id').size()).fillna(0).astype(int)
        for m in METRICS:
            d = tg[['team_id', 'opp_id', 'wk', 'loc', m]].dropna()
            if m == 'pace':
                d = d.drop_duplicates(['wk', 'team_id'])
            o_pr, d_pr = self.prior_vec(m, 'o'), self.prior_vec(m, 'd')
            if len(d) < 20:
                out[f'{m}_o'] = o_pr
                out[f'{m}_d'] = d_pr
                out[f'{m}_mu'] = p['league_mu'][m]
                continue
            obs = pd.DataFrame({'o': d.team_id.map(self.idx).values, 'd': d.opp_id.map(self.idx).values,
                                'y': d[m].values, 'loc': d['loc'].values})
            ok = obs.o.notna() & obs.d.notna()
            obs = obs[ok].astype({'o': int, 'd': int})
            w = p['decay'] ** (wk - 1 - d.wk.values[ok.values])
            lam = p['lam'][m]
            O, D, mu, h = ridge_ratings(obs, self.teams, o_pr, d_pr, lam, w)
            out[f'{m}_o'] = O
            out[f'{m}_d'] = D
            out[f'{m}_mu'] = mu
        return out


def final_ratings(g, tg, season, prior, params):
    """End-of-season ratings (all games including bowls)."""
    r = SeasonRater(g, tg, season, prior, params)
    wk = int(r.g.wk.max()) + 1
    return r.ratings(wk)


def make_prior(final, nxt_season, params, extra=None):
    """Next season's prior. Default: carry-over factor rho. If params has 'prior_coef'
    (fitted on past seasons), use last season's rating plus returning production,
    QB continuity and transfer talent."""
    pc = params.get('prior_coef')
    out = {}
    f = final.copy()
    if extra is not None:
        ex = extra[extra.season == nxt_season].set_index('team_id')
        f['ret_off'] = f.team_id.map(ex.ret_off) - 0.38
        f['qb_ret'] = f.team_id.map(ex.qb_ret).astype(float) - 0.4
        f['portal'] = f.team_id.map(ex.net_transfer_talent) / 500.0
        for c in ['ret_off', 'qb_ret', 'portal']:
            f[c] = f[c].fillna(0)
    for m in METRICS:
        o, d = f[f'{m}_o'].values, f[f'{m}_d'].values
        if pc is None or extra is None:
            rho = params['rho'][m]
            po, pd_ = rho * o, rho * d
        else:
            co, cd = pc[m]['o'], pc[m]['d']
            X = np.c_[f[['ret_off', 'qb_ret', 'portal']].values]
            po = co[0] * o + X @ np.array(co[1:])
            pd_ = cd[0] * d + X @ np.array(cd[1:])
        out[m] = pd.DataFrame({'team_id': f.team_id.values, 'o': po, 'd': pd_})
    return out
