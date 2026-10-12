"""Blend component ratings into one predicted margin; leave-one-season-out backtest."""
import numpy as np
import pandas as pd

BASE = ['x_epa', 'x_sr', 'x_ypp', 'x_pts', 'x_mkt', 'x_fpi', 'loc']


def bucket(wk):
    return np.select([wk <= 3, wk <= 6], [0, 1], 2)


def design(d, feats):
    X = d[feats].fillna(0).values
    return np.c_[X, np.ones(len(d))]


def fit(d, feats, ridge=1.0):
    X = design(d, feats)
    y = d.margin.values
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[-1, -1] -= ridge
    return np.linalg.solve(A, X.T @ y)


def predict(d, feats, co):
    return design(d, feats) @ co


def loso(d, feats, test_seasons, train_from=2019, by_bucket=True):
    out = []
    coefs = {}
    for s in test_seasons:
        tr = d[(d.season != s) & (d.season >= train_from) & (d.season < 2026)]
        te = d[d.season == s].copy()
        te['pred'] = np.nan
        for b in ([0, 1, 2] if by_bucket else [None]):
            trb = tr if b is None else tr[bucket(tr.wk) == b]
            m = np.ones(len(te), bool) if b is None else (bucket(te.wk) == b)
            co = fit(trb, feats)
            te.loc[m, 'pred'] = predict(te[m], feats, co)
            coefs[(s, b)] = co
        out.append(te)
    return pd.concat(out), coefs
