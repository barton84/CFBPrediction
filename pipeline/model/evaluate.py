"""Join features to results and lines; derive model inputs."""
import numpy as np
import pandas as pd


def frame(work, F=None):
    g = pd.read_parquet(f'{work}/games.parquet')
    if F is None:
        F = pd.read_parquet(f'{work}/feats.parquet')
    cols = ['game_id', 'home_division', 'away_division', 'neutral_site', 'home_points', 'away_points', 'home_line',
            'over_under', 'fpi_home_margin', 'start_date', 'venue', 'venue_id', 'home_team', 'away_team',
            'conference_game', 'completed', 'season_type', 'home_conference', 'away_conference']
    d = F.merge(g[cols], on='game_id', how='left')
    d['neutral_site'] = d.neutral_site.fillna(False).astype(bool)
    d['fbs_both'] = (d.home_division == 'fbs') & (d.away_division == 'fbs')
    d['margin'] = d.home_points - d.away_points
    d['total'] = d.home_points + d.away_points
    d['loc'] = (~d.neutral_site).astype(float)
    for m in ['epa', 'sr', 'ypp', 'pts']:
        # home offense vs away defense, minus away offense vs home defense
        d[f'x_{m}'] = (d[f'{m}_ho'] + d[f'{m}_ad']) - (d[f'{m}_ao'] + d[f'{m}_hd'])
    d['x_mkt'] = d.mkt_h - d.mkt_a
    d['x_fpi'] = d.fpi_home_margin
    # expected points for each side and expected pace, for totals
    d['xp_home'] = d.pts_mu + d.pts_ho + d.pts_ad
    d['xp_away'] = d.pts_mu + d.pts_ao + d.pts_hd
    d['x_pace'] = d.pace_mu + d.pace_ho + d.pace_ao + d.pace_hd + d.pace_ad
    d['x_tot_eff'] = (d.epa_ho + d.epa_ad) + (d.epa_ao + d.epa_hd)
    return d


def ats(d, pred_col, line_col='home_line', res_col='margin'):
    """pred = predicted home margin. Bet home if pred > -line."""
    x = d[d[line_col].notna() & d[res_col].notna()].copy()
    x['edge'] = x[pred_col] + x[line_col]           # >0 model likes home
    x['cover'] = x[res_col] + x[line_col]           # >0 home covered
    x = x[x.cover != 0]
    x['win'] = np.sign(x.edge) == np.sign(x.cover)
    x['abs_edge'] = x.edge.abs()
    return x
