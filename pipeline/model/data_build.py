"""Turn raw season files into the tables the model uses.

team_games.parquet : one row per offense per game (non-garbage efficiency + turnovers)
games.parquet      : schedule, scores, closing lines (ESPN), ESPN FPI game projection
starters.parquet   : the QB with the most dropbacks for each team in each game
returning.parquet  : returning offensive production and QB continuity by season
"""
import os
import numpy as np
import pandas as pd

PBP_COLS = ['game_id', 'season', 'week', 'seasonType', 'pos_team_id', 'def_pos_team_id', 'period', 'pos_score_diff_start',
            'rush', 'pass', 'EPA', 'EPA_success', 'statYardage', 'kneel_down', 'fumble_vec', 'int', 'fumble_lost',
            'homeTeamSpread', 'gameSpreadAvailable', 'passer_player_id', 'passer_player_name', 'rusher_player_id',
            'receiver_player_id']


def _garbage(df):
    d = df['pos_score_diff_start'].abs()
    q = df['period']
    return ((q == 2) & (d > 38)) | ((q == 3) & (d > 28)) | ((q >= 4) & (d > 22))


def _read_pbp(raw, y):
    f = f'{raw}/pbp_{y}.parquet'
    if not os.path.exists(f):
        return None
    import pyarrow.parquet as pq
    have = set(pq.read_schema(f).names)
    p = pd.read_parquet(f, columns=[c for c in PBP_COLS if c in have])
    p['game_id'] = pd.to_numeric(p.game_id).astype('int64')
    return p


def team_games(p, y):
    p = p.copy()
    p['EPA_success'] = p['EPA_success'].astype(float)
    p['scrim'] = ((p['rush'] == 1) | (p['pass'] == 1)) & (p['kneel_down'] != 1) & p['EPA'].notna()
    p['garb'] = _garbage(p)
    s = p[p.scrim]
    s_ng = s[~s['garb']]
    g = s_ng.groupby(['game_id', 'pos_team_id', 'def_pos_team_id'])
    a = g.agg(plays=('EPA', 'size'), epa=('EPA', 'mean'), sr=('EPA_success', 'mean'), ypp=('statYardage', 'mean'),
              expl=('EPA', lambda x: (x > 1.5).mean()))
    rp = s_ng[s_ng.rush == 1].groupby(['game_id', 'pos_team_id']).EPA.mean().rename('rush_epa')
    pp = s_ng[s_ng['pass'] == 1].groupby(['game_id', 'pos_team_id']).EPA.mean().rename('pass_epa')
    a = a.reset_index().merge(rp.reset_index(), how='left').merge(pp.reset_index(), how='left')
    allg = s.groupby(['game_id', 'pos_team_id']).agg(plays_all=('EPA', 'size'), fumbles=('fumble_vec', 'sum'),
                                                      fum_lost=('fumble_lost', 'sum'), ints=('int', 'sum')).reset_index()
    a = a.merge(allg, how='left')
    a['season'] = y
    a = a.rename(columns={'pos_team_id': 'team_id', 'def_pos_team_id': 'opp_id'})
    for c in ['team_id', 'opp_id']:
        a[c] = pd.to_numeric(a[c]).astype('int64')
    return a


def starters(p, y):
    q = p[(p['pass'] == 1) & p.passer_player_id.notna()]
    c = q.groupby(['game_id', 'pos_team_id', 'passer_player_id', 'passer_player_name']).agg(db=('EPA', 'size'), epa=('EPA', 'mean')).reset_index()
    c = c.sort_values('db', ascending=False).drop_duplicates(['game_id', 'pos_team_id'])
    c['season'] = y
    c = c.rename(columns={'pos_team_id': 'team_id', 'passer_player_id': 'qb_id', 'passer_player_name': 'qb_name', 'db': 'qb_db', 'epa': 'qb_epa'})
    c['team_id'] = pd.to_numeric(c.team_id).astype('int64')
    return c


def games(raw, y, p):
    s = pd.read_parquet(f'{raw}/cfb_schedules_{y}.parquet')
    s['game_id'] = pd.to_numeric(s.game_id).astype('int64')
    try:
        b = pd.read_parquet(f'{raw}/betting_{y}.parquet')[['game_id', 'home_team_spread', 'over_under', 'game_spread_available', 'odds_source']]
        b['game_id'] = pd.to_numeric(b.game_id).astype('int64')
        s = s.merge(b.drop_duplicates('game_id'), on='game_id', how='left')
    except Exception:
        s['home_team_spread'] = np.nan
        s['over_under'] = np.nan
    try:
        pi = pd.read_parquet(f'{raw}/power_index_{y}.parquet')
        pi['team_id'] = pd.to_numeric(pi.team_id).astype('int64')
        pi['game_id'] = pd.to_numeric(pi.game_id).astype('int64')
        pi = pi.drop_duplicates(['game_id', 'team_id'])
        s = s.merge(pi[['game_id', 'team_id', 'teampredptdiff']].rename(columns={'team_id': 'home_id', 'teampredptdiff': 'fpi_home_margin'}),
                    on=['game_id', 'home_id'], how='left')
    except Exception:
        s['fpi_home_margin'] = np.nan
    for c in ['home_id', 'away_id']:
        s[c] = pd.to_numeric(s[c]).astype('int64')
    if p is not None and 'homeTeamSpread' in p:
        sp = p[['game_id', 'homeTeamSpread', 'gameSpreadAvailable']].drop_duplicates('game_id')
        sp = sp[sp.gameSpreadAvailable == True]
        s = s.merge(sp[['game_id', 'homeTeamSpread']], on='game_id', how='left')
    else:
        s['homeTeamSpread'] = np.nan
    s['home_line'] = -s['homeTeamSpread']          # negative = home favored
    miss = s.home_line.isna()
    s.loc[miss, 'home_line'] = s.loc[miss, 'home_team_spread']
    return s


def returning(raw, seasons):
    """Share of last season's offensive touches by players seen in a team's first two games."""
    def touches(p):
        parts = []
        for c in ['passer_player_id', 'rusher_player_id', 'receiver_player_id']:
            x = p[['game_id', 'pos_team_id', 'week', 'seasonType', c]].dropna(subset=[c]).rename(columns={c: 'pid'})
            x['role'] = c.split('_')[0]
            parts.append(x)
        return pd.concat(parts)
    rows = []
    for y in seasons:
        prev_p, cur_p = _read_pbp(raw, y - 1), _read_pbp(raw, y)
        if prev_p is None or cur_p is None:
            continue
        prev, cur = touches(prev_p), touches(cur_p)
        first = cur[cur.seasonType == 2].sort_values('week')
        g2 = first.drop_duplicates(['pos_team_id', 'game_id']).groupby('pos_team_id').head(2)[['pos_team_id', 'game_id']]
        early = first.merge(g2, on=['pos_team_id', 'game_id'])
        present = set(zip(early.pos_team_id, early.pid))
        prev['ret'] = [(t, q) in present for t, q in zip(prev.pos_team_id, prev.pid)]
        ret = prev.groupby('pos_team_id').ret.mean()
        qb = prev[prev.role == 'passer'].groupby(['pos_team_id', 'pid']).size().reset_index(name='n').sort_values('n', ascending=False).drop_duplicates('pos_team_id')
        qb['qb_ret'] = [(t, q) in present for t, q in zip(qb.pos_team_id, qb.pid)]
        r = pd.DataFrame({'ret_off': ret}).join(qb.set_index('pos_team_id').qb_ret).reset_index().rename(columns={'pos_team_id': 'team_id'})
        r['season'] = y
        rows.append(r)
    out = pd.concat(rows) if rows else pd.DataFrame(columns=['team_id', 'ret_off', 'qb_ret', 'season'])
    out['team_id'] = pd.to_numeric(out.team_id).astype('int64')
    return out


def extra_table(raw, work, seasons):
    r = returning(raw, seasons)
    ps = []
    for y in seasons:
        f = f'{raw}/cfb_team_portal_{y}.parquet'
        if os.path.exists(f):
            ps.append(pd.read_parquet(f)[['season', 'team_id', 'net_transfer_talent']])
    p = pd.concat(ps) if ps else pd.DataFrame(columns=['season', 'team_id', 'net_transfer_talent'])
    p['team_id'] = pd.to_numeric(p.team_id).astype('int64')
    ex = r.merge(p, on=['season', 'team_id'], how='outer')
    ex.to_parquet(f'{work}/extra.parquet')
    return ex


def build_all(raw, work, seasons):
    os.makedirs(work, exist_ok=True)
    tg, gm, st = [], [], []
    for y in seasons:
        p = _read_pbp(raw, y)
        if p is not None:
            tg.append(team_games(p, y))
            st.append(starters(p, y))
        gm.append(games(raw, y, p))
        print('built', y, flush=True)
    pd.concat(tg, ignore_index=True).to_parquet(f'{work}/team_games.parquet')
    pd.concat(gm, ignore_index=True).to_parquet(f'{work}/games.parquet')
    pd.concat(st, ignore_index=True).to_parquet(f'{work}/starters.parquet')
