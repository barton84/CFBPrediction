"""Pregame situational features: travel, time zones, altitude, rest, kickoff window,
QB changes and season-to-date turnover luck. Everything uses only prior games."""
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

TZ_OFF = {'America/New_York': -5, 'America/Detroit': -5, 'America/Indiana/Indianapolis': -5, 'America/Kentucky/Louisville': -5,
          'America/Chicago': -6, 'America/Denver': -7, 'America/Boise': -7, 'America/Phoenix': -7,
          'America/Los_Angeles': -8, 'Pacific/Honolulu': -10}


def team_info(raw):
    parts = []
    for y in [2026, 2024, 2019]:
        try:
            parts.append(pd.read_parquet(f'{raw}/team_info_{y}.parquet'))
        except Exception:
            pass
    t = pd.concat(parts).drop_duplicates('team_id')
    t['team_id'] = t.team_id.astype('int64')
    for c in ['latitude', 'longitude', 'elevation', 'venue_id']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t['dome'] = t.dome.fillna(False).astype(bool)
    return t[['team_id', 'school', 'abbreviation', 'color', 'alt_color', 'logo', 'venue_id', 'latitude', 'longitude',
              'elevation', 'timezone', 'dome', 'conference', 'classification', 'city', 'state', 'venue_name', 'grass']]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def venue_table(ti, venues_json=None):
    v = ti.dropna(subset=['venue_id']).drop_duplicates('venue_id')[['venue_id', 'latitude', 'longitude', 'elevation', 'timezone', 'dome']]
    if venues_json:
        import json, os
        if os.path.exists(venues_json):
            extra = pd.DataFrame(json.load(open(venues_json)))
            if len(extra):
                extra['venue_id'] = pd.to_numeric(extra.venue_id, errors='coerce')
                for c in ['latitude', 'longitude', 'elevation']:
                    extra[c] = pd.to_numeric(extra[c], errors='coerce')
                extra['dome'] = extra.dome.fillna(False).astype(bool)
                # CFBD wins for dome flags and fills venues the team table does not have
                v = pd.concat([extra[['venue_id', 'latitude', 'longitude', 'elevation', 'timezone', 'dome']], v]).drop_duplicates('venue_id')
    return v.set_index('venue_id')


def build(g, raw, work, venues_json=None):
    ti = team_info(raw).set_index('team_id')
    venue_loc = venue_table(ti.reset_index(), venues_json)
    d = g[['game_id', 'season', 'wk', 'start_date', 'start_time_tbd', 'home_id', 'away_id', 'neutral_site', 'venue_id']].copy()
    # game location: the venue if it is some team's home stadium, else the home team's stadium when not neutral
    vl = d.venue_id.map(venue_loc.latitude)
    vo = d.venue_id.map(venue_loc.longitude)
    hl, ho = d.home_id.map(ti.latitude), d.home_id.map(ti.longitude)
    d['g_lat'] = np.where(vl.notna(), vl, np.where(~d.neutral_site, hl, np.nan))
    d['g_lon'] = np.where(vo.notna(), vo, np.where(~d.neutral_site, ho, np.nan))
    d['g_elev'] = d.venue_id.map(venue_loc.elevation).fillna(d.home_id.map(ti.elevation).where(~d.neutral_site))
    d['g_tz'] = d.venue_id.map(venue_loc.timezone).fillna(d.home_id.map(ti.timezone).where(~d.neutral_site))
    d['dome'] = d.venue_id.map(venue_loc.dome).fillna(False).astype(bool)
    d['g_known'] = d.g_lat.notna()
    for s, col in [('h', 'home_id'), ('a', 'away_id')]:
        d[f'trav_{s}'] = haversine(d[col].map(ti.latitude), d[col].map(ti.longitude), d.g_lat, d.g_lon)
        tz_team = d[col].map(ti.timezone).map(TZ_OFF)
        d[f'tz_{s}'] = (d.g_tz.map(TZ_OFF) - tz_team)          # + = traveling east
        d[f'elev_{s}'] = d[col].map(ti.elevation)
    d['trav_diff'] = (d.trav_a.fillna(0) - d.trav_h.fillna(0)) / 1000.0     # thousands of km, + favors home
    d['tz_abs_a'] = d.tz_a.abs().fillna(0) - d.tz_h.abs().fillna(0)
    # altitude: game above ~1,500 m and visiting team lives below 600 m
    d['alt_edge'] = (((d.g_elev > 1500) & (d.elev_a < 600)).astype(float) - ((d.g_elev > 1500) & (d.elev_h < 600)).astype(float))
    # kickoff windows in Eastern time
    ts = pd.to_datetime(d.start_date, utc=True, format='ISO8601')
    et = ts.dt.tz_convert(ZoneInfo('America/New_York'))
    d['ko_et_hour'] = et.dt.hour + et.dt.minute / 60
    d.loc[d.start_time_tbd == True, 'ko_et_hour'] = np.nan
    d['ko_night'] = (d.ko_et_hour >= 19).astype(float)
    d['ko_early'] = (d.ko_et_hour < 13).astype(float)
    d['weekday'] = (~et.dt.dayofweek.isin([5])).astype(float)   # not Saturday
    # west-coast team playing an early eastern kickoff
    d['body_clock'] = (((d.tz_a >= 2) & (d.ko_et_hour < 13)).astype(float) - ((d.tz_h >= 2) & (d.ko_et_hour < 13)).astype(float))
    # rest days
    long = pd.concat([d[['game_id', 'season', 'start_date', 'home_id']].rename(columns={'home_id': 'team_id'}),
                      d[['game_id', 'season', 'start_date', 'away_id']].rename(columns={'away_id': 'team_id'})])
    long['t'] = pd.to_datetime(long.start_date, utc=True, format='ISO8601')
    long = long.sort_values('t')
    long['rest'] = long.groupby(['season', 'team_id']).t.diff().dt.total_seconds() / 86400
    rest = long.set_index(['game_id', 'team_id']).rest
    d['rest_h'] = [rest.get((gid, t), np.nan) for gid, t in zip(d.game_id, d.home_id)]
    d['rest_a'] = [rest.get((gid, t), np.nan) for gid, t in zip(d.game_id, d.away_id)]
    rh, ra = d.rest_h.fillna(14).clip(3, 15), d.rest_a.fillna(14).clip(3, 15)
    d['rest_diff'] = rh - ra
    d['bye_edge'] = (rh >= 12).astype(float) - (ra >= 12).astype(float)
    d['short_edge'] = (rh <= 5).astype(float) - (ra <= 5).astype(float)

    # QB change and turnover luck from season-to-date play-by-play
    st = pd.read_parquet(f'{work}/starters.parquet')
    tg = pd.read_parquet(f'{work}/team_games.parquet')
    sched = d[['game_id', 'season', 'start_date']].copy()
    st = st.merge(sched, on=['game_id', 'season'])
    st['t'] = pd.to_datetime(st.start_date, utc=True, format='ISO8601')
    st = st.sort_values('t')
    qb_flag, qb_prev = {}, {}
    for (s, team), x in st.groupby(['season', 'team_id']):
        counts = {}
        last = None
        for gid, qb, db in zip(x.game_id, x.qb_id, x.qb_db):
            if sum(counts.values()) >= 60:      # need a real sample first
                primary = max(counts, key=counts.get)
                qb_flag[(gid, team)] = float(qb != primary)
                # knowable before kickoff: last game's starter is not the season's main QB
                qb_prev[(gid, team)] = float(last is not None and last != primary)
            counts[qb] = counts.get(qb, 0) + db
            last = qb
    d['qbchg_h'] = [qb_flag.get((gid, t), 0.0) for gid, t in zip(d.game_id, d.home_id)]
    d['qbchg_a'] = [qb_flag.get((gid, t), 0.0) for gid, t in zip(d.game_id, d.away_id)]
    d['qbchg_edge'] = d.qbchg_a - d.qbchg_h
    d['qbprev_h'] = [qb_prev.get((gid, t), 0.0) for gid, t in zip(d.game_id, d.home_id)]
    d['qbprev_a'] = [qb_prev.get((gid, t), 0.0) for gid, t in zip(d.game_id, d.away_id)]
    d['qbprev_edge'] = d.qbprev_a - d.qbprev_h

    # turnovers: committed by team in a game = ints + fumbles lost (non-garbage + garbage)
    tg = tg.merge(sched, on=['game_id', 'season'])
    tg['t'] = pd.to_datetime(tg.start_date, utc=True, format='ISO8601')
    tg['to'] = tg.ints.fillna(0) + tg.fum_lost.fillna(0)
    opp = tg[['game_id', 'team_id', 'to', 'fumbles', 'fum_lost']].rename(columns={'team_id': 'opp_id', 'to': 'to_forced', 'fumbles': 'opp_fum', 'fum_lost': 'opp_fum_lost'})
    tg = tg.merge(opp, on=['game_id', 'opp_id'], how='left')
    tg = tg.sort_values('t')
    grp = tg.groupby(['season', 'team_id'])
    for c in ['to', 'to_forced', 'fumbles', 'fum_lost', 'opp_fum', 'opp_fum_lost']:
        tg[f'c_{c}'] = grp[c].cumsum() - tg[c]
    tg['c_n'] = grp.cumcount()
    tg['to_margin_pg'] = (tg.c_to_forced - tg.c_to) / tg.c_n.replace(0, np.nan)
    # fumble recovery luck: share of all fumbles in your games that you recovered, vs 50%
    all_f = tg.c_fumbles + tg.c_opp_fum
    recovered = (tg.c_fumbles - tg.c_fum_lost) + tg.c_opp_fum_lost
    tg['fum_luck'] = (recovered / all_f.replace(0, np.nan) - 0.5) * (all_f.clip(upper=30) / 30)
    tl = tg.set_index(['game_id', 'team_id'])
    for s, col in [('h', 'home_id'), ('a', 'away_id')]:
        key = list(zip(d.game_id, d[col]))
        d[f'tom_{s}'] = tl.to_margin_pg.reindex(key).values
        d[f'fl_{s}'] = tl.fum_luck.reindex(key).values
    d['tom_diff'] = d.tom_h.fillna(0) - d.tom_a.fillna(0)
    d['fluck_diff'] = d.fl_h.fillna(0) - d.fl_a.fillna(0)
    keep = ['game_id', 'trav_diff', 'tz_abs_a', 'alt_edge', 'ko_et_hour', 'ko_night', 'ko_early', 'weekday', 'body_clock',
            'rest_diff', 'bye_edge', 'short_edge', 'qbchg_h', 'qbchg_a', 'qbchg_edge', 'qbprev_h', 'qbprev_a', 'qbprev_edge', 'tom_diff', 'fluck_diff', 'dome',
            'g_lat', 'g_lon', 'g_elev', 'g_tz', 'trav_h', 'trav_a', 'rest_h', 'rest_a', 'g_known']
    return d[keep]


if __name__ == '__main__':
    import sys
    from core import load
    g, _ = load(sys.argv[2])
    s = build(g, sys.argv[1], sys.argv[2])
    s.to_parquet(f'{sys.argv[2]}/situ.parquet')
    print(s.describe().T[['count', 'mean', 'std']])
