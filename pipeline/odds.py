"""DraftKings lines from the-odds-api.com (needs ODDS_API_KEY).

One call pulls every upcoming college game with spreads and totals and costs 2 credits
(2 markets x 1 region). The daily job makes one call, so a month is about 60-70 credits,
well inside the free 500.
"""
import difflib
import json
import os
import re
import unicodedata
import urllib.request
import pandas as pd

URL = ('https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/?apiKey={key}'
       '&regions=us&markets=spreads,totals&bookmakers=draftkings&oddsFormat=american&dateFormat=iso')

ALIASES = {
    # Odds API name fragment -> ESPN school name, for the ones fuzzy matching gets wrong
    'miami hurricanes': 'miami', 'miami (oh) redhawks': 'miami (oh)', 'massachusetts minutemen': 'umass',
    'southern mississippi golden eagles': 'southern miss', 'ul monroe warhawks': 'ul monroe',
    'louisiana monroe warhawks': 'ul monroe', 'appalachian state mountaineers': 'app state',
    'connecticut huskies': 'uconn', 'florida international panthers': 'florida international',
    'texas el paso miners': 'utep', 'texas san antonio roadrunners': 'utsa', 'central florida knights': 'ucf',
    'louisiana ragin cajuns': 'louisiana', 'sam houston state bearkats': 'sam houston', 'sam houston bearkats': 'sam houston',
    'hawaii rainbow warriors': 'hawaii', 'san jose state spartans': 'san jose state', 'nc state wolfpack': 'nc state',
    'north carolina state wolfpack': 'nc state', 'ole miss rebels': 'ole miss', 'mississippi rebels': 'ole miss',
    'pittsburgh panthers': 'pittsburgh', 'usc trojans': 'usc', 'southern california trojans': 'usc',
    'lsu tigers': 'lsu', 'byu cougars': 'byu', 'smu mustangs': 'smu', 'tcu horned frogs': 'tcu', 'uab blazers': 'uab',
    'kennesaw state owls': 'kennesaw state', 'delaware blue hens': 'delaware', 'missouri state bears': 'missouri state',
}


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    s = s.lower().replace('&', 'and')
    s = re.sub(r"[\.'’]", '', s)
    s = re.sub(r'[^a-z0-9() ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def fetch(key=None):
    key = key or os.environ.get('ODDS_API_KEY')
    if not key:
        print('ODDS_API_KEY not set; skipping DraftKings lines')
        return None, None
    try:
        req = urllib.request.Request(URL.format(key=key), headers={'User-Agent': 'cfb-prediction-model'})
        with urllib.request.urlopen(req, timeout=60) as r:
            remaining = r.headers.get('x-requests-remaining')
            used = r.headers.get('x-requests-used')
            data = json.loads(r.read().decode())
        print(f'odds api: {len(data)} events, credits used {used}, remaining {remaining}')
        return data, {'used': used, 'remaining': remaining}
    except Exception as e:
        print('odds api failed:', e)
        return None, None


def parse(events):
    rows = []
    for ev in events or []:
        dk = next((b for b in ev.get('bookmakers', []) if b.get('key') == 'draftkings'), None)
        if not dk:
            continue
        r = {'event_id': ev['id'], 'commence': ev['commence_time'], 'home_name': ev['home_team'], 'away_name': ev['away_team'],
             'home_spread': None, 'home_spread_price': None, 'away_spread_price': None, 'total': None,
             'over_price': None, 'under_price': None, 'dk_updated': dk.get('last_update')}
        for m in dk.get('markets', []):
            if m['key'] == 'spreads':
                for o in m['outcomes']:
                    if o['name'] == ev['home_team']:
                        r['home_spread'], r['home_spread_price'] = o.get('point'), o.get('price')
                    elif o['name'] == ev['away_team']:
                        r['away_spread_price'] = o.get('price')
            elif m['key'] == 'totals':
                for o in m['outcomes']:
                    if o['name'].lower() == 'over':
                        r['total'], r['over_price'] = o.get('point'), o.get('price')
                    elif o['name'].lower() == 'under':
                        r['under_price'] = o.get('price')
        rows.append(r)
    return pd.DataFrame(rows)


def _names(ti):
    """Candidate strings per ESPN team id: school, school + mascot, abbreviation, alternates."""
    out = {}
    for _, t in ti.iterrows():
        c = {norm(t.school)}
        if isinstance(t.get('mascot'), str):
            c.add(norm(f'{t.school} {t.mascot}'))
        for k in ['abbreviation', 'alt_name1', 'alt_name2', 'alt_name3']:
            v = t.get(k)
            if isinstance(v, str) and v:
                c.add(norm(v))
        out[int(t.team_id)] = c
    return out


def _score(odds_name, cands):
    n = norm(odds_name)
    if n in ALIASES:
        n2 = ALIASES[n]
        if n2 in cands:
            return 1.0
    best = 0.0
    for c in cands:
        if n == c:
            return 1.0
        if n.startswith(c + ' ') and len(c) >= 4:
            best = max(best, 0.9 + 0.05 * min(len(c), 20) / 20)
        best = max(best, difflib.SequenceMatcher(None, n, c).ratio() * 0.9)
    return best


def match(lines, games, ti):
    """Attach each DraftKings event to a schedule game: same kickoff day (±1) and best name match."""
    if lines is None or lines.empty:
        return pd.DataFrame(), []
    names = _names(ti)
    g = games.copy()
    g['ts'] = pd.to_datetime(g.start_date, utc=True, format='ISO8601')
    matched, unmatched = [], []
    for _, r in lines.iterrows():
        ts = pd.Timestamp(r.commence)
        cand = g[(g.ts - ts).abs() <= pd.Timedelta(hours=36)]
        best, best_s, flipped = None, 0, False
        for _, c in cand.iterrows():
            hn, an = names.get(int(c.home_id), set()), names.get(int(c.away_id), set())
            s1 = _score(r.home_name, hn) + _score(r.away_name, an)
            s2 = _score(r.home_name, an) + _score(r.away_name, hn)
            if s1 > best_s:
                best, best_s, flipped = c, s1, False
            if s2 > best_s:
                best, best_s, flipped = c, s2, True
        if best is None or best_s < 1.55:
            unmatched.append(f"{r.away_name} @ {r.home_name}")
            continue
        spread = r.home_spread
        if flipped and spread is not None:
            spread = -spread           # the book listed the teams the other way round
        matched.append({'game_id': int(best.game_id), 'dk_home_line': spread, 'dk_total': r.total,
                        'dk_home_price': r.away_spread_price if flipped else r.home_spread_price,
                        'dk_away_price': r.home_spread_price if flipped else r.away_spread_price,
                        'dk_over_price': r.over_price, 'dk_under_price': r.under_price, 'dk_updated': r.dk_updated,
                        'match_score': round(best_s, 2)})
    return pd.DataFrame(matched), unmatched
