"""CollegeFootballData.com helpers (needs CFBD_API_KEY). Used for stadium locations,
elevation and dome status, mainly so neutral-site games get weather and travel."""
import json
import os
import urllib.request

API = 'https://api.collegefootballdata.com'


def _get(path, key):
    req = urllib.request.Request(API + path, headers={'Authorization': f'Bearer {key}', 'accept': 'application/json',
                                                      'User-Agent': 'cfb-prediction-model'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _latlon(v):
    lat, lon = v.get('latitude'), v.get('longitude')
    loc = v.get('location')
    if (lat is None or lon is None) and isinstance(loc, dict):
        a, b = loc.get('x'), loc.get('y')
        if a is not None and b is not None:
            # the longitude is the negative one for US stadiums
            lat, lon = (a, b) if abs(b) > abs(a) else (b, a)
    return lat, lon


def refresh_venues(state_dir, key=None):
    key = key or os.environ.get('CFBD_API_KEY')
    path = os.path.join(state_dir, 'venues.json')
    if not key:
        print('CFBD_API_KEY not set; neutral-site venues will use what is cached')
        return False
    try:
        vs = _get('/venues', key)
    except Exception as e:
        print('CFBD venues failed:', e)
        return False
    out = []
    for v in vs:
        lat, lon = _latlon(v)
        out.append({'venue_id': v.get('id'), 'name': v.get('name'), 'latitude': lat, 'longitude': lon,
                    'elevation': _num(v.get('elevation')), 'dome': bool(v.get('dome')), 'timezone': v.get('timezone'),
                    'city': v.get('city'), 'state': v.get('state')})
    json.dump(out, open(path, 'w'))
    print('saved', len(out), 'venues')
    return True


def _num(x):
    try:
        return float(x)
    except Exception:
        return None
