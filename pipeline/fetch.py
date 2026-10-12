"""Download season files from the public SportsDataverse repositories (no key needed).

Play-by-play, schedules, ESPN betting lines, ESPN FPI and transfer data come from
sportsdataverse/cfbfastR-cfb-data. Stadium locations come from cfbfastR-data.
"""
import os
import time
import urllib.request

DATA = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-data/main/cfb'
INFO = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/team_info/parquet'

FILES = {
    'pbp': ('pbp/parquet/play_by_play_{y}.parquet', 'pbp_{y}.parquet'),
    'cfb_schedules': ('cfb_schedules/parquet/cfb_schedules_{y}.parquet', 'cfb_schedules_{y}.parquet'),
    'betting': ('betting/parquet/betting_{y}.parquet', 'betting_{y}.parquet'),
    'power_index': ('power_index/parquet/power_index_{y}.parquet', 'power_index_{y}.parquet'),
    'cfb_team_portal': ('cfb_team_portal/parquet/cfb_team_portal_{y}.parquet', 'cfb_team_portal_{y}.parquet'),
}


def get(url, dest, tries=3, max_age_hours=None):
    if max_age_hours is not None and os.path.exists(dest) and (time.time() - os.path.getmtime(dest)) < max_age_hours * 3600:
        return True
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'cfb-prediction-model'})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest + '.tmp', 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + '.tmp', dest)
            return True
        except Exception as e:
            print(f'  retry {i + 1} for {url}: {e}', flush=True)
            time.sleep(5 * (i + 1))
    return False


def fetch_seasons(raw, seasons, kinds=None, max_age_hours=None):
    os.makedirs(raw, exist_ok=True)
    for y in seasons:
        for k, (src, dst) in FILES.items():
            if kinds and k not in kinds:
                continue
            ok = get(f'{DATA}/{src.format(y=y)}', f'{raw}/{dst.format(y=y)}', max_age_hours=max_age_hours)
            print(('ok   ' if ok else 'MISS ') + dst.format(y=y), flush=True)
    for y in [2019, 2024, 2026]:
        get(f'{INFO}/cfb_team_info_{y}.parquet', f'{raw}/team_info_{y}.parquet', max_age_hours=24 * 30)


if __name__ == '__main__':
    import sys
    a, b = int(sys.argv[2]), int(sys.argv[3])
    fetch_seasons(sys.argv[1], range(a, b + 1))
