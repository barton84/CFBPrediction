import datetime as dt
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def current_season(today=None):
    today = today or dt.date.today()
    # the season runs Aug to Jan; January belongs to the previous season
    return today.year if today.month >= 7 else today.year - 1


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


def clean(o):
    """Replace NaN with None recursively so the JSON is valid."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and (o != o):
        return None
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(obj, path):
    with open(path, 'w') as f:
        json.dump(clean(obj), f, cls=NpEncoder, separators=(',', ':'))
