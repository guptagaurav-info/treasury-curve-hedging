"""Download the US Treasury daily par yield curve (pinned mirror) and verify its sha256.

The file is pinned to a dataset revision. If the bytes ever differ from the recorded hash,
the run stops: a changed input should be a conscious re-pin, never a silent change.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request

import pandas as pd

from . import config

TENOR_COLS = {
    0.25: "bc_3month", 0.5: "bc_6month", 1: "bc_1year", 2: "bc_2year", 3: "bc_3year",
    5: "bc_5year", 7: "bc_7year", 10: "bc_10year", 20: "bc_20year", 30: "bc_30year",
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cfg: dict) -> "pathlib.Path":
    src = cfg["source"]
    dest = cfg["raw_dir"] / src["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {src['url']}")
        urllib.request.urlretrieve(src["url"], dest)
    got = sha256(dest)
    if got != src["sha256"]:
        sys.exit(f"sha256 mismatch for {dest}: {got} != {src['sha256']} (source changed; re-pin consciously)")
    return dest


def load_curve(cfg: dict) -> pd.DataFrame:
    """Return a date-indexed frame of par yields in decimals, one column per tenor in years."""
    raw = pd.read_parquet(fetch(cfg))
    raw["report_date"] = pd.to_datetime(raw["report_date"])
    raw = raw.sort_values("report_date").drop_duplicates("report_date", keep="last")
    out = pd.DataFrame(index=pd.DatetimeIndex(raw["report_date"], name="date"))
    for tenor, col in TENOR_COLS.items():
        out[tenor] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
    return out


if __name__ == "__main__":
    c = config.load()
    df = load_curve(c)
    print(df.shape, df.index.min().date(), df.index.max().date())
