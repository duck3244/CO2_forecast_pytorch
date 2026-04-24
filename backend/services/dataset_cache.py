"""NOAA CO2 data loader with on-disk caching.

First request downloads from the NOAA URL and stores a parsed CSV under
``backend/.data_cache/co2_mm_mlo.csv``. Subsequent requests read the cache
unless ``force_refresh=True``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from api.config import backend_root, load_config

CACHE_DIR = backend_root() / ".data_cache"
CACHE_FILE = CACHE_DIR / "co2_mm_mlo.csv"


def _download_from_noaa(url: str) -> pd.DataFrame:
    """Download and parse NOAA monthly CO2 text format into a DataFrame indexed by date."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows = []
    for line in response.text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        year = int(parts[0])
        month = int(parts[1])
        co2 = float(parts[3]) if parts[3] != "-99.99" else np.nan
        rows.append((year, month, co2))

    df = pd.DataFrame(rows, columns=["year", "month", "co2"])
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.set_index("date")[["co2"]].dropna()
    return df


def get_co2_dataframe(force_refresh: bool = False) -> Tuple[pd.DataFrame, str]:
    """Return (df, source) where source is 'cache' | 'noaa'."""
    if CACHE_FILE.exists() and not force_refresh:
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"], index_col="date")
        return df, "cache"

    config = load_config()
    url = config["data"]["url"]
    df = _download_from_noaa(url)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index_label="date")
    return df, "noaa"


def cache_path() -> Path:
    return CACHE_FILE
