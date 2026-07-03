"""Typed loaders for the raw IMDC 2026 data files.

Each loader returns a DataFrame with parsed dates and no other transformation -
aggregation, joining and fold-cutoff filtering happen in downstream modules
(see aggregate.py, folds.py).

Caching: the hot, moderately-sized loaders (cases, ocean, environ, population,
geo_map) are memoized so a run gunzips+parses each file once instead of the ~17
times the pipeline would otherwise. Each public loader returns a defensive
`.copy()` of the cached frame so callers can filter/mutate freely without
corrupting the shared cache (the copy is milliseconds vs. seconds to re-parse).
The three largest files (climate, forecasting_climate, access_afya) are left
uncached: climate/forecasting_climate are aggregated once into the parquet cache
in features/panel.py, and access_afya is not used in modeling - caching them
would hold gigabytes on an 8 GB machine for little benefit.
"""
from functools import lru_cache

import pandas as pd

from imdc.config import RAW_FILES


@lru_cache(maxsize=None)
def _cases_cached(disease: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES[disease])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_cases(disease: str = "dengue") -> pd.DataFrame:
    """Load weekly case counts per municipality for 'dengue' or 'chikungunya'."""
    return _cases_cached(disease).copy()


def load_climate() -> pd.DataFrame:
    """Weekly ERA5 reanalysis climate per municipality (observed only). Uncached (large)."""
    df = pd.read_csv(RAW_FILES["climate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_forecasting_climate() -> pd.DataFrame:
    """Monthly ECMWF seasonal climate forecast per municipality. Uncached (large).

    `reference_month` is the forecast origin; `forecast_months_ahead` (1-6)
    indicates how far into the future each row's values apply.
    """
    df = pd.read_csv(RAW_FILES["forecasting_climate"])
    # reference_month is stored as a mix of "YYYY-MM-DD" and
    # "YYYY-MM-DD 00:00:00" strings in the raw file - format="mixed" handles both.
    df["reference_month"] = pd.to_datetime(df["reference_month"], format="mixed")
    return df


@lru_cache(maxsize=1)
def _ocean_cached() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["ocean"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_ocean_indices() -> pd.DataFrame:
    """Weekly ENSO / IOD / PDO ocean oscillation indices."""
    return _ocean_cached().copy()


@lru_cache(maxsize=1)
def _environ_cached() -> pd.DataFrame:
    return pd.read_csv(RAW_FILES["environ_vars"])


def load_environ_vars() -> pd.DataFrame:
    """Static per-municipality Koppen climate class and biome."""
    return _environ_cached().copy()


@lru_cache(maxsize=1)
def _population_cached() -> pd.DataFrame:
    return pd.read_csv(RAW_FILES["population"])


def load_population() -> pd.DataFrame:
    """Yearly population per municipality, 2001-2025."""
    return _population_cached().copy()


def load_access_afya(disease: str = "Dengue") -> pd.DataFrame:
    """Daily Afya Whitebook search-access counts per municipality. Uncached (large)."""
    df = pd.read_csv(RAW_FILES["access_afya"])
    df["access_date"] = pd.to_datetime(df["access_date"])
    if disease is not None:
        df = df[df["accessed_disease"] == disease]
    return df


@lru_cache(maxsize=1)
def _geo_map_cached() -> pd.DataFrame:
    return pd.read_csv(RAW_FILES["map_regional_health"])


def load_geo_map() -> pd.DataFrame:
    """Municipality -> regional/macroregional health-district hierarchy."""
    return _geo_map_cached().copy()
