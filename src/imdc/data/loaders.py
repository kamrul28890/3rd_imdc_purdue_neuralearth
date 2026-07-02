"""Typed loaders for the raw IMDC 2026 data files.

Each loader returns a DataFrame with parsed dates and no other
transformation - aggregation, joining and fold-cutoff filtering happen in
downstream modules (see aggregate.py, folds.py).
"""
import pandas as pd

from imdc.config import RAW_FILES


def load_cases(disease: str = "dengue") -> pd.DataFrame:
    """Load weekly case counts per municipality for 'dengue' or 'chikungunya'."""
    path = RAW_FILES[disease]
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_climate() -> pd.DataFrame:
    """Weekly ERA5 reanalysis climate per municipality (observed only)."""
    df = pd.read_csv(RAW_FILES["climate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_forecasting_climate() -> pd.DataFrame:
    """Monthly ECMWF seasonal climate forecast per municipality.

    `reference_month` is the forecast origin; `forecast_months_ahead` (1-6)
    indicates how far into the future each row's values apply.
    """
    df = pd.read_csv(RAW_FILES["forecasting_climate"])
    # reference_month is stored as a mix of "YYYY-MM-DD" and
    # "YYYY-MM-DD 00:00:00" strings in the raw file - format="mixed" handles both.
    df["reference_month"] = pd.to_datetime(df["reference_month"], format="mixed")
    return df


def load_ocean_indices() -> pd.DataFrame:
    """Weekly ENSO / IOD / PDO ocean oscillation indices."""
    df = pd.read_csv(RAW_FILES["ocean"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_environ_vars() -> pd.DataFrame:
    """Static per-municipality Koppen climate class and biome."""
    return pd.read_csv(RAW_FILES["environ_vars"])


def load_population() -> pd.DataFrame:
    """Yearly population per municipality, 2001-2025."""
    return pd.read_csv(RAW_FILES["population"])


def load_access_afya(disease: str = "Dengue") -> pd.DataFrame:
    """Daily Afya Whitebook search-access counts per municipality."""
    df = pd.read_csv(RAW_FILES["access_afya"])
    df["access_date"] = pd.to_datetime(df["access_date"])
    if disease is not None:
        df = df[df["accessed_disease"] == disease]
    return df


def load_geo_map() -> pd.DataFrame:
    """Municipality -> regional/macroregional health-district hierarchy."""
    return pd.read_csv(RAW_FILES["map_regional_health"])
