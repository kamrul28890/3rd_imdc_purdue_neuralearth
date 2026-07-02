"""Fold boundary derivation and leakage-safe date filtering.

Fold boundaries are derived from the train_N/target_N flags in the raw case
data at runtime rather than hardcoded, so the pipeline stays correct if the
data file is refreshed with different dates.

IMPORTANT: train_N/target_N flags are only present on the case-count tables
(dengue.csv.gz, chikungunya.csv.gz). Every other joined table (climate,
ocean, forecasting_climate, access_afya) carries no fold flags at all and
must be filtered explicitly by date using `cutoff_filter` /
`cutoff_filter_forecasting_climate` - never by re-deriving a "not target"
condition, since there is an unflagged gap of ~15 weeks between each
train_N cutoff and the corresponding target_N start that must also be
excluded from training data.
"""
from dataclasses import dataclass

import pandas as pd

from imdc.config import N_FOLDS
from imdc.data.loaders import load_cases


@dataclass(frozen=True)
class Fold:
    id: int
    train_cutoff: pd.Timestamp  # last date allowed in training data (inclusive)
    target_start: pd.Timestamp
    target_end: pd.Timestamp


def get_folds(disease: str = "dengue") -> list[Fold]:
    """Derive the N_FOLDS official backtest windows from train_N/target_N flags."""
    df = load_cases(disease)
    folds = []
    for i in range(1, N_FOLDS + 1):
        train_col = f"train_{i}"
        target_col = f"target_{i}"
        train_cutoff = df.loc[df[train_col], "date"].max()
        target_start = df.loc[df[target_col], "date"].min()
        target_end = df.loc[df[target_col], "date"].max()
        folds.append(
            Fold(id=i, train_cutoff=train_cutoff, target_start=target_start, target_end=target_end)
        )
    return folds


def get_forecast_origin(disease: str = "dengue") -> pd.Timestamp:
    """Latest available observed date - the origin for the real 2026-27 forecast."""
    df = load_cases(disease)
    return df["date"].max()


def cutoff_filter(df: pd.DataFrame, cutoff: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """Keep only rows with date_col <= cutoff (inclusive). Use for every joined table."""
    return df[df[date_col] <= cutoff]


def cutoff_filter_forecasting_climate(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Filter the ECMWF seasonal forecast product by forecast origin month.

    forecasting_climate.csv.gz is itself a forecast (reference_month = origin,
    forecast_months_ahead = 1-6). It's legitimate future-covariate information
    IF the reference_month is at or before the fold's train_cutoff month - i.e.
    a forecast issued on or before the cutoff, not a forecast issued later that
    happens to describe the target period.
    """
    origin_month = pd.Timestamp(cutoff.year, cutoff.month, 1)
    return df[df["reference_month"] <= origin_month]


def target_window(df: pd.DataFrame, fold: Fold, date_col: str = "date") -> pd.DataFrame:
    """Rows within [target_start, target_end] for a given fold - the evaluation window."""
    return df[(df[date_col] >= fold.target_start) & (df[date_col] <= fold.target_end)]
