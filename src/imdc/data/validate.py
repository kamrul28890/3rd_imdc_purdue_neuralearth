"""Data-quality and leakage assertions.

These are meant to be called both as one-off audit checks (Phase 1 EDA) and
as hard runtime guards inside the feature-building pipeline (Phase 2+), so a
future refactor can't silently reintroduce leakage.
"""
import pandas as pd

from imdc.data.folds import Fold


class LeakageError(AssertionError):
    pass


def assert_no_leakage(df: pd.DataFrame, cutoff: pd.Timestamp, date_col: str = "date", name: str = "frame") -> None:
    """Raise if any row in df has date_col beyond the fold's train_cutoff."""
    max_date = df[date_col].max()
    if pd.notna(max_date) and max_date > cutoff:
        raise LeakageError(
            f"{name}: found dates up to {max_date} beyond train_cutoff {cutoff} - leakage."
        )


def assert_gap_weeks_absent(df: pd.DataFrame, fold: Fold, date_col: str = "date") -> None:
    """Raise if any date strictly between train_cutoff and target_start survives.

    Confirmed empirically: there is an ~15-week gap per fold that is flagged
    False for both train_N and target_N. A pipeline that filters as
    `NOT target_N` (instead of an explicit date <= train_cutoff) will leak
    these weeks into training.
    """
    gap = df[(df[date_col] > fold.train_cutoff) & (df[date_col] < fold.target_start)]
    if len(gap) > 0:
        raise LeakageError(
            f"fold {fold.id}: {len(gap)} gap-week rows between train_cutoff "
            f"{fold.train_cutoff} and target_start {fold.target_start} leaked into frame."
        )


def assert_all_sundays(df: pd.DataFrame, date_col: str = "date", name: str = "frame") -> None:
    non_sunday = df[df[date_col].dt.weekday != 6]
    if len(non_sunday) > 0:
        raise AssertionError(f"{name}: {len(non_sunday)} rows with non-Sunday dates.")


def assert_no_duplicate_series(df: pd.DataFrame, key_cols: list[str], name: str = "frame") -> None:
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes > 0:
        raise AssertionError(f"{name}: {dupes} duplicate rows on {key_cols}.")


def assert_non_negative(df: pd.DataFrame, col: str, name: str = "frame") -> None:
    n_neg = (df[col] < 0).sum()
    if n_neg > 0:
        raise AssertionError(f"{name}: {n_neg} negative values in column '{col}'.")
