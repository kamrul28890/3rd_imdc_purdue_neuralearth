"""Shared post-processing: quantile monotonicity enforcement and wide-format pivoting.

Centralized here so no individual model implementation has to (re-)get this
right - ML/DL quantile heads trained independently per level can produce
crossing quantiles, and the wide submission format is only ever built once,
from the canonical long format, by to_submission_wide.
"""
import numpy as np
import pandas as pd

from imdc.config import QUANTILE_COLUMNS, QUANTILE_LEVEL_TO_COLUMN


def enforce_monotonicity(df: pd.DataFrame, columns: list = QUANTILE_COLUMNS) -> pd.DataFrame:
    """Sort-and-reassign rearrangement (Chernozhukov, Fernandez-Val & Galichon 2010).

    For each row, sort the quantile values ascending and reassign them back to
    `columns` (already in ascending quantile-level order) - guarantees a
    monotonic result. Also returns the fraction of rows that needed reordering
    as `frac_rows_needing_reordering` on the returned frame's attrs, a small
    calibration diagnostic worth reporting per model.
    """
    values = df[columns].to_numpy(dtype=float)
    sorted_values = np.sort(values, axis=1)
    needed_reorder = ~np.all(values == sorted_values, axis=1)

    out = df.copy()
    out[columns] = sorted_values
    out.attrs["frac_rows_needing_reordering"] = float(np.mean(needed_reorder))
    return out


def to_submission_wide(
    long_df: pd.DataFrame,
    index_cols: list = ("uf", "date"),
    quantile_col: str = "quantile_level",
    value_col: str = "predicted_value",
) -> pd.DataFrame:
    """Pivot the canonical long quantile format to the wide lower_*/pred/upper_* submission format."""
    pivot = long_df.pivot_table(index=list(index_cols), columns=quantile_col, values=value_col)
    pivot = pivot.rename(columns=QUANTILE_LEVEL_TO_COLUMN)
    missing = set(QUANTILE_COLUMNS) - set(pivot.columns)
    if missing:
        raise ValueError(f"to_submission_wide: missing quantile levels for columns {missing}")
    pivot = pivot[QUANTILE_COLUMNS].reset_index()
    return pivot
