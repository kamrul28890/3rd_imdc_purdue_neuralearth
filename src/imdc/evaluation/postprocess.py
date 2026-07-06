"""Shared post-processing: quantile monotonicity enforcement and wide-format pivoting.

Centralized here so no individual model implementation has to (re-)get this
right - ML/DL quantile heads trained independently per level can produce
crossing quantiles, and the wide submission format is only ever built once,
from the canonical long format, by to_submission_wide.
"""
import numpy as np
import pandas as pd

from imdc.config import INTERVAL_LEVELS, QUANTILE_COLUMNS, QUANTILE_LEVEL_TO_COLUMN


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


def conformal_widen_factors(
    calib: pd.DataFrame, interval_levels: list = INTERVAL_LEVELS, eps: float = 1e-9
) -> dict:
    """Per-interval multiplicative widening factors that hit nominal coverage on `calib`.

    A multiplicative analogue of conformalized quantile regression (Romano et al. 2019),
    robust to the huge cross-state scale differences (SP vs AC) that an additive conformal
    score handles poorly. For central interval level L with bounds [lower_L, upper_L] and
    median `pred`, each calibration row's "multiple needed to just cover y" is

        r = (y - pred) / (upper_L - pred)   if y > pred      (upper side)
            (pred - y) / (pred - lower_L)   if y < pred      (lower side)
            0                               otherwise

    and the factor s_L is the L/100 empirical quantile of r, so widening the interval about
    the median by s_L yields ~L% empirical coverage. `calib` needs columns pred, observed_value,
    and lower_L/upper_L for each level. Factors < 1 (tightening an over-covered interval) are
    allowed; the median is never touched.
    """
    y = calib["observed_value"].to_numpy(dtype=float)
    p = calib["pred"].to_numpy(dtype=float)
    hi, lo = y > p, y < p
    factors = {}
    for level in interval_levels:
        upper = calib[f"upper_{level}"].to_numpy(dtype=float)
        lower = calib[f"lower_{level}"].to_numpy(dtype=float)
        r = np.zeros(len(y))
        r[hi] = (y[hi] - p[hi]) / np.maximum(upper[hi] - p[hi], eps)
        r[lo] = (p[lo] - y[lo]) / np.maximum(p[lo] - lower[lo], eps)
        factors[level] = float(np.quantile(r, level / 100))
    return factors


def apply_conformal_widen(
    preds_wide: pd.DataFrame, factors: dict, interval_levels: list = INTERVAL_LEVELS
) -> pd.DataFrame:
    """Widen each central interval about the median by its factor; re-nest for monotonicity.

    Point forecast (`pred`) is unchanged. `enforce_monotonicity` repairs any crossing that
    unequal per-level factors could introduce.
    """
    out = preds_wide.copy()
    p = out["pred"].to_numpy(dtype=float)
    for level in interval_levels:
        s = factors[level]
        out[f"upper_{level}"] = p + s * (out[f"upper_{level}"].to_numpy(dtype=float) - p)
        out[f"lower_{level}"] = p - s * (p - out[f"lower_{level}"].to_numpy(dtype=float))
    return enforce_monotonicity(out)


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
