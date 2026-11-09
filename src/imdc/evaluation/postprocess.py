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
    unequal per-level factors could introduce. Note: a widening factor > 1 (e.g. the fitted
    95% factor of ~1.73, see docs/PLAN.md Phase 6) can push a lower bound below zero for
    small-count states/cities - callers building an actual submission (where values are
    non-negative case counts) should follow this with `clip_nonnegative`; left out of this
    function itself since it is domain-agnostic (also used for generic calibration/scoring
    where negative values are legitimate, see tests/test_postprocess.py).
    """
    out = preds_wide.copy()
    p = out["pred"].to_numpy(dtype=float)
    for level in interval_levels:
        s = factors[level]
        out[f"upper_{level}"] = p + s * (out[f"upper_{level}"].to_numpy(dtype=float) - p)
        out[f"lower_{level}"] = p - s * (p - out[f"lower_{level}"].to_numpy(dtype=float))
    return enforce_monotonicity(out)


# ---------------------------------------------------------------------------------------------
# Size-aware (burden-scaled) interval recalibration.
#
# The global conformal factors above are fitted to pooled coverage, which the high-burden states
# dominate; scored per state (each state weighted equally, as the IMDC organizers' Rv does), a
# single factor over-widens the low-burden states. Here each state's whole predictive distribution
# is rescaled about its median in log1p space by a factor that depends on the state's historical
# burden: s_u = clip(a + b * z_u), with z_u the standardized log mean weekly cases before the
# tuning fold's training cutoff. b > 0 sharpens small states and widens large ones; (a, b) are
# chosen once, on the tuning fold, and frozen (like the conformal factors).
SIZE_AWARE_BOUNDS = (0.4, 1.4)
SIZE_AWARE_A_GRID = tuple(np.round(np.arange(0.50, 1.501, 0.05), 2))
SIZE_AWARE_B_GRID = tuple(np.round(np.arange(-0.20, 0.401, 0.05), 2))


def state_burden_index(state_cases: pd.DataFrame, cutoff, start="2015-01-01") -> pd.Series:
    """Standardized log mean weekly cases per state over [start, cutoff] (index: uf).

    `state_cases` has columns uf, date, casos (aggregate_cases_to_state output). Uses only data
    on or before `cutoff`, so it is leakage-safe for any fold whose training cutoff is >= cutoff.
    """
    d = state_cases[(state_cases["date"] >= pd.Timestamp(start)) & (state_cases["date"] <= pd.Timestamp(cutoff))]
    b = np.log(d.groupby("uf")["casos"].mean().clip(lower=1e-3))
    return ((b - b.mean()) / b.std()).rename("burden_z")


def apply_size_aware_scale(
    preds_wide: pd.DataFrame, burden_z: pd.Series, a: float, b: float,
    bounds: tuple = SIZE_AWARE_BOUNDS, uf_col: str = "uf",
) -> pd.DataFrame:
    """Rescale every quantile about the median in log1p space: q' = expm1(m + s_u (log1p q - m)).

    m = log1p(median), s_u = clip(a + b * burden_z[uf], *bounds). The median is unchanged; s_u < 1
    sharpens, > 1 widens. Working in log space keeps the lower bounds non-negative and scales the
    skewed upper tail proportionally. Quantiles are clipped to >= 0 first (case counts).
    """
    z = preds_wide[uf_col].map(burden_z)
    if z.isna().any():
        raise ValueError(f"apply_size_aware_scale: no burden index for {sorted(preds_wide.loc[z.isna(), uf_col].unique())}")
    s = np.clip(a + b * z.to_numpy(dtype=float), *bounds)[:, None]
    q = np.log1p(preds_wide[QUANTILE_COLUMNS].clip(lower=0.0).to_numpy(dtype=float))
    m = np.log1p(preds_wide["pred"].clip(lower=0.0).to_numpy(dtype=float))[:, None]
    out = preds_wide.copy()
    out[QUANTILE_COLUMNS] = np.expm1(m + s * (q - m))
    return enforce_monotonicity(out)


def fit_size_aware_scale(
    calib: pd.DataFrame, burden_z: pd.Series, a_grid=SIZE_AWARE_A_GRID, b_grid=SIZE_AWARE_B_GRID,
    bounds: tuple = SIZE_AWARE_BOUNDS,
) -> dict:
    """Pick (a, b) on the calibration rows, by a per-state criterion under a pooled-WIS constraint.

    Objective: mean over states of log(mean WIS of the state) - every state weighted equally.
    Its argmin is identical to that of a geometric-mean WIS ratio against ANY fixed reference
    forecast (the reference's log WIS is a constant), so it reproduces the organizers' Rv
    selection without needing their reference model. Constraint: the pooled mean WIS (which the
    high-burden states dominate) must be no worse than the unscaled forecast's, so the per-state
    gain is never bought with a national loss. (a, b) = (1, 0) is the identity and always feasible.
    Returns {"a", "b", "objective", "pooled_wis", "objective_unscaled", "pooled_wis_unscaled"}.
    """
    from imdc.evaluation.metrics import wis_from_intervals

    y = calib["observed_value"].to_numpy(dtype=float)
    uf = calib["uf"].to_numpy()

    def _score(df):
        w = pd.Series(wis_from_intervals(df, y), index=calib.index)
        per_state = w.groupby(uf).mean()
        return float(np.log(per_state.clip(lower=1e-9)).mean()), float(w.mean())

    obj0, pooled0 = _score(calib)
    best = {"a": 1.0, "b": 0.0, "objective": obj0, "pooled_wis": pooled0}
    for a in a_grid:
        for b in b_grid:
            obj, pooled = _score(apply_size_aware_scale(calib, burden_z, a, b, bounds))
            if pooled <= pooled0 and obj < best["objective"] - 1e-12:
                best = {"a": float(a), "b": float(b), "objective": obj, "pooled_wis": pooled}
    best.update(objective_unscaled=obj0, pooled_wis_unscaled=pooled0)
    return best


def clip_nonnegative(df: pd.DataFrame, columns: list = QUANTILE_COLUMNS) -> pd.DataFrame:
    """Clip quantile columns to >= 0. Preserves monotonicity (clipping is order-preserving).

    Required before any submission is validated (the platform rejects negative values) but
    not folded into `apply_conformal_widen`/`enforce_monotonicity` since those are also used
    for generic scoring/calibration on data where negative values are legitimate.
    """
    out = df.copy()
    out[columns] = out[columns].clip(lower=0.0)
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
