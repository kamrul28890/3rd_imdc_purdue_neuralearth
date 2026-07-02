"""Probabilistic forecast scoring: WIS/CRPS, coverage, MASE, peak errors.

WIS is implemented as a thin wrapper around pinball loss evaluated at the 9
QUANTILE_LEVELS (config.py) - this is not a stylistic choice, it's the exact
identity IS_alpha(l,u,y) = (2/alpha)*[PB_{alpha/2}(l,y) + PB_{1-alpha/2}(u,y)],
which makes wis_from_quantiles a structural CRPS approximation rather than an
incidental one. wis_from_intervals is the primary entry point since our
prediction dataframes are wide (pred/lower_X/upper_X); it is numerically
identical to wis_from_quantiles on the same 9 levels (see
tests/test_metrics.py for the cross-check).

Verified against mosqlient==2.5.2's mosqlient.scoring.score.compute_wis /
compute_interval_score (installed package source read directly) - our
formulas match theirs exactly.
"""
import numpy as np
import pandas as pd

from imdc.config import INTERVAL_LEVELS, QUANTILE_LEVELS

_EPS = 1e-9


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Quantile (pinball) loss at quantile level tau in (0,1)."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """IS_alpha(l,u,y) = (u-l) + (2/alpha)*max(0,l-y) + (2/alpha)*max(0,y-u)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    width = upper - lower
    penalty_lower = (2 / alpha) * np.maximum(0.0, lower - y)
    penalty_upper = (2 / alpha) * np.maximum(0.0, y - upper)
    return width + penalty_lower + penalty_upper


def wis_from_intervals(
    pred: pd.DataFrame, y: np.ndarray, interval_levels: list = INTERVAL_LEVELS
) -> np.ndarray:
    """WIS from a wide dataframe with columns pred, lower_{L}, upper_{L} for L in interval_levels.

    WIS(F,y) = [w_0*|y-median| + sum_k w_k*IS_{alpha_k}(l_k,u_k,y)] / (K+0.5),
    w_0=1/2, w_k=alpha_k/2, alpha_k = 1 - L_k/100.
    """
    y = np.asarray(y, dtype=float)
    K = len(interval_levels)
    median_term = 0.5 * np.abs(y - pred["pred"].to_numpy(dtype=float))
    total = median_term.copy()
    for level in interval_levels:
        alpha = 1 - level / 100
        w_k = alpha / 2
        is_k = interval_score(
            pred[f"lower_{level}"].to_numpy(dtype=float),
            pred[f"upper_{level}"].to_numpy(dtype=float),
            y,
            alpha,
        )
        total += w_k * is_k
    return total / (K + 0.5)


def wis_from_quantiles(
    quantile_values: np.ndarray, quantile_levels: list, y: np.ndarray
) -> np.ndarray:
    """WIS as sum(pinball loss over quantile_levels) / (K+0.5) (structural CRPS-approximation identity).

    quantile_values: array shaped (..., len(quantile_levels)), matching quantile_levels order.
    quantile_levels must be the 2K+1 median-symmetric levels (K central intervals + median).
    Numerically identical to wis_from_intervals when quantile_levels == QUANTILE_LEVELS
    and quantile_values are the corresponding 9 columns - NOT a plain mean: dividing by
    len(quantile_levels) (9) instead of (K+0.5) (4.5) silently halves the result, since
    9 == 2*(K+0.5). Verified against wis_from_intervals in tests/test_metrics.py.
    """
    y = np.asarray(y, dtype=float)
    quantile_values = np.asarray(quantile_values, dtype=float)
    K = (len(quantile_levels) - 1) / 2
    losses = np.stack(
        [pinball_loss(y, quantile_values[..., i], tau) for i, tau in enumerate(quantile_levels)],
        axis=-1,
    )
    return losses.sum(axis=-1) / (K + 0.5)


def wis_decomposition(
    pred: pd.DataFrame, y: np.ndarray, interval_levels: list = INTERVAL_LEVELS
) -> pd.DataFrame:
    """Per-row WIS split into dispersion / overprediction / underprediction, summing to WIS.

    The median term is folded into dispersion (it is itself a zero-width,
    zero-alpha-limit "interval", i.e. pure sharpness cost with no direction).
    """
    y = np.asarray(y, dtype=float)
    K = len(interval_levels)
    dispersion = 0.5 * np.abs(y - pred["pred"].to_numpy(dtype=float))
    overprediction = np.zeros(len(pred))
    underprediction = np.zeros(len(pred))
    for level in interval_levels:
        alpha = 1 - level / 100
        w_k = alpha / 2
        lower = pred[f"lower_{level}"].to_numpy(dtype=float)
        upper = pred[f"upper_{level}"].to_numpy(dtype=float)
        dispersion += w_k * (upper - lower)
        overprediction += w_k * (2 / alpha) * np.maximum(0.0, lower - y)
        underprediction += w_k * (2 / alpha) * np.maximum(0.0, y - upper)
    denom = K + 0.5
    return pd.DataFrame(
        {
            "dispersion": dispersion / denom,
            "overprediction": overprediction / denom,
            "underprediction": underprediction / denom,
        }
    )


def coverage(pred: pd.DataFrame, y: np.ndarray, interval_levels: list = INTERVAL_LEVELS) -> dict:
    """Empirical coverage per nominal interval level: fraction of y within [lower_L, upper_L]."""
    y = np.asarray(y, dtype=float)
    out = {}
    for level in interval_levels:
        lower = pred[f"lower_{level}"].to_numpy(dtype=float)
        upper = pred[f"upper_{level}"].to_numpy(dtype=float)
        out[f"coverage_{level}"] = float(np.mean((y >= lower) & (y <= upper)))
    return out


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 52) -> float:
    """Mean absolute scaled error; denominator is the seasonal-naive MAE on the training series."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale < _EPS:
        return float("nan")
    return float(mae / scale)


def peak_timing_error(pred_median: pd.Series, observed: pd.Series) -> int:
    """argmax_week(median forecast) - argmax_week(observed), in index positions (weeks)."""
    pred_peak = int(np.argmax(pred_median.to_numpy()))
    obs_peak = int(np.argmax(observed.to_numpy()))
    return pred_peak - obs_peak


def peak_magnitude_error(pred_median: pd.Series, observed: pd.Series) -> float:
    """log(pred_median_at_true_peak_week / observed_peak_value); evaluated at the TRUE peak week."""
    obs_values = observed.to_numpy()
    obs_peak_idx = int(np.argmax(obs_values))
    obs_peak = obs_values[obs_peak_idx]
    pred_at_true_peak = pred_median.to_numpy()[obs_peak_idx]
    if obs_peak < _EPS or pred_at_true_peak < _EPS:
        return float("nan")
    return float(np.log(pred_at_true_peak / obs_peak))


def relative_wis(scored_long: pd.DataFrame, baseline_model: str, group_cols: list) -> pd.Series:
    """Pairwise-comparison relative WIS (Cramer et al. 2022), scaled to baseline_model=1.0.

    scored_long must have columns `model`, `wis`, and the columns in group_cols
    (the forecasting units to average the log-ratio over, e.g. ["fold_id","uf","horizon_weeks"]).
    Only units where every model has a score are used (fully-crossed comparison).
    """
    pivot = scored_long.pivot_table(index=group_cols, columns="model", values="wis")
    pivot = pivot.dropna(axis=0, how="any")
    pivot = pivot.clip(lower=_EPS)
    models = list(pivot.columns)

    theta = {}
    for m in models:
        log_ratios = [
            np.mean(np.log(pivot[m]) - np.log(pivot[m2])) for m2 in models if m2 != m
        ]
        theta[m] = np.exp(np.mean(log_ratios))

    baseline_theta = theta[baseline_model]
    return pd.Series({m: theta[m] / baseline_theta for m in models}, name="relative_wis")
