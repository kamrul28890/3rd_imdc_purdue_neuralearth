"""Forecast ensembling: quantile-median (Vincentization) and inverse-WIS weighting.

Operates on the canonical wide per-model prediction tables saved by each model
family's backtest run (results/metrics/*_scored.csv), so ensembling never
re-runs a model. Ensemble weights, when used, are tuned ONLY on the designated
tuning fold (fold 1) and frozen for the headline folds (2-3) and the prospective
fold (4) - the discipline the plan requires to avoid overfitting the ensemble to
the folds it is reported on.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from imdc.config import QUANTILE_COLUMNS, QUANTILE_LEVELS
from imdc.evaluation.postprocess import enforce_monotonicity

_KEYS = ["uf", "date", "fold_id", "horizon_weeks", "observed_value"]
_INDEX_COLS = ["uf", "date", "fold_id", "horizon_weeks"]
_LEVEL_Z = norm.ppf(np.array(QUANTILE_LEVELS))
_MIN_SIGMA = 1e-3


def _median_ensemble(stacked: pd.DataFrame, index_cols: list, passthrough_cols: list = None) -> pd.DataFrame:
    """Per-quantile median across rows sharing `index_cols` (Gneiting-style Vincentization).

    `index_cols` must be non-null for every row being ensembled - deliberately NOT the old
    `_KEYS` (which put `observed_value` in the groupby key itself): grouping on a column that
    can be NaN silently *drops* those groups, which is exactly what happened on the
    prediction-only/forecast path (no observed value exists yet there), forcing a separate
    near-duplicate implementation (submission/forecast.py's old `_vincentize_wide`). Any such
    column belongs in `passthrough_cols` instead and is attached via a first-value merge after
    the groupby - safe because it's constant within a group by construction (the actual
    outcome at a given uf/date/fold/horizon doesn't depend on which model predicted it).

    The median of monotone-ordered quantile vectors is itself monotone, so no crossing is
    introduced; enforce_monotonicity is applied as a safeguard regardless.
    """
    combined = stacked.groupby(index_cols)[QUANTILE_COLUMNS].median().reset_index()
    if passthrough_cols:
        extra = stacked.groupby(index_cols)[passthrough_cols].first().reset_index()
        combined = combined.merge(extra, on=index_cols, how="left")
    return enforce_monotonicity(combined)


def vincentization(preds_wide: pd.DataFrame, models: list) -> pd.DataFrame:
    """Per-quantile median across models (Gneiting-style Vincentization) for scored backtest
    tables (has `observed_value`) or plain forecast tables (doesn't) alike - see
    `_median_ensemble` for why this doesn't just group by `_KEYS`.
    """
    sub = preds_wide[preds_wide["model"].isin(models)]
    index_cols = [c for c in _INDEX_COLS if c in sub.columns]
    passthrough = ["observed_value"] if "observed_value" in sub.columns else []
    combined = _median_ensemble(sub, index_cols, passthrough)
    combined["model"] = "ensemble_vincent"
    return combined


def inverse_wis_weights(scored: pd.DataFrame, models: list, tuning_fold: int = 1, power: float = 1.0) -> dict:
    """Weights proportional to (1 / mean-WIS-on-tuning-fold)^power, normalized to sum 1."""
    tune = scored[(scored["fold_id"] == tuning_fold) & (scored["model"].isin(models))]
    mean_wis = tune.groupby("model")["wis"].mean()
    raw = (1.0 / mean_wis) ** power
    w = (raw / raw.sum()).to_dict()
    return {m: w.get(m, 0.0) for m in models}


def weighted_ensemble(preds_wide: pd.DataFrame, weights: dict, name: str = "ensemble_invwis") -> pd.DataFrame:
    """Per-quantile weighted mean across models (weights need not sum to 1; normalized here).

    A weighted average of monotone-ordered quantile vectors preserves order, so
    the result stays monotone; enforce_monotonicity is a safeguard.
    """
    models = list(weights.keys())
    sub = preds_wide[preds_wide["model"].isin(models)].copy()
    sub["_w"] = sub["model"].map(weights)
    total_w = sub.groupby(_KEYS)["_w"].transform("sum")
    for col in QUANTILE_COLUMNS:
        sub[col] = sub[col] * sub["_w"]
    agg = sub.groupby(_KEYS)[QUANTILE_COLUMNS + ["_w"]].sum().reset_index()
    for col in QUANTILE_COLUMNS:
        agg[col] = agg[col] / agg["_w"]
    agg = agg.drop(columns="_w")
    agg = enforce_monotonicity(agg)
    agg["model"] = name
    return agg


def _fit_shifted_lognormal(values: np.ndarray) -> tuple:
    """OLS fit of log1p(y) = mu + sigma*z across the 9 canonical quantile levels, per row.

    `z` (the standard-normal quantile of each level) is fixed and shared across rows, so this
    is a vectorized simple linear regression rather than a per-row loop. `sigma` is floored at
    `_MIN_SIGMA` so a near-degenerate row (e.g. the all-zero-median fix's flat quantile vector)
    doesn't blow up the precision-weighted pooling below.
    """
    y = np.log1p(values)
    zc = _LEVEL_Z - _LEVEL_Z.mean()
    yc = y - y.mean(axis=1, keepdims=True)
    sigma = (yc @ zc) / np.sum(zc**2)
    sigma = np.maximum(sigma, _MIN_SIGMA)
    mu = y.mean(axis=1) - sigma * _LEVEL_Z.mean()
    return mu, sigma


def log_linear_pool(
    preds_wide: pd.DataFrame, models: list, weights: dict = None, name: str = "ensemble_loglinear"
) -> pd.DataFrame:
    """Combine models by log-linear (logarithmic) pooling of a per-row shifted-lognormal
    approximation to each model's predictive distribution - the ensemble methodology of the
    inaugural IMDC24 sprint (Araujo et al. 2026; logarithmic pooling per Carvalho et al. 2023),
    offered here as an alternative to this project's default per-quantile-median Vincentization.

    Vincentization combines quantiles as order statistics (a per-quantile median); log-linear
    pooling instead combines full predictive distributions, f_pool(y) proportional to
    prod_k f_k(y)^w_k, which for (shifted-)lognormal components has a closed form: a
    precision-weighted combination of each model's fitted (mu, sigma). That guarantees a
    unimodal combined distribution by construction, which Vincentization does not.

    `weights` defaults to equal weights (mirrors IMDC24's E1); pass fold-1-only weights (e.g.
    from `inverse_wis_weights`) to mirror its CRPS-weighted E2 - this project's own fold-1
    tuning discipline applies here exactly as it does to `weighted_ensemble`.
    """
    sub = preds_wide[preds_wide["model"].isin(models)].copy()
    index_cols = [c for c in _INDEX_COLS if c in sub.columns]
    passthrough = ["observed_value"] if "observed_value" in sub.columns else []

    if weights is None:
        weights = {m: 1.0 / len(models) for m in models}
    total_w = sum(weights.get(m, 0.0) for m in models)
    weights = {m: weights.get(m, 0.0) / total_w for m in models}

    mu, sigma = _fit_shifted_lognormal(sub[QUANTILE_COLUMNS].to_numpy(dtype=float))
    sub["_precision"] = sub["model"].map(weights) / sigma**2
    sub["_precision_mu"] = sub["_precision"] * mu

    agg = sub.groupby(index_cols)[["_precision", "_precision_mu"]].sum()
    tau_pool = agg["_precision"]
    mu_pool = agg["_precision_mu"] / tau_pool
    sigma_pool = np.sqrt(1.0 / tau_pool)

    combined = pd.DataFrame(index=agg.index)
    for level, col in zip(QUANTILE_LEVELS, QUANTILE_COLUMNS):
        combined[col] = np.clip(np.expm1(mu_pool + sigma_pool * norm.ppf(level)), 0.0, None)
    combined = combined.reset_index()

    if passthrough:
        extra = sub.groupby(index_cols)[passthrough].first().reset_index()
        combined = combined.merge(extra, on=index_cols, how="left")

    combined["model"] = name
    return enforce_monotonicity(combined)


def score_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Attach WIS / decomposition / coverage to a wide ensemble prediction table."""
    from imdc.evaluation.metrics import wis_decomposition, wis_from_intervals

    y = wide["observed_value"].to_numpy()
    out = wide.copy()
    out["wis"] = wis_from_intervals(out, y)
    decomp = wis_decomposition(out, y)
    out = pd.concat([out, decomp], axis=1)
    out["ae_median"] = np.abs(y - out["pred"].to_numpy())
    for level in [50, 80, 90, 95]:
        out[f"coverage_{level}"] = (y >= out[f"lower_{level}"].to_numpy()) & (y <= out[f"upper_{level}"].to_numpy())
    return out
