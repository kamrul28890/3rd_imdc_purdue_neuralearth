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

from imdc.config import QUANTILE_COLUMNS
from imdc.evaluation.postprocess import enforce_monotonicity

_KEYS = ["uf", "date", "fold_id", "horizon_weeks", "observed_value"]
_INDEX_COLS = ["uf", "date", "fold_id", "horizon_weeks"]


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
