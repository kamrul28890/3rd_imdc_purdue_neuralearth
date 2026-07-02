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


def vincentization(preds_wide: pd.DataFrame, models: list) -> pd.DataFrame:
    """Per-quantile median across models (Gneiting-style Vincentization).

    The median of monotone-ordered quantile vectors is itself monotone, so no
    crossing is introduced; enforce_monotonicity is applied as a safeguard.
    """
    sub = preds_wide[preds_wide["model"].isin(models)]
    combined = sub.groupby(_KEYS)[QUANTILE_COLUMNS].median().reset_index()
    combined = enforce_monotonicity(combined)
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
