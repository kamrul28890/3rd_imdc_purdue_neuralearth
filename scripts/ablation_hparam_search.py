"""Re-verify the paper's hyperparameter-tuning ablation (Table tab:ablation) on the current
XGBoost-based, current-data pipeline.

Original (LightGBM, pre-refresh) finding: a grid search on an internal time-block holdout
preferred a deeper config (~5% lower holdout pinball loss), but that config was WORSE on the
actual backtest (overfit the holdout, which doesn't represent forecasting a full future season).

Methodology (matches docs/PLAN.md's stated discipline: "any internal train/validation split for
hyperparameter tuning ... must be a contiguous time-block holdout"): fold 1 (the project's own
designated tuning fold) training panel, last 52 weeks of origins held out, everything before that
is "proper" train. Grid over max_depth/min_child_weight (XGBoost's depth/regularization knobs,
playing the role LightGBM's num_leaves/max_depth played in the original test). Whichever config
wins the internal holdout is the "tuned" candidate to then check against the true 4-fold backtest
in a follow-up run (ablation_hparam_true_backtest.py) - this script only does the fast internal
search.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ablation_hparam_search.py
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb

from imdc.config import QUANTILE_LEVELS
from imdc.data.folds import get_folds
from imdc.features.panel import build_panel
from imdc.models.xgb_quantile import DEFAULT_PARAMS, DEFAULT_N_ESTIMATORS

HOLDOUT_WEEKS = 52  # docs/PLAN.md's stated internal-tuning holdout convention

CANDIDATES = {
    "default (max_depth=5, min_child_weight=30)": {},
    "deeper_A (max_depth=7, min_child_weight=10)": {"max_depth": 7, "min_child_weight": 10},
    "deeper_B (max_depth=9, min_child_weight=5)": {"max_depth": 9, "min_child_weight": 5},
    "deeper_C (max_depth=7, min_child_weight=30)": {"max_depth": 7, "min_child_weight": 30},
    "shallower (max_depth=3, min_child_weight=50)": {"max_depth": 3, "min_child_weight": 50},
}


def pinball_loss(y: np.ndarray, preds: np.ndarray, taus: list) -> float:
    """Mean pinball (quantile) loss over all quantile levels and rows. preds: (n, len(taus))."""
    total = 0.0
    for j, tau in enumerate(taus):
        diff = y - preds[:, j]
        total += np.mean(np.maximum(tau * diff, (tau - 1) * diff))
    return total / len(taus)


def main():
    fold1 = get_folds("dengue")[0]
    panel, feature_cols = build_panel(fold1, disease="dengue")

    split = fold1.train_cutoff - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    proper = panel[panel["origin_date"] < split]
    holdout = panel[panel["origin_date"] >= split]
    print(f"proper rows: {len(proper)}, holdout rows: {len(holdout)} "
          f"(split date {split.date()}, cutoff {fold1.train_cutoff.date()})")

    X_train, y_train = proper[feature_cols], proper["label"].to_numpy()
    X_hold, y_hold = holdout[feature_cols], holdout["label"].to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train)

    results = {}
    for name, overrides in CANDIDATES.items():
        t0 = time.time()
        params = {**DEFAULT_PARAMS, **overrides, "quantile_alpha": np.array(QUANTILE_LEVELS)}
        booster = xgb.train(params, dtrain, num_boost_round=DEFAULT_N_ESTIMATORS)
        preds = np.atleast_2d(booster.inplace_predict(X_hold))
        loss = pinball_loss(y_hold, preds, QUANTILE_LEVELS)
        results[name] = loss
        print(f"{name}: holdout pinball={loss:.6f} ({time.time()-t0:.1f}s)")

    best = min(results, key=results.get)
    default_loss = results["default (max_depth=5, min_child_weight=30)"]
    print(f"\nbest on internal holdout: {best} ({results[best]:.6f} vs default {default_loss:.6f}, "
          f"{100*(default_loss-results[best])/default_loss:.1f}% lower)")


if __name__ == "__main__":
    main()
