"""Test the COMBINED best-known XGBoost config: no observed-climate covariates
(ablation_climate_covariates.py: WIS 1166.3) + shallower hyperparameters
(ablation_hparam_true_backtest.py: WIS 1152.4), together. The two haven't been tested
together - tree-model hyperparameter optima can shift when the feature set changes, so this
checks whether they stack (better than both), interact negatively (worse than one alone), or
are roughly independent, rather than assuming.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ablation_combined_best.py
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import normalized_wis, run_backtest, score_backtest, summarize
from imdc.features.panel import OBSERVED_CLIMATE_COLS
from imdc.models.xgb_quantile import XGBQuantileModel

TUNED_PARAMS = {"max_depth": 3, "min_child_weight": 50}


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(
        lambda: XGBQuantileModel(disease="dengue", params=TUNED_PARAMS, exclude_cols=OBSERVED_CLIMATE_COLS),
        folds, disease="dengue", ufs=MANDATORY_UFS,
    )
    scored = score_backtest(preds, disease="dengue", folds=folds)
    print(f"xgb_combined_best backtest: {len(scored)} scored rows in {time.time()-t0:.1f}s")

    out = METRICS_DIR / "ablation_xgb_combined_best_scored.csv"
    scored.to_csv(out, index=False)
    print(f"wrote {out}")

    overall = summarize(scored, by=["model"])
    by_fold = summarize(scored, by=["model", "fold_id"])
    norm_all = normalized_wis(scored, by=["model"])
    norm_ex2024 = normalized_wis(scored, by=["model"], exclude_folds=(2,))

    print("\n=== xgb_combined_best: overall ===")
    print(overall.to_string(index=False))
    print("\n=== by fold ===")
    print(by_fold.sort_values("fold_id").to_string(index=False))
    print("\nnormWIS_all:", norm_all["normalized_wis"].iloc[0])
    print("normWIS_ex2024:", norm_ex2024["normalized_wis"].iloc[0])


if __name__ == "__main__":
    main()
