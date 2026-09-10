"""Follow-up to ablation_hparam_search.py: the internal fold-1 holdout (last 52 weeks)
preferred a SHALLOWER XGBoost config (max_depth=3, min_child_weight=50, pinball 0.1645) over
the deployed default (max_depth=5, min_child_weight=30, pinball 0.1688) - a 2.5% improvement.
This runs that "tuned" config through the actual full 4-fold/26-state backtest to check
whether the holdout's preference generalizes, replicating the paper's methodological point
either way (LightGBM's original story was the opposite direction: a deeper config won its
holdout and then lost on the true backtest).

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ablation_hparam_true_backtest.py
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import normalized_wis, run_backtest, score_backtest, summarize
from imdc.models.xgb_quantile import XGBQuantileModel

TUNED_PARAMS = {"max_depth": 3, "min_child_weight": 50}


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(
        lambda: XGBQuantileModel(disease="dengue", params=TUNED_PARAMS),
        folds, disease="dengue", ufs=MANDATORY_UFS,
    )
    scored = score_backtest(preds, disease="dengue", folds=folds)
    print(f"xgb_holdout_tuned backtest: {len(scored)} scored rows in {time.time()-t0:.1f}s")

    out = METRICS_DIR / "ablation_xgb_holdout_tuned_scored.csv"
    scored.to_csv(out, index=False)
    print(f"wrote {out}")

    overall = summarize(scored, by=["model"])
    by_fold = summarize(scored, by=["model", "fold_id"])
    norm_all = normalized_wis(scored, by=["model"])
    norm_ex2024 = normalized_wis(scored, by=["model"], exclude_folds=(2,))

    print("\n=== xgb_holdout_tuned: overall ===")
    print(overall.to_string(index=False))
    print("\n=== by fold ===")
    print(by_fold.sort_values("fold_id").to_string(index=False))
    print("\nnormWIS_all:", norm_all["normalized_wis"].iloc[0])
    print("normWIS_ex2024:", norm_ex2024["normalized_wis"].iloc[0])


if __name__ == "__main__":
    main()
