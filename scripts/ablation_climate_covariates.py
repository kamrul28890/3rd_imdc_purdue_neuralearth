"""Re-verify the paper's climate-covariate ablation (Table tab:ablation) on the current
XGBoost-based, current-data pipeline.

Original (LightGBM, pre-refresh) finding: adding observed-climate covariates to the GBM
worsened backtest WIS by 3.3% (1299 -> 1342). This script reproduces the same comparison on
XGBoost: the "with climate" arm is just the already-deployed xgb_quantile result
(results/metrics/final_scored.csv or the current WIS=1190.1 reported in FUTURE_WORK.md/
IMPROVEMENTS.md), so only the "without observed climate" arm needs an actual run here.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ablation_climate_covariates.py
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.features.panel import OBSERVED_CLIMATE_COLS
from imdc.models.xgb_quantile import XGBQuantileModel


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(
        lambda: XGBQuantileModel(disease="dengue", exclude_cols=OBSERVED_CLIMATE_COLS),
        folds, disease="dengue", ufs=MANDATORY_UFS,
    )
    scored = score_backtest(preds, disease="dengue", folds=folds)
    print(f"xgb_no_climate backtest: {len(scored)} scored rows in {time.time()-t0:.1f}s")

    out = METRICS_DIR / "ablation_xgb_no_climate_scored.csv"
    scored.to_csv(out, index=False)
    print(f"wrote {out}")

    overall = summarize(scored, by=["model"])
    by_fold = summarize(scored, by=["model", "fold_id"])
    from imdc.evaluation.harness import normalized_wis
    norm_all = normalized_wis(scored, by=["model"])
    norm_ex2024 = normalized_wis(scored, by=["model"], exclude_folds=(2,))

    print("\n=== xgb_no_climate: overall ===")
    print(overall.to_string(index=False))
    print("\n=== by fold ===")
    print(by_fold.sort_values("fold_id").to_string(index=False))
    print("\nnormWIS_all:", norm_all["normalized_wis"].iloc[0])
    print("normWIS_ex2024:", norm_ex2024["normalized_wis"].iloc[0])


if __name__ == "__main__":
    main()
