"""Run the LightGBM quantile model across all 4 folds x 26 mandatory states.

Appends to the baseline leaderboard for a combined comparison. Run as:
    KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_ml
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.models.ml_boosted import LGBMQuantileModel


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(lambda: LGBMQuantileModel(disease="dengue"), folds, disease="dengue", ufs=MANDATORY_UFS)
    scored = score_backtest(preds, disease="dengue", folds=folds)
    print(f"LGBM backtest: {len(scored)} scored rows in {time.time()-t0:.1f}s")

    scored.to_csv(METRICS_DIR / "lgbm_scored.csv", index=False)

    # Combined leaderboard with baselines, if present
    frames = [scored]
    baseline_path = METRICS_DIR / "baselines_scored.csv"
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path, parse_dates=["date", "origin_date"]))
    combined = pd.concat(frames, ignore_index=True)

    by_model = summarize(combined, by=["model"]).sort_values("wis")
    by_model_fold = summarize(combined, by=["model", "fold_id"]).sort_values(["fold_id", "wis"])
    by_model.to_csv(METRICS_DIR / "combined_leaderboard.csv", index=False)
    by_model_fold.to_csv(METRICS_DIR / "combined_leaderboard_by_fold.csv", index=False)

    print("\n=== Combined leaderboard (mean over all states/folds/horizons) ===")
    print(by_model.to_string(index=False))
    print("\n=== By fold ===")
    print(by_model_fold.to_string(index=False))


if __name__ == "__main__":
    main()
