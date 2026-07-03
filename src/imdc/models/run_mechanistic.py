"""Run the mechanistic trajectory model across all folds x 26 states; save scored predictions.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_mechanistic
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.models.mechanistic import MechanisticTrajectoryModel


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(lambda: MechanisticTrajectoryModel(disease="dengue"), folds,
                         disease="dengue", ufs=MANDATORY_UFS)
    scored = score_backtest(preds, disease="dengue", folds=folds)
    scored.to_csv(METRICS_DIR / "mechanistic_scored.csv", index=False)
    print(f"Mechanistic backtest: {len(scored)} rows in {time.time()-t0:.1f}s")
    print(summarize(scored, by=["model", "fold_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
