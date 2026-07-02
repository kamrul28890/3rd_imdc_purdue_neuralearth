"""Run the GRU NegBinom sequence model across all 4 folds x 26 states.

Appends to the combined leaderboard. Run as:
    KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_dl
"""
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.models.dl_sequence import DLSequenceModel

N_ENSEMBLE = 5
EPOCHS = 30


def main():
    folds = get_folds("dengue")
    t0 = time.time()
    preds = run_backtest(
        lambda: DLSequenceModel(disease="dengue", n_ensemble=N_ENSEMBLE, epochs=EPOCHS),
        folds, disease="dengue", ufs=MANDATORY_UFS,
    )
    scored = score_backtest(preds, disease="dengue", folds=folds)
    print(f"GRU backtest: {len(scored)} scored rows in {time.time()-t0:.1f}s")
    scored.to_csv(METRICS_DIR / "gru_scored.csv", index=False)

    frames = [scored]
    for name in ["baselines_scored.csv", "lgbm_scored.csv"]:
        p = METRICS_DIR / name
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["date", "origin_date"]))
    combined = pd.concat(frames, ignore_index=True)

    by_model = summarize(combined, by=["model"]).sort_values("wis")
    by_model_fold = summarize(combined, by=["model", "fold_id"]).sort_values(["fold_id", "wis"])
    by_model.to_csv(METRICS_DIR / "combined_leaderboard.csv", index=False)
    by_model_fold.to_csv(METRICS_DIR / "combined_leaderboard_by_fold.csv", index=False)

    print("\n=== Combined leaderboard ===")
    print(by_model.to_string(index=False))
    print("\n=== By fold ===")
    print(by_model_fold.to_string(index=False))


if __name__ == "__main__":
    main()
