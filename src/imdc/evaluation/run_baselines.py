"""Run all baseline models across all 4 official folds, all 26 mandatory states.

Produces results/metrics/baselines_scored.csv (per-unit scores) and
results/metrics/baselines_leaderboard.csv (mean WIS etc. by model and by
model x fold), the first real backtest artifact for the paper's results
tables. Run as: python -m imdc.evaluation.run_baselines
"""
import time

from imdc.config import METRICS_DIR, MANDATORY_UFS
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.evaluation.harness import run_backtest, score_backtest, summarize

MODELS = [NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel]


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    folds = get_folds("dengue")

    all_scored = []
    for model_cls in MODELS:
        t0 = time.time()
        preds = run_backtest(model_cls, folds, disease="dengue", ufs=MANDATORY_UFS)
        scored = score_backtest(preds, disease="dengue", folds=folds)
        all_scored.append(scored)
        print(f"{model_cls.__name__}: {len(scored)} scored rows in {time.time()-t0:.1f}s")

    import pandas as pd

    scored_all = pd.concat(all_scored, ignore_index=True)
    scored_all.to_csv(METRICS_DIR / "baselines_scored.csv", index=False)

    by_model = summarize(scored_all, by=["model"])
    by_model_fold = summarize(scored_all, by=["model", "fold_id"])
    by_model.to_csv(METRICS_DIR / "baselines_leaderboard.csv", index=False)
    by_model_fold.to_csv(METRICS_DIR / "baselines_leaderboard_by_fold.csv", index=False)

    print("\n=== Leaderboard (mean over all states/folds/horizons) ===")
    print(by_model.sort_values("wis").to_string(index=False))
    print("\n=== By fold ===")
    print(by_model_fold.sort_values(["fold_id", "wis"]).to_string(index=False))


if __name__ == "__main__":
    main()
