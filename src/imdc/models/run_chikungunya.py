"""Chikungunya state-level track: baselines + mechanistic + LightGBM + ensemble.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_chikungunya
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.models.ensemble import score_wide, vincentization
from imdc.models.mechanistic import MechanisticTrajectoryModel
from imdc.models.ml_boosted import LGBMQuantileModel

MEMBERS = ["climatological_quantile", "lgbm_quantile", "mechanistic_traj"]


def main():
    folds = get_folds("chikungunya")
    specs = [NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel,
             lambda: MechanisticTrajectoryModel(disease="chikungunya"),
             lambda: LGBMQuantileModel(disease="chikungunya")]
    frames = []
    for fac in specs:
        preds = run_backtest(fac, folds, disease="chikungunya", ufs=MANDATORY_UFS)
        frames.append(score_backtest(preds, disease="chikungunya", folds=folds))
    allc = pd.concat(frames, ignore_index=True)

    ens = score_wide(vincentization(allc, MEMBERS)); ens["model"] = "ensemble_vincent"
    combined = pd.concat([allc, ens], ignore_index=True)
    combined.to_csv(METRICS_DIR / "chik_final_scored.csv", index=False)
    combined.groupby("model")[["wis", "coverage_50", "coverage_90"]].mean().sort_values("wis")\
        .to_csv(METRICS_DIR / "chik_final_leaderboard.csv")
    print(summarize(combined, by=["model"])[["model", "wis"]].sort_values("wis").to_string(index=False))


if __name__ == "__main__":
    main()
