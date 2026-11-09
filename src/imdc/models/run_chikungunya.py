"""Chikungunya state-level track: baselines + mechanistic + LightGBM + ensemble.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_chikungunya [--rescore]

--rescore skips the backtests and rebuilds only the derived models (ensemble, size-aware
recalibration) from the per-model rows already saved in chik_final_scored.csv.
"""
import argparse
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
from imdc.models.run_ensemble import size_aware_recalibrate

MEMBERS = ["climatological_quantile", "lgbm_quantile", "mechanistic_traj"]
BASE_MODELS = ["naive", "seasonal_naive", "climatological_quantile", "mechanistic_traj", "lgbm_quantile"]


def _backtest():
    folds = get_folds("chikungunya")
    specs = [NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel,
             lambda: MechanisticTrajectoryModel(disease="chikungunya"),
             lambda: LGBMQuantileModel(disease="chikungunya")]
    frames = []
    for fac in specs:
        preds = run_backtest(fac, folds, disease="chikungunya", ufs=MANDATORY_UFS)
        frames.append(score_backtest(preds, disease="chikungunya", folds=folds))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    if args.rescore:
        allc = pd.read_csv(METRICS_DIR / "chik_final_scored.csv", parse_dates=["date", "origin_date"])
        allc = allc[allc["model"].isin(BASE_MODELS)].reset_index(drop=True)
    else:
        allc = _backtest()

    ens = score_wide(vincentization(allc, MEMBERS)); ens["model"] = "ensemble_vincent"
    # LightGBM alone is the deployed chikungunya model (it beats the ensemble); its size-aware
    # recalibration is tuned on fold 1 exactly as for the dengue ensemble (run_ensemble.py).
    lgbm_sa, sa_params, sa_fit = size_aware_recalibrate(allc, "lgbm_quantile", "lgbm_size_aware", "chikungunya")
    lgbm_sa["disease"] = "chikungunya"
    sa_params.to_csv(METRICS_DIR / "chik_size_aware_params.csv", index=False)
    print(f"size-aware scale (tuned on fold 1): a={sa_fit['a']:.2f} b={sa_fit['b']:.2f} "
          f"(fold-1 pooled WIS {sa_fit['pooled_wis_unscaled']:.2f} -> {sa_fit['pooled_wis']:.2f})")
    combined = pd.concat([allc, ens, lgbm_sa], ignore_index=True)
    combined.to_csv(METRICS_DIR / "chik_final_scored.csv", index=False)
    combined.groupby("model")[["wis", "coverage_50", "coverage_90"]].mean().sort_values("wis")\
        .to_csv(METRICS_DIR / "chik_final_leaderboard.csv")
    print(summarize(combined, by=["model"])[["model", "wis"]].sort_values("wis").to_string(index=False))


if __name__ == "__main__":
    main()
