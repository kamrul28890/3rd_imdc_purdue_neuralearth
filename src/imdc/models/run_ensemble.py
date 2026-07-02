"""Build and score ensembles from saved per-model predictions; write final leaderboard.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_ensemble
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import METRICS_DIR
from imdc.models.ensemble import inverse_wis_weights, score_wide, vincentization, weighted_ensemble

MEMBERS = ["climatological_quantile", "lgbm_quantile", "gru_negbin"]
SCORE_COLS = ["wis", "coverage_50", "coverage_80", "coverage_90", "coverage_95"]


def _load_all():
    frames = []
    for f in ["baselines_scored.csv", "lgbm_scored.csv", "gru_scored.csv"]:
        frames.append(pd.read_csv(METRICS_DIR / f, parse_dates=["date", "origin_date"]))
    return pd.concat(frames, ignore_index=True)


def main():
    allpreds = _load_all()

    vincent = score_wide(vincentization(allpreds, MEMBERS))
    weights = inverse_wis_weights(allpreds, MEMBERS, tuning_fold=1)
    invwis = score_wide(weighted_ensemble(allpreds, weights))

    combined = pd.concat([allpreds, vincent, invwis], ignore_index=True)
    combined.to_csv(METRICS_DIR / "final_scored.csv", index=False)

    leaderboard = combined.groupby("model")[SCORE_COLS].mean().reset_index().sort_values("wis")
    leaderboard.to_csv(METRICS_DIR / "final_leaderboard.csv", index=False)
    by_fold = combined.groupby(["model", "fold_id"])["wis"].mean().unstack("fold_id")
    by_fold.to_csv(METRICS_DIR / "final_leaderboard_by_fold.csv")

    print("Inverse-WIS weights (tuned on fold 1):", {k: round(v, 3) for k, v in weights.items()})
    print("\n=== Final leaderboard (overall mean WIS) ===")
    print(leaderboard.to_string(index=False))
    print("\n=== WIS by fold ===")
    print(by_fold.round(0).to_string())

    # SI robustness: inverse-WIS weights tuned on each fold in turn (leave-one-out sensitivity)
    print("\n=== Weight-tuning robustness (invWIS weights by tuning fold) ===")
    for tf in [1, 2, 3]:
        w = inverse_wis_weights(allpreds, MEMBERS, tuning_fold=tf)
        parts = ", ".join("{}:{:.2f}".format(k.split("_")[0], v) for k, v in w.items())
        print(f"  tune on fold {tf}: {{{parts}}}")


if __name__ == "__main__":
    main()
