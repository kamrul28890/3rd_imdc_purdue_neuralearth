"""Build and score ensembles from saved per-model predictions; write final leaderboard.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_ensemble
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import METRICS_DIR
from imdc.evaluation.harness import normalized_wis
from imdc.evaluation.postprocess import apply_conformal_widen, conformal_widen_factors
from imdc.models.ensemble import inverse_wis_weights, score_wide, vincentization, weighted_ensemble

# Fold used to calibrate ensemble weights AND the conformal recalibration factors (kept out of
# the headline folds it is reported on, per the tuning-fold discipline).
CALIB_FOLD = 1

# Ensemble members chosen by fold-1 (tuning) performance + scale-free relative-WIS, NOT by
# overall mean WIS (which is dominated by the outlier fold). The mechanistic model is loaded
# as a single-model comparison but deliberately excluded from the ensemble: adding it hurts
# both the fold-1 score and the relative-WIS on the headline folds (it is strong on raw mean
# only because of the outlier fold's magnitude).
#
# 2026-09-08: lgbm_quantile -> xgb_quantile. Found during an overnight model sweep (see
# docs/FUTURE_WORK.md Sec 6c) that XGBoost's native multi-quantile objective, on the identical
# feature set, is a clean win as a drop-in replacement for LightGBM: better on fold 1 (the
# tuning fold - wis 337.3 vs 340.8, nWIS 0.306 vs 0.309) AND on the full-aggregate metrics
# (wis 1162.2 vs 1173.4, normWIS_all 0.561 vs 0.566, normWIS_ex2024 0.375 vs 0.381) - a genuine
# single-substitution improvement validated under the same tuning-fold discipline as every other
# ensemble-composition decision here, not a search artifact (contrast with several other overnight
# findings that looked good on aggregate metrics but failed the fold-1 check). lgbm_quantile is
# still loaded below for the leaderboard's own comparison table.
MEMBERS = ["climatological_quantile", "xgb_quantile", "gru_negbin"]
SCORE_COLS = ["wis", "coverage_50", "coverage_80", "coverage_90", "coverage_95"]


def _load_all():
    frames = []
    for f in ["baselines_scored.csv", "lgbm_scored.csv", "xgb_scored.csv", "gru_scored.csv",
              "mechanistic_scored.csv"]:
        p = METRICS_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["date", "origin_date"]))
    return pd.concat(frames, ignore_index=True)


def main():
    allpreds = _load_all()

    vincent_wide = vincentization(allpreds, MEMBERS)
    vincent = score_wide(vincent_wide)
    weights = inverse_wis_weights(allpreds, MEMBERS, tuning_fold=CALIB_FOLD)
    invwis = score_wide(weighted_ensemble(allpreds, weights))

    # Conformal recalibration of the ensemble's intervals: factors tuned on the tuning fold and
    # applied to all folds. Widens mainly the outer tails, insuring against catastrophic
    # underprediction in an extreme season (2024) for a small cost in normal ones -> better WIS
    # and near-nominal coverage. This is the recommended submission variant.
    cfactors = conformal_widen_factors(vincent_wide[vincent_wide["fold_id"] == CALIB_FOLD])
    vincent_cal = score_wide(apply_conformal_widen(vincent_wide, cfactors))
    vincent_cal["model"] = "ensemble_conformal"
    pd.Series(cfactors, name="factor").rename_axis("interval_level").to_csv(METRICS_DIR / "conformal_factors.csv")

    combined = pd.concat([allpreds, vincent, invwis, vincent_cal], ignore_index=True)
    combined.to_csv(METRICS_DIR / "final_scored.csv", index=False)

    leaderboard = combined.groupby("model")[SCORE_COLS].mean().reset_index().sort_values("wis")
    leaderboard.to_csv(METRICS_DIR / "final_leaderboard.csv", index=False)
    by_fold = combined.groupby(["model", "fold_id"])["wis"].mean().unstack("fold_id")
    by_fold.to_csv(METRICS_DIR / "final_leaderboard_by_fold.csv")

    # Official challenge metric: normalized WIS (sum WIS / sum cases), reported all-folds and
    # excluding the atypical 2024 outlier (fold 2), which dominates the raw mean. This de-biases
    # the leaderboard away from high-burden states and matches how the organizers rank models.
    nwis = (
        normalized_wis(combined, by=["model"]).rename(columns={"normalized_wis": "normWIS_all"})
        .merge(
            normalized_wis(combined, by=["model"], exclude_folds=(2,))
            .rename(columns={"normalized_wis": "normWIS_ex2024"}),
            on="model",
        )
        .sort_values("normWIS_all")
    )
    nwis.to_csv(METRICS_DIR / "final_leaderboard_normalized.csv", index=False)

    print(f"Inverse-WIS weights (tuned on fold {CALIB_FOLD}):", {k: round(v, 3) for k, v in weights.items()})
    print(f"Conformal widen factors (tuned on fold {CALIB_FOLD}):", {k: round(v, 2) for k, v in cfactors.items()})
    print("\n=== Final leaderboard (overall mean WIS) ===")
    print(leaderboard.to_string(index=False))
    print("\n=== Official NORMALIZED WIS (sum WIS / sum cases; lower=better) ===")
    print(nwis.round(4).to_string(index=False))
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
