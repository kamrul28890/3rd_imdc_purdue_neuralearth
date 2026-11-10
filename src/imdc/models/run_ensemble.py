"""Build and score ensembles from saved per-model predictions; write final leaderboard.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.models.run_ensemble
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import METRICS_DIR, QUANTILE_COLUMNS
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import get_folds
from imdc.data.loaders import load_cases
from imdc.evaluation.harness import normalized_wis
from imdc.evaluation.postprocess import SIZE_AWARE_BOUNDS, apply_conformal_widen, apply_size_aware_scale, \
    conformal_widen_factors, fit_size_aware_scale, state_burden_index
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
# 2026-11: SARIMAX-augmented ensemble (paper Sec. "sizeaware"). The per-fold-order SARIMAX v2 is
# erratic alone but strongest exactly where the three members fail (2025-26), so as a fourth
# member under the per-quantile median it improves the median rather than just the intervals.
# Adding it lowers fold-1 (tuning) pooled WIS, 337.9 -> 327.6, so the tuning-fold rule accepts
# it; but it was tried after the organizers' validation results, so it is a post-hoc variant
# whose clean test is the frozen 2026-27 forecast, not the deployed ensemble.
SARIMAX_MEMBER = "sarimax_v2"
MEMBERS_SARIMAX = MEMBERS + [SARIMAX_MEMBER]
SCORE_COLS = ["wis", "coverage_50", "coverage_80", "coverage_90", "coverage_95"]


def size_aware_recalibrate(scored: pd.DataFrame, base_model: str, name: str, disease: str,
                           calib_fold: int = CALIB_FOLD) -> tuple:
    """Size-aware (burden-scaled) recalibration of `base_model`'s scored backtest predictions.

    Burden index from state cases up to the calibration fold's training cutoff; (a, b) fitted on
    the calibration fold only (see postprocess.fit_size_aware_scale), then applied unchanged to
    every fold. Returns (scored frame labelled `name`, params frame: one row per UF with a, b,
    burden_z and the resulting scale s_u).
    """
    cutoff = next(f for f in get_folds(disease) if f.id == calib_fold).train_cutoff
    z = state_burden_index(aggregate_cases_to_state(load_cases(disease)), cutoff)
    base = scored[scored["model"] == base_model].drop(columns=["model"]).reset_index(drop=True)
    keep = [c for c in base.columns if c in ("disease", "uf", "date", "fold_id", "horizon_weeks",
                                               "origin_date", "observed_value") or c in QUANTILE_COLUMNS]
    base = base[keep]
    fit = fit_size_aware_scale(base[base["fold_id"] == calib_fold].reset_index(drop=True), z)
    out = score_wide(apply_size_aware_scale(base, z, fit["a"], fit["b"]))
    out["model"] = name
    params = z.rename_axis("uf").reset_index().assign(
        disease=disease, base_model=base_model, a=fit["a"], b=fit["b"], calib_fold=calib_fold,
        burden_cutoff=cutoff.date().isoformat())
    params["scale"] = (params["a"] + params["b"] * params["burden_z"]).clip(*SIZE_AWARE_BOUNDS)
    return out, params, fit


def _load_all():
    frames = []
    for f in ["baselines_scored.csv", "lgbm_scored.csv", "xgb_scored.csv", "gru_scored.csv",
              "mechanistic_scored.csv", "sarimax_v2_perfold_scored.csv"]:
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

    # Size-aware recalibration (2026-11): the conformal factors above are fit to pooled coverage,
    # which SP/MG dominate, and over-widen low-burden states under a per-state metric such as the
    # organizers' Rv (paper Sec. "official"). This variant rescales each state's distribution by a
    # burden-dependent factor chosen on the tuning fold under a no-worse-pooled-WIS constraint.
    vincent_sa, sa_params, sa_fit = size_aware_recalibrate(vincent, "ensemble_vincent",
                                                           "ensemble_size_aware", "dengue")
    sa_params.to_csv(METRICS_DIR / "size_aware_params.csv", index=False)

    # SARIMAX-augmented ensemble: plain, conformal (factors refit on the tuning fold) and size-aware
    # ((a, b) refit on the tuning fold), so each recalibration is compared on the same median.
    sx_wide = vincentization(allpreds, MEMBERS_SARIMAX).assign(model="ensemble_sarimax")
    sx = score_wide(sx_wide.drop(columns=["model"])).assign(model="ensemble_sarimax")
    sx_cfactors = conformal_widen_factors(sx_wide[sx_wide["fold_id"] == CALIB_FOLD])
    sx_cal = score_wide(apply_conformal_widen(sx_wide.drop(columns=["model"]), sx_cfactors))
    sx_cal["model"] = "ensemble_sarimax_conformal"
    sx_sa, sx_sa_params, sx_sa_fit = size_aware_recalibrate(sx, "ensemble_sarimax",
                                                            "ensemble_sarimax_size_aware", "dengue")
    sx_sa_params.to_csv(METRICS_DIR / "size_aware_sarimax_params.csv", index=False)

    combined = pd.concat([allpreds, vincent, invwis, vincent_cal, vincent_sa, sx, sx_cal, sx_sa],
                         ignore_index=True)
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
    print(f"Size-aware scale (tuned on fold {CALIB_FOLD}): a={sa_fit['a']:.2f} b={sa_fit['b']:.2f} "
          f"(fold-1 pooled WIS {sa_fit['pooled_wis_unscaled']:.1f} -> {sa_fit['pooled_wis']:.1f})")
    print(f"SARIMAX-augmented: conformal factors {({k: round(v, 2) for k, v in sx_cfactors.items()})}; "
          f"size-aware a={sx_sa_fit['a']:.2f} b={sx_sa_fit['b']:.2f} "
          f"(fold-1 pooled WIS {sx_sa_fit['pooled_wis_unscaled']:.1f} -> {sx_sa_fit['pooled_wis']:.1f})")
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
