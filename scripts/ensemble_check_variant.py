"""Check candidate XGBoost variants (from the ablation scripts) at the ENSEMBLE level, under
this project's fold-1 tuning discipline - a standalone-model improvement doesn't always survive
being combined into the ensemble (dl_sequence_wis.py's gru_wis was a worse ensemble member
despite being a better standalone model; see docs/FUTURE_WORK.md Sec 6c).

For each candidate xgb_*_scored.csv, rebuilds the exact same ensemble run_ensemble.py does
(climatological_quantile + <candidate xgb> + gru_negbin, Vincentized, conformal-recalibrated on
fold 1) and reports fold-1 WIS + full-aggregate WIS + normWIS, so they can be compared against
the currently deployed ensemble (fold-1 wis=337.27, all-fold wis=1162.20, normWIS_all=0.561).

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ensemble_check_variant.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import METRICS_DIR
from imdc.evaluation.harness import normalized_wis
from imdc.evaluation.postprocess import apply_conformal_widen, conformal_widen_factors
from imdc.models.ensemble import score_wide, vincentization

CALIB_FOLD = 1
MEMBERS = ["climatological_quantile", "xgb_quantile", "gru_negbin"]

CANDIDATES = {
    "deployed_default": "xgb_scored.csv",
    "no_climate": "ablation_xgb_no_climate_scored.csv",
    "shallower_hparams": "ablation_xgb_holdout_tuned_scored.csv",
    "combined_best": "ablation_xgb_combined_best_scored.csv",
}


def _load(name):
    p = METRICS_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date", "origin_date"])


def main():
    climatological = _load("baselines_scored.csv")
    climatological = climatological[climatological["model"] == "climatological_quantile"]
    gru = _load("gru_scored.csv")

    results = []
    for label, fname in CANDIDATES.items():
        xgb = _load(fname)
        if xgb is None:
            print(f"{label}: SKIPPED ({fname} not found yet)")
            continue
        xgb = xgb.copy()
        xgb["model"] = "xgb_quantile"  # normalize candidate's model tag to match MEMBERS

        allpreds = pd.concat([climatological, xgb, gru], ignore_index=True)
        vincent_wide = vincentization(allpreds, MEMBERS)
        cfactors = conformal_widen_factors(vincent_wide[vincent_wide["fold_id"] == CALIB_FOLD])
        vincent_cal = score_wide(apply_conformal_widen(vincent_wide, cfactors))

        fold1 = vincent_cal[vincent_cal["fold_id"] == CALIB_FOLD]["wis"].mean()
        all_wis = vincent_cal["wis"].mean()
        n_all = normalized_wis(vincent_cal.assign(model="x"), by=["model"])["normalized_wis"].iloc[0]
        n_ex2024 = normalized_wis(vincent_cal.assign(model="x"), by=["model"], exclude_folds=(2,))["normalized_wis"].iloc[0]
        results.append((label, fold1, all_wis, n_all, n_ex2024))
        print(f"{label}: fold1_wis={fold1:.2f}  all_wis={all_wis:.2f}  "
              f"normWIS_all={n_all:.4f}  normWIS_ex2024={n_ex2024:.4f}  "
              f"conformal_factors={ {k: round(v,3) for k,v in cfactors.items()} }")

    print("\n(compare against currently deployed ensemble: fold1_wis=337.27, all_wis=1162.20, "
          "normWIS_all=0.5610, normWIS_ex2024=0.3746)")
    if results:
        best = min(results, key=lambda r: r[1])  # best on fold 1 - the tuning discipline
        print(f"\nbest on fold 1 (the tuning-discipline metric): {best[0]}")


if __name__ == "__main__":
    main()
