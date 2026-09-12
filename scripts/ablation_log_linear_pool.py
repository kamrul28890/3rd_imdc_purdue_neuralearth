"""Check log-linear (logarithmic) pooling as an alternative to Vincentization for the deployed
ensemble, under this project's fold-1 tuning discipline.

Motivated by reading the inaugural IMDC24 sprint's own paper (Araujo et al. 2026), whose ensemble
methodology combines models via log-linear pooling of parametric predictive distributions rather
than a per-quantile median - a genuinely different combination rule (precision-weighted
distributional pooling vs. an order-statistic median), not just different weights. We already
tested weight choice (inverse_wis_weights vs. equal) on top of Vincentization; this checks the
combination RULE itself.

Four variants of `log_linear_pool` (imdc.models.ensemble), each with and without the same
conformal recalibration used on the deployed ensemble:
  - equal weights (mirrors IMDC24's E1)
  - fold-1 inverse-WIS weights (mirrors IMDC24's E2, but tuned only on the designated tuning
    fold rather than "whichever past season", the exact overfitting IMDC24's own E2 fell into)

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/ablation_log_linear_pool.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import METRICS_DIR
from imdc.evaluation.harness import normalized_wis
from imdc.evaluation.postprocess import apply_conformal_widen, conformal_widen_factors
from imdc.models.ensemble import inverse_wis_weights, log_linear_pool, score_wide, vincentization

CALIB_FOLD = 1
MEMBERS = ["climatological_quantile", "xgb_quantile", "gru_negbin"]


def _load_all():
    frames = []
    for f in ["baselines_scored.csv", "xgb_scored.csv", "gru_scored.csv"]:
        frames.append(pd.read_csv(METRICS_DIR / f, parse_dates=["date", "origin_date"]))
    allpreds = pd.concat(frames, ignore_index=True)
    return allpreds[allpreds["model"].isin(MEMBERS)]


def _report(label, wide):
    cfactors = conformal_widen_factors(wide[wide["fold_id"] == CALIB_FOLD])
    raw = score_wide(wide)
    cal = score_wide(apply_conformal_widen(wide, cfactors))

    def _summary(scored):
        fold1 = scored[scored["fold_id"] == CALIB_FOLD]["wis"].mean()
        all_wis = scored["wis"].mean()
        n_all = normalized_wis(scored.assign(model="x"), by=["model"])["normalized_wis"].iloc[0]
        n_ex2024 = normalized_wis(scored.assign(model="x"), by=["model"], exclude_folds=(2,))["normalized_wis"].iloc[0]
        return fold1, all_wis, n_all, n_ex2024

    f1, a1, n1, ne1 = _summary(raw)
    print(f"{label:32s} raw:       fold1={f1:7.2f}  all={a1:8.2f}  normWIS_all={n1:.4f}  normWIS_ex2024={ne1:.4f}")
    f2, a2, n2, ne2 = _summary(cal)
    print(f"{label:32s} conformal: fold1={f2:7.2f}  all={a2:8.2f}  normWIS_all={n2:.4f}  normWIS_ex2024={ne2:.4f}  "
          f"factors={ {k: round(v, 3) for k, v in cfactors.items()} }")


def main():
    allpreds = _load_all()

    _report("vincentization (deployed)", vincentization(allpreds, MEMBERS))

    _report("log-linear, equal weights", log_linear_pool(allpreds, MEMBERS))

    invwis_w = inverse_wis_weights(allpreds, MEMBERS, tuning_fold=CALIB_FOLD)
    print(f"\nfold-{CALIB_FOLD} inverse-WIS weights: {{{', '.join(f'{k}:{v:.3f}' for k, v in invwis_w.items())}}}\n")
    _report("log-linear, fold-1 invWIS weights", log_linear_pool(allpreds, MEMBERS, weights=invwis_w))

    print("\n(compare against the deployed conformal ensemble: fold1_wis=337.27, all_wis=1162.20, "
          "normWIS_all=0.5610, normWIS_ex2024=0.3746)")


if __name__ == "__main__":
    main()
