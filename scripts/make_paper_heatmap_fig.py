"""Model x state performance heatmap (SI figure).

Per-state model comparison is currently only a big table; a heatmap of each model's skill
relative to the seasonal-naive baseline, one row per model and one column per state, is far more
scannable (following the same idea as the IMDC organizers' own per-state chikungunya heatmap shown
at the 2026-07-31 validation webinar). Color is the same log(relative WIS) used in
`_stability()` (make_figures.py) and Fig 3 of the main text, but broken out by state instead of
collapsed to a mean/std - so this SI figure and the main-text stability figure are two views of
the same underlying quantity.

Run as: python scripts/make_paper_heatmap_fig.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR, METRICS_DIR
from imdc.evaluation.metrics import relative_wis

BASELINE = "seasonal_naive"
MODELS = ["climatological_quantile", "xgb_quantile", "gru_negbin", "mechanistic_traj",
          "ensemble_vincent", "ensemble_conformal"]
LAB = {"climatological_quantile": "Climatological", "xgb_quantile": "XGBoost",
       "gru_negbin": "GRU", "mechanistic_traj": "Mechanistic",
       "ensemble_vincent": "Ensemble (median)", "ensemble_conformal": "Ensemble (conformal)"}


def _per_state_log_rw(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for f in sorted(df["fold_id"].dropna().unique()):
        sub = df[df["fold_id"] == f]
        for uf, g in sub.groupby("uf"):
            rw = relative_wis(g, baseline_model=BASELINE, group_cols=["horizon_weeks"])
            records += [{"model": m, "uf": uf, "fold_id": f, "log_rw": np.log(v)} for m, v in rw.items()]
    long = pd.DataFrame(records)
    return long.groupby(["model", "uf"])["log_rw"].mean().unstack("uf")


def main():
    df = pd.read_csv(METRICS_DIR / "final_scored.csv", low_memory=False)
    pivot = _per_state_log_rw(df[df["model"].isin(MODELS + [BASELINE])])
    pivot = pivot.reindex(index=MODELS)
    pivot = pivot[sorted(pivot.columns)]
    pivot.to_csv(METRICS_DIR / "log_relative_wis_by_state.csv")

    fig, ax = plt.subplots(figsize=(12, 3.2))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([LAB[m] for m in pivot.index], fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Mean log(relative WIS vs. seasonal-naive)\n(green = better, red = worse)", fontsize=8)
    ax.set_title("Per-state skill relative to the seasonal-naive baseline, averaged over four seasons",
                 loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_state_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote paper_state_heatmap.png")


if __name__ == "__main__":
    main()
