"""Two Tier-1 paper figures: statistical significance and robustness (horizon + decomposition)."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR, METRICS_DIR

INK, MUTED = "#0b0b0b", "#898781"
LAB = {"ensemble_conformal": "Ensemble (conformal)", "ensemble_vincent": "Ensemble (median)",
       "xgb_quantile": "XGBoost", "mechanistic_traj": "Mechanistic",
       "climatological_quantile": "Climatological", "gru_negbin": "GRU",
       "seasonal_naive": "Seasonal-naive", "naive": "Naive"}
COL = {"ensemble_conformal": "#0b0b0b", "gru_negbin": "#1baf7a", "xgb_quantile": "#e34948",
       "climatological_quantile": "#2a78d6"}

# Bootstrap results from paper_tier1_stats.py (block bootstrap over (state,season) tasks, seed 42).
NWIS_CI = {  # model: (point, lo, hi)
    "ensemble_conformal": (0.561, 0.423, 0.639), "ensemble_vincent": (0.595, 0.438, 0.684),
    "xgb_quantile": (0.575, 0.447, 0.654), "mechanistic_traj": (0.598, 0.491, 0.657),
    "climatological_quantile": (0.610, 0.475, 0.687), "gru_negbin": (0.673, 0.460, 0.797),
    "seasonal_naive": (0.666, 0.535, 0.737), "naive": (0.752, 0.683, 0.802)}
PAIRED = [  # label, diff, lo, hi
    ("Conformal $-$ median ensemble", -71.3, -87.8, -56.9),
    ("Conformal $-$ XGBoost", -27.9, -47.3, -8.8),
    ("GRU $-$ XGBoost (ordinary seasons)", -48.9, -87.4, -14.9),
    ("XGBoost $-$ LightGBM", -111.1, -143.0, -81.6)]


def significance_fig():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    order = sorted(NWIS_CI, key=lambda m: NWIS_CI[m][0])
    y = np.arange(len(order))
    for i, m in enumerate(order):
        p, lo, hi = NWIS_CI[m]
        axA.plot([lo, hi], [i, i], color=MUTED, lw=1.5)
        axA.plot(p, i, "o", color=COL.get(m, INK), markersize=6)
    axA.set_yticks(y); axA.set_yticklabels([LAB[m] for m in order], fontsize=9)
    axA.invert_yaxis(); axA.set_xlabel("Normalized WIS (95% bootstrap CI)")
    axA.set_title("A  Model skill with uncertainty", loc="left", fontsize=11, fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)

    yb = np.arange(len(PAIRED))
    for i, (lab, d, lo, hi) in enumerate(PAIRED):
        axB.plot([lo, hi], [i, i], color=INK, lw=1.6)
        axB.plot(d, i, "s", color="#1b6ca8", markersize=7)
    axB.axvline(0, color="#c0392b", ls="--", lw=1.2)
    axB.set_yticks(yb); axB.set_yticklabels([p[0] for p in PAIRED], fontsize=9)
    axB.invert_yaxis(); axB.set_xlabel("Paired difference in mean WIS (95% CI)")
    axB.set_title("B  Paired comparisons (left of 0 favors first)", loc="left", fontsize=11, fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_significance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper_significance.png")


def robustness_fig():
    dg = pd.read_csv(METRICS_DIR / "final_scored.csv", low_memory=False)
    for c in ["wis", "observed_value", "fold_id", "horizon_weeks", "dispersion", "overprediction", "underprediction"]:
        dg[c] = pd.to_numeric(dg[c], errors="coerce")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))
    ordn = dg[dg.fold_id != 2].copy()
    bins, labels = [15, 27, 39, 51, 67], ["16-27", "28-39", "40-51", "52-67"]
    ordn["hbin"] = pd.cut(ordn.horizon_weeks, bins, labels=labels)
    for m in ["ensemble_conformal", "gru_negbin", "xgb_quantile", "climatological_quantile"]:
        s = ordn[ordn.model == m]
        v = s.groupby("hbin", observed=True).apply(lambda x: x.wis.sum() / x.observed_value.sum(), include_groups=False)
        axA.plot(labels, [v.get(b, np.nan) for b in labels], "o-", color=COL[m], label=LAB[m], lw=2, markersize=5)
    axA.set_xlabel("Forecast horizon (weeks ahead)"); axA.set_ylabel("Normalized WIS (ordinary seasons)")
    axA.set_title("A  Skill by horizon", loc="left", fontsize=11, fontweight="bold")
    axA.legend(frameon=False, fontsize=8); axA.spines[["top", "right"]].set_visible(False)

    ec = dg[dg.model == "ensemble_conformal"]
    dec = ec.groupby("fold_id")[["dispersion", "underprediction", "overprediction"]].mean()
    folds = [1, 2, 3, 4]; x = np.arange(4)
    disp = [dec.loc[f, "dispersion"] for f in folds]
    und = [dec.loc[f, "underprediction"] for f in folds]
    ov = [dec.loc[f, "overprediction"] for f in folds]
    axB.bar(x, disp, 0.6, label="dispersion", color="#9aa7b0")
    axB.bar(x, und, 0.6, bottom=disp, label="under-prediction", color="#c0392b")
    axB.bar(x, ov, 0.6, bottom=np.array(disp) + np.array(und), label="over-prediction", color="#e1a100")
    axB.set_yscale("log"); axB.set_ylim(1, 4000)
    axB.set_xticks(x); axB.set_xticklabels(["Fold 1", "Fold 2\n(2024)", "Fold 3", "Fold 4"], fontsize=9)
    axB.set_ylabel("WIS component (log scale)")
    axB.set_title("B  WIS decomposition (conformal ensemble)", loc="left", fontsize=11, fontweight="bold")
    axB.legend(frameon=False, fontsize=8); axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_robustness.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper_robustness.png")


if __name__ == "__main__":
    significance_fig()
    robustness_fig()
