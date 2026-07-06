"""Regenerate the paper figures from the committed scored predictions.

Run as: python scripts/make_figures.py

Figures written to results/figures/:
  paper_wis_by_fold.png    mean WIS by model and season (log), including the conformal ensemble
  paper_coverage.png       interval calibration by model
  paper_relative_wis.png   metric-disagreement slope chart (mean WIS vs normalized WIS)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR, METRICS_DIR

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
C = {"naive": "#b8b6ae", "seasonal_naive": "#eda100", "climatological_quantile": "#2a78d6",
     "lgbm_quantile": "#e34948", "gru_negbin": "#1baf7a", "mechanistic_traj": "#4a3aa7",
     "ensemble_vincent": "#898781", "ensemble_conformal": "#0b0b0b"}
LAB = {"naive": "Naive", "seasonal_naive": "Seasonal-naive", "climatological_quantile": "Climatological",
       "lgbm_quantile": "LightGBM", "gru_negbin": "GRU", "mechanistic_traj": "Mechanistic",
       "ensemble_vincent": "Ensemble (median)", "ensemble_conformal": "Ensemble (conformal)"}
ORDER = list(C.keys())
SLOPE = ["ensemble_conformal", "ensemble_vincent", "climatological_quantile",
         "lgbm_quantile", "mechanistic_traj", "gru_negbin"]


def _style():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "font.size": 11})


def _wis_by_fold(df):
    byf = df.groupby(["model", "fold_id"])["wis"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    folds = [1, 2, 3, 4]
    n = len(ORDER)
    w = 0.80 / n
    for i, m in enumerate(ORDER):
        ys = [byf[(byf.model == m) & (byf.fold_id == f)]["wis"].values for f in folds]
        ys = [y[0] if len(y) else np.nan for y in ys]
        ax.bar([f + (i - (n - 1) / 2) * w for f in folds], ys, w, color=C[m], label=LAB[m])
    ax.set_yscale("log")
    ax.set_xticks(folds)
    ax.set_xticklabels(["Fold 1\n2022-23", "Fold 2\n2023-24 (2024 outlier)", "Fold 3\n2024-25", "Fold 4\n2025-26 (partial)"])
    ax.set_ylabel("Mean WIS (log scale)")
    ax.set_title("Weighted interval score by model and season", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_wis_by_fold.png", dpi=150, bbox_inches="tight")
    plt.close()


def _coverage(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    noms = [50, 80, 90, 95]
    for m in ORDER:
        emp = [df[df.model == m][f"coverage_{L}"].mean() * 100 for L in noms]
        ax.plot(noms, emp, marker="o", color=C[m], label=LAB[m], lw=1.8, markersize=5)
    ax.plot([45, 100], [45, 100], color=MUTED, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Interval calibration", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_coverage.png", dpi=150)
    plt.close()


def _metric_disagreement(df):
    """Slope chart: model rank by magnitude-weighted mean WIS (all seasons) vs by
    normalized WIS on the ordinary seasons. Lines that cross show the reordering."""
    raw = df.groupby("model")["wis"].mean()
    g = df[df.fold_id != 2].groupby("model")
    nrm = g["wis"].sum() / g["observed_value"].sum()
    raw_rank = raw[SLOPE].rank()
    nrm_rank = nrm[SLOPE].rank()

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in SLOPE:
        y0, y1 = raw_rank[m], nrm_rank[m]
        ax.plot([0, 1], [y0, y1], "-o", color=C[m], lw=2.2, markersize=7)
        ax.text(-0.04, y0, f"{LAB[m]}  ({raw[m]:.0f})", ha="right", va="center", fontsize=9.5, color=C[m])
        ax.text(1.04, y1, f"{LAB[m]}  ({nrm[m]:.3f})", ha="left", va="center", fontsize=9.5, color=C[m])
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(len(SLOPE) + 0.5, 0.5)  # rank 1 (best) at top
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mean WIS\n(all seasons)", "Normalized WIS\n(ordinary seasons)"], fontsize=10)
    ax.set_yticks([])
    ax.set_title("The best model depends on the metric", loc="left", fontsize=12)
    ax.text(0.5, 0.2, "rank 1 = best (top)", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_relative_wis.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    _style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_DIR / "final_scored.csv", parse_dates=["date"], low_memory=False)
    for c in ["wis", "observed_value", "fold_id"] + [f"coverage_{L}" for L in [50, 80, 90, 95]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _wis_by_fold(df)
    _coverage(df)
    _metric_disagreement(df)
    print("Wrote paper_wis_by_fold.png, paper_coverage.png, paper_relative_wis.png")


if __name__ == "__main__":
    main()
