"""Regenerate the paper figures from the committed scored predictions.

Run as: python scripts/make_figures.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR, METRICS_DIR
from imdc.evaluation.metrics import relative_wis

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
C = {"naive": "#898781", "seasonal_naive": "#eda100", "climatological_quantile": "#2a78d6",
     "lgbm_quantile": "#e34948", "gru_negbin": "#1baf7a", "mechanistic_traj": "#4a3aa7",
     "ensemble_vincent": "#0b0b0b"}
LAB = {"naive": "Naive", "seasonal_naive": "Seasonal-naive", "climatological_quantile": "Climatological",
       "lgbm_quantile": "LightGBM", "gru_negbin": "GRU", "mechanistic_traj": "Mechanistic",
       "ensemble_vincent": "Ensemble"}
ORDER = list(C.keys())


def _style():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "font.size": 11})


def main():
    _style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_DIR / "final_scored.csv", parse_dates=["date"], low_memory=False)

    # Fig: WIS by model x fold (log)
    byf = df.groupby(["model", "fold_id"])["wis"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5)); folds = [1, 2, 3, 4]; w = 0.11
    for i, m in enumerate(ORDER):
        ys = [byf[(byf.model == m) & (byf.fold_id == f)]["wis"].values for f in folds]
        ys = [y[0] if len(y) else np.nan for y in ys]
        ax.bar([f + (i - 3) * w for f in folds], ys, w, color=C[m], label=LAB[m])
    ax.set_yscale("log"); ax.set_xticks(folds)
    ax.set_xticklabels(["Fold 1\n2022-23", "Fold 2\n2023-24 (2024 outlier)", "Fold 3\n2024-25", "Fold 4\n2025-26 (partial)"])
    ax.set_ylabel("Mean WIS (log)"); ax.set_title("Weighted interval score by model and season", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_wis_by_fold.png", dpi=150, bbox_inches="tight"); plt.close()

    # Fig: coverage diagram
    fig, ax = plt.subplots(figsize=(6, 6)); noms = [50, 80, 90, 95]
    for m in ORDER:
        emp = [df[df.model == m][f"coverage_{L}"].mean() * 100 for L in noms]
        ax.plot(noms, emp, marker="o", color=C[m], label=LAB[m], lw=2, markersize=5)
    ax.plot([45, 100], [45, 100], color=MUTED, ls="--", lw=1, label="Perfect")
    ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Interval calibration", loc="left", fontsize=12); ax.legend(frameon=False, fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.5); plt.tight_layout(); plt.savefig(FIGURES_DIR / "paper_coverage.png", dpi=150); plt.close()

    # Fig: relative WIS on headline folds 2-3
    h = df[df.fold_id.isin([2, 3])]
    rel = relative_wis(h, baseline_model="naive", group_cols=["fold_id", "uf", "horizon_weeks"])
    ms = [m for m in ORDER if m in rel.index]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(range(len(ms)), [rel[m] for m in ms], color=[C[m] for m in ms])
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[m] for m in ms])
    ax.set_xlabel("Relative WIS vs naive (folds 2-3, scale-free; lower=better)")
    ax.set_title("Scale-free skill on headline seasons", loc="left", fontsize=12)
    for i, m in enumerate(ms):
        ax.text(rel[m] + 0.008, i, f"{rel[m]:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "paper_relative_wis.png", dpi=150); plt.close()
    print("Wrote paper_wis_by_fold.png, paper_coverage.png, paper_relative_wis.png")


if __name__ == "__main__":
    main()
