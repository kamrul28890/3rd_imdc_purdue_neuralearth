"""Figure: effect of conformal recalibration on the dengue ensemble.

Panel A: mean WIS by season, Vincentization ensemble vs conformal-recalibrated ensemble (log y).
Panel B: reliability curve (empirical vs nominal coverage) for both, against the diagonal.
Reads the committed results/metrics/final_scored.csv. Publication style: clean, grayscale-safe.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR

LEVELS = [50, 80, 90, 95]
FOLDS = {1: "2022-23", 2: "2023-24\n(2024 outlier)", 3: "2024-25", 4: "2025-26\n(partial)"}
# Entity colours must match make_paper_tier1_figs.py's COL map, or the same model wears two
# different colours across the paper's figures. ensemble_conformal is INK there, so it is INK here.
# ensemble_vincent has no entity colour assigned, and MUTED is apt for it: this figure's job is
# before/after, so the superseded version should read as recessive. INK against MUTED also
# separates by lightness, which survives greyscale printing and every common CVD type, where the
# previous blue-against-grey pairing did not.
C_VINC, C_CONF = "#898781", "#0b0b0b"

df = pd.read_csv("results/metrics/final_scored.csv", low_memory=False)
for c in ["wis", "fold_id"] + [f"coverage_{L}" for L in LEVELS]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

# Panel A: WIS by fold
vinc = df[df.model == "ensemble_vincent"].groupby("fold_id")["wis"].mean()
conf = df[df.model == "ensemble_conformal"].groupby("fold_id")["wis"].mean()
x = np.arange(len(FOLDS)); w = 0.38
axA.bar(x - w / 2, [vinc[f] for f in FOLDS], w, label="Vincentization ensemble", color=C_VINC)
axA.bar(x + w / 2, [conf[f] for f in FOLDS], w, label="Conformal-recalibrated", color=C_CONF)
axA.set_yscale("log"); axA.set_ylabel("Mean WIS (log scale)")
axA.set_xticks(x); axA.set_xticklabels(FOLDS.values(), fontsize=8)
axA.set_title("A  Accuracy by season", loc="left", fontsize=11, fontweight="bold")
axA.legend(frameon=False, fontsize=8, loc="upper left")
axA.spines[["top", "right"]].set_visible(False)

# Panel B: reliability
def cov(model):
    s = df[df.model == model]
    return [s[f"coverage_{L}"].mean() * 100 for L in LEVELS]
axB.plot([40, 100], [40, 100], ls="--", color="0.6", lw=1, label="perfect calibration")
axB.plot(LEVELS, cov("ensemble_vincent"), "o-", color=C_VINC, label="Vincentization ensemble")
axB.plot(LEVELS, cov("ensemble_conformal"), "s-", color=C_CONF, label="Conformal-recalibrated")
axB.set_xlabel("Nominal coverage (%)"); axB.set_ylabel("Empirical coverage (%)")
axB.set_xticks(LEVELS); axB.set_title("B  Interval calibration", loc="left", fontsize=11, fontweight="bold")
axB.legend(frameon=False, fontsize=8, loc="upper left")
axB.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = FIGURES_DIR / "paper_conformal.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
