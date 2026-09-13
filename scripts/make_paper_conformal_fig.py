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
INK = "#0b0b0b"
C_VINC, C_CONF = "#898781", "#0b0b0b"

df = pd.read_csv("results/metrics/final_scored.csv", low_memory=False)
for c in ["wis", "fold_id"] + [f"coverage_{L}" for L in LEVELS]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

# Panel A: per-season CHANGE from recalibration, not the two levels side by side.
# Plotting the two levels on a log axis spanning ~300 to ~3600 made the actual finding invisible:
# the 7.8% outbreak-season gain and the 1-2% ordinary-season costs all render as equal-height bar
# pairs. The claim is about the asymmetry between them, so plot the difference directly, in WIS
# units, which is the scale the insurance argument is actually made in (a 277-unit gain against
# costs of 5 and 9 units).
vinc = df[df.model == "ensemble_vincent"].groupby("fold_id")["wis"].mean()
conf = df[df.model == "ensemble_conformal"].groupby("fold_id")["wis"].mean()
delta = [conf[f] - vinc[f] for f in FOLDS]
pct = [100 * (conf[f] - vinc[f]) / vinc[f] for f in FOLDS]
x = np.arange(len(FOLDS))
axA.bar(x, delta, 0.55, color=[C_CONF if d < 0 else C_VINC for d in delta])
axA.axhline(0, color=INK, lw=0.9)
for xi, (d, p) in enumerate(zip(delta, pct)):
    below = d < 0
    axA.annotate(
        f"{d:+.0f}\n({p:+.1f}%)",
        xy=(xi, d),
        xytext=(0, -20 if below else 6),
        textcoords="offset points",
        ha="center",
        fontsize=7.5,
        color=C_CONF if below else C_VINC,
        fontweight="bold" if below else "normal",
    )
axA.set_ylabel("Change in mean WIS\n(negative = recalibration helps)")
axA.set_xticks(x); axA.set_xticklabels(FOLDS.values(), fontsize=8)
axA.set_ylim(min(delta) * 1.35, max(max(delta) * 3.2, 60))
axA.set_title("A  Effect of recalibration by season", loc="left", fontsize=11, fontweight="bold")
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
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
