"""Figure: probability integral transform (PIT) calibration histograms, one panel per model.

A calibrated probabilistic forecast puts the observation uniformly across its own predictive
distribution, so each PIT bin should hold its nominal share. This complements the reliability
curve in paper_conformal.png: that plot summarizes calibration at four interval levels, while
this one shows WHERE in the distribution the miscalibration sits (a U shape means intervals are
too narrow, a central hump means too wide, a tilt means bias).

Two details this implementation gets right, both of which matter for count data reported as
quantiles rather than as a full predictive CDF:

1. The nine submitted quantile levels cut the distribution into ten bins of UNEQUAL width (the
   central bin spans 25 percentage points, the tail bins 2.5). Plotting raw counts would make the
   wide central bins look over-full by construction, so every panel plots the ratio of observed to
   expected share, where 1.0 is perfect calibration regardless of bin width.
2. Case counts are discrete and frequently tied against their own predicted quantiles (an
   all-zero-median forecast for a sparse city-week is the extreme case). A naive bin assignment
   would charge every such tie to the lowest bin and manufacture a spike there. We use the
   non-randomized PIT for discrete data instead: an observation tied against k quantiles
   contributes 1/(k+1) to each bin it could belong to, which is deterministic and reproducible,
   unlike the randomized-PIT alternative.

Run as: python scripts/make_paper_pit_fig.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imdc.config import FIGURES_DIR, QUANTILE_COLUMNS, QUANTILE_LEVELS

# Panels chosen to carry the calibration story of the main text: the overconfident deep network,
# the two calibrated single models, the mechanistic model's wide intervals, and the ensemble
# before and after conformal recalibration.
PANELS = [
    ("gru_negbin", "GRU (deep learning)"),
    ("xgb_quantile", "XGBoost"),
    ("climatological_quantile", "Climatological"),
    ("mechanistic_traj", "Mechanistic"),
    ("ensemble_vincent", "Ensemble (Vincentization)"),
    ("ensemble_conformal", "Ensemble (conformal)"),
]
C_BAR, C_REF = "#1b6ca8", "#8c8c8c"

# Bin i spans (tau_i, tau_{i+1}] with tau_0 = 0 and tau_10 = 1, so widths are the level gaps.
EDGES = np.array([0.0] + list(QUANTILE_LEVELS) + [1.0])
EXPECTED = np.diff(EDGES)
BIN_LABELS = ["<2.5", "2.5-5", "5-10", "10-25", "25-50", "50-75", "75-90", "90-95", "95-97.5", ">97.5"]


def pit_ratio(sub: pd.DataFrame) -> np.ndarray:
    """Observed-to-expected share per PIT bin, using the non-randomized PIT for discrete data."""
    q = sub[QUANTILE_COLUMNS].to_numpy(dtype=float)
    y = sub["observed_value"].to_numpy(dtype=float)[:, None]
    n_lt = (q < y).sum(axis=1)  # first bin the observation could occupy
    n_leq = (q <= y).sum(axis=1)  # last bin it could occupy; equal to n_lt when there are no ties

    mass = np.zeros(len(EXPECTED))
    for lo, hi in zip(n_lt, n_leq):
        share = 1.0 / (hi - lo + 1)
        mass[lo : hi + 1] += share
    return (mass / mass.sum()) / EXPECTED


df = pd.read_csv("results/metrics/final_scored.csv", low_memory=False)
df = df[df["observed_value"].notna()]
for c in QUANTILE_COLUMNS + ["observed_value"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=QUANTILE_COLUMNS + ["observed_value"])

fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharey=True)
x = np.arange(len(EXPECTED))

for ax, (model, title) in zip(axes.ravel(), PANELS):
    ratio = pit_ratio(df[df.model == model])
    ax.bar(x, ratio, width=0.78, color=C_BAR)
    ax.axhline(1.0, ls="--", lw=1, color=C_REF)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=6, rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    # One direct label per panel: the worst-calibrated bin. A shared y-axis keeps panels
    # comparable but is set by the GRU's outlier, so labelling the tallest bar recovers the
    # number that matters in the panels the shared scale compresses.
    top = int(np.argmax(ratio))
    ax.annotate(
        f"{ratio[top]:.1f}x",
        xy=(top, ratio[top]),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        fontsize=7.5,
        fontweight="bold",
        color=C_BAR,
    )

for ax in axes[:, 0]:
    ax.set_ylabel("Observed / expected")
for ax in axes[1, :]:
    ax.set_xlabel("Percentile of the predictive distribution", fontsize=8)

axes[0, 0].text(
    0.02, 0.93, "1.0 = calibrated", transform=axes[0, 0].transAxes, fontsize=7, color=C_REF, va="top"
)

fig.tight_layout()
out = FIGURES_DIR / "paper_pit.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)

for model, title in PANELS:
    r = pit_ratio(df[df.model == model])
    print(f"{title:28s} tails(<2.5%, >97.5%)={r[0]:.2f},{r[-1]:.2f}  centre(25-75%)={r[4]:.2f},{r[5]:.2f}")
