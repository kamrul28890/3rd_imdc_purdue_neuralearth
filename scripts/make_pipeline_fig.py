"""End-to-end pipeline overview figure for the Methods section.

Matches the visual style of the other paper figures (make_figures.py: same ink/muted/grid
palette, same font size) so it reads as part of the same figure set rather than a one-off.

Run as: python scripts/make_pipeline_fig.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from imdc.config import FIGURES_DIR

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
ACCENT = "#0b0b0b"


def _box(ax, xy, w, h, text, fc="#fcfcfb", ec=INK, fontsize=9.5, fontweight="normal"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                          linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=INK, fontweight=fontweight, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def _arrow(ax, b1, b2, y_off1=0.0, y_off2=0.0):
    x1 = b1[0] + b1[2]
    y1 = b1[1] + b1[3] / 2 + y_off1
    x2 = b2[0]
    y2 = b2[1] + b2[3] / 2 + y_off2
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                  linewidth=1.1, color=MUTED, zorder=1))


def main():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "font.size": 9.5})
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    b_data = _box(ax, (0.15, 1.85), 1.55, 0.9,
                  "Raw surveillance\n+ covariates\n(2010–2026)")
    b_cut = _box(ax, (1.95, 1.85), 1.55, 0.9,
                 "Leakage-safe\ncutoff filter\n(per fold)")
    b_panel = _box(ax, (3.75, 1.85), 1.65, 0.9,
                    "Synthetic-origin\ntraining panel\n(weekly origins ×\nhorizons 1–67)")

    b_base = _box(ax, (5.65, 3.55), 1.9, 0.6, "Climatological\n(+ naive baselines)", fontsize=8.5)
    b_lgbm = _box(ax, (5.65, 2.85), 1.9, 0.6, "XGBoost + CQR", fontsize=8.7)
    b_gru = _box(ax, (5.65, 2.15), 1.9, 0.6, "GRU deep ensemble", fontsize=8.7)
    b_mech = _box(ax, (5.65, 1.45), 1.9, 0.6, "Mechanistic\n(Richards bootstrap)", fontsize=8.5)

    b_ens = _box(ax, (7.85, 2.15), 1.35, 0.9,
                 "Ensemble\n(Vincentization:\nper-quantile\nmedian)", fontsize=8.5)
    b_conf = _box(ax, (9.5, 2.15), 1.35, 0.9,
                  "Conformal\nrecalibration\n(fold-1 factors)", fontsize=8.5)

    b_sub = _box(ax, (7.85, 0.35), 3.0, 0.9,
                 "Submission: 9 quantiles\n× state/city × week\n(validated against platform schema)",
                 fontsize=8.7)

    _arrow(ax, b_data, b_cut)
    _arrow(ax, b_cut, b_panel)
    for b in (b_base, b_lgbm, b_gru, b_mech):
        _arrow(ax, b_panel, b, y_off2=0)
    for b in (b_base, b_lgbm, b_gru):  # mechanistic is evaluated separately, not an ensemble member
        _arrow(ax, b, b_ens)
    _arrow(ax, b_ens, b_conf)

    x1 = b_conf[0] + b_conf[2] / 2
    y1 = b_conf[1]
    x2 = b_sub[0] + b_sub[2] / 2
    y2 = b_sub[1] + b_sub[3]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                  linewidth=1.1, color=MUTED,
                                  connectionstyle="arc3,rad=-0.25", zorder=1))

    ax.text(6.6, 4.35, "Five model families (fit independently per fold)", ha="center",
            fontsize=8.5, color=MUTED, style="italic")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paper_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote paper_pipeline.png")


if __name__ == "__main__":
    main()
