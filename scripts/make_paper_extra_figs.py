"""Two additional paper figures.

paper_design.png     study design: national weekly dengue trend + the four-fold backtest timeline.
paper_skill_map.png  state-level forecast skill of the conformal ensemble (normalized WIS choropleth).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from imdc.config import FIGURES_DIR, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.data.loaders import load_cases

INK, MUTED = "#0b0b0b", "#898781"
C_TARGET, C_GAP, C_LINE = "#2a78d6", "#f0c419", "#1b4f8a"
SHAPE = "data/raw/data_imdc_2026/shape_muni.gpkg"


def design_figure():
    cases = load_cases("dengue")
    nat = cases.groupby("date", as_index=False)["casos"].sum().sort_values("date")
    folds = get_folds("dengue")

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 6.6), gridspec_kw={"height_ratios": [2, 1.15]})
    axA.fill_between(nat["date"], nat["casos"], color=C_TARGET, alpha=0.22)
    axA.plot(nat["date"], nat["casos"], color=C_LINE, lw=1.1)
    peak = nat.loc[nat["casos"].idxmax()]
    axA.annotate(f"2024 season\n{int(peak['casos']):,} cases in one week",
                 xy=(peak["date"], peak["casos"]),
                 xytext=(peak["date"] - pd.Timedelta(days=1600), peak["casos"] * 0.82),
                 arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=9)
    axA.set_ylabel("Weekly dengue cases (national)")
    axA.set_title("A  National weekly dengue incidence, 2010-2026", loc="left", fontsize=12, fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)

    for f in folds:
        y = f.id
        c, ts, te = mdates.date2num(f.train_cutoff), mdates.date2num(f.target_start), mdates.date2num(f.target_end)
        axB.broken_barh([(c, ts - c)], (y - 0.28, 0.56), facecolors=C_GAP, alpha=0.6)
        axB.broken_barh([(ts, te - ts)], (y - 0.28, 0.56), facecolors=C_TARGET)
        axB.plot([c, c], [y - 0.36, y + 0.36], color=INK, lw=1.6)
    axB.set_yticks([1, 2, 3, 4])
    axB.set_yticklabels(["Fold 1", "Fold 2\n(2024)", "Fold 3", "Fold 4\n(partial)"], fontsize=9)
    axB.invert_yaxis()
    axB.set_xlim(mdates.date2num(pd.Timestamp("2022-01-01")), mdates.date2num(pd.Timestamp("2026-07-01")))
    axB.xaxis_date()
    axB.set_title("B  Backtest design: training cutoff, 15-week reporting gap, evaluation window",
                  loc="left", fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axB.legend(handles=[Line2D([0], [0], color=INK, lw=1.6, label="training cutoff"),
                        Patch(facecolor=C_GAP, alpha=0.6, label="15-week gap"),
                        Patch(facecolor=C_TARGET, label="evaluation window")],
               frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axB.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_design.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper_design.png")


def skill_map():
    df = pd.read_csv(METRICS_DIR / "final_scored.csv", low_memory=False)
    d = df[df.model == "ensemble_conformal"].copy()
    d["wis"] = pd.to_numeric(d["wis"], errors="coerce")
    d["observed_value"] = pd.to_numeric(d["observed_value"], errors="coerce")
    g = d.groupby("uf")
    skill = (g["wis"].sum() / g["observed_value"].sum()).rename("nwis").reset_index()

    muni = gpd.read_file(SHAPE)
    # Repair invalid polygons, then buffer out before dissolving and back in after, which closes the
    # sub-kilometre slivers between adjacent municipalities that otherwise render as white speckle.
    muni["geometry"] = muni.geometry.buffer(0).buffer(0.01)
    states = muni.dissolve(by="uf", as_index=False)[["uf", "geometry"]]
    states["geometry"] = states.geometry.buffer(-0.01).simplify(0.01)
    states = states.merge(skill, on="uf", how="left")

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    states.plot(column="nwis", cmap="YlOrRd", ax=ax, edgecolor="white", lw=0.4, legend=True,
                missing_kwds={"color": "#e0e0e0", "label": "not forecast"},
                legend_kwds={"label": "Normalized WIS (lower = better)", "shrink": 0.55})
    ax.set_title("State-level forecast skill, conformal ensemble", loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paper_skill_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper_skill_map.png")


if __name__ == "__main__":
    design_figure()
    skill_map()
