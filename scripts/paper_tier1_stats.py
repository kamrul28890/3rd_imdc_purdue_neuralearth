"""Tier-1 analytical additions for the paper: bootstrap significance, horizon stratification,
WIS decomposition, city-track and chikungunya leaderboards. Prints numbers for the manuscript.
"""
import numpy as np
import pandas as pd

M = "results/metrics"
RNG = np.random.default_rng(42)
LEVELS = [50, 80, 90, 95]


def _num(d, cols):
    for c in cols:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def leaderboard(df, models, cov=True):
    rows = []
    for m in models:
        s = df[df.model == m]
        r = {"model": m, "wis": s.wis.mean(), "nwis": s.wis.sum() / s.observed_value.sum()}
        if cov:
            for L in LEVELS:
                if f"coverage_{L}" in s:
                    r[f"c{L}"] = s[f"coverage_{L}"].mean() * 100
        rows.append(r)
    return pd.DataFrame(rows).sort_values("wis")


def bootstrap_ci(df, models, n=2000):
    """Block bootstrap over (uf, fold) tasks; 95% CI on mean WIS and normalized WIS."""
    df = df.copy()
    df["task"] = df.uf.astype(str) + "_" + df.fold_id.astype(str)
    tasks = df.task.unique()
    by_task = {t: g for t, g in df.groupby("task")}
    out = {m: {"wis": [], "nwis": []} for m in models}
    for _ in range(n):
        samp = pd.concat([by_task[t] for t in RNG.choice(tasks, len(tasks), replace=True)])
        for m in models:
            s = samp[samp.model == m]
            out[m]["wis"].append(s.wis.mean())
            out[m]["nwis"].append(s.wis.sum() / s.observed_value.sum())
    ci = {}
    for m in models:
        ci[m] = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out[m].items()}
    return ci


def paired_boot(df, a, b, n=2000, folds=None):
    """Bootstrap 95% CI on mean per-unit WIS difference (a - b); negative favors a."""
    d = df[df.fold_id.isin(folds)] if folds else df
    wide = d.pivot_table(index=["uf", "fold_id", "horizon_weeks"], columns="model", values="wis")
    wide = wide[[a, b]].dropna()
    diff = (wide[a] - wide[b]).to_numpy()
    idx = np.arange(len(diff))
    means = [diff[RNG.choice(idx, len(idx), replace=True)].mean() for _ in range(n)]
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- DENGUE STATE ----------
dg = _num(pd.read_csv(f"{M}/final_scored.csv", low_memory=False),
          ["wis", "observed_value", "fold_id", "horizon_weeks", "dispersion", "overprediction", "underprediction"] + [f"coverage_{L}" for L in LEVELS])
MODELS = ["ensemble_conformal", "ensemble_vincent", "lgbm_quantile", "mechanistic_traj",
          "climatological_quantile", "gru_negbin", "seasonal_naive", "naive"]

print("=== DENGUE bootstrap 95% CI (mean WIS; normalized WIS) ===")
ci = bootstrap_ci(dg, MODELS)
lb = leaderboard(dg, MODELS)
for _, r in lb.iterrows():
    m = r.model
    print(f"  {m:24s} WIS {r.wis:6.0f} [{ci[m]['wis'][0]:.0f},{ci[m]['wis'][1]:.0f}]   nWIS {r.nwis:.3f} [{ci[m]['nwis'][0]:.3f},{ci[m]['nwis'][1]:.3f}]")

print("\n=== DENGUE paired bootstrap (mean per-unit WIS diff, 95% CI) ===")
for a, b, fl, lab in [("ensemble_conformal", "ensemble_vincent", None, "conformal vs median (all)"),
                      ("ensemble_conformal", "lgbm_quantile", None, "conformal vs LightGBM (all)"),
                      ("gru_negbin", "lgbm_quantile", [1, 3, 4], "GRU vs LightGBM (ordinary)"),
                      ("ensemble_conformal", "naive", None, "conformal vs naive (all)")]:
    d, lo, hi = paired_boot(dg, a, b, folds=fl)
    sig = "significant" if (lo < 0) == (hi < 0) else "n.s."
    print(f"  {lab:32s} diff {d:8.1f}  [{lo:8.1f}, {hi:8.1f}]  {sig}")

print("\n=== DENGUE normalized WIS by horizon bin (ordinary seasons, folds 1/3/4) ===")
ordn = dg[dg.fold_id != 2].copy()
ordn["hbin"] = pd.cut(ordn.horizon_weeks, [15, 27, 39, 51, 67], labels=["16-27", "28-39", "40-51", "52-67"])
for m in ["ensemble_conformal", "gru_negbin", "lgbm_quantile", "climatological_quantile"]:
    s = ordn[ordn.model == m]
    vals = s.groupby("hbin", observed=True).apply(lambda x: x.wis.sum() / x.observed_value.sum(), include_groups=False)
    print(f"  {m:24s} " + "  ".join(f"{b}:{vals.get(b, float('nan')):.3f}" for b in ["16-27", "28-39", "40-51", "52-67"]))

print("\n=== DENGUE WIS decomposition by fold (conformal ensemble; mean per unit) ===")
ec = dg[dg.model == "ensemble_conformal"]
dec = ec.groupby("fold_id")[["dispersion", "overprediction", "underprediction"]].mean()
print(dec.round(0).to_string())

# ---------- CHIKUNGUNYA STATE ----------
ck = _num(pd.read_csv(f"{M}/chik_final_scored.csv", low_memory=False),
          ["wis", "observed_value", "fold_id"] + [f"coverage_{L}" for L in LEVELS])
CK_MODELS = ["lgbm_quantile", "ensemble_vincent", "climatological_quantile", "seasonal_naive", "mechanistic_traj", "naive"]
print("\n=== CHIKUNGUNYA overall + by fold (WIS) ===")
print(leaderboard(ck, CK_MODELS).round(2).to_string(index=False))
ckf = ck.groupby(["model", "fold_id"])["wis"].mean().unstack("fold_id")
print(ckf.reindex(CK_MODELS).round(1).to_string())

# ---------- CITY TRACKS ----------
for disease, f in [("dengue", "city_dengue_scored"), ("chikungunya", "city_chikungunya_scored")]:
    c = _num(pd.read_csv(f"{M}/{f}.csv", low_memory=False), ["wis", "observed_value"] + [f"coverage_{L}" for L in LEVELS])
    print(f"\n=== CITY {disease} leaderboard ===")
    print(leaderboard(c, ["climatological_quantile", "seasonal_naive", "naive"]).round(2).to_string(index=False))
    print(f"  n cities (uf-groups): {c.uf.nunique()}  rows: {len(c)}")
