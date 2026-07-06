"""Hierarchical (empirical-Bayes) seasonal climatology with spatial partial pooling.

A principled, deterministic cousin of the winning 2024-sprint approach (GHR/BSC's Bayesian
spatiotemporal model): instead of MCMC/INLA, it partial-pools each state's seasonal incidence
distribution toward its macroregion via James-Stein-style shrinkage, while large states keep
their own well-estimated distribution.

ABLATION / COMPARISON MODEL - deliberately NOT an ensemble member. Empirically (dengue backtest,
all folds): standalone it edges the fully-unpooled ClimatologicalQuantileModel overall
(normWIS 0.606 -> 0.590), but that gain comes from the incidence-space population adjustment,
NOT the pooling: shrinking a small state (RR, AP) toward a macroregion dominated by a large,
differently-behaving state (AM) INTRODUCES bias and makes small states worse (small-state
normWIS 0.376 -> 0.449). Adding it to the conformal ensemble hurts (1216 -> 1241), whether as a
4th member or replacing climatology - it is too correlated with the climatological member.
Kept as a reproducible ablation for the paper (Brazil's states are too heterogeneous within a
macroregion for naive seasonal pooling to help).

Design (all in per-100k incidence space, so states of very different size are comparable):
  * per (state, epiweek +/-2 window): empirical incidence quantile function q_state
  * per (macroregion, epiweek +/-2 window): pooled incidence quantile function q_region
  * shrunk quantile function q = lambda_s * q_state + (1 - lambda_s) * q_region, with
    lambda_s = N_s / (N_s + kappa) where N_s is the state's historical case volume (a
    reliability proxy) and kappa defaults to the median N_s (so a median-volume state pools
    ~50/50). A convex blend of two monotone quantile functions is monotone -> no crossing.
  * predictions convert incidence back to counts with the target year's population.

Leakage-safe: consumes only the harness's cutoff-filtered train_df plus static population and
geography. Implements the shared fit/predict protocol (see baselines.py).
"""
import numpy as np
import pandas as pd

from imdc.config import QUANTILE_LEVELS
from imdc.data.loaders import load_cases, load_geo_map, load_population
from imdc.evaluation.baselines import _epiweek_of_year

INCIDENCE_SCALE = 1e5


def _windowed_samples(by_week: dict, window: int = 2) -> dict:
    """For each epiweek 1..53, pool samples over the circular +/-window neighborhood."""
    out = {}
    span = 2 * window + 1
    for w in range(1, 54):
        weeks = [((w - window + k - 1) % 53) + 1 for k in range(span)]
        s = []
        for ww in weeks:
            s.extend(by_week.get(ww, []))
        out[w] = np.asarray(s, dtype=float) if s else np.array([0.0])
    return out


class HierarchicalClimatologicalModel:
    name = "hierarchical_climatological"

    def __init__(self, kappa: float = None, window: int = 2):
        self.kappa = kappa      # shrinkage strength; None -> median state case-volume
        self.window = window

    def _state_population(self) -> pd.DataFrame:
        pop = load_population()
        geo_uf = load_cases("dengue")[["geocode", "uf"]].drop_duplicates()
        merged = pop.merge(geo_uf, on="geocode", how="inner")
        return merged.groupby(["uf", "year"], as_index=False)["population"].sum()

    def fit(self, train_df: pd.DataFrame, fold) -> "HierarchicalClimatologicalModel":
        geo = load_geo_map()[["uf", "macroregion_code"]].drop_duplicates()
        self._uf_region = dict(zip(geo["uf"], geo["macroregion_code"]))

        state_pop = self._state_population()
        self._pop_lookup = {(r.uf, int(r.year)): float(r.population) for r in state_pop.itertuples()}
        self._pop_years = (int(state_pop["year"].min()), int(state_pop["year"].max()))

        df = train_df.copy()
        df["epiweek"] = _epiweek_of_year(df["date"])
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["pop_year"] = df["year"].clip(*self._pop_years)
        df = df.merge(state_pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
        df = df[df["population"].notna() & (df["population"] > 0)]
        df["inc"] = df["casos"] / df["population"] * INCIDENCE_SCALE
        df["region"] = df["uf"].map(self._uf_region)

        # per-state and per-region windowed incidence quantile support
        self._state_samples = {}
        for uf, g in df.groupby("uf"):
            self._state_samples[uf] = _windowed_samples(
                g.groupby("epiweek")["inc"].apply(list).to_dict(), self.window)
        self._region_samples = {}
        for region, g in df.groupby("region"):
            self._region_samples[region] = _windowed_samples(
                g.groupby("epiweek")["inc"].apply(list).to_dict(), self.window)

        # reliability-based shrinkage weight per state
        vol = df.groupby("uf")["casos"].sum()
        kappa = self.kappa if self.kappa is not None else float(vol.median())
        self._lambda = {uf: float(v / (v + kappa)) for uf, v in vol.items()}
        self._kappa = kappa
        return self

    def _pop_for(self, uf: str, year: int) -> float:
        return self._pop_lookup.get((uf, int(np.clip(year, *self._pop_years))), np.nan)

    def predict(self, target_dates: pd.DataFrame, quantile_levels: list = QUANTILE_LEVELS) -> pd.DataFrame:
        td = target_dates.copy()
        td["epiweek"] = _epiweek_of_year(td["date"])
        td["year"] = pd.to_datetime(td["date"]).dt.year
        ql = np.asarray(quantile_levels)
        rows = []
        for _, row in td.iterrows():
            uf, date, w, year = row["uf"], row["date"], int(row["epiweek"]), int(row["year"])
            region = self._uf_region.get(uf)
            q_state = np.quantile(self._state_samples.get(uf, {}).get(w, np.array([0.0])), ql)
            q_region = np.quantile(self._region_samples.get(region, {}).get(w, np.array([0.0])), ql)
            lam = self._lambda.get(uf, 0.5)
            inc_q = lam * q_state + (1.0 - lam) * q_region      # shrunk incidence quantiles
            pop = self._pop_for(uf, year)
            counts = np.maximum(0.0, inc_q * pop / INCIDENCE_SCALE) if np.isfinite(pop) else np.zeros_like(inc_q)
            for tau, v in zip(quantile_levels, counts):
                rows.append({"uf": uf, "date": date, "quantile_level": tau, "predicted_value": float(v)})
        return pd.DataFrame(rows)
