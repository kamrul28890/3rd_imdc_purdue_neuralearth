"""Semi-mechanistic epidemic-trajectory model (Phase 5).

Originally planned around EpiScanner's Richards-curve fits pulled via the
Mosqlimate API; that endpoint is down (HTTP 500) and, more importantly, the
fits are a curve-fitting procedure over case data we already hold locally, so
this module reimplements the mechanistic core itself - no API dependency.

Design (plan Sec 5.2, adapted): each completed dengue season (EW41->EW40) is a
coherent epidemic trajectory. To forecast a target season we (1) bootstrap-
resample whole historical season trajectories observed before the forecast
origin, (2) convert incidence to counts with the target-year population, and
(3) layer a Negative-Binomial observation model (dispersion estimated from the
data) so a handful of historical curves yields smooth, non-lumpy predictive
quantiles. A Richards growth curve is also fit per season to recover the
EpiScanner-style parameters (peak week, growth rate, final size, shape) for the
paper - fitting is best-effort and never blocks forecasting.

This is a genuinely different, interpretable comparison point: it models the
season as a coherent epidemic object with an explicit observation model, rather
than as independent per-week quantiles (the climatological baseline) or a
feature regression (LGBM/GRU).
"""
import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.optimize import curve_fit
from scipy.stats import nbinom

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage
from imdc.features.panel import INCIDENCE_SCALE, state_population

SEASON_LEN = 53  # allow for 53-week epi years (e.g. 2026); 52-week seasons leave index 52 empty


def season_week_from_date(date) -> int:
    """Within-season index (EW41 -> 1) derived from the date, so 53-week years don't collide.

    A pure epiweek->index map is not correct: in a 53-week epi year both EW53 and the next
    year's EW1 map to the same index, colliding within one season. Computing the index from
    weeks-since-season-start (EW41 of the season's first year) is injective and handles both
    52- and 53-week seasons.
    """
    date = pd.Timestamp(date)
    ew = Week.fromdate(date)
    start_year = ew.year if ew.week >= 41 else ew.year - 1
    season_start = pd.Timestamp(Week(start_year, 41).startdate())
    return (date - season_start).days // 7 + 1


def _season_start_year(date: pd.Timestamp) -> int:
    ew = Week.fromdate(date)
    return ew.year if ew.week >= 41 else ew.year - 1


def richards_cumulative(t, K, r, tp, alpha):
    """Richards growth model for cumulative cases (the EpiScanner functional form)."""
    return K / np.power(1.0 + alpha * np.exp(-r * (t - tp)), 1.0 / alpha)


def fit_richards(incidence_curve: np.ndarray):
    """Best-effort Richards fit to a season's cumulative incidence; returns params or None."""
    y = np.cumsum(np.nan_to_num(incidence_curve))
    t = np.arange(1, len(y) + 1, dtype=float)
    if y[-1] <= 0:
        return None
    try:
        K0, tp0 = y[-1] * 1.05, float(t[np.argmax(np.diff(y, prepend=0))])
        popt, _ = curve_fit(
            richards_cumulative, t, y, p0=[K0, 0.3, tp0, 1.0],
            bounds=([y[-1] * 0.5, 1e-3, 1, 1e-3], [y[-1] * 5, 5.0, len(y), 50]),
            maxfev=5000,
        )
        K, r, tp, alpha = popt
        return {"final_size": K, "growth_rate": r, "peak_week": tp, "shape_alpha": alpha,
                "obs_peak_week": int(np.argmax(incidence_curve)) + 1}
    except Exception:
        return {"final_size": float(y[-1]), "growth_rate": np.nan, "peak_week": np.nan,
                "shape_alpha": np.nan, "obs_peak_week": int(np.argmax(incidence_curve)) + 1}


class MechanisticTrajectoryModel:
    """Bootstrap ensemble of historical season trajectories with a NegBinom observation model."""

    name = "mechanistic_traj"

    def __init__(self, disease: str = "dengue", n_boot: int = 800, seed: int = 0,
                 quantile_levels: list = QUANTILE_LEVELS, fit_curves: bool = False):
        self.disease = disease
        self.n_boot = n_boot
        self.seed = seed
        self.quantile_levels = quantile_levels
        self.fit_curves = fit_curves
        self._fold = None

    def _state_incidence(self, fold) -> pd.DataFrame:
        cases = cutoff_filter(load_cases(self.disease), fold.train_cutoff)
        assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} mechanistic")
        state = aggregate_cases_to_state(cases)
        pop = state_population()
        df = state.copy()
        df["year"] = df["date"].dt.year
        ymin, ymax = pop["year"].min(), pop["year"].max()
        df["pop_year"] = df["year"].clip(ymin, ymax)
        df = df.merge(pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
        df["incidence"] = df["casos"] / df["population"] * INCIDENCE_SCALE
        df["season_start_year"] = [_season_start_year(d) for d in df["date"]]
        df["season_week"] = [season_week_from_date(d) for d in df["date"]]
        return df

    def fit(self, train_df, fold, covariates=None):
        self._fold = fold
        df = self._state_incidence(fold)
        target_season_year = _season_start_year(fold.target_start)

        # historical season trajectories: (uf) -> array (n_seasons, SEASON_LEN) of incidence,
        # keeping only complete seasons strictly before the target season
        self._trajectories = {}
        self._richards = []
        for uf, g in df.groupby("uf"):
            mats = []
            for syear, sg in g.groupby("season_start_year"):
                if syear >= target_season_year:
                    continue
                sg = sg[(sg["season_week"] >= 1) & (sg["season_week"] <= SEASON_LEN)]
                if sg["season_week"].nunique() < SEASON_LEN * 0.9:  # require a near-complete season
                    continue
                curve = np.full(SEASON_LEN, np.nan)
                curve[sg["season_week"].to_numpy() - 1] = sg["incidence"].to_numpy()
                curve = pd.Series(curve).interpolate(limit_direction="both").to_numpy()
                mats.append(curve)
                if self.fit_curves:
                    params = fit_richards(curve)
                    if params:
                        self._richards.append({"uf": uf, "season": int(syear), **params})
            if mats:
                self._trajectories[uf] = np.vstack(mats)

        # global NegBinom dispersion from historical count over-dispersion (var = mu + alpha*mu^2)
        self._alpha = self._estimate_dispersion(df)
        return self

    def _estimate_dispersion(self, df: pd.DataFrame) -> float:
        stats = df.groupby(["uf", "season_week"])["casos"].agg(["mean", "var"]).dropna()
        stats = stats[stats["mean"] > 5]
        if len(stats) == 0:
            return 0.1
        alpha = ((stats["var"] - stats["mean"]) / stats["mean"] ** 2).clip(lower=1e-3, upper=10)
        return float(np.median(alpha))

    def predict(self, target_grid, quantile_levels=None):
        quantile_levels = quantile_levels or self.quantile_levels
        rng = np.random.default_rng(self.seed)
        pop = state_population()
        ymin, ymax = pop["year"].min(), pop["year"].max()

        rows = []
        grid = target_grid.copy()
        grid["date"] = pd.to_datetime(grid["date"])
        grid["season_week"] = [season_week_from_date(d) for d in grid["date"]]

        for uf, gg in grid.groupby("uf"):
            traj = self._trajectories.get(uf)
            if traj is None or len(traj) == 0:
                continue
            n_seasons = traj.shape[0]
            boot_idx = rng.integers(0, n_seasons, size=self.n_boot)  # resample whole seasons
            for _, row in gg.iterrows():
                sw = int(row["season_week"])
                if sw < 1 or sw > SEASON_LEN:
                    continue
                inc_draws = traj[boot_idx, sw - 1]  # incidence from resampled seasons at this season-week
                year = min(max(int(row["date"].year), ymin), ymax)
                pmap = pop[pop["uf"] == uf].set_index("year")["population"]
                popn = pmap.get(year, pmap.iloc[-1])
                mu = np.maximum(inc_draws * popn / INCIDENCE_SCALE, 1e-6)
                r = 1.0 / self._alpha
                p = r / (r + mu)
                count_draws = nbinom.rvs(r, p, random_state=rng)  # NegBinom observation noise
                qs = np.quantile(count_draws, quantile_levels)
                qs = np.maximum(0.0, np.sort(qs))
                for tau, v in zip(quantile_levels, qs):
                    rows.append({"uf": uf, "date": row["date"], "quantile_level": tau, "predicted_value": v})
        return pd.DataFrame(rows)
