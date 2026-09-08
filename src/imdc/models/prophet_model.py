"""Prophet baseline - univariate, per state, no covariates.

Added after evidence (not just a priori reasoning) that it can work well on this exact task: the
IMDC 2026 validation webinar named a competing team's Prophet model one of the 6 best-overall
dengue-state models (docs cite `imdc2026webinar` in the paper). Kept deliberately simple (no
covariates) to match how it's typically used - Prophet's own trend+seasonality decomposition is the
whole point, not a vehicle for the climate/ocean/ECMWF features the other models already use.

Uses `predictive_samples` (Prophet's built-in simulated posterior trajectories, available even
without MCMC fitting) rather than re-fitting per interval width, so all 9 required quantile levels
come from one fit per state.

Fit on log1p(case count), not the raw count: a first version fit raw counts directly and scored far
worse than even the naive baseline (WIS 2196 vs naive's 1664, normalized WIS above 1.0). The cause
was visible directly in the predictions - the 10th/25th percentiles were pinned to exactly 0 even
with a large median, because Prophet's default additive Gaussian noise model has no notion that
case counts can't go negative: for a right-skewed count series it puts a large fraction of its
simulated samples below zero, which then all clip to the same 0 floor and collapse the lower
quantiles. Modeling log1p(count) keeps the noise symmetric on a scale where it can't do that.
"""
import numpy as np
import pandas as pd

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage

N_SAMPLES = 1000


class ProphetModel:
    name = "prophet"

    def __init__(self, disease: str = "dengue", ufs: list = MANDATORY_UFS,
                 quantile_levels: list = QUANTILE_LEVELS, seed: int = 0):
        self.disease = disease
        self.ufs = list(ufs)
        self.quantile_levels = quantile_levels
        self.seed = seed
        self._models = {}

    def fit(self, train_df, fold, covariates=None):
        from prophet import Prophet

        cases = cutoff_filter(load_cases(self.disease), fold.train_cutoff)
        assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} prophet")
        state = aggregate_cases_to_state(cases)

        self._models = {}
        for uf, g in state.groupby("uf"):
            if uf not in self.ufs:
                continue
            g = g.sort_values("date")
            df = pd.DataFrame({"ds": g["date"], "y": np.log1p(g["casos"].astype(float))})
            if len(df) < 104 or g["casos"].sum() <= 0:
                continue
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            try:
                m.fit(df)
                self._models[uf] = m
            except Exception:
                continue
        return self

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        quantile_levels = list(quantile_levels or self.quantile_levels)
        grid = target_grid.copy()
        grid["date"] = pd.to_datetime(grid["date"])
        rng = np.random.default_rng(self.seed)

        rows = []
        for uf, gg in grid.groupby("uf"):
            m = self._models.get(uf)
            if m is None:
                continue
            dates = pd.DataFrame({"ds": pd.to_datetime(gg["date"].unique())}).sort_values("ds")
            samples = m.predictive_samples(dates)["yhat"]  # (n_dates, N_SAMPLES), log1p(count) scale
            samples = np.maximum(np.expm1(samples), 0.0)
            qmat = np.quantile(samples, quantile_levels, axis=1)  # (n_q, n_dates)
            date_to_col = {d: i for i, d in enumerate(dates["ds"])}
            for _, row in gg.iterrows():
                col = date_to_col[pd.Timestamp(row["date"])]
                vals = np.sort(qmat[:, col])
                for tau, v in zip(quantile_levels, vals):
                    rows.append({"uf": uf, "date": row["date"], "quantile_level": tau,
                                 "predicted_value": float(v)})
        return pd.DataFrame(rows)
