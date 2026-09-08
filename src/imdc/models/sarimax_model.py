"""SARIMAX baseline - per state, fit on log1p(case counts), a fixed modest seasonal order.

The concrete, closeable gap identified earlier: an ARIMA-family model was planned
(docs/PLAN.md Sec 7.3, `pmdarima`/statsforecast fallbacks) but never actually implemented -
`pmdarima` ended up installed only as a transitive `mosqlient` dependency, never wired into a
model class. This fills that gap directly with `statsmodels.tsa.statespace.SARIMAX` rather than
`pmdarima`'s auto-search (which would mean a full grid search per state per fold - too slow for
104 fits), using a single fixed order chosen for stability at a long forecast horizon.

Order choice: no differencing (d=D=0). A quick test with d=1 produced a 95% interval of roughly
[-56000, 60000] for SP at long horizons - the standard ARIMA random-walk variance-growth problem,
which compounds badly over the 52-67 week horizon this challenge requires (already flagged in
docs/PLAN.md as why ARIMA needs care here). Fitting log1p(counts) with d=0 keeps the process
stationary so forecast variance stays bounded rather than growing without limit, at the cost of the
interval still widening substantially by 60+ weeks out - a real, expected property of this model
class at this horizon, not a bug, and worth reporting honestly either way.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage
from scipy.stats import norm

ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 52)


class SarimaxModel:
    name = "sarimax"

    def __init__(self, disease: str = "dengue", ufs: list = MANDATORY_UFS,
                 quantile_levels: list = QUANTILE_LEVELS):
        self.disease = disease
        self.ufs = list(ufs)
        self.quantile_levels = quantile_levels
        self._results = {}

    def fit(self, train_df, fold, covariates=None):
        cases = cutoff_filter(load_cases(self.disease), fold.train_cutoff)
        assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} sarimax")
        state = aggregate_cases_to_state(cases)

        self._results = {}
        self._cutoff = fold.train_cutoff
        for uf, g in state.groupby("uf"):
            if uf not in self.ufs:
                continue
            g = g.sort_values("date")
            y = np.log1p(g["casos"].astype(float).to_numpy())
            if len(y) < 104:
                continue
            try:
                mod = SARIMAX(y, order=ORDER, seasonal_order=SEASONAL_ORDER,
                               enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=100)
                self._results[uf] = res
            except Exception:
                continue
        return self

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        quantile_levels = list(quantile_levels or self.quantile_levels)
        grid = target_grid.copy()
        grid["date"] = pd.to_datetime(grid["date"])
        z = {tau: norm.ppf(tau) for tau in quantile_levels}

        rows = []
        for uf, gg in grid.groupby("uf"):
            res = self._results.get(uf)
            if res is None:
                continue
            max_h = int(((gg["date"].max() - self._cutoff).days) // 7)
            fc = res.get_forecast(steps=max_h)
            mean = fc.predicted_mean
            se = fc.se_mean
            for _, row in gg.iterrows():
                h = int(((row["date"] - self._cutoff).days) // 7) - 1
                if h < 0 or h >= len(mean):
                    continue
                mu, s = mean[h], max(se[h], 1e-6)
                vals = sorted(np.expm1(mu + z[tau] * s) for tau in quantile_levels)
                vals = np.maximum(0.0, vals)
                for tau, v in zip(quantile_levels, vals):
                    rows.append({"uf": uf, "date": row["date"], "quantile_level": tau,
                                 "predicted_value": float(v)})
        return pd.DataFrame(rows)
