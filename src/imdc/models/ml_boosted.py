"""LightGBM quantile-regression forecaster (classical-ML model family).

Direct multi-horizon: one pooled model per quantile level across all states and
all horizons, with horizon and target-epiweek as explicit features (plan Sec
3.4). Models log1p(incidence) internally; converts predictions back to raw
counts for the submission format. Quantile monotonicity is guaranteed by
sorting the 9 per-row predictions (a monotone transform of a monotone-sorted
set stays sorted, so sorting in log-incidence space is valid).

Requires KMP_DUPLICATE_LIB_OK=TRUE in the environment on this machine (libomp
is linked by both lightgbm and torch); set it before importing if needed.
"""
import numpy as np
import pandas as pd

import lightgbm as lgb

from imdc.config import QUANTILE_LEVELS
from imdc.features.panel import build_panel, build_prediction_features
from imdc.features.panel import INCIDENCE_SCALE

# Conservative defaults, kept after a tuning experiment. A grid search on the fold-2 panel's
# internal time-block holdout preferred a deeper config (num_leaves=63, max_depth=7, ~5% lower
# holdout pinball), but that config was WORSE on the actual backtest (lgbm WIS 1317 vs 1309;
# ensemble 1294 vs 1288): it overfit the internal holdout, which does not represent forecasting
# a full future season from EW25. This is the exact risk the plan flagged, so the conservative
# defaults are retained. See reports/modeling_results_report for the finding.
DEFAULT_PARAMS = {
    "objective": "quantile",
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 30,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
    # Reproducibility: fixed seeds + deterministic mode make boosting bit-identical
    # across re-runs on the same platform (force_row_wise removes the row/col-wise
    # heuristic's thread-count dependence).
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
DEFAULT_N_ESTIMATORS = 400

# Quantile-level pairs bounding each central interval, for CQR calibration.
# Hardcoded to the exact QUANTILE_LEVELS values (float arithmetic on
# (1-level/100)/2 does not land exactly on 0.10 / 0.90 etc.).
_INTERVAL_BOUNDS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}


class LGBMQuantileModel:
    """One LightGBM booster per quantile level; direct multi-horizon, pooled across states.

    Optionally conformalized (CQR, Romano et al. 2019): a contiguous time-block
    holdout (the last `calib_weeks` of origins) calibrates an additive per-interval
    widening in log-incidence space, correcting the well-known undercoverage of
    independent GBM quantile fits. Monotonicity is guaranteed by sorting the 9
    per-row predictions after adjustment.
    """

    name = "lgbm_quantile"

    def __init__(self, params: dict = None, n_estimators: int = DEFAULT_N_ESTIMATORS,
                 quantile_levels: list = QUANTILE_LEVELS, disease: str = "dengue",
                 calibrate: bool = True, calib_weeks: int = 78):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.quantile_levels = quantile_levels
        self.disease = disease
        self.calibrate = calibrate
        self.calib_weeks = calib_weeks
        self._boosters = {}
        self._cqr_adjust = {tau: 0.0 for tau in quantile_levels}
        self._feature_cols = None
        self._fold = None

    def fit(self, train_df: pd.DataFrame, fold, covariates=None) -> "LGBMQuantileModel":
        self._fold = fold
        panel, feature_cols = build_panel(fold, disease=self.disease)
        self._feature_cols = feature_cols

        if self.calibrate:
            split = fold.train_cutoff - pd.Timedelta(weeks=self.calib_weeks)
            proper = panel[panel["origin_date"] < split]
            calib = panel[panel["origin_date"] >= split]
        else:
            proper, calib = panel, panel.iloc[:0]

        X = proper[feature_cols]
        y = proper["label"].to_numpy()
        for tau in self.quantile_levels:
            params = {**self.params, "alpha": tau}
            self._boosters[tau] = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=self.n_estimators)

        if self.calibrate and len(calib) > 200:
            self._fit_cqr(calib, feature_cols)
        return self

    def _fit_cqr(self, calib: pd.DataFrame, feature_cols: list) -> None:
        """Additive per-interval conformity widening in log-incidence space (symmetric CQR)."""
        Xc = calib[feature_cols]
        yc = calib["label"].to_numpy()
        pred_c = {tau: self._boosters[tau].predict(Xc) for tau in self.quantile_levels}
        n = len(yc)
        for level, (lo_tau, hi_tau) in _INTERVAL_BOUNDS.items():
            qlo, qhi = pred_c[lo_tau], pred_c[hi_tau]
            scores = np.maximum(qlo - yc, yc - qhi)  # >0 when y outside [qlo,qhi]
            target_cov = level / 100
            k = int(np.ceil((n + 1) * target_cov))
            q = np.sort(scores)[min(k, n) - 1]
            self._cqr_adjust[lo_tau] = -q  # widen lower bound down
            self._cqr_adjust[hi_tau] = +q  # widen upper bound up

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        feats, feature_cols = build_prediction_features(self._fold, target_grid, disease=self.disease)
        X = feats[feature_cols]

        preds_log = np.column_stack([
            self._boosters[tau].predict(X) + self._cqr_adjust[tau] for tau in self.quantile_levels
        ])
        preds_log = np.sort(preds_log, axis=1)  # enforce quantile monotonicity in log-incidence space

        incidence = np.expm1(preds_log)
        population = feats["population"].to_numpy()[:, None]
        counts = np.maximum(0.0, incidence * population / INCIDENCE_SCALE)

        out_rows = []
        ufs = feats["uf"].to_numpy()
        dates = feats["target_date"].to_numpy()
        for j, tau in enumerate(self.quantile_levels):
            out_rows.append(pd.DataFrame({
                "uf": ufs, "date": dates, "quantile_level": tau, "predicted_value": counts[:, j],
            }))
        return pd.concat(out_rows, ignore_index=True)
