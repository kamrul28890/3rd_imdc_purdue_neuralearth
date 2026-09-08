"""XGBoost quantile-regression forecaster - a second classical-ML family alongside LightGBM.

Motivated by the "XGBSillas" (FGV-EMAP) team's use of XGBoost in this same challenge (see
docs/FUTURE_WORK.md Sec 6b) - reviewed for the choice of model family, not their code. Reuses
this project's existing leakage-safe origin-anchored panel (imdc.features.panel), exactly as
ml_boosted.py's LightGBM model does, so this is a fair apples-to-apples comparison of GBM
implementations on identical features, not a new feature-engineering experiment.

Key difference from ml_boosted.py: XGBoost >=2.0's native multi-quantile objective
(`reg:quantileerror` with an array `quantile_alpha`) fits ALL quantile levels as one boosted
model with shared trees (`multi_strategy="multi_output_tree"`), rather than nine independent
boosters. This shares statistical strength across quantiles and is more efficient - the tradeoff
identified in XGBoost's own docs is that extreme quantiles can be noisier when forced to share
tree structure with the median; CQR calibration (identical scheme to ml_boosted.py) plus a final
sort corrects for both under-coverage and any residual crossing.
"""
import numpy as np
import pandas as pd

import xgboost as xgb

from imdc.config import QUANTILE_LEVELS
from imdc.features.panel import build_panel, build_prediction_features
from imdc.features.panel import INCIDENCE_SCALE

DEFAULT_PARAMS = {
    "tree_method": "hist",
    "objective": "reg:quantileerror",
    "multi_strategy": "multi_output_tree",
    "max_depth": 5,
    "min_child_weight": 30,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
DEFAULT_N_ESTIMATORS = 400

_INTERVAL_BOUNDS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}


class XGBQuantileModel:
    """One multi-output XGBoost booster covering all quantile levels; direct multi-horizon,
    pooled across states. Optional CQR calibration, identical scheme to LGBMQuantileModel."""

    name = "xgb_quantile"

    def __init__(self, params: dict = None, n_estimators: int = DEFAULT_N_ESTIMATORS,
                 quantile_levels: list = QUANTILE_LEVELS, disease: str = "dengue",
                 calibrate: bool = True, calib_weeks: int = 78, sillas_weights: bool = False,
                 sillas_recency: bool = True, sillas_extreme: bool = True):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.quantile_levels = list(quantile_levels)
        self.disease = disease
        self.calibrate = calibrate
        self.calib_weeks = calib_weeks
        # Sample-weighting scheme from the "XGBSillas" (FGV-EMAp) team, reviewed for the idea
        # (not their code) - see docs/FUTURE_WORK.md Sec 6b. Two components: (1) recency -
        # up-weight rows whose origin is close to the fold's own cutoff, since those origins'
        # climate/case regime is most representative of what the model will actually see at
        # forecast time; (2) extreme-value - up-weight rows whose label is in the upper tail of
        # that state's historical distribution, since a pooled quantile model trained mostly on
        # ordinary weeks otherwise under-fits rare, high-impact outbreak weeks.
        self.sillas_weights = sillas_weights
        self.sillas_recency = sillas_recency
        self.sillas_extreme = sillas_extreme
        self._booster = None
        self._cqr_adjust = {tau: 0.0 for tau in self.quantile_levels}
        self._feature_cols = None
        self._fold = None

    def _sample_weights(self, proper: pd.DataFrame, fold) -> np.ndarray:
        w = np.ones(len(proper))
        if self.sillas_recency:
            dist_weeks = (fold.train_cutoff - proper["origin_date"]).dt.days.to_numpy() / 7.0
            w *= 0.3 + 0.7 * np.exp(-dist_weeks / 26.0)
        if self.sillas_extreme:
            p75 = proper.groupby("uf")["label"].transform(lambda s: s.quantile(0.75))
            p90 = proper.groupby("uf")["label"].transform(lambda s: s.quantile(0.90))
            p95 = proper.groupby("uf")["label"].transform(lambda s: s.quantile(0.95))
            mult = np.ones(len(proper))
            label = proper["label"].to_numpy()
            mult = np.where(label > p75.to_numpy(), 2.0, mult)
            mult = np.where(label > p90.to_numpy(), 4.0, mult)
            mult = np.where(label > p95.to_numpy(), 6.0, mult)
            w = w * mult
        return w

    def fit(self, train_df: pd.DataFrame, fold, covariates=None) -> "XGBQuantileModel":
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
        weight = self._sample_weights(proper, fold) if self.sillas_weights else None
        params = {**self.params, "quantile_alpha": np.array(self.quantile_levels)}
        dtrain = xgb.DMatrix(X, label=y, weight=weight)
        self._booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

        if self.calibrate and len(calib) > 200:
            self._fit_cqr(calib, feature_cols)
        return self

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_quantiles) log-incidence predictions, columns matching self.quantile_levels."""
        preds = self._booster.inplace_predict(X)
        return np.atleast_2d(preds)

    def _fit_cqr(self, calib: pd.DataFrame, feature_cols: list) -> None:
        Xc = calib[feature_cols]
        yc = calib["label"].to_numpy()
        pred_c = self._predict_raw(Xc)
        pred_by_tau = {tau: pred_c[:, j] for j, tau in enumerate(self.quantile_levels)}
        n = len(yc)
        for level, (lo_tau, hi_tau) in _INTERVAL_BOUNDS.items():
            qlo, qhi = pred_by_tau[lo_tau], pred_by_tau[hi_tau]
            scores = np.maximum(qlo - yc, yc - qhi)
            target_cov = level / 100
            k = int(np.ceil((n + 1) * target_cov))
            q = np.sort(scores)[min(k, n) - 1]
            self._cqr_adjust[lo_tau] = -q
            self._cqr_adjust[hi_tau] = +q

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        feats, feature_cols = build_prediction_features(self._fold, target_grid, disease=self.disease)
        X = feats[feature_cols]

        preds_log = self._predict_raw(X)
        adjust = np.array([self._cqr_adjust[tau] for tau in self.quantile_levels])
        preds_log = preds_log + adjust[None, :]
        preds_log = np.sort(preds_log, axis=1)

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
