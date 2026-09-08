"""Prophet v2 - rebuilt after reviewing why a competing team's Prophet model outperformed our v1.

v1 (prophet_model.py, kept for the record) fit univariate log1p(cases) with Prophet's defaults and
scored far worse than naive (WIS 2175 vs 1664). Reviewing the public repo of a team whose Prophet
model was named a best-overall dengue-state model at the 2026-07-31 validation webinar
(github.com/EzequielEBS/3rd_imdc_emap_epidematicos_prophet - reviewed for the approach, not copied;
see docs/FUTURE_WORK.md Sec 6b for the full comparison) identified three concrete gaps, all fixed
here:

1. `growth="flat"`: dengue has no secular trend, only seasonality. Prophet's default linear trend
   + changepoints can extrapolate spurious drift over our 67-week horizon.
2. Real regressors, not univariate: reuses the already leakage-tested origin-anchored features from
   imdc.features.panel (lag_52, climate rolling means, ocean-index lags) as Prophet regressors,
   computed identically for training (each historical date as its own origin) and prediction (all
   target dates anchored at the fold's actual cutoff) - no separate leakage logic to get wrong.
3. A custom seasonally-stratified residual bootstrap for intervals, replacing Prophet's built-in
   `predictive_samples` (whose generic Gaussian noise model produced negative samples that clipped
   to zero and collapsed v1's lower quantiles). In-sample residuals are tagged by epiweek; a target
   date's predictive samples are the point forecast plus residuals resampled from a +/-4-week
   window around its own epiweek - the same "circular same-season window" idea `panel.py`'s
   climatological baseline already uses, applied to residuals instead of raw incidence.
"""
import numpy as np
import pandas as pd
from epiweeks import Week

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.features.panel import _origin_anchored_series, build_prediction_features

N_SAMPLES = 1000
EPIWEEK_WINDOW = 6
REGRESSOR_COLS = ["lag_52", "temp_med_roll4", "precip_med_roll4",
                  "rel_humid_med_roll4", "temp_anomaly", "enso_lag0", "iod_lag0", "pdo_lag0"]
# Deliberately no lag_1/lag_4/lag_8, and no panel.py's roll_mean_4/roll_std_4/roll_mean_8/
# roll_max_12: those rolling stats are computed as an UNSHIFTED rolling window over log_inc
# (includes the current row's own value), only leakage-safe in an origin/horizon panel structure
# (LightGBM's) where the label is a separately joined future row, never the origin row itself.
# Tried adding roll_mean_4 here anyway: catastrophic, not marginal (WIS 1520 -> 2139, coverage_50
# 0.30 -> 0.07) - confirms it's genuine train-time circularity, not just added noise.
#
# Two further structural changes were tried after diagnosing severe, systematic under-prediction
# in the largest states (SP fold 2: predicted 5716 vs actual 42237; MG fold 2: 1942 vs 34241), and
# both were reverted because they empirically regressed overall performance despite each being
# individually well-motivated (full account in docs/FUTURE_WORK.md Sec 6b):
# 1. Swapping lag_52 for `seasonal_anchor`, since lag_52 is origin-anchored - held at a single
#    constant value across the entire 52-67 week forecast horizon at prediction time, while at
#    training time it genuinely varies week to week - a real train/predict mismatch. The diagnosis
#    was correct as far as it went, but seasonal_anchor made results worse in isolation
#    (WIS 1520 -> 1739): whatever noise seasonal_anchor carries (one specific past year's value at
#    exactly one epiweek, versus lag_52's more mechanical week-count shift) apparently outweighs
#    the staleness problem it was meant to fix.
# 2. growth="linear" with a heavily regularized changepoint_prior_scale=0.01, after confirming via
#    Prophet's own component decomposition (not a guess) that growth="flat" gives SP/MG a
#    demonstrably poor *training* fit, not just poor extrapolation. Tried three ways, all
#    rejected:
#    a. combined with the seasonal_anchor change above: WIS 1520 -> 1783.
#    b. alone, blanket across all states (lag_52 kept): WIS 1520 -> 1575. Fold 2 (the outbreak
#       fold) did improve (3874 -> 3806), confirming the diagnosis has some truth, but the
#       linear trend overfit/drifted on the smaller, stable folds (fold 4 WIS 346 -> 626) for a
#       net loss.
#    c. per-state, linear only for the two states above a 20M-population threshold (SP, MG) and
#       flat elsewhere: WIS 1520 -> 1514, a small aggregate gain, and fold 2 improved further
#       (3874 -> 3779) - but the harness's own prospective fold 4 got markedly worse (346 -> 526)
#       and the ex-2024 "normal season" nWIS regressed (0.648 -> 0.674). Confining the linear
#       trend to just the states that need it removes (b)'s worst side effect on OTHER states,
#       but doesn't fix the side effect it causes within SP/MG themselves - the same regularized
#       trend that helps them in an outbreak year still drifts in their own ordinary years.
# 3. Recency-weighting the residual bootstrap pool (exponential decay by years-before-training-end,
#    half-life 5 or 2 years) instead of sampling all past years at a given epiweek uniformly - a
#    softer way to reflect the same real level-shift-over-time as (2), applied only to interval
#    width/center rather than the point forecast. Made things monotonically worse as the half-life
#    shortened (WIS 1520 -> 1526 at 5y -> 1547 at 2y): the outbreak fold barely moved either way,
#    so the point-forecast under-prediction - not the residual pool - is the actual bottleneck.
# The SP/MG outbreak-fold failure that motivated all these attempts looks, after this, like a
# genuine structural limitation shared with this paper's GRU model: neither has a mechanism to
# represent "this season is unlike any prior season," and nothing tried here fixed it without a
# worse cost elsewhere.


def _epiweek(dates) -> np.ndarray:
    return np.array([Week.fromdate(d).week for d in pd.to_datetime(dates)])


class ProphetV2Model:
    name = "prophet_v2"

    def __init__(self, disease: str = "dengue", ufs: list = MANDATORY_UFS,
                 quantile_levels: list = QUANTILE_LEVELS, seed: int = 0):
        self.disease = disease
        self.ufs = list(ufs)
        self.quantile_levels = quantile_levels
        self.seed = seed
        self._models = {}
        self._resid_by_epiweek = {}

    def fit(self, train_df, fold, covariates=None):
        from prophet import Prophet

        self._fold = fold
        origin_df = _origin_anchored_series(fold, self.disease)
        origin_df = origin_df[origin_df["uf"].isin(self.ufs)]

        self._models = {}
        self._resid_by_epiweek = {}
        for uf, g in origin_df.groupby("uf"):
            g = g.dropna(subset=REGRESSOR_COLS)
            if len(g) < 104 or g["log_inc"].sum() <= 0:
                continue
            df = g[["date"] + REGRESSOR_COLS].rename(columns={"date": "ds"}).copy()
            df["y"] = g["log_inc"].to_numpy()

            # additive mode (the default) is correct here, not multiplicative: y is already
            # log1p(incidence), so an additive effect in log-space is already a multiplicative
            # effect in the original count scale (exp(trend+seasonal) = exp(trend)*exp(seasonal));
            # applying Prophet's "multiplicative" mode on top would double up that effect.
            # growth="flat": see the module docstring for the linear-trend alternatives tried
            # (blanket and per-state-by-population) and why both were rejected.
            m = Prophet(growth="flat",
                        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            m.add_seasonality(name="yearly", period=365.25, fourier_order=6)
            for col in REGRESSOR_COLS:
                m.add_regressor(col, standardize=True)
            try:
                m.fit(df)
            except Exception:
                continue
            self._models[uf] = m

            fitted = m.predict(df)["yhat"].to_numpy()
            resid = df["y"].to_numpy() - fitted
            ew = _epiweek(df["ds"])
            self._resid_by_epiweek[uf] = pd.Series(resid, index=ew)
        return self

    def _resample_residuals(self, uf: str, target_epiweek: int, rng) -> np.ndarray:
        resid = self._resid_by_epiweek.get(uf)
        if resid is None or len(resid) == 0:
            return np.zeros(N_SAMPLES)
        diff = np.minimum((resid.index - target_epiweek) % 52, (target_epiweek - resid.index) % 52)
        pool = resid[diff <= EPIWEEK_WINDOW].to_numpy()
        if len(pool) == 0:
            pool = resid.to_numpy()
        return rng.choice(pool, size=N_SAMPLES, replace=True)

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        quantile_levels = list(quantile_levels or self.quantile_levels)
        rng = np.random.default_rng(self.seed)
        feats, _ = build_prediction_features(self._fold, target_grid, self.disease)

        rows = []
        for uf, gg in feats.groupby("uf"):
            m = self._models.get(uf)
            if m is None:
                continue
            df = gg[["target_date"] + REGRESSOR_COLS].rename(columns={"target_date": "ds"}).copy()
            df[REGRESSOR_COLS] = df[REGRESSOR_COLS].fillna(0.0)
            yhat = m.predict(df)["yhat"].to_numpy()
            target_ew = _epiweek(df["ds"])
            pop = gg["population"].to_numpy()
            for i, (_, row) in enumerate(gg.iterrows()):
                samples = yhat[i] + self._resample_residuals(uf, int(target_ew[i]), rng)
                # samples are log1p(incidence per 100k) - convert back to raw case counts
                samples = np.maximum(np.expm1(samples) * pop[i] / 1e5, 0.0)
                vals = np.sort(np.quantile(samples, quantile_levels))
                for tau, v in zip(quantile_levels, vals):
                    rows.append({"uf": uf, "date": row["target_date"], "quantile_level": tau,
                                 "predicted_value": float(v)})
        return pd.DataFrame(rows)
