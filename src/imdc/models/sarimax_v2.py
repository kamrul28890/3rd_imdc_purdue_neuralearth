"""SARIMAX v2 - rebuilt after reviewing why a competing team's SARIMAX model outperformed our v1.

v1 (sarimax_model.py, kept for the record) fit a bare SARIMA, one fixed order forced onto all 26
states, with a Gaussian analytic confidence interval back-transformed through expm1 - and scored
badly (normalized WIS ex-2024 worse than naive). Reviewing the public repo of a team whose SARIMAX
model was named a best-overall dengue-state model at the 2026-07-31 validation webinar
(github.com/EzequielEBS/3rd_imdc_emap_epidematicos_sarimax_state - reviewed for the approach, not
copied; see docs/FUTURE_WORK.md Sec 6b) identified three gaps, all fixed here:

1. Actual exogenous regressors (the "X" in SARIMAX): reuses the same leakage-tested origin-anchored
   climate/ocean features as Prophet v2 and every ML model in this project, rather than a bare
   univariate SARIMA.
2. Per-state order selection instead of one order forced onto all 26 states.
3. Bootstrap-simulated forecast paths (via the fitted state-space model's own `.simulate`) instead
   of a Gaussian analytic CI back-transformed through expm1 - the latter is exactly what produced
   v1's absurd ~[-56000, 60000] interval, a symmetric log-space Gaussian exploding once exponentiated.

Adaptation for cost: the competitor's per-state order search was WIS-ranked and (implicitly) run
per state per fold. A true per-fold search here would cost roughly 8x this project's already-slow
per-fold SARIMAX fit time (~215s for 26 states), 4 folds over - multiple hours. Instead, orders are
selected ONCE via a held-out-tail validation split on fold 1's data, then frozen and reused for
every fold's actual fit. This is the same "tune once on a designated fold, freeze, reuse" discipline
already used in this project for conformal recalibration factors and inverse-WIS ensemble weights,
not an ad hoc shortcut.

Tune-once (this file's default) scored worse than v1 (WIS 1751.9 vs 1418.6): the order selected on
fold 1's validation tail doesn't generalize to fold 3's different regime (fold 3 WIS blew up to
2372.3 vs v1's 1463.9) - a real lesson that "tune once, freeze" doesn't automatically extend from a
single scalar (a conformal factor, an ensemble weight) to a higher-dimensional structural choice
like ARIMA order. Two follow-ups, both in docs/FUTURE_WORK.md Sec 6b:
- A full per-fold order search (select_orders=True every fold, accepting the multi-hour cost) to
  see whether re-selecting the order each fold fixes the generalization failure.
- Before that finished, a suspicion that EXOG_COLS' own staleness might be a bigger problem than
  the order itself: like Prophet v2's lag_52, predict() held the origin's last known exog snapshot
  constant across the entire 52-67 week horizon, which for temp_med_roll4/precip_med_roll4 means
  ignoring their real annual seasonal cycle for everything beyond the first few weeks. Tried
  replacing those two (not enso_lag0, which has no within-year seasonal cycle to speak of) with a
  per-uf, per-target-epiweek climatological mean computed from training history: no real change
  (WIS 1751.9 -> 1756.4, fold 3 if anything slightly worse at 2526.9) - ruled out as the bottleneck
  for the tune-once failure mode. Reverted in favor of the simpler constant-exog design.

Full per-fold order search (select_orders=True every fold, ~2h43m total) confirmed the diagnosis:
WIS 1751.9 -> 1527.6, a 13% reduction, and fold 3 (the regime the frozen fold-1 order failed on)
dropped from 2372.3 to 1527.98 - almost exactly matching v1's own fold 3 (1463.9), i.e. per-fold
selection recovers essentially all of the generalization loss tune-once introduced. But the result
is a genuine tradeoff against v1 rather than a clean win: nWIS(ex-2024) improves on v1 (0.776 vs
0.835 - better), while nWIS(all) and raw WIS are both still slightly worse than v1 (0.737 vs 0.685;
1527.6 vs 1418.6). Per the project's own history with similar tradeoffs (see mechanistic_nsub.py),
this is not promoted over v1 or into the ensemble - v1 remains the simpler, cheaper, and still
marginally better-on-balance SARIMAX result. select_orders=True is kept as an opt-in for anyone who
wants to reproduce or build on the per-fold result; it is NOT the default (tune-once, cheap, run in
under an hour) because the 2h43m cost buys mostly a wash, not a win.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from imdc.config import MANDATORY_UFS, QUANTILE_LEVELS
from imdc.features.panel import _origin_anchored_series, build_prediction_features

EXOG_COLS = ["temp_med_roll4", "precip_med_roll4", "enso_lag0"]
CANDIDATE_ORDERS = [
    ((1, 0, 0), (1, 0, 0, 52)), ((2, 0, 0), (1, 0, 0, 52)),
    ((1, 0, 1), (1, 0, 0, 52)), ((2, 0, 1), (1, 0, 0, 52)),
    ((1, 0, 0), (1, 0, 1, 52)), ((2, 0, 1), (1, 0, 1, 52)),
]
VALIDATION_TAIL_WEEKS = 26
N_SIM = 1000

_FROZEN_ORDERS: dict = {}  # uf -> (order, seasonal_order), selected once, module-level cache


def _fit_series(y, exog, order, seasonal_order):
    mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
    return mod.fit(disp=False, maxiter=100)


def _select_order(y, exog) -> tuple:
    """Grid-search order selection via a held-out tail, scored by mean absolute error on the log
    scale (a cheap proxy for pinball/WIS that avoids fitting each candidate at every quantile)."""
    if len(y) < VALIDATION_TAIL_WEEKS * 3:
        return CANDIDATE_ORDERS[0]
    split = len(y) - VALIDATION_TAIL_WEEKS
    y_tr, y_val = y[:split], y[split:]
    exog_tr = exog[:split] if exog is not None else None
    exog_val = exog[split:] if exog is not None else None

    best_score, best_order = np.inf, CANDIDATE_ORDERS[0]
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            res = _fit_series(y_tr, exog_tr, order, seasonal_order)
            fc = res.get_forecast(steps=len(y_val), exog=exog_val).predicted_mean
            score = float(np.mean(np.abs(fc - y_val)))
        except Exception:
            score = np.inf
        if score < best_score:
            best_score, best_order = score, (order, seasonal_order)
    return best_order


class SarimaxV2Model:
    name = "sarimax_v2"

    def __init__(self, disease: str = "dengue", ufs: list = MANDATORY_UFS,
                 quantile_levels: list = QUANTILE_LEVELS, select_orders: bool = False, seed: int = 0):
        self.disease = disease
        self.ufs = list(ufs)
        self.quantile_levels = quantile_levels
        self.select_orders = select_orders  # True only for the one designated tuning fold
        self.seed = seed
        self._results = {}

    def fit(self, train_df, fold) -> "SarimaxV2Model":
        self._fold = fold
        origin_df = _origin_anchored_series(fold, self.disease)
        origin_df = origin_df[origin_df["uf"].isin(self.ufs)]

        self._results = {}
        for uf, g in origin_df.groupby("uf"):
            g = g.sort_values("date").dropna(subset=EXOG_COLS)
            if len(g) < 104:
                continue
            y = g["log_inc"].to_numpy()
            exog = g[EXOG_COLS].to_numpy()

            if self.select_orders or uf not in _FROZEN_ORDERS:
                _FROZEN_ORDERS[uf] = _select_order(y, exog)
            order, seasonal_order = _FROZEN_ORDERS[uf]

            try:
                res = _fit_series(y, exog, order, seasonal_order)
                self._results[uf] = res
            except Exception:
                continue
        return self

    def predict(self, target_grid: pd.DataFrame, quantile_levels: list = None) -> pd.DataFrame:
        quantile_levels = list(quantile_levels or self.quantile_levels)
        feats, _ = build_prediction_features(self._fold, target_grid, self.disease)
        rng = np.random.default_rng(self.seed)

        rows = []
        for uf, gg in feats.groupby("uf"):
            res = self._results.get(uf)
            if res is None:
                continue
            gg = gg.sort_values("horizon_weeks")
            max_h = int(gg["horizon_weeks"].max())
            exog_future = gg[EXOG_COLS].fillna(0.0).drop_duplicates().iloc[[0]].to_numpy()
            exog_future = np.repeat(exog_future, max_h, axis=0)  # constant origin-anchored exog

            try:
                sims = res.simulate(nsimulations=max_h, repetitions=N_SIM, anchor="end",
                                     exog=exog_future, random_state=rng)
            except Exception:
                continue
            sims = np.asarray(sims)
            if sims.ndim == 3:
                sims = sims[:, 0, :]  # (max_h, N_SIM)
            pop = gg["population"].to_numpy()
            counts = np.maximum(np.expm1(sims) * pop[:1] / 1e5, 0.0)  # broadcast pop (same origin)

            for _, row in gg.iterrows():
                h = int(row["horizon_weeks"]) - 1
                if h < 0 or h >= counts.shape[0]:
                    continue
                vals = np.sort(np.quantile(counts[h], quantile_levels))
                for tau, v in zip(quantile_levels, vals):
                    rows.append({"uf": uf, "date": row["target_date"], "quantile_level": tau,
                                 "predicted_value": float(v)})
        return pd.DataFrame(rows)
