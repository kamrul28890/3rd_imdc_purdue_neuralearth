"""Baseline forecasters: Naive, SeasonalNaive, ClimatologicalQuantile.

Every model implements the shared protocol used throughout the harness:
    fit(train_df, fold) -> self
    predict(target_dates, quantile_levels) -> long df [uf, date, quantile_level, predicted_value]

train_df must already be leakage-safe-filtered by the caller (state-level
aggregated case series with columns uf, date, casos, date <= fold.train_cutoff)
- baselines never re-derive their own cutoff logic, per the harness design.
Epiweeks (not ISO weeks) are used throughout to match official reporting.
"""
import numpy as np
import pandas as pd
from epiweeks import Week

from imdc.config import QUANTILE_LEVELS


def _epiweek_of_year(dates: pd.Series) -> np.ndarray:
    return np.array([Week.fromdate(d).week for d in pd.to_datetime(dates)])


class NaiveModel:
    """Median = last observed value per state. Spread from empirical h-step-difference quantiles."""

    name = "naive"

    def fit(self, train_df: pd.DataFrame, fold) -> "NaiveModel":
        self._last_value = train_df.sort_values("date").groupby("uf")["casos"].last()
        self._history = {uf: g.sort_values("date")["casos"].to_numpy() for uf, g in train_df.groupby("uf")}
        return self

    def _h_step_diff_quantiles(self, uf: str, h: int, quantile_levels: list) -> np.ndarray:
        series = self._history[uf]
        if len(series) <= h:
            h = max(1, len(series) - 1)
        diffs = series[h:] - series[:-h] if h > 0 and len(series) > h else np.array([0.0])
        if len(diffs) == 0:
            diffs = np.array([0.0])
        return np.quantile(diffs, quantile_levels)

    def predict(self, target_dates: pd.DataFrame, quantile_levels: list = QUANTILE_LEVELS) -> pd.DataFrame:
        """target_dates: df with columns uf, date, horizon_weeks."""
        rows = []
        for _, row in target_dates.iterrows():
            uf, date, h = row["uf"], row["date"], int(row["horizon_weeks"])
            last = self._last_value.get(uf, 0.0)
            diff_q = self._h_step_diff_quantiles(uf, h, quantile_levels)
            values = np.maximum(0.0, last + diff_q)
            for tau, v in zip(quantile_levels, values):
                rows.append({"uf": uf, "date": date, "quantile_level": tau, "predicted_value": v})
        return pd.DataFrame(rows)


class SeasonalNaiveModel:
    """Median = median of same-epiweek history across years. Spread from pooled residual quantiles."""

    name = "seasonal_naive"

    def fit(self, train_df: pd.DataFrame, fold) -> "SeasonalNaiveModel":
        df = train_df.copy()
        df["epiweek"] = _epiweek_of_year(df["date"])
        self._seasonal_median = df.groupby(["uf", "epiweek"])["casos"].median()

        residuals = {}
        for uf, g in df.groupby("uf"):
            med = g["epiweek"].map(lambda w: self._seasonal_median.get((uf, w), np.nan))
            res = (g["casos"].to_numpy() - med.to_numpy())
            residuals[uf] = res[~np.isnan(res)]
        self._residuals = residuals
        return self

    def predict(self, target_dates: pd.DataFrame, quantile_levels: list = QUANTILE_LEVELS) -> pd.DataFrame:
        rows = []
        target_dates = target_dates.copy()
        target_dates["epiweek"] = _epiweek_of_year(target_dates["date"])
        for _, row in target_dates.iterrows():
            uf, date, w = row["uf"], row["date"], int(row["epiweek"])
            median = self._seasonal_median.get((uf, w), np.nan)
            if np.isnan(median):
                median = self._seasonal_median.xs(uf, level="uf").median() if uf in self._seasonal_median.index.get_level_values("uf") else 0.0
            res = self._residuals.get(uf, np.array([0.0]))
            if len(res) == 0:
                res = np.array([0.0])
            values = np.maximum(0.0, median + np.quantile(res, quantile_levels))
            for tau, v in zip(quantile_levels, values):
                rows.append({"uf": uf, "date": date, "quantile_level": tau, "predicted_value": v})
        return pd.DataFrame(rows)


class ClimatologicalQuantileModel:
    """Per (state, epiweek +/-2 window), direct order-statistic empirical quantiles of raw counts.

    Horizon-invariant by design: the same per-epiweek quantile function is used
    regardless of horizon. This is the harness's validation client - pure order
    statistics, zero optimization, so any WIS discrepancy is unambiguously a
    harness bug rather than a modeling artifact.
    """

    name = "climatological_quantile"

    def fit(self, train_df: pd.DataFrame, fold) -> "ClimatologicalQuantileModel":
        df = train_df.copy()
        df["epiweek"] = _epiweek_of_year(df["date"])
        pooled = {}
        for uf, g in df.groupby("uf"):
            by_week = g.groupby("epiweek")["casos"].apply(list).to_dict()
            pooled_by_week = {}
            for w in range(1, 54):
                window_weeks = [((w - 3 + k - 1) % 53) + 1 for k in range(5)]  # w-2..w+2, circular over 1..53
                samples = []
                for ww in window_weeks:
                    samples.extend(by_week.get(ww, []))
                pooled_by_week[w] = np.array(samples) if samples else np.array([0.0])
            pooled[uf] = pooled_by_week
        self._pooled = pooled
        return self

    def predict(self, target_dates: pd.DataFrame, quantile_levels: list = QUANTILE_LEVELS) -> pd.DataFrame:
        rows = []
        target_dates = target_dates.copy()
        target_dates["epiweek"] = _epiweek_of_year(target_dates["date"])
        for _, row in target_dates.iterrows():
            uf, date, w = row["uf"], row["date"], int(row["epiweek"])
            samples = self._pooled.get(uf, {}).get(w, np.array([0.0]))
            values = np.maximum(0.0, np.quantile(samples, quantile_levels))
            for tau, v in zip(quantile_levels, values):
                rows.append({"uf": uf, "date": date, "quantile_level": tau, "predicted_value": v})
        return pd.DataFrame(rows)
