"""Climate-conditioned surge template - extends surge_template.py with a climate-modulated gain
term, inspired by "LNCC CLIDENGO"'s climate-modulated growth-rate ODE (see docs/FUTURE_WORK.md
Sec 6c) - reimplemented independently as a much simpler linear conditioning, not their Brière-curve
ODE system (which would require numerically integrating a compartmental model, out of scope for a
quick, well-scoped test of the underlying idea: does knowing a season is warmer/cooler than normal
help predict how big that season's outbreak gain will be?).

Design: `surge_template.py` samples each forecast season's peak-height "gain" from a single
unconditional log-normal fit to historical peak heights. This file instead regresses each
historical season's log peak-height against that season's own mean temperature anomaly
(`log_gain ~ b0 + b1 * mean_anomaly`, ordinary least squares per state), then centers the sampled
gain distribution on the regression's prediction for the *target* season's anomaly, rather than on
the unconditional historical mean. Since we cannot observe the target season's future climate at
forecast time, the target-season anomaly is proxied by the average temperature anomaly over the
8 weeks immediately before the fold's cutoff - a real, leakage-safe, available-at-forecast-time
signal (temperature anomalies, unlike case counts, have enough persistence over months for a
recent-past reading to carry some information about the season ahead, e.g. an ENSO-driven warm
spell). If a state has too few historical seasons (<6) to fit a stable regression, or the fit
would be driven by 1-2 outlier seasons, this falls back to `surge_template.py`'s unconditional
gain distribution for that state (b1=0).
"""
import numpy as np
import pandas as pd
from epiweeks import Week

from imdc.data.folds import cutoff_filter
from imdc.features.panel import state_climate_full
from imdc.models.mechanistic import SEASON_LEN, _season_start_year, season_week_from_date
from imdc.models.surge_template import SurgeTemplateModel

_MIN_SEASONS_FOR_CLIMATE_FIT = 6
_RECENT_WEEKS_FOR_TARGET_PROXY = 8


def _season_mean_anomaly(fold) -> pd.DataFrame:
    """(uf, season_start_year) -> mean temp anomaly across that season's weeks, from
    origin-anchored (leakage-safe, cutoff-filtered) climate history."""
    climate = cutoff_filter(state_climate_full(), fold.train_cutoff).sort_values(["uf", "date"]).reset_index(drop=True)
    climate["epiweek"] = [Week.fromdate(d).week for d in climate["date"]]
    clim_by_ew = climate.groupby(["uf", "epiweek"])["temp_med"].transform("mean")
    climate["temp_anomaly"] = climate["temp_med"] - clim_by_ew
    climate["season_start_year"] = [_season_start_year(d) for d in climate["date"]]
    climate["season_week"] = [season_week_from_date(d) for d in climate["date"]]
    return climate


class ClimateSurgeModel(SurgeTemplateModel):
    name = "climate_surge"

    def fit(self, train_df, fold) -> "ClimateSurgeModel":
        super().fit(train_df, fold)
        climate = _season_mean_anomaly(fold)
        in_season = climate[(climate["season_week"] >= 1) & (climate["season_week"] <= SEASON_LEN)]
        season_anom = in_season.groupby(["uf", "season_start_year"])["temp_anomaly"].mean()

        recent = climate[climate["date"] > fold.train_cutoff - pd.Timedelta(weeks=_RECENT_WEEKS_FOR_TARGET_PROXY)]
        self._target_anomaly = recent.groupby("uf")["temp_anomaly"].mean().to_dict()

        self._climate_beta = {}
        for uf, years in self._season_years.items():
            if uf not in self._templates:
                continue
            valid_peak = self._peak_offsets[uf]  # same filter (peak_val>0) already applied in super().fit()
            if len(valid_peak) != len(years):
                # super().fit() may have dropped some seasons (peak_val<=0); realign by recomputing
                # which years survived is not tracked post-filter, so skip climate conditioning for
                # this state rather than risk misaligning years to the wrong gain values.
                continue
            anom = np.array([season_anom.get((uf, int(y)), np.nan) for y in years])
            valid = ~np.isnan(anom)
            if valid.sum() < _MIN_SEASONS_FOR_CLIMATE_FIT:
                self._climate_beta[uf] = 0.0
                continue
            # recover each season's own log_gain (peak height) - not stored individually by the
            # base class, so recompute from the trajectory matrix directly, same definition
            # (np.max per row) as SurgeTemplateModel.fit() used before averaging into mu/sigma.
            traj = self._trajectories[uf]
            peak_val = np.max(traj, axis=1)
            pos = peak_val > 0
            if pos.sum() != len(years):
                self._climate_beta[uf] = 0.0
                continue
            log_gain_per_season = np.log(peak_val[pos])
            a, m = anom[valid], log_gain_per_season[valid]
            if np.std(a) < 1e-6:
                self._climate_beta[uf] = 0.0
                continue
            b1, b0 = np.polyfit(a, m, 1)
            self._climate_beta[uf] = float(b1)
            self._climate_intercept = getattr(self, "_climate_intercept", {})
            self._climate_intercept[uf] = float(b0)
        return self

    def predict(self, target_grid, quantile_levels=None):
        # Temporarily shift each uf's gain-distribution mean to the climate-conditioned prediction,
        # then delegate to the parent's predict() for the actual sampling/NegBinom layering -
        # avoids duplicating that logic.
        original_mu = dict(self._log_gain_mu)
        intercepts = getattr(self, "_climate_intercept", {})
        for uf, beta in self._climate_beta.items():
            if beta == 0.0 or uf not in intercepts:
                continue
            target_anom = self._target_anomaly.get(uf)
            if target_anom is None or np.isnan(target_anom):
                continue
            self._log_gain_mu[uf] = intercepts[uf] + beta * target_anom
        try:
            return super().predict(target_grid, quantile_levels)
        finally:
            self._log_gain_mu = original_mu
