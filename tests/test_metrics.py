import numpy as np
import pandas as pd
import pytest

from imdc.config import QUANTILE_LEVELS
from imdc.evaluation.metrics import (
    coverage,
    interval_score,
    mase,
    peak_magnitude_error,
    peak_timing_error,
    pinball_loss,
    relative_wis,
    wis_decomposition,
    wis_from_intervals,
    wis_from_quantiles,
)


def _reference_wis_bracher(pred_row: dict, y: float) -> float:
    """Independent re-derivation of WIS (Bracher et al. 2021), vendored for cross-checking
    our implementation against a second, from-scratch source rather than testing the
    implementation against itself. Also matches mosqlient==2.5.2's
    mosqlient.scoring.score.compute_wis (installed package source read directly)."""
    K = 4
    w0 = 0.5
    total = w0 * abs(y - pred_row["pred"])
    for level, alpha in [(50, 0.5), (80, 0.2), (90, 0.1), (95, 0.05)]:
        l, u = pred_row[f"lower_{level}"], pred_row[f"upper_{level}"]
        is_alpha = (u - l) + (2 / alpha) * max(0, l - y) + (2 / alpha) * max(0, y - u)
        total += (alpha / 2) * is_alpha
    return total / (K + 0.5)


class TestWisToyExample:
    """K=2-style hand-computed example (using the full 4-interval format with wide
    outer intervals so only 50 and 90 actually bind) - see plan Sec 2.4."""

    def setup_method(self):
        self.pred = pd.DataFrame([{
            "pred": 10, "lower_50": 6, "upper_50": 14,
            "lower_80": 4, "upper_80": 16,
            "lower_90": 2, "upper_90": 20,
            "lower_95": 0, "upper_95": 24,
        }])
        self.y = np.array([15.0])

    def test_hand_computed_value(self):
        wis = wis_from_intervals(self.pred, self.y)[0]
        # median: 0.5*|15-10| = 2.5
        # 50%: y=15>14 -> IS=(14-6)+4*(15-14)=12; w=0.25 -> 3.0
        # 80%: y=15<16 -> IS=(16-4)=12; w=0.1 -> 1.2
        # 90%: y=15<20 -> IS=(20-2)=18; w=0.05 -> 0.9
        # 95%: y=15<24 -> IS=(24-0)=24; w=0.025 -> 0.6
        # total = (2.5+3.0+1.2+0.9+0.6)/4.5 = 8.2/4.5
        expected = (2.5 + 3.0 + 1.2 + 0.9 + 0.6) / 4.5
        assert wis == pytest.approx(expected, abs=1e-9)

    def test_matches_independent_reference_implementation(self):
        wis = wis_from_intervals(self.pred, self.y)[0]
        ref = _reference_wis_bracher(self.pred.iloc[0].to_dict(), self.y[0])
        assert wis == pytest.approx(ref, abs=1e-9)

    def test_matches_pinball_loss_identity(self):
        """WIS must equal mean pinball loss over the 9 QUANTILE_LEVELS (structural identity)."""
        row = self.pred.iloc[0]
        quantile_values = np.array([
            row["lower_95"], row["lower_90"], row["lower_80"], row["lower_50"],
            row["pred"],
            row["upper_50"], row["upper_80"], row["upper_90"], row["upper_95"],
        ])
        wis_via_quantiles = wis_from_quantiles(quantile_values, QUANTILE_LEVELS, self.y[0])
        wis_via_intervals = wis_from_intervals(self.pred, self.y)[0]
        assert wis_via_quantiles == pytest.approx(wis_via_intervals, abs=1e-9)


class TestWisProperties:
    def test_zero_intervals_reduces_to_absolute_error(self):
        pred = pd.DataFrame([{
            "pred": 10.0,
            "lower_50": 10, "upper_50": 10, "lower_80": 10, "upper_80": 10,
            "lower_90": 10, "upper_90": 10, "lower_95": 10, "upper_95": 10,
        }])
        y = np.array([17.0])
        assert wis_from_intervals(pred, y)[0] == pytest.approx(abs(17.0 - 10.0))

    def test_perfect_forecast_has_zero_wis(self):
        pred = pd.DataFrame([{
            "pred": 5.0,
            "lower_50": 5, "upper_50": 5, "lower_80": 5, "upper_80": 5,
            "lower_90": 5, "upper_90": 5, "lower_95": 5, "upper_95": 5,
        }])
        assert wis_from_intervals(pred, np.array([5.0]))[0] == pytest.approx(0.0)

    def test_wis_decomposition_sums_to_wis(self):
        pred = pd.DataFrame([{
            "pred": 10, "lower_50": 6, "upper_50": 14, "lower_80": 4, "upper_80": 16,
            "lower_90": 2, "upper_90": 20, "lower_95": 0, "upper_95": 24,
        }])
        y = np.array([15.0])
        wis = wis_from_intervals(pred, y)
        decomp = wis_decomposition(pred, y)
        total = decomp["dispersion"] + decomp["overprediction"] + decomp["underprediction"]
        assert total.iloc[0] == pytest.approx(wis[0], abs=1e-9)

    def test_monte_carlo_coverage_converges_to_nominal(self):
        rng = np.random.default_rng(0)
        from scipy.stats import nbinom

        mu, alpha = 50.0, 0.1
        n = 1 / alpha
        p = n / (n + mu)
        draws = nbinom.rvs(n, p, size=5000, random_state=rng)

        levels = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}
        rows = {"pred": nbinom.ppf(0.5, n, p)}
        for level, (lo_tau, hi_tau) in levels.items():
            rows[f"lower_{level}"] = nbinom.ppf(lo_tau, n, p)
            rows[f"upper_{level}"] = nbinom.ppf(hi_tau, n, p)
        pred = pd.DataFrame([rows] * len(draws))

        cov = coverage(pred, draws)
        for level in [50, 80, 90, 95]:
            assert abs(cov[f"coverage_{level}"] - level / 100) < 0.03


class TestPinballAndIntervalScore:
    def test_pinball_loss_median_is_half_absolute_error(self):
        y, q = np.array([10.0]), np.array([7.0])
        assert pinball_loss(y, q, 0.5)[0] == pytest.approx(0.5 * abs(10 - 7))

    def test_interval_score_zero_penalty_when_inside(self):
        score = interval_score(np.array([5.0]), np.array([15.0]), np.array([10.0]), alpha=0.1)
        assert score[0] == pytest.approx(10.0)  # just the width, no penalty


class TestMase:
    def test_perfect_forecast_is_zero(self):
        y_train = np.sin(np.linspace(0, 20, 200)) * 10 + 20
        assert mase(y_train[-10:], y_train[-10:], y_train, m=52) == pytest.approx(0.0)


class TestPeakErrors:
    def test_timing_and_magnitude_error_on_shifted_peak(self):
        observed = pd.Series([1, 2, 10, 3, 1])
        pred_median = pd.Series([1, 2, 3, 10, 1])  # peak shifted 1 week late
        assert peak_timing_error(pred_median, observed) == 1
        # at the true peak week (idx 2), predicted value is 3, observed peak is 10
        mag = peak_magnitude_error(pred_median, observed)
        assert mag == pytest.approx(np.log(3 / 10))


class TestRelativeWis:
    def test_baseline_has_relative_wis_one(self):
        df = pd.DataFrame([
            {"model": "a", "unit": 1, "wis": 2.0}, {"model": "a", "unit": 2, "wis": 4.0},
            {"model": "b", "unit": 1, "wis": 1.0}, {"model": "b", "unit": 2, "wis": 2.0},
        ])
        rel = relative_wis(df, baseline_model="a", group_cols=["unit"])
        assert rel["a"] == pytest.approx(1.0)
        assert rel["b"] < rel["a"]  # b is uniformly better (half the WIS everywhere)
