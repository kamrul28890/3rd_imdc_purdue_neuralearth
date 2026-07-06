"""Conformal interval recalibration: factor computation, widening, monotonicity, coverage."""
import numpy as np
import pandas as pd
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.evaluation.postprocess import apply_conformal_widen, conformal_widen_factors

LEVELS = [50, 80, 90, 95]


def test_widen_factors_are_multiplicative_cqr_quantiles():
    # pred=0 with symmetric unit intervals -> the multiple needed to cover y is |y|,
    # so s_L must equal the L/100 empirical quantile of |y|.
    y = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    calib = pd.DataFrame({"pred": np.zeros(len(y)), "observed_value": y})
    for level in LEVELS:
        calib[f"upper_{level}"], calib[f"lower_{level}"] = 1.0, -1.0
    factors = conformal_widen_factors(calib)
    for level in LEVELS:
        assert factors[level] == pytest.approx(np.quantile(np.abs(y), level / 100))


def test_apply_widens_about_median_and_stays_monotone():
    wide = pd.DataFrame([{
        "pred": 10.0,
        "lower_50": 8, "upper_50": 12, "lower_80": 6, "upper_80": 14,
        "lower_90": 4, "upper_90": 16, "lower_95": 2, "upper_95": 18,
    }])
    out = apply_conformal_widen(wide, {level: 2.0 for level in LEVELS})
    assert out["pred"].iloc[0] == 10.0                       # median untouched
    assert out["upper_50"].iloc[0] == pytest.approx(14)      # 10 + 2*(12-10)
    assert out["lower_50"].iloc[0] == pytest.approx(6)       # 10 - 2*(10-8)
    vals = out[QUANTILE_COLUMNS].iloc[0].to_numpy(dtype=float)
    assert np.all(np.diff(vals) >= 0)                        # nested / monotone


def test_widening_restores_nominal_coverage_in_sample():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 4000)
    calib = pd.DataFrame({"pred": np.zeros(len(y)), "observed_value": y})
    for level, hw in [(50, 0.1), (80, 0.2), (90, 0.3), (95, 0.4)]:  # far too narrow
        calib[f"upper_{level}"], calib[f"lower_{level}"] = hw, -hw
    before = {level: float(((y >= calib[f"lower_{level}"]) & (y <= calib[f"upper_{level}"])).mean()) for level in LEVELS}
    out = apply_conformal_widen(calib, conformal_widen_factors(calib))
    for level in LEVELS:
        cov = float(((y >= out[f"lower_{level}"]) & (y <= out[f"upper_{level}"])).mean())
        assert before[level] < level / 100                   # started overconfident
        assert cov == pytest.approx(level / 100, abs=0.03)   # recalibrated to nominal
