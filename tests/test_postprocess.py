"""Conformal interval recalibration: factor computation, widening, monotonicity, coverage."""
import numpy as np
import pandas as pd
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.evaluation.postprocess import apply_conformal_widen, clip_nonnegative, conformal_widen_factors

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


def test_clip_nonnegative_floors_at_zero_and_stays_monotone():
    # A widening factor > 1 on a small pred can push a lower bound negative (real case:
    # the fitted 95% conformal factor is ~1.73) - clip_nonnegative must floor it at 0
    # without breaking the nesting that apply_conformal_widen already established.
    wide = pd.DataFrame([{
        "pred": 1.0,
        "lower_50": 0.5, "upper_50": 1.5, "lower_80": -1.0, "upper_80": 3.0,
        "lower_90": -3.0, "upper_90": 5.0, "lower_95": -6.0, "upper_95": 8.0,
    }])
    out = clip_nonnegative(wide)
    vals = out[QUANTILE_COLUMNS].iloc[0].to_numpy(dtype=float)
    assert vals.min() == 0.0
    assert out["upper_95"].iloc[0] == 8.0   # positive values untouched
    assert np.all(np.diff(vals) >= 0)       # still nested after clipping


# --- size-aware (burden-scaled) recalibration -------------------------------------------------
from imdc.evaluation.postprocess import apply_size_aware_scale, fit_size_aware_scale, state_burden_index


def _wide_row(uf, pred, lo95, hi95, y=None):
    row = {"uf": uf, "pred": float(pred)}
    for i, level in enumerate([50, 80, 90, 95]):
        frac = (i + 1) / 4
        row[f"lower_{level}"] = pred - frac * (pred - lo95)
        row[f"upper_{level}"] = pred + frac * (hi95 - pred)
    if y is not None:
        row["observed_value"] = float(y)
    return row


def test_size_aware_scale_is_log_space_about_median_and_burden_dependent():
    wide = pd.DataFrame([_wide_row("SMALL", 100, 20, 500), _wide_row("BIG", 100, 20, 500)])
    z = pd.Series({"SMALL": -1.0, "BIG": 1.0})
    out = apply_size_aware_scale(wide, z, a=1.0, b=0.2)
    np.testing.assert_allclose(out["pred"], 100.0)                       # median untouched
    m = np.log1p(100)
    for uf, s in [("SMALL", 0.8), ("BIG", 1.2)]:
        r = out[out["uf"] == uf].iloc[0]
        assert r["upper_95"] == pytest.approx(np.expm1(m + s * (np.log1p(500) - m)))
        assert r["lower_95"] == pytest.approx(np.expm1(m + s * (np.log1p(20) - m)))
        assert r["lower_95"] >= 0 and np.all(np.diff(r[QUANTILE_COLUMNS].to_numpy(dtype=float)) >= 0)
    small, big = out.iloc[0], out.iloc[1]
    assert small["upper_95"] - small["lower_95"] < 480 < big["upper_95"] - big["lower_95"]


def test_size_aware_identity_and_missing_state():
    wide = pd.DataFrame([_wide_row("AC", 10, 2, 40)])
    out = apply_size_aware_scale(wide, pd.Series({"AC": 0.3}), a=1.0, b=0.0)
    np.testing.assert_allclose(out[QUANTILE_COLUMNS].to_numpy(float), wide[QUANTILE_COLUMNS].to_numpy(float))
    with pytest.raises(ValueError):
        apply_size_aware_scale(wide, pd.Series({"SP": 0.0}), a=1.0, b=0.1)


def test_fit_size_aware_sharpens_overwide_small_state_without_hurting_pooled_wis():
    # Small state: intervals far too wide for observations at the median -> sharpening helps it.
    # Big state: observations land outside the intervals -> it needs widening, and dominates the
    # pooled WIS, so a uniform sharpening would be rejected by the constraint but b > 0 is not.
    rows = [_wide_row("SMALL", 10, 1, 200, y=10) for _ in range(20)]
    rows += [_wide_row("BIG", 10_000, 8_000, 12_000, y=14_000) for _ in range(20)]
    calib = pd.DataFrame(rows)
    z = pd.Series({"SMALL": -1.0, "BIG": 1.0})
    fit = fit_size_aware_scale(calib, z)
    assert fit["b"] > 0
    assert fit["objective"] < fit["objective_unscaled"]
    assert fit["pooled_wis"] <= fit["pooled_wis_unscaled"]


def test_state_burden_index_uses_only_data_up_to_cutoff():
    dates = pd.date_range("2016-01-03", periods=10, freq="W")
    cases = pd.concat([pd.DataFrame({"uf": "A", "date": dates, "casos": 10}),
                       pd.DataFrame({"uf": "B", "date": dates, "casos": 1000})])
    late = pd.DataFrame({"uf": "A", "date": [pd.Timestamp("2030-01-06")], "casos": [10**9]})
    z = state_burden_index(pd.concat([cases, late]), cutoff=dates[-1])
    assert z["A"] < 0 < z["B"] and z.mean() == pytest.approx(0)
    assert z.equals(state_burden_index(cases, cutoff=dates[-1]))
