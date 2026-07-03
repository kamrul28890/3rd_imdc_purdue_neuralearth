"""Tests for the semi-mechanistic trajectory model: season-week mapping, Richards
fit sanity, and end-to-end protocol conformance."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest
from imdc.evaluation.postprocess import enforce_monotonicity
from imdc.models.mechanistic import MechanisticTrajectoryModel, fit_richards, richards_cumulative, season_week

SMALL_UFS = ["SP", "RJ", "AC"]


def test_season_week_mapping():
    # EW41 starts a season (week 1); EW52 -> 12; EW1 -> 13; EW40 -> 52
    assert season_week(41) == 1
    assert season_week(52) == 12
    assert season_week(1) == 13
    assert season_week(40) == 52


def test_richards_fit_recovers_synthetic_curve():
    t = np.arange(1, 53, dtype=float)
    true = dict(K=10000.0, r=0.4, tp=20.0, alpha=1.0)
    cumulative = richards_cumulative(t, **true)
    incidence = np.diff(cumulative, prepend=0.0)
    params = fit_richards(incidence)
    assert params is not None
    assert params["final_size"] == pytest.approx(true["K"], rel=0.15)
    # observed peak week should land near the true inflection
    assert abs(params["obs_peak_week"] - true["tp"]) <= 4


@pytest.fixture(scope="module")
def mech_scored():
    fold1 = [get_folds("dengue")[0]]
    preds = run_backtest(lambda: MechanisticTrajectoryModel(disease="dengue", n_boot=300),
                         fold1, disease="dengue", ufs=SMALL_UFS)
    return score_backtest(preds, disease="dengue", folds=fold1)


def test_predictions_non_negative_and_finite(mech_scored):
    vals = mech_scored[QUANTILE_COLUMNS].to_numpy()
    assert (vals >= 0).all()
    assert np.isfinite(vals).all()


def test_quantiles_monotonic(mech_scored):
    mono = enforce_monotonicity(mech_scored)
    assert mono.attrs["frac_rows_needing_reordering"] == pytest.approx(0.0)


def test_covers_all_states_and_finite_wis(mech_scored):
    assert set(mech_scored["uf"].unique()) == set(SMALL_UFS)
    assert np.isfinite(mech_scored["wis"]).all()
    assert (mech_scored["wis"] >= 0).all()
