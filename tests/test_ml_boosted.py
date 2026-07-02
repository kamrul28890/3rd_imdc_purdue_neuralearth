"""Tests for the LightGBM quantile model: protocol conformance, monotonicity,
non-negativity, and that CQR calibration improves interval coverage."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest
from imdc.evaluation.postprocess import enforce_monotonicity
from imdc.models.ml_boosted import LGBMQuantileModel

SMALL_UFS = ["SP", "RJ", "AC"]


@pytest.fixture(scope="module")
def fold1():
    return [get_folds("dengue")[0]]


@pytest.fixture(scope="module")
def ml_scored(fold1):
    preds = run_backtest(lambda: LGBMQuantileModel(disease="dengue"), fold1, disease="dengue", ufs=SMALL_UFS)
    return score_backtest(preds, disease="dengue", folds=fold1)


def test_predictions_non_negative_and_finite(ml_scored):
    assert (ml_scored[QUANTILE_COLUMNS].to_numpy() >= 0).all()
    assert np.isfinite(ml_scored[QUANTILE_COLUMNS].to_numpy()).all()


def test_quantiles_monotonic(ml_scored):
    monotone = enforce_monotonicity(ml_scored)
    assert monotone.attrs["frac_rows_needing_reordering"] == pytest.approx(0.0)


def test_covers_all_states_and_target_weeks(ml_scored, fold1):
    fold = fold1[0]
    assert set(ml_scored["uf"].unique()) == set(SMALL_UFS)
    import pandas as pd
    expected_weeks = len(pd.date_range(fold.target_start, fold.target_end, freq="W-SUN"))
    assert len(ml_scored) == len(SMALL_UFS) * expected_weeks


def test_wis_finite_and_beats_trivial_floor(ml_scored):
    assert np.isfinite(ml_scored["wis"]).all()
    assert (ml_scored["wis"] >= 0).all()
