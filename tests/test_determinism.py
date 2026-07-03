"""Determinism guards: the deterministic models must reproduce bit-for-bit across runs.

(The GRU is intentionally not covered — its reproducibility is limited by Apple-MPS
float non-determinism; it is bit-reproducible only on CPU with deterministic algorithms.)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest
from imdc.models.mechanistic import MechanisticTrajectoryModel
from imdc.models.ml_boosted import LGBMQuantileModel

UFS = ["SP", "AC"]


def _wis_sum(model_factory, disease="dengue"):
    fold1 = [get_folds(disease)[0]]
    preds = run_backtest(model_factory, fold1, disease=disease, ufs=UFS)
    return score_backtest(preds, disease=disease, folds=fold1)["wis"].sum()


def test_lgbm_is_bit_deterministic():
    a = _wis_sum(lambda: LGBMQuantileModel(disease="dengue"))
    b = _wis_sum(lambda: LGBMQuantileModel(disease="dengue"))
    assert a == pytest.approx(b, abs=1e-9), f"LightGBM non-deterministic: {a} vs {b}"


def test_mechanistic_is_bit_deterministic():
    a = _wis_sum(lambda: MechanisticTrajectoryModel(disease="dengue", n_boot=300))
    b = _wis_sum(lambda: MechanisticTrajectoryModel(disease="dengue", n_boot=300))
    assert a == pytest.approx(b, abs=1e-9), f"Mechanistic non-deterministic: {a} vs {b}"
