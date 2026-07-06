"""Hierarchical empirical-Bayes climatology: schema, monotonicity, shrinkage sanity.

Kept as a paper ablation (not an ensemble member) - see models/hierarchical.py.
"""
import numpy as np
import pytest

from imdc.config import QUANTILE_LEVELS
from imdc.data.folds import get_folds
from imdc.evaluation.harness import _target_date_grid, build_state_training_frame
from imdc.models.hierarchical import HierarchicalClimatologicalModel

UFS = ["SP", "RJ", "AC"]  # large / mid / small


@pytest.fixture(scope="module")
def fitted():
    fold = get_folds("dengue")[0]
    model = HierarchicalClimatologicalModel().fit(build_state_training_frame(fold, "dengue"), fold)
    grid = _target_date_grid(fold, ufs=UFS)
    return model, grid, model.predict(grid)


def test_schema_and_non_negative(fitted):
    _, grid, preds = fitted
    assert {"uf", "date", "quantile_level", "predicted_value"}.issubset(preds.columns)
    assert (preds["predicted_value"] >= 0).all()
    assert len(preds) == len(grid) * len(QUANTILE_LEVELS)


def test_quantiles_monotone_per_unit(fitted):
    _, _, preds = fitted
    for _, g in preds.groupby(["uf", "date"]):
        v = g.sort_values("quantile_level")["predicted_value"].to_numpy()
        assert np.all(np.diff(v) >= -1e-9)


def test_shrinkage_weights_valid_and_volume_ordered(fitted):
    model, _, _ = fitted
    lam = model._lambda
    assert all(0.0 <= v <= 1.0 for v in lam.values())
    # high-volume state pools less toward the region (larger lambda) than a low-volume one
    assert lam["SP"] > lam["AC"]
