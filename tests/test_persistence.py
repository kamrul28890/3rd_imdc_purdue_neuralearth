"""Model persistence round-trips: a loaded model predicts identically to the saved one."""
import numpy as np
import pandas as pd
import pytest

from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.evaluation.harness import _target_date_grid, build_state_training_frame
from imdc.models.mechanistic import MechanisticTrajectoryModel
from imdc.models.persistence import load_model, save_model

UFS = ["SP", "AC"]


def _fit_and_grid(model_factory):
    fold = get_folds("dengue")[0]
    train = build_state_training_frame(fold, "dengue")
    model = model_factory()
    model.fit(train, fold)
    grid = _target_date_grid(fold, ufs=UFS)
    return model, grid


@pytest.mark.parametrize("factory", [
    ClimatologicalQuantileModel,
    lambda: MechanisticTrajectoryModel(disease="dengue", n_boot=200),
])
def test_save_load_predicts_identically(factory, tmp_path):
    model, grid = _fit_and_grid(factory)
    before = model.predict(grid).sort_values(["uf", "date", "quantile_level"]).reset_index(drop=True)

    path = save_model(model, tmp_path / "m.joblib")
    assert path.exists()
    loaded = load_model(path)
    after = loaded.predict(grid).sort_values(["uf", "date", "quantile_level"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(after["predicted_value"]).all()
