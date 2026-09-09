"""The `Forecaster` protocol every model in `imdc.models`/`imdc.evaluation.baselines`
implements (IMPROVEMENTS.md Sec 3.1).

This was previously an unenforced, docstring-only convention. Making it an explicit
`typing.Protocol` lets every model be type-checked against the same contract without
requiring a common base class (structural typing - a model "is a" Forecaster if its
methods match, nothing needs to subclass this).
"""
from typing import Protocol, runtime_checkable

import pandas as pd

from imdc.data.folds import Fold
from imdc.config import QUANTILE_LEVELS


@runtime_checkable
class Forecaster(Protocol):
    """fit/predict contract shared by every model family in this project.

    fit: trains the model on `train_df` (the harness's leakage-safe training frame for
    `fold`) and returns self, so `model_factory().fit(train_df, fold)` chains.
    predict: given a target grid (uf/geocode, date, horizon_weeks rows), returns the
    long-format predictions frame (one row per target unit x quantile_level) the
    harness/submission code expects.
    """

    def fit(self, train_df: pd.DataFrame, fold: Fold) -> "Forecaster": ...

    def predict(self, target_grid: pd.DataFrame,
                quantile_levels: list = QUANTILE_LEVELS) -> pd.DataFrame: ...
