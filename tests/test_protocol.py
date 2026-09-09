"""Every model family must structurally satisfy `imdc.protocol.Forecaster`
(IMPROVEMENTS.md Sec 3.1: "no formal model interface" was a P1 gap).

Deliberately only checks construction + protocol conformance (fit/predict exist with
the right arity), not fit/predict *behavior* - that's what each model's own test file
(where one exists) covers. This test's job is to make the shared contract enforceable:
add a model here and it fails fast if it stops looking like a Forecaster, e.g. someone
reintroduces a dead `covariates` param or a model forgets to return `self` from `fit`.
"""
import inspect

import pytest

from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.models.climate_surge import ClimateSurgeModel
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.dl_sequence_wis import DLSequenceWISModel
from imdc.models.hierarchical import HierarchicalClimatologicalModel
from imdc.models.mechanistic import MechanisticTrajectoryModel
from imdc.models.mechanistic_nsub import NSubEpidemicMechanisticModel
from imdc.models.ml_boosted import LGBMQuantileModel
from imdc.models.prophet_model import ProphetModel
from imdc.models.prophet_v2 import ProphetV2Model
from imdc.models.sarimax_model import SarimaxModel
from imdc.models.sarimax_v2 import SarimaxV2Model
from imdc.models.surge_template import SurgeTemplateModel
from imdc.models.xgb_quantile import XGBQuantileModel
from imdc.protocol import Forecaster

ALL_MODEL_FACTORIES = [
    NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel,
    ClimateSurgeModel, DLSequenceModel, DLSequenceWISModel,
    HierarchicalClimatologicalModel, MechanisticTrajectoryModel,
    NSubEpidemicMechanisticModel, LGBMQuantileModel, ProphetModel,
    ProphetV2Model, SarimaxModel, SarimaxV2Model, SurgeTemplateModel,
    XGBQuantileModel,
]


@pytest.mark.parametrize("factory", ALL_MODEL_FACTORIES, ids=lambda f: f.__name__)
def test_model_satisfies_forecaster_protocol(factory):
    model = factory()
    assert isinstance(model, Forecaster), (
        f"{factory.__name__} does not structurally satisfy Forecaster "
        f"(missing/mismatched fit or predict)"
    )


@pytest.mark.parametrize("factory", ALL_MODEL_FACTORIES, ids=lambda f: f.__name__)
def test_fit_signature_has_no_dead_covariates_param(factory):
    """Regression guard for IMPROVEMENTS.md Sec 3.1's dead `covariates=None` params."""
    params = inspect.signature(factory.fit).parameters
    assert "covariates" not in params, (
        f"{factory.__name__}.fit still has a covariates param - "
        f"either use it for real or remove it, don't leave it dead"
    )
