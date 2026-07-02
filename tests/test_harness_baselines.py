import numpy as np
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.evaluation.harness import run_backtest, score_backtest, summarize
from imdc.evaluation.postprocess import enforce_monotonicity

SMALL_UFS = ["SP", "RJ", "AC"]  # one large, one mid, one small state - fast test


@pytest.fixture(scope="module")
def fold1():
    return [get_folds("dengue")[0]]


def _run_and_score(model_cls, fold1):
    preds = run_backtest(model_cls, fold1, disease="dengue", ufs=SMALL_UFS)
    assert set(preds["uf"].unique()) == set(SMALL_UFS)
    assert (preds["predicted_value"] >= 0).all()

    scored = score_backtest(preds, disease="dengue", folds=fold1)
    return scored


class TestClimatologicalQuantileEndToEnd:
    def test_schema_and_shape(self, fold1):
        scored = _run_and_score(ClimatologicalQuantileModel, fold1)
        assert set(QUANTILE_COLUMNS).issubset(scored.columns)
        n_dates = len(scored["date"].unique())
        assert len(scored) == len(SMALL_UFS) * n_dates

    def test_quantiles_are_monotonic(self, fold1):
        scored = _run_and_score(ClimatologicalQuantileModel, fold1)
        monotone = enforce_monotonicity(scored)
        assert monotone.attrs["frac_rows_needing_reordering"] == pytest.approx(0.0)

    def test_wis_finite_and_non_negative(self, fold1):
        scored = _run_and_score(ClimatologicalQuantileModel, fold1)
        assert np.isfinite(scored["wis"]).all()
        assert (scored["wis"] >= 0).all()

    def test_beats_naive_on_strongly_seasonal_state(self, fold1):
        """SP has a strong, consistent seasonal signal (see EDA) - climatology should
        beat a flat persistence forecast there."""
        clim = _run_and_score(ClimatologicalQuantileModel, fold1)
        naive = _run_and_score(NaiveModel, fold1)
        clim_sp = clim[clim["uf"] == "SP"]["wis"].mean()
        naive_sp = naive[naive["uf"] == "SP"]["wis"].mean()
        assert clim_sp < naive_sp


class TestAllBaselinesRunCleanly:
    @pytest.mark.parametrize("model_cls", [NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel])
    def test_runs_without_error_and_produces_valid_scores(self, model_cls, fold1):
        scored = _run_and_score(model_cls, fold1)
        assert len(scored) > 0
        assert np.isfinite(scored["wis"]).all()


class TestSummarize:
    def test_summarize_by_model_and_fold(self, fold1):
        scored = _run_and_score(ClimatologicalQuantileModel, fold1)
        summary = summarize(scored, by=["fold_id"])
        assert "wis" in summary.columns
        assert len(summary) == 1
