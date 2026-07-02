"""Tests for the GRU NegBinom model: protocol conformance, monotonicity,
non-negativity, and the NB negative-log-likelihood's correctness.

Uses a tiny config (1 ensemble member, few epochs) for speed - correctness of
the plumbing, not forecast quality, is what's under test here.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from imdc.config import QUANTILE_COLUMNS
from imdc.data.folds import get_folds
from imdc.evaluation.harness import run_backtest, score_backtest
from imdc.evaluation.postprocess import enforce_monotonicity
from imdc.models.dl_sequence import DLSequenceModel, _negbin_nll

SMALL_UFS = ["SP", "RJ", "AC"]


def test_negbin_nll_matches_scipy():
    """The hand-implemented NB2 NLL must match scipy's on random (y, mu, alpha)."""
    from scipy.stats import nbinom

    rng = np.random.default_rng(0)
    y = rng.integers(0, 500, size=50).astype(float)
    mu = rng.uniform(1, 400, size=50)
    alpha = rng.uniform(0.05, 2.0, size=50)
    ours = _negbin_nll(torch.tensor(y), torch.tensor(mu), torch.tensor(np.log(alpha))).numpy()
    r = 1.0 / alpha
    p = r / (r + mu)
    scipy_nll = -nbinom.logpmf(y, r, p)
    assert np.allclose(ours, scipy_nll, atol=1e-4)


@pytest.fixture(scope="module")
def dl_scored():
    fold1 = [get_folds("dengue")[0]]
    model_factory = lambda: DLSequenceModel(disease="dengue", n_ensemble=1, epochs=3)
    preds = run_backtest(model_factory, fold1, disease="dengue", ufs=SMALL_UFS)
    return score_backtest(preds, disease="dengue", folds=fold1)


def test_predictions_non_negative_and_finite(dl_scored):
    vals = dl_scored[QUANTILE_COLUMNS].to_numpy()
    assert (vals >= 0).all()
    assert np.isfinite(vals).all()


def test_quantiles_monotonic(dl_scored):
    monotone = enforce_monotonicity(dl_scored)
    assert monotone.attrs["frac_rows_needing_reordering"] == pytest.approx(0.0)


def test_covers_all_requested_states(dl_scored):
    assert set(dl_scored["uf"].unique()) == set(SMALL_UFS)


def test_wis_finite(dl_scored):
    assert np.isfinite(dl_scored["wis"]).all()
    assert (dl_scored["wis"] >= 0).all()
