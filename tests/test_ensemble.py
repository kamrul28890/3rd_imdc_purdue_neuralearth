"""Tests for the ensembling logic: monotonicity, identity, weighting correctness."""
import numpy as np
import pandas as pd
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.evaluation.postprocess import enforce_monotonicity
from imdc.models.ensemble import inverse_wis_weights, vincentization, weighted_ensemble


def _mk(model, uf, fold, base, observed):
    """One wide prediction row with monotone quantiles centered on `base`.

    `observed` is the actual outcome - identical across models for a given
    (uf, date, fold, horizon), since it is a grouping key, not a prediction."""
    offs = [-9, -6, -4, -2, 0, 2, 4, 6, 9]
    row = {"model": model, "uf": uf, "date": pd.Timestamp("2023-01-01"), "fold_id": fold,
           "horizon_weeks": 16, "observed_value": observed}
    for col, o in zip(QUANTILE_COLUMNS, offs):
        row[col] = base + o
    return row


@pytest.fixture
def preds():
    rows = []
    for uf, observed in [("SP", 115), ("RJ", 115)]:
        rows.append(_mk("m_a", uf, 1, 100, observed))
        rows.append(_mk("m_b", uf, 1, 120, observed))
        rows.append(_mk("m_c", uf, 1, 140, observed))
    return pd.DataFrame(rows)


def test_vincentization_is_monotone(preds):
    ens = vincentization(preds, ["m_a", "m_b", "m_c"])
    mono = enforce_monotonicity(ens)
    assert mono.attrs["frac_rows_needing_reordering"] == pytest.approx(0.0)


def test_vincentization_of_single_model_is_that_model(preds):
    ens = vincentization(preds, ["m_b"])
    only_b = preds[preds["model"] == "m_b"].set_index(["uf"])
    ens = ens.set_index("uf")
    for uf in ["SP", "RJ"]:
        for col in QUANTILE_COLUMNS:
            assert ens.loc[uf, col] == pytest.approx(only_b.loc[uf, col])


def test_vincentization_takes_per_quantile_median(preds):
    ens = vincentization(preds, ["m_a", "m_b", "m_c"]).set_index("uf")
    # median of {100,120,140}-centered rows is the m_b (120) row
    assert ens.loc["SP", "pred"] == pytest.approx(120)


def test_weighted_ensemble_equal_weights_is_mean(preds):
    w = {"m_a": 1.0, "m_b": 1.0, "m_c": 1.0}
    ens = weighted_ensemble(preds, w).set_index("uf")
    assert ens.loc["SP", "pred"] == pytest.approx(120)  # mean of 100,120,140


def test_weighted_ensemble_respects_weights(preds):
    w = {"m_a": 8.0, "m_b": 1.0, "m_c": 1.0}  # heavily weight m_a (100)
    ens = weighted_ensemble(preds, w).set_index("uf")
    assert ens.loc["SP", "pred"] < 110  # pulled toward 100


def test_inverse_wis_weights_favor_lower_wis():
    scored = pd.DataFrame([
        {"model": "good", "fold_id": 1, "wis": 1.0},
        {"model": "bad", "fold_id": 1, "wis": 4.0},
    ])
    w = inverse_wis_weights(scored, ["good", "bad"], tuning_fold=1)
    assert w["good"] > w["bad"]
    assert w["good"] + w["bad"] == pytest.approx(1.0)
    assert w["good"] == pytest.approx(0.8)  # (1/1)/((1/1)+(1/4)) = 0.8
