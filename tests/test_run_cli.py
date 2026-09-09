"""Tests for the unified `python -m imdc.run` CLI (IMPROVEMENTS.md Sec 3.3).

Only exercises the CLI plumbing (registry resolution, argument parsing, end-to-end run
on a fast model over a tiny state subset) - not every model's forecast quality, which is
each model's own test file's job.
"""
import pandas as pd
import pytest

from imdc import run as run_cli
from imdc.evaluation.baselines import NaiveModel


def test_registry_covers_expected_models():
    expected = {"naive", "seasonal_naive", "climatological", "lgbm", "xgb", "gru", "gru_wis",
                "mechanistic", "mechanistic_nsub", "hierarchical", "prophet", "prophet_v2",
                "sarimax", "sarimax_v2", "surge_template", "climate_surge"}
    assert expected <= set(run_cli.MODEL_REGISTRY)


def test_registry_factories_construct_without_fitting():
    """Every factory must at least construct (bad import path, wrong kwarg, etc. fails here)
    without needing to actually fit - fitting each of the 16 is what the smoke tests below
    (and each model's own test file) are for, kept separate so this stays fast.
    """
    for name, factory in run_cli.MODEL_REGISTRY.items():
        model = factory("dengue")
        assert hasattr(model, "fit") and hasattr(model, "predict"), name


def test_cli_runs_a_fast_model_end_to_end(tmp_path, capsys):
    out = tmp_path / "naive_scored.csv"
    run_cli.main(["--model", "naive", "--disease", "dengue", "--ufs", "SP", "RJ", "AC",
                  "--out", str(out)])
    scored = pd.read_csv(out)
    assert set(scored["uf"].unique()) == {"SP", "RJ", "AC"}
    assert (scored["wis"] >= 0).all()
    printed = capsys.readouterr().out
    assert "Leaderboard" in printed


def test_cli_list_prints_registry(capsys):
    run_cli.main(["--list"])
    printed = capsys.readouterr().out.split()
    assert "naive" in printed and "xgb" in printed


def test_cli_requires_model_unless_listing():
    with pytest.raises(SystemExit):
        run_cli.main([])
