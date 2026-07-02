import pandas as pd
import pytest

from imdc.config import N_FOLDS
from imdc.data.folds import cutoff_filter, cutoff_filter_forecasting_climate, get_folds
from imdc.data.loaders import load_cases, load_climate, load_forecasting_climate
from imdc.data.validate import (
    LeakageError,
    assert_all_sundays,
    assert_gap_weeks_absent,
    assert_no_duplicate_series,
    assert_no_leakage,
    assert_non_negative,
)


@pytest.fixture(scope="module")
def dengue_df():
    return load_cases("dengue")


@pytest.fixture(scope="module")
def folds():
    return get_folds("dengue")


def test_four_folds_derived(folds):
    assert len(folds) == N_FOLDS
    for f in folds:
        assert f.train_cutoff < f.target_start <= f.target_end


def test_fold4_matches_known_boundaries(folds):
    # Empirically confirmed boundaries at time of writing; a data refresh
    # extending fold 4 further should still satisfy the ordering above even
    # if these exact dates change.
    f4 = folds[3]
    assert f4.train_cutoff == pd.Timestamp("2025-06-15")
    assert f4.target_start == pd.Timestamp("2025-10-05")


def test_case_data_structural_sanity(dengue_df):
    assert_all_sundays(dengue_df, name="dengue")
    assert_no_duplicate_series(dengue_df, ["geocode", "date"], name="dengue")
    assert_non_negative(dengue_df, "casos", name="dengue")


def test_cutoff_filter_excludes_future_rows(dengue_df, folds):
    fold = folds[0]
    filtered = cutoff_filter(dengue_df, fold.train_cutoff)
    assert_no_leakage(filtered, fold.train_cutoff, name="fold1 dengue")
    assert filtered["date"].max() <= fold.train_cutoff


def test_naive_not_target_filter_leaks_gap_weeks(dengue_df, folds):
    """Regression test for the confirmed leakage trap: filtering by
    `~target_1` instead of an explicit cutoff silently includes the 15
    gap weeks between train_1's cutoff and target_1's start."""
    fold = folds[0]
    naively_filtered = dengue_df[~dengue_df["target_1"]]
    with pytest.raises(LeakageError):
        assert_gap_weeks_absent(naively_filtered, fold)


def test_explicit_cutoff_has_no_gap_week_leakage(dengue_df, folds):
    fold = folds[0]
    correctly_filtered = cutoff_filter(dengue_df, fold.train_cutoff)
    assert_gap_weeks_absent(correctly_filtered, fold)


def test_climate_table_has_no_fold_flags_and_needs_manual_cutoff(folds):
    climate = load_climate()
    assert "train_1" not in climate.columns
    fold = folds[0]
    filtered = cutoff_filter(climate, fold.train_cutoff)
    assert_no_leakage(filtered, fold.train_cutoff, name="climate fold1")


def test_forecasting_climate_cutoff_uses_reference_month(folds):
    fc = load_forecasting_climate()
    fold = folds[0]
    filtered = cutoff_filter_forecasting_climate(fc, fold.train_cutoff)
    origin_month = pd.Timestamp(fold.train_cutoff.year, fold.train_cutoff.month, 1)
    assert filtered["reference_month"].max() <= origin_month
