"""Regression tests for the Phase-3 panel's two leakage disciplines.

The confirmed 15-week gap between train_cutoff and target_start is a
withheld-*evaluation* gap (simulating reporting lag), not a missing-*observation*
gap: the underlying weekly case series is continuous up to the cutoff. Lag
features must therefore be built from that continuous cutoff-filtered history,
never from the gapped target window - the second, distinct leakage trap flagged
in the plan (Sec 3.2).
"""
import numpy as np
import pandas as pd
import pytest

from imdc.data.folds import get_folds
from imdc.features.panel import build_panel

SMALL_UFS = ["SP", "RJ"]


@pytest.fixture(scope="module")
def panel_fold1():
    fold = get_folds("dengue")[0]
    panel, feature_cols = build_panel(fold, ufs=SMALL_UFS)
    return fold, panel, feature_cols


def test_no_label_or_origin_beyond_cutoff(panel_fold1):
    fold, panel, _ = panel_fold1
    assert panel["target_date"].max() <= fold.train_cutoff
    assert panel["origin_date"].max() <= fold.train_cutoff


def test_lags_come_from_continuous_history_not_gapped_window(panel_fold1):
    """Origins in the ~15 weeks just before the cutoff must still have fully
    populated lag_1..lag_4 - which is only possible if lags were built from the
    continuous history. If lags had been built from target_window() output
    (which starts at target_start, 15 weeks after the cutoff), these recent
    origins would have all-NaN lags."""
    fold, panel, _ = panel_fold1
    recent = panel[panel["origin_date"] >= fold.train_cutoff - pd.Timedelta(weeks=10)]
    assert len(recent) > 0
    for lag_col in ["lag_1", "lag_2", "lag_3", "lag_4"]:
        assert recent[lag_col].notna().all(), f"{lag_col} has NaNs for recent origins -> gapped history"


def test_lag_1_equals_origin_observed_value(panel_fold1):
    """lag_1 must equal the actual observed log-incidence at the origin date -
    a direct check that the origin-anchored series is the real continuous series."""
    from imdc.data.aggregate import aggregate_cases_to_state
    from imdc.data.folds import cutoff_filter
    from imdc.data.loaders import load_cases
    from imdc.features.panel import _add_incidence, state_population

    fold, panel, _ = panel_fold1
    cases = cutoff_filter(load_cases("dengue"), fold.train_cutoff)
    state = aggregate_cases_to_state(cases)
    truth = _add_incidence(state, state_population())
    truth = truth[truth["uf"].isin(SMALL_UFS)].set_index(["uf", "date"])["log_inc"]

    sample = panel.drop_duplicates(["uf", "origin_date"]).sample(min(200, len(panel)), random_state=0)
    for _, row in sample.iterrows():
        expected = truth.get((row["uf"], row["origin_date"]), np.nan)
        if not np.isnan(expected):
            assert row["lag_1"] == pytest.approx(expected, abs=1e-9)


def test_no_gap_week_rows_as_origin(panel_fold1):
    """No origin may fall in the 15-week gap between this fold's cutoff and target_start
    (the gap weeks are strictly after the cutoff, so cutoff-filtering already excludes them)."""
    fold, panel, _ = panel_fold1
    gap_mask = (panel["origin_date"] > fold.train_cutoff) & (panel["origin_date"] < fold.target_start)
    assert gap_mask.sum() == 0
