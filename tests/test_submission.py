"""Tests for submission building + validation against the platform's exact rules."""
import numpy as np
import pandas as pd
import pytest

from imdc.config import QUANTILE_COLUMNS
from imdc.submission.build import build_submission_frame, season_date_range
from imdc.submission.forecast import _vincentize_wide
from imdc.submission.validate import SubmissionError, is_valid, validate_submission


def test_vincentize_wide_takes_per_quantile_median_of_member_forecasts():
    """`_vincentize_wide` (used by the real forecast-phase submission, no observed_value/
    fold_id available) must agree with `models.ensemble.vincentization`'s median logic -
    they share `_median_ensemble` (IMPROVEMENTS.md Sec 3.2 dedup) and should never diverge.
    """
    dates = season_date_range(2027)[:2]
    members = []
    for base in (100, 120, 140):
        w = pd.DataFrame({"uf": ["SP"] * 2, "date": dates})
        for col, off in zip(QUANTILE_COLUMNS, [-9, -6, -4, -2, 0, 2, 4, 6, 9]):
            w[col] = base + off
        members.append(w)
    ens = _vincentize_wide(members)
    assert ens.set_index("uf")["pred"].iloc[0] == pytest.approx(120)


def test_season_date_range_matches_platform_epiweeks():
    # season 2024 => EW41 2023 .. EW40 2024 (verified against epiweeks.Week / mosqlient)
    rng = season_date_range(2024)
    assert rng[0] == pd.Timestamp("2023-10-08")
    assert rng[-1] == pd.Timestamp("2024-09-29")
    assert len(rng) == 52
    assert (rng.weekday == 6).all()  # all Sundays


def _valid_frame(season_year=2024):
    dates = season_date_range(season_year)
    n = len(dates)
    base = np.linspace(100, 200, n)
    frame = pd.DataFrame({"date": dates})
    offs = {"lower_95": -40, "lower_90": -30, "lower_80": -20, "lower_50": -10,
            "pred": 0, "upper_50": 10, "upper_80": 20, "upper_90": 30, "upper_95": 40}
    for col, o in offs.items():
        frame[col] = base + o
    return frame


def test_valid_frame_passes():
    validate_submission(_valid_frame(), 2024)  # should not raise
    assert is_valid(_valid_frame(), 2024)


def test_missing_column_fails():
    f = _valid_frame().drop(columns=["upper_95"])
    with pytest.raises(SubmissionError, match="missing columns"):
        validate_submission(f, 2024)


def test_date_gap_fails():
    f = _valid_frame().drop(index=5).reset_index(drop=True)
    with pytest.raises(SubmissionError, match="dates do not match"):
        validate_submission(f, 2024)


def test_negative_value_fails():
    f = _valid_frame()
    f.loc[0, "lower_95"] = -1.0
    with pytest.raises(SubmissionError, match="negative"):
        validate_submission(f, 2024)


def test_nesting_violation_fails():
    f = _valid_frame()
    f.loc[3, "upper_50"] = f.loc[3, "lower_50"] - 5  # cross the median interval
    with pytest.raises(SubmissionError, match="nested-bound"):
        validate_submission(f, 2024)


def test_build_frame_reindexes_to_full_season():
    dates = season_date_range(2024)
    partial = pd.DataFrame({"uf": "SP", "date": dates[:10]})
    for col in QUANTILE_COLUMNS:
        partial[col] = 100.0
    frame = build_submission_frame(partial, "SP", 2024)
    assert len(frame) == 52  # reindexed to the full season
    assert frame[QUANTILE_COLUMNS].iloc[10:].isna().all().all()  # missing weeks surfaced as NaN
