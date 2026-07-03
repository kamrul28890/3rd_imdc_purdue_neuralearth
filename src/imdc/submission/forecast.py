"""Generate FULL-SEASON submission forecasts (all 52 weeks EW41->EW40).

The backtest harness truncates each fold's target window to the observed data,
which is fine for scoring but wrong for a submission: the 2025-26 season (fold 4)
is ongoing, and the platform requires a complete EW41..EW40 forecast. This module
fits the submission model(s) on a fold's training data and predicts the full
season grid, so fold 4 is a genuine full-season forecast rather than a truncated one.
"""
import numpy as np
import pandas as pd

from imdc.config import MANDATORY_UFS, QUANTILE_COLUMNS
from imdc.evaluation.city import _city_cases
from imdc.evaluation.harness import build_state_training_frame
from imdc.evaluation.postprocess import enforce_monotonicity, to_submission_wide
from imdc.submission.build import season_date_range

FOLD_SEASON = {1: 2023, 2: 2024, 3: 2025, 4: 2026}


def _full_season_grid(fold, season_year: int, geographies: list) -> pd.DataFrame:
    dates = season_date_range(season_year)
    hz = ((pd.DatetimeIndex(dates) - fold.train_cutoff).days // 7).astype(int)
    frames = [pd.DataFrame({"uf": g, "date": dates, "horizon_weeks": hz}) for g in geographies]
    return pd.concat(frames, ignore_index=True)


def _vincentize_wide(member_wides: list) -> pd.DataFrame:
    stacked = pd.concat([w[["uf", "date"] + QUANTILE_COLUMNS] for w in member_wides], ignore_index=True)
    ens = stacked.groupby(["uf", "date"], as_index=False)[QUANTILE_COLUMNS].median()
    return enforce_monotonicity(ens)


def state_ensemble_forecast(fold, disease: str, member_factories: list, ufs=MANDATORY_UFS) -> pd.DataFrame:
    """Full-season per-quantile-median ensemble forecast for one fold, wide format."""
    season = FOLD_SEASON[fold.id]
    train = build_state_training_frame(fold, disease)
    grid = _full_season_grid(fold, season, ufs)
    wides = []
    for _, fac in member_factories:
        model = fac()
        model.fit(train, fold)
        wides.append(to_submission_wide(model.predict(grid)))
    return _vincentize_wide(wides)


def state_single_forecast(fold, disease: str, model_factory, ufs=MANDATORY_UFS) -> pd.DataFrame:
    """Full-season forecast from a single state model, wide format."""
    train = build_state_training_frame(fold, disease)
    grid = _full_season_grid(fold, FOLD_SEASON[fold.id], ufs)
    model = model_factory()
    model.fit(train, fold)
    wide = to_submission_wide(model.predict(grid))
    return enforce_monotonicity(wide)


def city_single_forecast(fold, disease: str, model_factory, geocodes: list) -> pd.DataFrame:
    """Full-season forecast for the target cities from a single model, wide format."""
    train = _city_cases(fold, disease, geocodes)
    grid = _full_season_grid(fold, FOLD_SEASON[fold.id], [str(g) for g in geocodes])
    model = model_factory()
    model.fit(train, fold)
    wide = to_submission_wide(model.predict(grid))
    return enforce_monotonicity(wide)
