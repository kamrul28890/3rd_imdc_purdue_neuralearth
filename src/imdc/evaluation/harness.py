"""Backtesting harness: run models over official folds, score, and summarize.

The harness - not individual models - is responsible for leakage-safe cutoff
filtering (imdc.data.folds) and for enforcing it (imdc.data.validate) before
any model ever sees a covariate frame. Every model family plugs in via the
shared fit/predict protocol documented in baselines.py.
"""
import numpy as np
import pandas as pd

from imdc.config import MANDATORY_UFS, QUANTILE_LEVEL_TO_COLUMN
from imdc.data.aggregate import aggregate_cases_to_state
from imdc.data.folds import Fold, cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_gap_weeks_absent, assert_no_leakage
from imdc.evaluation.metrics import wis_decomposition, wis_from_intervals


def _target_date_grid(fold: Fold, ufs: list = MANDATORY_UFS) -> pd.DataFrame:
    """uf x date grid for a fold's target window, with horizon_weeks relative to train_cutoff."""
    dates = pd.date_range(fold.target_start, fold.target_end, freq="W-SUN")
    horizon_weeks = ((dates - fold.train_cutoff).days // 7).astype(int)
    grid = pd.DataFrame(
        {"date": np.tile(dates, len(ufs)), "horizon_weeks": np.tile(horizon_weeks, len(ufs))}
    )
    grid["uf"] = np.repeat(ufs, len(dates))
    return grid[["uf", "date", "horizon_weeks"]]


def build_state_training_frame(fold: Fold, disease: str = "dengue") -> pd.DataFrame:
    """Leakage-safe state-aggregated training series for one fold: uf, date, casos, date <= train_cutoff."""
    cases = load_cases(disease)
    filtered = cutoff_filter(cases, fold.train_cutoff)
    assert_no_leakage(filtered, fold.train_cutoff, name=f"fold{fold.id} {disease} training frame")
    assert_gap_weeks_absent(filtered, fold)
    return aggregate_cases_to_state(filtered)


def build_state_observed_frame(fold: Fold, disease: str = "dengue") -> pd.DataFrame:
    """State-aggregated observed series for a fold's full target window (for scoring only)."""
    cases = load_cases(disease)
    window = cases[(cases["date"] >= fold.target_start) & (cases["date"] <= fold.target_end)]
    return aggregate_cases_to_state(window)


def _run_backtest_generic(model_factory, folds: list, disease: str, build_train, build_grid,
                          extra_cols: dict = None) -> pd.DataFrame:
    """Shared fit/predict loop behind both `run_backtest` (state) and `city.run_city_backtest`
    (IMPROVEMENTS.md Sec 3.2 dedup - the two used to be near-identical copies).

    `build_train(fold, disease)` -> training frame; `build_grid(fold)` -> target grid (uf,
    date, horizon_weeks). `extra_cols`: {column_name: f(fold)} for anything beyond
    model/disease/fold_id every row should carry (state stamps `origin_date`; city doesn't).
    """
    extra_cols = extra_cols or {}
    all_predictions = []
    for fold in folds:
        train_df = build_train(fold, disease)
        model = model_factory()
        model.fit(train_df, fold)

        grid = build_grid(fold)
        preds = model.predict(grid)
        preds = preds.merge(grid, on=["uf", "date"], how="left")
        preds["model"] = getattr(model, "name", model.__class__.__name__)
        preds["disease"] = disease
        preds["fold_id"] = fold.id
        for col, fn in extra_cols.items():
            preds[col] = fn(fold)
        all_predictions.append(preds)

    return pd.concat(all_predictions, ignore_index=True)


def run_backtest(
    model_factory, folds: list, disease: str = "dengue", ufs: list = MANDATORY_UFS
) -> pd.DataFrame:
    """Fit+predict model_factory() across every fold; returns the canonical long predictions df.

    model_factory: a zero-arg callable returning a fresh, unfit model instance
    (fresh per fold, since fit() is stateful).
    """
    return _run_backtest_generic(
        model_factory, folds, disease,
        build_train=build_state_training_frame,
        build_grid=lambda fold: _target_date_grid(fold, ufs),
        extra_cols={"origin_date": lambda fold: fold.train_cutoff},
    )


def _score_generic(predictions_long: pd.DataFrame, folds: list, index_cols: list, build_observed) -> pd.DataFrame:
    """Shared pivot + join-observed + WIS/coverage scoring behind both `score_backtest` (state)
    and `city.score_city_backtest` (IMPROVEMENTS.md Sec 3.2 dedup).

    `build_observed(fold)` -> DataFrame with columns [uf, date, observed_value] for that fold's
    target window - the only piece that's actually geography/disease-specific; `fold_id` is
    added here, not by the caller.
    """
    fold_by_id = {f.id: f for f in folds}
    wide = predictions_long.pivot_table(index=index_cols, columns="quantile_level", values="predicted_value")
    wide = wide.rename(columns=QUANTILE_LEVEL_TO_COLUMN).reset_index()

    observed_frames = []
    for fold_id in wide["fold_id"].unique():
        obs = build_observed(fold_by_id[fold_id]).copy()
        obs["fold_id"] = fold_id
        observed_frames.append(obs)
    observed = pd.concat(observed_frames, ignore_index=True)

    scored = wide.merge(observed, on=["fold_id", "uf", "date"], how="inner")
    y = scored["observed_value"].to_numpy()

    scored["wis"] = wis_from_intervals(scored, y)
    decomp = wis_decomposition(scored, y)
    scored = pd.concat([scored, decomp], axis=1)
    scored["ae_median"] = np.abs(y - scored["pred"].to_numpy())

    for level in [50, 80, 90, 95]:
        lower = scored[f"lower_{level}"].to_numpy()
        upper = scored[f"upper_{level}"].to_numpy()
        scored[f"coverage_{level}"] = (y >= lower) & (y <= upper)

    return scored


def score_backtest(predictions_long: pd.DataFrame, disease: str = "dengue", folds: list = None) -> pd.DataFrame:
    """Pivot predictions to wide, join observed values, compute WIS/coverage/decomposition per unit."""
    if folds is None:
        raise ValueError("score_backtest requires the same `folds` list used in run_backtest")

    index_cols = ["model", "disease", "fold_id", "uf", "date", "horizon_weeks", "origin_date"]

    def build_observed(fold):
        return build_state_observed_frame(fold, disease).rename(columns={"casos": "observed_value"})

    return _score_generic(predictions_long, folds, index_cols, build_observed)


# Folds whose target season is not fully resolved in the data (prospective/partial).
# Fold 4 (2025-26) only has data through ~2026-03; averaging it into a headline number
# mixes a partial season with complete ones. Headline/paper aggregations should pass
# `exclude_folds=PROSPECTIVE_FOLDS`; per-fold tables keep it to show fold 4 separately.
PROSPECTIVE_FOLDS = (4,)


def summarize(scored: pd.DataFrame, by: list, exclude_folds: tuple = ()) -> pd.DataFrame:
    """Aggregate scored predictions (mean WIS/coverage/etc.) by arbitrary grouping columns.

    `exclude_folds` drops those fold_ids before aggregating (e.g. the prospective fold 4
    for a headline number). Default keeps every fold, so existing callers are unchanged.
    """
    if exclude_folds:
        scored = scored[~scored["fold_id"].isin(exclude_folds)]
    metric_cols = [
        "wis", "dispersion", "overprediction", "underprediction", "ae_median",
        "coverage_50", "coverage_80", "coverage_90", "coverage_95",
    ]
    return scored.groupby(by)[metric_cols].mean().reset_index()


def normalized_wis(scored: pd.DataFrame, by: list, exclude_folds: tuple = ()) -> pd.DataFrame:
    """The organizers' headline score: normalized WIS = sum(WIS) / sum(observed cases), by group.

    Unlike `summarize` (a plain mean of per-unit WIS, dominated by high-burden states and the
    atypical 2024 fold), this ratio-of-sums matches the official 2nd/3rd-IMDC ensemble methodology
    and is scale-free across regions. Lower is better. `exclude_folds` drops folds first (e.g. the
    2024 outlier fold 2) so a normal-season number can be reported alongside the all-fold one.
    """
    if exclude_folds:
        scored = scored[~scored["fold_id"].isin(exclude_folds)]
    g = scored.groupby(by)
    out = (g["wis"].sum() / g["observed_value"].sum()).reset_index(name="normalized_wis")
    return out.sort_values("normalized_wis").reset_index(drop=True)
