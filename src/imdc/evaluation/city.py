"""City-level backtesting (optional challenge tracks).

The state harness aggregates municipalities to UF; the city tracks instead
forecast a fixed set of target municipalities directly. We relabel each city's
geocode into the `uf` column so the geography-agnostic models (the baselines and
climatological-quantile) run unchanged. Feature-heavy models (LGBM/GRU) key
their covariates on state geography and would need municipal features, so the
city tracks use the baseline/climatological models here.
"""
import pandas as pd

from imdc.data.folds import cutoff_filter
from imdc.data.loaders import load_cases
from imdc.data.validate import assert_no_leakage
from imdc.evaluation.harness import _run_backtest_generic, _score_generic


def _city_cases(fold, disease: str, geocodes: list) -> pd.DataFrame:
    """Leakage-safe city case series (geocode relabeled to `uf`), date <= train_cutoff."""
    cases = cutoff_filter(load_cases(disease), fold.train_cutoff)
    assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} city cases")
    sub = cases[cases["geocode"].isin(geocodes)][["geocode", "date", "casos"]].copy()
    sub = sub.rename(columns={"geocode": "uf"})
    sub["uf"] = sub["uf"].astype(str)
    return sub.sort_values(["uf", "date"]).reset_index(drop=True)


def _city_target_grid(fold, geocodes: list) -> pd.DataFrame:
    dates = pd.date_range(fold.target_start, fold.target_end, freq="W-SUN")
    hz = ((dates - fold.train_cutoff).days // 7).astype(int)
    frames = [pd.DataFrame({"uf": str(g), "date": dates, "horizon_weeks": hz}) for g in geocodes]
    return pd.concat(frames, ignore_index=True)


def run_city_backtest(model_factory, folds, disease: str, geocodes: list) -> pd.DataFrame:
    """Same fit/predict loop as `harness.run_backtest`, over city geocodes instead of states -
    see `_run_backtest_generic` (IMPROVEMENTS.md Sec 3.2 dedup) for the shared implementation.
    """
    return _run_backtest_generic(
        model_factory, folds, disease,
        build_train=lambda fold, disease: _city_cases(fold, disease, geocodes),
        build_grid=lambda fold: _city_target_grid(fold, geocodes),
    )


def score_city_backtest(preds_long: pd.DataFrame, folds, disease: str, geocodes: list) -> pd.DataFrame:
    """Same pivot/WIS/coverage scoring as `harness.score_backtest`, over city geocodes instead
    of states - see `_score_generic` (IMPROVEMENTS.md Sec 3.2 dedup) for the shared
    implementation. No `origin_date` column, unlike the state version.
    """
    index_cols = ["model", "disease", "fold_id", "uf", "date", "horizon_weeks"]
    cases = load_cases(disease)

    def build_observed(fold):
        w = cases[(cases["date"] >= fold.target_start) & (cases["date"] <= fold.target_end)
                  & (cases["geocode"].isin(geocodes))][["geocode", "date", "casos"]].copy()
        w = w.rename(columns={"geocode": "uf", "casos": "observed_value"})
        w["uf"] = w["uf"].astype(str)
        return w

    return _score_generic(preds_long, folds, index_cols, build_observed)
