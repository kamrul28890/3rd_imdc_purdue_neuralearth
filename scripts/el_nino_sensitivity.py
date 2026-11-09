"""How much of our 2026-27 forecast is El Nino? A counterfactual-input sensitivity analysis.

The IMDC 2026 results webinar (Nov 2026, "Super El Nino" slide) singled out PURDUE as the model
with "strong amplification" under El Nino: 2-3x the official ensemble in PR, SP, MG, the
Centro-Oeste and BA. This script quantifies how much of that comes from the climate-forecast
inputs, by re-predicting the exact deployed 2026-27 forecast (scripts/generate_forecast_2026_27.py:
same members, same training data through EW25 2026, same conformal factors) with those inputs
replaced at prediction time only - the fitted models are untouched:

  deployed            as submitted
  enso_neutral        ENSO index (all lags) set to 0 (neutral); IOD/PDO/ECMWF as observed
  ocean_neutral       ENSO, IOD and PDO (all lags) set to 0
  no_climate_forecast ocean indices neutral AND the ECMWF seasonal forecast removed (NaN,
                      as the models see it beyond the forecast's 6-month range)

Only the gradient-boosted members (XGBoost for dengue, LightGBM for chikungunya) consume these
inputs; the climatological and GRU dengue members do not, so they are fitted once and reused.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/el_nino_sensitivity.py
Writes results/metrics/el_nino_sensitivity_state.csv and el_nino_sensitivity_national.csv.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd

import imdc.models.ml_boosted as ml_boosted
import imdc.models.xgb_quantile as xgb_quantile
from imdc.config import INTERVAL_LEVELS, MANDATORY_UFS, METRICS_DIR, UF_TO_ADM1
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.evaluation.harness import build_state_training_frame
from imdc.evaluation.postprocess import apply_conformal_widen, clip_nonnegative, enforce_monotonicity, \
    to_submission_wide
from imdc.features.panel import ECMWF_COLS, _OCEAN_LAGS, build_prediction_features
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.ml_boosted import LGBMQuantileModel
from imdc.models.xgb_quantile import XGBQuantileModel
from imdc.submission.forecast import FOLD_SEASON, _full_season_grid, _vincentize_wide

import generate_forecast_2026_27 as deployed  # same folder; reuse its fold + conformal factors

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "data" / "external" / "platform_predictions"
OFFICIAL_ENSEMBLE_ID = 93
REFERENCE_ID = 85

SCENARIOS = {
    "deployed": dict(neutral=(), drop_ecmwf=False),
    "enso_neutral": dict(neutral=("enso",), drop_ecmwf=False),
    "ocean_neutral": dict(neutral=("enso", "iod", "pdo"), drop_ecmwf=False),
    "no_climate_forecast": dict(neutral=("enso", "iod", "pdo"), drop_ecmwf=True),
}
_active = {"scenario": "deployed"}


def _scenario_features(*args, **kwargs):
    feats, cols = build_prediction_features(*args, **kwargs)
    spec = SCENARIOS[_active["scenario"]]
    feats = feats.copy()
    for idx in spec["neutral"]:
        for lag in _OCEAN_LAGS:
            feats[f"{idx}_lag{lag}"] = 0.0
    if spec["drop_ecmwf"]:
        feats[ECMWF_COLS] = np.nan
    return feats, cols


# Both boosted models call the name imported into their own module; patch it there.
xgb_quantile.build_prediction_features = _scenario_features
ml_boosted.build_prediction_features = _scenario_features


def _fit(fold, disease, factory):
    model = factory()
    model.fit(build_state_training_frame(fold, disease), fold)
    return model


def _predict_all_scenarios(model, grid) -> dict:
    out = {}
    for name in SCENARIOS:
        _active["scenario"] = name
        out[name] = to_submission_wide(model.predict(grid))
    _active["scenario"] = "deployed"
    return out


def _summarise(wide: pd.DataFrame, disease: str, scenario: str, source: str) -> pd.DataFrame:
    g = wide.groupby("uf")
    s = pd.DataFrame({"total_median": g["pred"].sum(), "peak_median": g["pred"].max(),
                      "total_upper_90": g["upper_90"].sum()}).reset_index()
    return s.assign(disease=disease, scenario=scenario, source=source)


def _platform_season(model_id: int, disease_code: str) -> pd.DataFrame:
    df = pd.read_csv(PLATFORM / f"model_{model_id}.csv.gz", parse_dates=["date"])
    df = df[(df["disease"] == disease_code) & (df["date"] >= "2026-09-01")]
    latest = df.groupby("adm_1")["prediction_id"].transform("max")
    df = df[df["prediction_id"] == latest]
    df["uf"] = df["adm_1"].map({v: k for k, v in UF_TO_ADM1.items()})
    return df


def main():
    rows = []
    factors = deployed._load_conformal_factors()

    # ---- dengue: climatological + XGBoost + GRU, Vincentized, conformal-widened (as deployed)
    d5 = deployed._real_fold("dengue")
    grid = _full_season_grid(d5, FOLD_SEASON[5], MANDATORY_UFS)
    print("dengue: fitting climatological, GRU (once) and XGBoost...", flush=True)
    clim = to_submission_wide(_fit(d5, "dengue", ClimatologicalQuantileModel).predict(grid))
    gru = to_submission_wide(_fit(d5, "dengue", lambda: DLSequenceModel(
        disease="dengue", n_ensemble=5, epochs=30)).predict(grid))
    xgb_by = _predict_all_scenarios(_fit(d5, "dengue", lambda: XGBQuantileModel(disease="dengue")), grid)
    rows += [_summarise(clim, "dengue", "any", "member:climatological"),
             _summarise(gru, "dengue", "any", "member:gru")]
    for name, xw in xgb_by.items():
        ens = clip_nonnegative(apply_conformal_widen(_vincentize_wide([clim, xw, gru]), factors,
                                                     INTERVAL_LEVELS))
        rows += [_summarise(xw, "dengue", name, "member:xgboost"),
                 _summarise(ens, "dengue", name, "purdue")]
        if name == "deployed":
            sub = pd.concat([pd.read_csv(ROOT / "submissions" / "validation" / "dengue" / "season_2027"
                                         / f"{uf}.csv").assign(uf=uf) for uf in MANDATORY_UFS])
            print(f"  reproduction check: submitted national median total {sub['pred'].sum():,.0f} vs "
                  f"regenerated {ens['pred'].sum():,.0f}", flush=True)

    # ---- chikungunya: LightGBM alone (as deployed)
    c5 = deployed._real_fold("chikungunya")
    cgrid = _full_season_grid(c5, FOLD_SEASON[5], MANDATORY_UFS)
    print("chikungunya: fitting LightGBM...", flush=True)
    lgbm_by = _predict_all_scenarios(_fit(c5, "chikungunya",
                                          lambda: LGBMQuantileModel(disease="chikungunya")), cgrid)
    for name, w in lgbm_by.items():
        rows.append(_summarise(enforce_monotonicity(w), "chikungunya", name, "purdue"))

    # ---- comparators: the official IMDC ensemble and the reference model, same season
    for mid, label in [(OFFICIAL_ENSEMBLE_ID, "official_ensemble"), (REFERENCE_ID, "reference_model")]:
        for code, disease in [("A90", "dengue"), ("A92.0", "chikungunya")]:
            p = _platform_season(mid, code)
            if len(p):
                rows.append(_summarise(p, disease, "n/a", label))

    state = pd.concat(rows, ignore_index=True)
    state.to_csv(METRICS_DIR / "el_nino_sensitivity_state.csv", index=False)
    national = (state.groupby(["disease", "source", "scenario"])[["total_median", "total_upper_90"]]
                .sum().reset_index())
    national.to_csv(METRICS_DIR / "el_nino_sensitivity_national.csv", index=False)
    pd.set_option("display.width", 200)
    print(national.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    main()
