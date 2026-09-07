"""Generate the real forecast-phase submission: the 2026-27 season (EW41 2026 -> EW40 2027).

Unlike fold 4 (scripts/generate_fold4.py), this is not a backtest fold - there is
no held-out observed outcome yet. The training cutoff is the latest data actually
available (EW25 2026, after scripts/refresh_data.py), and the target window is a
genuine future 52-week horizon. It reuses the exact same models/harness as every
backtest fold via a synthetic Fold(id=5, ...) registered as FOLD_SEASON[5] = 2027
in imdc/submission/forecast.py, so no model code needed to change.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_forecast_2026_27.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from epiweeks import Week

from imdc.config import CHIKUNGUNYA_TARGET_CITIES, DENGUE_TARGET_CITIES, INTERVAL_LEVELS, MANDATORY_UFS, \
    METRICS_DIR, SUBMISSIONS_DIR
from imdc.data.folds import Fold, get_forecast_origin
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.evaluation.postprocess import apply_conformal_widen, clip_nonnegative
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.ml_boosted import LGBMQuantileModel
from imdc.submission.build import build_submission_frame
from imdc.submission.forecast import city_single_forecast, state_ensemble_forecast, state_single_forecast
from imdc.submission.validate import SubmissionError, validate_submission

SEASON = 2027


def _load_conformal_factors() -> dict:
    """Fold-1-tuned multiplicative widening factors (see docs/PLAN.md Phase 6 / run_ensemble.py).

    This is the "ensemble + conformal recalibration" model the README/paper document as the
    chosen dengue-state model (WIS 1216 vs 1281 for plain Vincentization) - applying it here
    (not just at backtest-scoring time) is what actually makes the real forecast match that
    documented methodology.
    """
    s = pd.read_csv(METRICS_DIR / "conformal_factors.csv", index_col="interval_level")["factor"]
    return {int(k): float(v) for k, v in s.items()}


def _real_fold(disease: str) -> Fold:
    cutoff = get_forecast_origin(disease)
    return Fold(
        id=5,
        train_cutoff=cutoff,
        target_start=pd.Timestamp(Week(SEASON - 1, 41).startdate()),
        target_end=pd.Timestamp(Week(SEASON, 40).startdate()),
    )


def _write(wide, subdir, geographies):
    d = SUBMISSIONS_DIR / "validation" / subdir / f"season_{SEASON}"
    d.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for g in geographies:
        frame = build_submission_frame(wide, str(g), SEASON)
        try:
            validate_submission(frame, SEASON, name=f"{subdir}/{SEASON}/{g}")
            frame.to_csv(d / f"{g}.csv", index=False); ok += 1
        except SubmissionError as e:
            fail += 1; print("  FAIL:", e)
    print(f"{subdir}: {ok} valid, {fail} failed")


def main():
    d5 = _real_fold("dengue")
    c5 = _real_fold("chikungunya")
    print(f"forecast origin (train_cutoff): dengue={d5.train_cutoff.date()} "
          f"chikungunya={c5.train_cutoff.date()}")
    print(f"target window: {d5.target_start.date()} -> {d5.target_end.date()}")

    print("\ndengue state (ensemble: climatological+lgbm+gru) — fitting on data through EW25 2026...")
    members = [
        ("climatological", ClimatologicalQuantileModel),
        ("lgbm", lambda: LGBMQuantileModel(disease="dengue")),
        ("gru", lambda: DLSequenceModel(disease="dengue", n_ensemble=5, epochs=30)),
    ]
    ensemble_wide = state_ensemble_forecast(d5, "dengue", members, ufs=MANDATORY_UFS)
    factors = _load_conformal_factors()
    print(f"  applying conformal recalibration factors: {factors}")
    ensemble_wide = clip_nonnegative(apply_conformal_widen(ensemble_wide, factors, INTERVAL_LEVELS))
    _write(ensemble_wide, "dengue", MANDATORY_UFS)

    print("\nchikungunya state (LightGBM)...")
    _write(state_single_forecast(c5, "chikungunya", lambda: LGBMQuantileModel(disease="chikungunya"),
                                 ufs=MANDATORY_UFS), "chikungunya", MANDATORY_UFS)

    print("\ndengue cities (climatological)...")
    _write(city_single_forecast(d5, "dengue", ClimatologicalQuantileModel, DENGUE_TARGET_CITIES),
           "dengue_cities", DENGUE_TARGET_CITIES)

    print("\nchikungunya cities (climatological)...")
    _write(city_single_forecast(c5, "chikungunya", ClimatologicalQuantileModel, CHIKUNGUNYA_TARGET_CITIES),
           "chikungunya_cities", CHIKUNGUNYA_TARGET_CITIES)


if __name__ == "__main__":
    main()
