"""Generate FULL-SEASON fold-4 (2025-26) submission files for every track.

Fold 4's season is ongoing, so the backtest truncates it; a submission needs all
52 weeks. Folds 1-3 are already complete (resolved seasons). This regenerates only
season_2026 for: dengue state (ensemble), chikungunya state (LightGBM), and both
city tracks (climatological).

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_fold4.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imdc.config import CHIKUNGUNYA_TARGET_CITIES, DENGUE_TARGET_CITIES, MANDATORY_UFS, SUBMISSIONS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.ml_boosted import LGBMQuantileModel
from imdc.submission.build import build_submission_frame
from imdc.submission.forecast import city_single_forecast, state_ensemble_forecast, state_single_forecast
from imdc.submission.validate import SubmissionError, validate_submission

SEASON = 2026


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
    d4 = get_folds("dengue")[3]
    c4 = get_folds("chikungunya")[3]

    print("dengue state (ensemble: climatological+lgbm+gru) — refitting on fold 4...")
    members = [
        ("climatological", ClimatologicalQuantileModel),
        ("lgbm", lambda: LGBMQuantileModel(disease="dengue")),
        ("gru", lambda: DLSequenceModel(disease="dengue", n_ensemble=5, epochs=30)),
    ]
    _write(state_ensemble_forecast(d4, "dengue", members, ufs=MANDATORY_UFS), "dengue", MANDATORY_UFS)

    print("chikungunya state (LightGBM)...")
    _write(state_single_forecast(c4, "chikungunya", lambda: LGBMQuantileModel(disease="chikungunya"),
                                 ufs=MANDATORY_UFS), "chikungunya", MANDATORY_UFS)

    print("dengue cities (climatological)...")
    _write(city_single_forecast(d4, "dengue", ClimatologicalQuantileModel, DENGUE_TARGET_CITIES),
           "dengue_cities", DENGUE_TARGET_CITIES)

    print("chikungunya cities (climatological)...")
    _write(city_single_forecast(c4, "chikungunya", ClimatologicalQuantileModel, CHIKUNGUNYA_TARGET_CITIES),
           "chikungunya_cities", CHIKUNGUNYA_TARGET_CITIES)


if __name__ == "__main__":
    main()
