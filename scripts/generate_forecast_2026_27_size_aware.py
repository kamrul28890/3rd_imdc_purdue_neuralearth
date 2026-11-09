"""Freeze a size-aware-recalibrated version of the 2026-27 dengue forecast, for prospective scoring.

The forecast actually submitted to the challenge (scripts/generate_forecast_2026_27.py, deadline
2026-09-10) carries the global conformal recalibration. The size-aware (burden-scaled)
recalibration (imdc.evaluation.postprocess.apply_size_aware_scale; paper Sec. "official") was
devised in November 2026, after the organizers' validation results, so its backtest numbers are
not a clean out-of-sample test of the idea. The 2026-27 season is: this script regenerates the
exact deployed ensemble (same members, same training data through EW25 2026, same fitting code;
the national median total is checked against the submitted files), applies the size-aware scale
with the parameters frozen by run_ensemble.py (results/metrics/size_aware_params.csv: a, b and the
fold-1 burden index), and writes the result next to, never over, the submission. It was
committed on 2026-11-09, four weeks into the target window but using no data after EW25 2026 and
before any 2026-27 observation was examined; that is what makes the comparison prospective. (A
strict variant scores only the target weeks after the commit date.)

Both versions share the same median; they differ only in their intervals. Chikungunya needs no
counterpart: its fold-1 rule selected the identity scale (a=1, b=0), so the size-aware forecast is
the submitted LightGBM forecast.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_forecast_2026_27_size_aware.py
Writes submissions/prospective/size_aware_2026_27/dengue/<UF>.csv and a provenance README.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
from pathlib import Path

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.evaluation.postprocess import apply_size_aware_scale, clip_nonnegative
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.xgb_quantile import XGBQuantileModel
from imdc.submission.build import build_submission_frame
from imdc.submission.forecast import state_ensemble_forecast
from imdc.submission.validate import validate_submission

import generate_forecast_2026_27 as deployed  # same folder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submissions" / "prospective" / "size_aware_2026_27" / "dengue"
SUBMITTED = ROOT / "submissions" / "validation" / "dengue" / f"season_{deployed.SEASON}"


def main():
    params = pd.read_csv(METRICS_DIR / "size_aware_params.csv")
    a, b = float(params["a"].iloc[0]), float(params["b"].iloc[0])
    z = params.set_index("uf")["burden_z"]
    print(f"frozen size-aware parameters: a={a:.2f} b={b:.2f} (fold {params['calib_fold'].iloc[0]}, "
          f"burden index through {params['burden_cutoff'].iloc[0]})")

    d5 = deployed._real_fold("dengue")
    members = [
        ("climatological", ClimatologicalQuantileModel),
        ("xgboost", lambda: XGBQuantileModel(disease="dengue")),
        ("gru", lambda: DLSequenceModel(disease="dengue", n_ensemble=5, epochs=30)),
    ]
    vincent = state_ensemble_forecast(d5, "dengue", members, ufs=MANDATORY_UFS)

    submitted = pd.concat([pd.read_csv(SUBMITTED / f"{uf}.csv").assign(uf=uf) for uf in MANDATORY_UFS])
    print(f"reproduction check: submitted national median total {submitted['pred'].sum():,.0f} vs "
          f"regenerated {vincent['pred'].sum():,.0f}")

    sized = clip_nonnegative(apply_size_aware_scale(vincent, z, a, b))
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for uf in MANDATORY_UFS:
        frame = build_submission_frame(sized, uf, deployed.SEASON)
        validate_submission(frame, deployed.SEASON, name=f"size_aware/{uf}")
        path = OUT / f"{uf}.csv"
        frame.to_csv(path, index=False)
        digest.update(path.read_bytes())

    nat = sized.groupby("date")[["pred", "lower_90", "upper_90"]].sum()
    sub_nat = submitted.groupby("date")[["lower_90", "upper_90"]].sum()
    (OUT.parent / "README.md").write_text(
        "# Size-aware 2026-27 dengue forecast (prospective comparison)\n\n"
        "Not a challenge submission. The submitted forecast is in "
        f"`submissions/validation/dengue/season_{deployed.SEASON}/` (global conformal recalibration). "
        "This is the same ensemble (identical medians) with the size-aware recalibration instead, "
        f"frozen with a = {a:.2f}, b = {b:.2f} (tuned on the 2022-23 season only) and generated "
        "using no data after the forecast origin. It was committed on 2026-11-09, four weeks into the "
        "target window, before any 2026-27 observation was examined. Scoring both once the season "
        "resolves is the out-of-sample test of the size-aware rule (strictly: on the target weeks "
        "after the commit date).\n\n"
        f"- generator: `scripts/generate_forecast_2026_27_size_aware.py`\n"
        f"- training data through: {d5.train_cutoff.date()}\n"
        f"- target window: {d5.target_start.date()} to {d5.target_end.date()}\n"
        f"- national median total: {sized['pred'].sum():,.0f} (submitted: {submitted['pred'].sum():,.0f})\n"
        f"- national sum of 90% bounds, season total: {nat['lower_90'].sum():,.0f} to "
        f"{nat['upper_90'].sum():,.0f} (submitted: {sub_nat['lower_90'].sum():,.0f} to "
        f"{sub_nat['upper_90'].sum():,.0f})\n"
        f"- SHA-256 of the 26 state files, concatenated in MANDATORY_UFS order: `{digest.hexdigest()}`\n",
        encoding="utf-8")
    print(f"wrote {len(MANDATORY_UFS)} files to {OUT}")


if __name__ == "__main__":
    main()
