"""Freeze the SARIMAX-augmented, size-aware 2026-27 dengue forecast, for prospective scoring.

The SARIMAX-augmented ensemble (run_ensemble.py: the three deployed members plus the per-fold-order
SARIMAX v2, combined by per-quantile median, then size-aware recalibrated; paper Sec. "sizeaware")
was tried in November 2026, after the organizers' validation results, so its backtest gain is not a
clean out-of-sample estimate. The 2026-27 season is. This script:

1. refits the three deployed members (climatological, XGBoost, GRU) on the real forecast fold
   exactly as scripts/generate_forecast_2026_27.py does, and checks that their median reproduces
   the submitted forecast (national median total);
2. fits SARIMAX v2 on the same fold, with ARIMA orders selected on that fold's own training data
   (select_orders=True, as in the backtest);
3. takes the per-quantile median of the four members and applies the size-aware scale with the
   parameters frozen by run_ensemble.py (results/metrics/size_aware_sarimax_params.csv);
4. writes the result next to, never over, the submission and the size-aware forecast, with the
   SARIMAX member's own forecast and a provenance README carrying a SHA-256 of the state files.

It uses no data after the forecast origin (EW25 2026) and was generated before any 2026-27
observation was examined. It is committed some weeks into the target window (which starts EW41,
2026-10-11), so the strict prospective comparison scores only the target weeks after the commit.

Run as: KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_forecast_2026_27_sarimax_size_aware.py
(from the scripts folder or with it on PYTHONPATH; about 45 minutes, mostly the SARIMAX fit).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import datetime
import hashlib
from pathlib import Path

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR, QUANTILE_COLUMNS
from imdc.evaluation.baselines import ClimatologicalQuantileModel
from imdc.evaluation.harness import build_state_training_frame
from imdc.evaluation.postprocess import apply_size_aware_scale, clip_nonnegative, to_submission_wide
from imdc.models.dl_sequence import DLSequenceModel
from imdc.models.sarimax_v2 import SarimaxV2Model
from imdc.models.xgb_quantile import XGBQuantileModel
from imdc.submission.build import build_submission_frame
from imdc.submission.forecast import FOLD_SEASON, _full_season_grid, _vincentize_wide
from imdc.submission.validate import validate_submission

import generate_forecast_2026_27 as deployed  # same folder

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "submissions" / "prospective" / "sarimax_size_aware_2026_27"
OUT = OUT_ROOT / "dengue"
SUBMITTED = ROOT / "submissions" / "validation" / "dengue" / f"season_{deployed.SEASON}"


def main():
    params = pd.read_csv(METRICS_DIR / "size_aware_sarimax_params.csv")
    a, b = float(params["a"].iloc[0]), float(params["b"].iloc[0])
    z = params.set_index("uf")["burden_z"]
    print(f"frozen size-aware parameters (SARIMAX-augmented): a={a:.2f} b={b:.2f} "
          f"(fold {params['calib_fold'].iloc[0]}, burden index through {params['burden_cutoff'].iloc[0]})")

    d5 = deployed._real_fold("dengue")
    train = build_state_training_frame(d5, "dengue")
    grid = _full_season_grid(d5, FOLD_SEASON[d5.id], MANDATORY_UFS)
    members = [
        ("climatological", ClimatologicalQuantileModel),
        ("xgboost", lambda: XGBQuantileModel(disease="dengue")),
        ("gru", lambda: DLSequenceModel(disease="dengue", n_ensemble=5, epochs=30)),
        ("sarimax_v2", lambda: SarimaxV2Model(disease="dengue", select_orders=True)),
    ]
    wides = {}
    for name, factory in members:
        model = factory()
        model.fit(train, d5)
        wides[name] = to_submission_wide(model.predict(grid))
        print(f"  fitted {name}: {wides[name]['uf'].nunique()} states, {len(wides[name])} rows")
    missing = set(MANDATORY_UFS) - set(wides["sarimax_v2"]["uf"])
    if missing:
        raise RuntimeError(f"SARIMAX produced no forecast for {sorted(missing)}")

    deployed3 = _vincentize_wide([wides[n] for n in ("climatological", "xgboost", "gru")])
    submitted = pd.concat([pd.read_csv(SUBMITTED / f"{uf}.csv").assign(uf=uf) for uf in MANDATORY_UFS])
    print(f"reproduction check: submitted national median total {submitted['pred'].sum():,.0f} vs "
          f"regenerated three-member {deployed3['pred'].sum():,.0f}")

    four = _vincentize_wide(list(wides.values()))
    sized = clip_nonnegative(apply_size_aware_scale(four, z, a, b))

    OUT.mkdir(parents=True, exist_ok=True)
    wides["sarimax_v2"][["uf", "date"] + QUANTILE_COLUMNS].to_csv(OUT_ROOT / "sarimax_v2_member.csv",
                                                                   index=False)
    digest = hashlib.sha256()
    for uf in MANDATORY_UFS:
        frame = build_submission_frame(sized, uf, deployed.SEASON)
        validate_submission(frame, deployed.SEASON, name=f"sarimax_size_aware/{uf}")
        path = OUT / f"{uf}.csv"
        frame.to_csv(path, index=False)
        digest.update(path.read_bytes())

    nat = sized.groupby("date")[["pred", "lower_90", "upper_90"]].sum()
    sub_nat = submitted.groupby("date")[["lower_90", "upper_90"]].sum()
    (OUT_ROOT / "README.md").write_text(
        "# SARIMAX-augmented, size-aware 2026-27 dengue forecast (prospective comparison)\n\n"
        "Not a challenge submission. The submitted forecast is in "
        f"`submissions/validation/dengue/season_{deployed.SEASON}/` (three-member ensemble, global "
        "conformal recalibration); a size-aware version of it is in "
        "`submissions/prospective/size_aware_2026_27/`. This one adds the per-fold-order SARIMAX v2 "
        "as a fourth member (per-quantile median), which changes the median as well as the "
        f"intervals, and applies the size-aware scale frozen with a = {a:.2f}, b = {b:.2f} "
        "(tuned on the 2022-23 season only). It uses no data after the forecast origin and was "
        f"generated on {datetime.date.today().isoformat()}, inside the target window but before "
        "any 2026-27 observation was examined; the git commit that adds it timestamps it. Scoring "
        "all three once the season resolves is the out-of-sample test of the post-hoc variants "
        "(strictly: on the target weeks after the commit date).\n\n"
        f"- generator: `scripts/generate_forecast_2026_27_sarimax_size_aware.py`\n"
        f"- training data through: {d5.train_cutoff.date()}\n"
        f"- target window: {d5.target_start.date()} to {d5.target_end.date()}\n"
        f"- national median total: {sized['pred'].sum():,.0f} (submitted: {submitted['pred'].sum():,.0f}; "
        f"SARIMAX member alone: {wides['sarimax_v2']['pred'].sum():,.0f})\n"
        f"- national sum of 90% bounds, season total: {nat['lower_90'].sum():,.0f} to "
        f"{nat['upper_90'].sum():,.0f} (submitted: {sub_nat['lower_90'].sum():,.0f} to "
        f"{sub_nat['upper_90'].sum():,.0f})\n"
        "- `sarimax_v2_member.csv`: the SARIMAX member's own forecast, for decomposing the result\n"
        f"- SHA-256 of the 26 state files, concatenated in MANDATORY_UFS order: `{digest.hexdigest()}`\n",
        encoding="utf-8")
    print(f"wrote {len(MANDATORY_UFS)} files to {OUT}")


if __name__ == "__main__":
    main()
