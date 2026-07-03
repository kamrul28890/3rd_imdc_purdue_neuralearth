"""Generate + locally validate validation-phase submission files from a model's predictions.

Writes one CSV per (disease, season, state) under submissions/validation/, and prints a
pass/fail report. Does NOT upload anything (see upload.py + confirm before publishing).

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.submission.generate
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import MANDATORY_UFS, METRICS_DIR, SUBMISSIONS_DIR
from imdc.submission.build import build_submission_frame
from imdc.submission.validate import validate_submission, SubmissionError

FOLD_SEASON = {1: 2023, 2: 2024, 3: 2025, 4: 2026}  # season_year = EW40 target year per fold


def generate(model: str = "ensemble_vincent", disease: str = "dengue", ufs=MANDATORY_UFS,
             scored_file: str = "final_scored.csv"):
    scored = pd.read_csv(METRICS_DIR / scored_file, parse_dates=["date"])
    preds = scored[scored["model"] == model]
    if preds.empty:
        raise SystemExit(f"model '{model}' not found in {scored_file}")

    out_root = SUBMISSIONS_DIR / "validation" / disease
    n_ok, n_fail = 0, 0
    failures = []
    for fold_id, season in FOLD_SEASON.items():
        fold_preds = preds[preds["fold_id"] == fold_id]
        if fold_preds.empty:
            continue
        season_dir = out_root / f"season_{season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        for uf in ufs:
            frame = build_submission_frame(fold_preds, uf, season)
            try:
                validate_submission(frame, season, name=f"{disease}/{season}/{uf}")
                frame.to_csv(season_dir / f"{uf}.csv", index=False)
                n_ok += 1
            except SubmissionError as e:
                n_fail += 1
                failures.append(str(e))

    print(f"model={model} disease={disease}: {n_ok} valid, {n_fail} failed")
    for f in failures[:10]:
        print("  FAIL:", f)
    return n_ok, n_fail


def generate_cities(disease: str, cities: list, model: str = "climatological_quantile"):
    """City-level submissions (adm_level 2), one file per target geocode per season."""
    scored = pd.read_csv(METRICS_DIR / f"city_{disease}_scored.csv", parse_dates=["date"])
    preds = scored[scored["model"] == model].copy()
    preds["uf"] = preds["uf"].astype(str)
    out_root = SUBMISSIONS_DIR / "validation" / f"{disease}_cities"
    n_ok = n_fail = 0
    for fold_id, season in FOLD_SEASON.items():
        fp = preds[preds["fold_id"] == fold_id]
        d = out_root / f"season_{season}"
        d.mkdir(parents=True, exist_ok=True)
        for g in cities:
            frame = build_submission_frame(fp, str(g), season)
            try:
                validate_submission(frame, season, name=f"{disease}_city/{season}/{g}")
                frame.to_csv(d / f"{g}.csv", index=False); n_ok += 1
            except SubmissionError:
                n_fail += 1
    print(f"{disease} cities: {n_ok} valid, {n_fail} failed")
    return n_ok, n_fail


def generate_all():
    """Regenerate every track's submission files (best model per track)."""
    from imdc.config import CHIKUNGUNYA_TARGET_CITIES, DENGUE_TARGET_CITIES
    generate(model="ensemble_vincent", disease="dengue", scored_file="final_scored.csv")
    generate(model="lgbm_quantile", disease="chikungunya", scored_file="chik_final_scored.csv")
    generate_cities("dengue", DENGUE_TARGET_CITIES)
    generate_cities("chikungunya", CHIKUNGUNYA_TARGET_CITIES)


if __name__ == "__main__":
    generate_all()
