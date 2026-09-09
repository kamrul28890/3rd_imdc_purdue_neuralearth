"""Replace the already-published season-2027 (forecast-phase) predictions with the
regenerated ones (climatological+XGBoost+GRU ensemble, post data-refresh).

mosqlient has no "update prediction data" call -- only delete + re-upload. This script
does that ONE TRACK AT A TIME (dengue-state, chikungunya-state, dengue-cities,
chikungunya-cities): delete that track's existing season-2027 predictions, verify they're
gone, upload the freshly regenerated CSVs, verify the new count matches the expected
geography count, then move to the next track. If a track's post-upload count doesn't
match, it stops rather than proceeding to delete the next track -- so at most one track
is ever left incomplete on the platform, and it's the one printed in the failure.

Run (when the API is up) as:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/resubmit_forecast_2026_27.py
"""
import os
import socket

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
socket.setdefaulttimeout(60)

import pandas as pd
from dotenv import load_dotenv

from imdc.config import (CHIKUNGUNYA_TARGET_CITIES, DENGUE_TARGET_CITIES, DISEASE_CODE,
                         MANDATORY_UFS, SUBMISSIONS_DIR, UF_TO_ADM1)
from imdc.submission.build import season_date_range
from imdc.submission.upload import git_commit_hash

REPO = "kamrul28890/3rd_imdc_purdue_neuralearth"
SEASON = 2027

# (label, subdir, disease, adm_level, geographies)
TRACKS = [
    ("dengue-state", "dengue", "dengue", 1, MANDATORY_UFS),
    ("chikungunya-state", "chikungunya", "chikungunya", 1, MANDATORY_UFS),
    ("dengue-cities", "dengue_cities", "dengue", 2, DENGUE_TARGET_CITIES),
    ("chikungunya-cities", "chikungunya_cities", "chikungunya", 2, CHIKUNGUNYA_TARGET_CITIES),
]


def _get_model(api_key: str):
    from mosqlient import get_all_models
    matches = [m for m in get_all_models(api_key=api_key)
               if "3rd_imdc_purdue_neuralearth" in str(m.repository)]
    if not matches:
        raise RuntimeError("no registered model found for this repository")
    return matches[0]


def _season_2027_predictions(model, api_key: str, disease_code: str, adm_level: int):
    start = str(season_date_range(SEASON)[0].date())
    preds = model.predictions(api_key=api_key)
    out = []
    for p in preds:
        if str(p.disease) != disease_code or str(p.start)[:10] != start:
            continue
        is_city = p.adm_2 is not None
        if is_city != (adm_level == 2):
            continue
        out.append(p)
    return out


def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.environ["MOSQLIMATE_API_KEY"]
    from mosqlient import upload_prediction

    commit = git_commit_hash(require_clean=True)
    print(f"commit: {commit}\n")

    for label, subdir, disease, adm_level, geos in TRACKS:
        code = DISEASE_CODE[disease]
        expected = len(geos)
        print(f"=== {label} (expect {expected} predictions) ===")

        model = _get_model(api_key)
        existing = _season_2027_predictions(model, api_key, code, adm_level)
        print(f"  found {len(existing)} existing season-{SEASON} predictions to replace")

        for p in existing:
            p.delete(api_key=api_key)
        model = _get_model(api_key)
        remaining = _season_2027_predictions(model, api_key, code, adm_level)
        if remaining:
            print(f"  ABORT: {len(remaining)} predictions still present after delete "
                  f"({label}). Not touching later tracks. Re-run to retry the delete.")
            return
        print(f"  deleted {len(existing)}, confirmed 0 remain")

        season_dir = SUBMISSIONS_DIR / "validation" / subdir / f"season_{SEASON}"
        uploaded = failed = 0
        for g in geos:
            path = season_dir / f"{g}.csv"
            if not path.exists():
                print(f"  MISSING FILE: {path}")
                failed += 1
                continue
            frame = pd.read_csv(path, parse_dates=["date"])
            kw = {"adm_1": UF_TO_ADM1[g]} if adm_level == 1 else \
                 {"adm_1": int(str(g)[:2]), "adm_2": int(g)}
            try:
                upload_prediction(api_key=api_key, repository=REPO, disease=code,
                                  description=f"IMDC 2026 {label} season {SEASON} "
                                              f"(climatological+xgboost+gru ensemble, "
                                              f"post-refresh data, commit {commit[:8]})",
                                  commit=commit, prediction=frame, adm_level=adm_level, **kw)
                uploaded += 1
            except Exception as e:
                print(f"  UPLOAD FAILED {g}: {repr(e)[:150]}")
                failed += 1

        model = _get_model(api_key)
        now_present = _season_2027_predictions(model, api_key, code, adm_level)
        print(f"  uploaded {uploaded}/{expected}, {failed} failed, "
              f"platform now shows {len(now_present)} for this track")
        if len(now_present) != expected:
            print(f"  ABORT: {label} ended with {len(now_present)}/{expected} on the "
                  f"platform (deleted old, new upload incomplete). Fix and re-run before "
                  f"continuing to the next track -- do NOT delete the next track's old "
                  f"predictions while this one is short.")
            return
        print(f"  {label} done: {expected}/{expected} on platform.\n")

    print("All four tracks replaced successfully.")


if __name__ == "__main__":
    main()
