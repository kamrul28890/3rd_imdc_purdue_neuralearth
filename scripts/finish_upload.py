"""Idempotently finish the submission: upload only the predictions not yet on the platform.

Robust to transient api.mosqlimate.org outages: it queries what is already published, computes
the missing (disease, adm_1, adm_2, season) set across ALL four tracks, and uploads only those.
Safe to re-run any number of times — it never duplicates.

Run (when the API is up) as:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/finish_upload.py
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
SEASONS = [2023, 2024, 2025, 2026, 2027]  # 2027 = the real forecast-phase submission (EW41 2026 -> EW40 2027)

# (track subdir, disease, adm_level, geographies)
TRACKS = [
    ("dengue", "dengue", 1, MANDATORY_UFS),
    ("chikungunya", "chikungunya", 1, MANDATORY_UFS),
    ("dengue_cities", "dengue", 2, DENGUE_TARGET_CITIES),
    ("chikungunya_cities", "chikungunya", 2, CHIKUNGUNYA_TARGET_CITIES),
]


def _present_keys(api_key):
    from mosqlient import get_all_models
    m = [x for x in get_all_models(api_key=api_key) if "3rd_imdc_purdue_neuralearth" in str(x.repository)][0]
    preds = m.predictions(api_key=api_key)
    keys = set()
    for p in preds:
        keys.add((p.disease, p.adm_1, getattr(p, "adm_2", None), str(p.start)[:10]))
    return m.predictions_count, keys


def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.environ["MOSQLIMATE_API_KEY"]
    from mosqlient import upload_prediction

    commit = git_commit_hash(require_clean=True)
    count, present = _present_keys(api_key)
    print(f"already published: {count}")

    missing = []
    for subdir, disease, adm_level, geos in TRACKS:
        code = DISEASE_CODE[disease]
        for season in SEASONS:
            start = str(season_date_range(season)[0].date())
            for g in geos:
                if adm_level == 1:
                    adm_1, adm_2 = UF_TO_ADM1[g], None
                else:
                    adm_1, adm_2 = int(str(g)[:2]), int(g)
                if (code, adm_1, adm_2, start) not in present:
                    missing.append((subdir, code, adm_level, adm_1, adm_2, g, season))

    print(f"missing: {len(missing)}")
    done = 0
    for subdir, code, adm_level, adm_1, adm_2, g, season in missing:
        path = SUBMISSIONS_DIR / "validation" / subdir / f"season_{season}" / f"{g}.csv"
        frame = pd.read_csv(path, parse_dates=["date"])
        kw = {"adm_1": adm_1} if adm_level == 1 else {"adm_1": adm_1, "adm_2": adm_2}
        try:
            upload_prediction(api_key=api_key, repository=REPO, disease=code,
                              description=f"IMDC 2026 {subdir} season {season}", commit=commit,
                              prediction=frame, adm_level=adm_level, **kw)
            done += 1
        except Exception as e:
            if "Duplication" in repr(e):
                done += 1
            else:
                print(f"  retry later: {subdir}/{g}/{season} -> {repr(e)[:70]}")
    print(f"uploaded/confirmed {done}/{len(missing)} this pass. Re-run if any remain.")


if __name__ == "__main__":
    main()
