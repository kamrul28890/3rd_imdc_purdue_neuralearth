"""Upload validated submissions to the Mosqlimate platform via mosqlient.

PREREQUISITE (one-time, web-UI only — the API cannot create models):
register a model at https://mosqlimate.org pointing at a public GitHub/GitLab
repo, for each (disease, adm_level) track you submit. Then pass that repo's
"owner/name" as `repository` here.

Safety:
  - Defaults to dry_run=True (calls Prediction.validate_prediction, which checks
    the data against your registered model WITHOUT publishing).
  - Refuses to run against a dirty working tree, so the recorded commit hash
    actually corresponds to the code/data that produced the numbers.
  - Set dry_run=False explicitly to publish (outward-facing, hard to undo).
"""
import os
import subprocess

import pandas as pd

from imdc.config import DISEASE_CODE, MANDATORY_UFS, SUBMISSIONS_DIR, UF_TO_ADM1
from imdc.submission.build import season_date_range
from imdc.submission.validate import validate_submission

FOLD_SEASON = {1: 2023, 2: 2024, 3: 2025, 4: 2026}


def git_commit_hash(require_clean: bool = True) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if require_clean and dirty:
        raise RuntimeError(
            "Working tree is dirty — commit or stash first so the uploaded commit hash "
            "matches the code/data that produced these predictions."
        )
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _load_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    key = os.environ.get("MOSQLIMATE_API_KEY")
    if not key:
        raise RuntimeError("MOSQLIMATE_API_KEY not set (create .env from .env.example)")
    return key


def upload_season(repository: str, disease: str, season: int, description: str,
                  ufs=MANDATORY_UFS, dry_run: bool = True, require_clean: bool = True) -> dict:
    """Validate (dry_run) or upload every state's submission for one season.

    Returns {uf: 'ok'|'error: ...'} per state.
    """
    from mosqlient import upload_prediction
    from mosqlient.registry.models import Prediction

    key = _load_key()
    commit = git_commit_hash(require_clean=require_clean)
    disease_code = DISEASE_CODE[disease]
    season_dir = SUBMISSIONS_DIR / "validation" / disease / f"season_{season}"

    results = {}
    for uf in ufs:
        path = season_dir / f"{uf}.csv"
        if not path.exists():
            results[uf] = "error: file missing (season may be incomplete)"
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        try:
            validate_submission(frame, season, name=f"{disease}/{season}/{uf}")
            fn = Prediction.validate_prediction if dry_run else upload_prediction
            fn(api_key=key, repository=repository, disease=disease_code,
               description=description, commit=commit, prediction=frame,
               adm_level=1, adm_1=UF_TO_ADM1[uf])
            results[uf] = "ok (validated)" if dry_run else "ok (uploaded)"
        except Exception as e:
            results[uf] = f"error: {repr(e)[:120]}"
    return results


# Track config: subdir under submissions/validation, disease, adm_level, and how to map a
# geography key to the adm_1 (state) or adm_2 (municipality geocode) argument.
TRACKS = {
    "dengue_state": {"subdir": "dengue", "disease": "dengue", "adm_level": 1},
    "chikungunya_state": {"subdir": "chikungunya", "disease": "chikungunya", "adm_level": 1},
    "dengue_cities": {"subdir": "dengue_cities", "disease": "dengue", "adm_level": 2},
    "chikungunya_cities": {"subdir": "chikungunya_cities", "disease": "chikungunya", "adm_level": 2},
}


def upload_track(repository: str, track: str, dry_run: bool = True, require_clean: bool = True) -> dict:
    """Validate/upload every geography and season for one track. Returns {geography_season: status}."""
    from mosqlient import upload_prediction
    from mosqlient.registry.models import Prediction

    cfg = TRACKS[track]
    key = _load_key()
    commit = git_commit_hash(require_clean=require_clean)
    disease_code = DISEASE_CODE[cfg["disease"]]
    track_dir = SUBMISSIONS_DIR / "validation" / cfg["subdir"]

    results = {}
    for season in FOLD_SEASON.values():
        season_dir = track_dir / f"season_{season}"
        if not season_dir.exists():
            continue
        for path in sorted(season_dir.glob("*.csv")):
            geo = path.stem  # "SP" for states, "3550308" (geocode) for cities
            frame = pd.read_csv(path, parse_dates=["date"])
            adm_kwargs = {"adm_1": UF_TO_ADM1[geo]} if cfg["adm_level"] == 1 else {"adm_2": int(geo)}
            try:
                validate_submission(frame, season, name=f"{track}/{season}/{geo}")
                fn = Prediction.validate_prediction if dry_run else upload_prediction
                fn(api_key=key, repository=repository, disease=disease_code,
                   description=f"IMDC 2026 {track} season {season}", commit=commit, prediction=frame,
                   adm_level=cfg["adm_level"], **adm_kwargs)
                results[f"{geo}_{season}"] = "ok"
            except Exception as e:
                msg = repr(e)[:100]
                results[f"{geo}_{season}"] = "ok (already uploaded)" if "Duplication" in msg else f"error: {msg}"
    return results


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else None
    if not repo:
        print("usage: python -m imdc.submission.upload <owner/repo> [--publish]")
        raise SystemExit(1)
    publish = "--publish" in sys.argv
    for fold, season in FOLD_SEASON.items():
        res = upload_season(repo, "dengue", season, f"IMDC 2026 dengue ensemble, season {season}",
                            dry_run=not publish, require_clean=not publish)
        ok = sum(1 for v in res.values() if v.startswith("ok"))
        print(f"season {season}: {ok}/{len(res)} ok" + ("" if ok == len(res) else f" | {res}"))
