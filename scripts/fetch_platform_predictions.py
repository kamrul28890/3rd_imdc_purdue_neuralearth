"""Download every 3rd-IMDC (2026) model's state-level predictions from the Mosqlimate platform.

Used by scripts/score_vs_reference.py to reproduce the organizers' official
"Rv = WIS model / WIS reference" evaluation (IMDC 2026 results webinar, Nov 2026)
with our own scorer, against their reference model (id 85, lsbastos/3rd_imdc_procc_bb_model,
the covariate-free Bayesian baseline the slides call "BB").

Each model is cached to data/external/platform_predictions/model_<id>.csv.gz (gitignored:
these are other teams' forecasts, already public on the platform/GitHub). Re-running skips
models already cached unless --refresh is passed.

Run as: python scripts/fetch_platform_predictions.py [--refresh] [--ids 83 85 ...]
Needs MOSQLIMATE_API_KEY in .env.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "platform_predictions"


def fetch_model(model, key: str) -> pd.DataFrame:
    rows = []
    for p in model.predictions(api_key=key):
        if p.adm_2 is not None or p.adm_1 is None:
            continue  # state level only
        df = p.to_dataframe()
        df["prediction_id"] = p.id
        df["disease"] = p.disease
        df["adm_1"] = int(p.adm_1)
        df["description"] = p.description
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "repository", str(model.repository))
    out.insert(0, "model_id", model.id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ["MOSQLIMATE_API_KEY"]
    from mosqlient import get_all_models

    models = [m for m in get_all_models(api_key=key) if m.imdc_year == 2026]
    if args.ids:
        models = [m for m in models if m.id in args.ids]
    OUT.mkdir(parents=True, exist_ok=True)
    for m in sorted(models, key=lambda m: m.id):
        path = OUT / f"model_{m.id}.csv.gz"
        if path.exists() and not args.refresh:
            print(f"{m.id:>3} cached  {m.repository}")
            continue
        df = fetch_model(m, key)
        df.to_csv(path, index=False)
        print(f"{m.id:>3} {len(df):>6} rows  {m.repository}", flush=True)


if __name__ == "__main__":
    main()
