"""Refresh the raw IMDC 2026 dataset with data through EW25 2026 for the forecast phase.

The committed archive under data/raw/data_imdc_2026/ ends 2026-03-08. The
official FTP mirror (info.dengue.mat.br/data_imdc_2026) publishes small
"*_update_2026.csv.gz" increments alongside the base tables that extend the
case/climate/ocean series through EW25 2026 (2026-06-21) - the cutoff the
forecast-phase submission (EW41 2026 -> EW40 2027) must train on. This script
downloads those increments and merges them into the base tables in place.

Files with no published update (access_afya, population, environ_vars,
map_regional_health, the shapefiles) are left untouched.

Run as: python scripts/refresh_data.py
Then:   python scripts/make_manifest.py   (rewrites CHECKSUMS.sha256 + RESULTS.md)
"""
import io
from ftplib import FTP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "data_imdc_2026"
FTP_HOST = "info.dengue.mat.br"
FTP_DIR = "data_imdc_2026"

# base filename -> (update filename, dedup key columns, {update_col: base_col} renames)
MERGE_SPECS = {
    "dengue.csv.gz": ("dengue_update_2026.csv.gz", ["geocode", "date"], {}),
    "chikungunya.csv.gz": ("chikungunya_update_2026.csv.gz", ["geocode", "date"], {}),
    "climate.csv.gz": ("climate_update_2026.csv.gz", ["geocode", "date"], {}),
    "forecasting_climate.csv.gz": (
        "forecasting_climate_update_2026.csv.gz",
        ["geocode", "reference_month", "forecast_months_ahead"],
        {"rel_umid_med": "umid_med"},
    ),
    "ocean_climate_oscillations.csv.gz": (
        "ocean_climate_oscillations_update_2026.csv.gz",
        ["date"],
        {},
    ),
}


def _fetch(ftp: FTP, name: str) -> bytes:
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {name}", buf.write)
    return buf.getvalue()


def _merge_one(base_path: Path, update_bytes: bytes, key_cols: list, rename: dict) -> tuple[int, int]:
    base = pd.read_csv(base_path)
    update = pd.read_csv(io.BytesIO(update_bytes), compression="gzip")
    update = update.rename(columns=rename)
    update = update.drop(columns=[c for c in update.columns if c.startswith("Unnamed")])

    missing = set(base.columns) - set(update.columns)
    if missing:
        raise ValueError(f"{base_path.name}: update file is missing columns {missing}")
    update = update.reindex(columns=base.columns)

    combined = pd.concat([base, update], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)

    assert not combined.duplicated(subset=key_cols).any(), f"{base_path.name}: duplicate keys after merge"

    combined.to_csv(base_path, index=False, compression={"method": "gzip", "compresslevel": 6})
    return len(update), len(combined) - len(base)


def main():
    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login()
    ftp.cwd(FTP_DIR)
    remote_files = set(ftp.nlst())

    for base_name, (update_name, key_cols, rename) in MERGE_SPECS.items():
        if update_name not in remote_files:
            print(f"skip {base_name}: no {update_name} found on {FTP_HOST}/{FTP_DIR}")
            continue
        print(f"fetching {update_name} ...")
        data = _fetch(ftp, update_name)
        n_update, n_new = _merge_one(RAW / base_name, data, key_cols, rename)
        print(f"  {base_name}: merged {n_update} update rows ({n_new} net-new rows after dedup)")

    ftp.quit()
    print("\nRaw data refreshed. Now run: python scripts/make_manifest.py")


if __name__ == "__main__":
    main()
