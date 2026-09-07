"""Extrapolate municipal population to 2026/2027 for the forecast-phase submission.

The official population table (datasus_population_2001_2025.csv.gz) ends in 2025,
but the forecast phase needs per-capita incidence normalization for target years
2026 and 2027 (EW41 2026 -> EW40 2027). Every consumer of this table currently
clips the target year to the table's own max year (`year.clip(upper=year_max)`),
so simply appending extrapolated 2026/2027 rows here makes the whole pipeline
pick them up with no other code changes.

Method: per municipality, a 3-year compound annual growth rate (CAGR) from 2022
to 2025 (chosen because Brazil's national growth has been flat at ~0.39-0.41%/yr
since 2021 regardless of a 2-5yr window - a short recent window tracks that
plateau better than a longer one skewed by the faster 2010s growth), clipped to
[-5%, +8%] per year as a safety net against noisy small-municipality estimates
(observed 2020-2025 per-geocode CAGR: 5th/95th pct well inside this band, so the
clip only guards outliers rather than shaping the bulk of the estimates). A
municipality with fewer than 2 years of history (one newly created geocode) is
carried forward flat (0% growth).

Run as: python scripts/extrapolate_population.py
Then:   python scripts/make_manifest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POP_PATH = ROOT / "data" / "raw" / "data_imdc_2026" / "datasus_population_2001_2025.csv.gz"

CAGR_WINDOW = (2022, 2025)
CAGR_CLIP = (-0.05, 0.08)
TARGET_YEARS = [2026, 2027]


def _cagr(pop: pd.DataFrame) -> pd.Series:
    y0, y1 = CAGR_WINDOW
    wide = pop.pivot(index="geocode", columns="year", values="population")
    rate = pd.Series(0.0, index=wide.index)
    have_both = wide[y0].notna() & wide[y1].notna() & (wide[y0] > 0)
    n_years = y1 - y0
    rate[have_both] = (wide.loc[have_both, y1] / wide.loc[have_both, y0]) ** (1 / n_years) - 1
    return rate.clip(*CAGR_CLIP)


def main():
    pop = pd.read_csv(POP_PATH)
    last_year = pop["year"].max()
    last = pop[pop["year"] == last_year].set_index("geocode")["population"]
    rate = _cagr(pop)

    new_rows = []
    for i, year in enumerate(TARGET_YEARS, start=1):
        projected = (last * (1 + rate.reindex(last.index).fillna(0.0)) ** i).round().astype(int)
        new_rows.append(pd.DataFrame({"geocode": projected.index, "year": year, "population": projected.values}))

    extended = pd.concat([pop] + new_rows, ignore_index=True)
    assert not extended.duplicated(subset=["geocode", "year"]).any()

    extended.to_csv(POP_PATH, index=False, compression={"method": "gzip", "compresslevel": 6})

    for year in TARGET_YEARS:
        total = extended.loc[extended["year"] == year, "population"].sum()
        base_total = last.sum()
        print(f"{year}: total population {total:,} ({(total / base_total - 1) * 100:+.3f}% vs {last_year})")
    print(f"\nWrote {len(pop)} + {sum(len(r) for r in new_rows)} rows to {POP_PATH.name}. "
          "Now run: python scripts/make_manifest.py")


if __name__ == "__main__":
    main()
