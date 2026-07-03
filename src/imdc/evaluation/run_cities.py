"""City-level optional tracks: baselines for dengue (15 cities) and chikungunya (10 cities).

Run as: KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.evaluation.run_cities
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from imdc.config import CHIKUNGUNYA_TARGET_CITIES, DENGUE_TARGET_CITIES, METRICS_DIR
from imdc.data.folds import get_folds
from imdc.evaluation.baselines import ClimatologicalQuantileModel, NaiveModel, SeasonalNaiveModel
from imdc.evaluation.city import run_city_backtest, score_city_backtest
from imdc.evaluation.harness import summarize


def main():
    specs = [NaiveModel, SeasonalNaiveModel, ClimatologicalQuantileModel]
    for disease, cities in [("dengue", DENGUE_TARGET_CITIES), ("chikungunya", CHIKUNGUNYA_TARGET_CITIES)]:
        folds = get_folds(disease)
        frames = []
        for fac in specs:
            preds = run_city_backtest(fac, folds, disease, cities)
            frames.append(score_city_backtest(preds, folds, disease, cities))
        allc = pd.concat(frames, ignore_index=True)
        allc.to_csv(METRICS_DIR / f"city_{disease}_scored.csv", index=False)
        print(f"\n{disease} cities ({len(cities)}):")
        print(summarize(allc, by=["model"])[["model", "wis"]].sort_values("wis").to_string(index=False))


if __name__ == "__main__":
    main()
