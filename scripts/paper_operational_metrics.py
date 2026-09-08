"""Operationally-relevant summary metrics: total-season cases, peak week/magnitude, onset week.

These four quantities (not raw WIS) are what health-system planning actually needs from a
forecast, and are explicitly what the IMDC organizers highlight as target outputs for the
platform's own ensemble (see the 2026-07-31 validation webinar, "Ensemble Model" slide). We
already had `peak_timing_error`/`peak_magnitude_error` in the harness but never reported them;
this adds `onset_week_error`/`total_cases_error` and reports all four together, per (fold, state)
unit, aggregated by model.

Run as: python scripts/paper_operational_metrics.py
"""
import numpy as np
import pandas as pd

from imdc.config import METRICS_DIR
from imdc.evaluation.metrics import (
    onset_week_error,
    peak_magnitude_error,
    peak_timing_error,
    total_cases_error,
)

MODELS = ["naive", "seasonal_naive", "climatological_quantile", "xgb_quantile", "gru_negbin",
          "mechanistic_traj", "ensemble_vincent", "ensemble_conformal"]


def _per_unit_errors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold, uf), g in df.groupby(["model", "fold_id", "uf"]):
        g = g.sort_values("date")
        pred, obs = g["pred"], g["observed_value"]
        rows.append({
            "model": model, "fold_id": fold, "uf": uf,
            "onset_week_error": onset_week_error(pred, obs),
            "peak_timing_error": peak_timing_error(pred, obs),
            "peak_magnitude_error": peak_magnitude_error(pred, obs),
            "total_cases_error": total_cases_error(pred, obs),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(METRICS_DIR / "final_scored.csv", parse_dates=["date"], low_memory=False)
    per_unit = _per_unit_errors(df[df["model"].isin(MODELS)])
    per_unit.to_csv(METRICS_DIR / "operational_errors_by_unit.csv", index=False)

    summary = per_unit.groupby("model").agg(
        onset_week_mae=("onset_week_error", lambda s: s.abs().mean()),
        onset_week_median=("onset_week_error", "median"),
        peak_week_mae=("peak_timing_error", lambda s: s.abs().mean()),
        peak_magnitude_median_log=("peak_magnitude_error", "median"),
        total_cases_median_log=("total_cases_error", "median"),
        total_cases_iqr_log=("total_cases_error", lambda s: s.quantile(0.75) - s.quantile(0.25)),
    ).reindex(MODELS).round(3)
    summary.to_csv(METRICS_DIR / "operational_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
