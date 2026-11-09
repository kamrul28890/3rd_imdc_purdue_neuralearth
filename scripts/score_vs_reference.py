"""Reproduce the organizers' official IMDC 2026 evaluation: Rv = WIS(model) / WIS(reference).

The 3rd-IMDC results webinar (F. C. Coelho, Nov 2026, slide "Performance e Consistência")
ranks models by Rv against the organizers' reference model (Mosqlimate model 85,
lsbastos/3rd_imdc_procc_bb_model: a Bayesian model with no covariates), computed per state
(UF) and validation season, and reports:
  * the geometric mean of Rv over all UFs (lower is better; < 1 beats the reference),
  * "UFs < 1": the number of states where the model beats the reference,
  * a stability plot of mean log(Rv) against its standard deviation across seasons.

This script recomputes all of that from the platform's own stored predictions
(scripts/fetch_platform_predictions.py) with our scorer (imdc.evaluation.metrics, verified
identical to mosqlient's WIS), so (a) the official numbers can be checked and cited with a
reproducible derivation, and (b) every one of OUR internal backtest models
(results/metrics/*_scored.csv) can be placed on the organizers' scale, against their
baseline instead of our seasonal-naive one.

Rv for a (model, UF, season) = mean WIS over the season's target weeks / the reference's
mean WIS over the same weeks. Aggregations (geometric means are means of log Rv):
  geo_rv          exp(mean log Rv over every UF x season pair)
  ufs_below_1     # UFs whose geometric mean Rv over the seasons is < 1
  mean_log_rv     mean log Rv over all UF x season pairs (= log geo_rv; the slide's x-axis)
  sd_log_rv       SD of log Rv over all UF x season pairs (the slide's y-axis, "stability";
                  lower = more stable). Verified to reproduce the slide: PURDUE 0.28,
                  LNCC_ARP26 0.47, IFGW 0.54, RKI_PH 0.34.
  sd_log_rv_seasons  SD across seasons of the per-season mean log Rv (a stricter
                  interannual-only variant, reported alongside)

Run as: python scripts/score_vs_reference.py
Writes results/metrics/official_rv_by_uf_fold.csv and results/metrics/official_rv_summary.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from imdc.config import METRICS_DIR, UF_TO_ADM1
from imdc.data.folds import get_folds
from imdc.evaluation.harness import build_state_observed_frame
from imdc.evaluation.metrics import wis_from_intervals

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "data" / "external" / "platform_predictions"
REFERENCE_ID = 85
OUR_ID = 83
DISEASES = {"A90": "dengue", "A92.0": "chikungunya"}
ADM1_TO_UF = {v: k for k, v in UF_TO_ADM1.items()}
VALIDATION_FOLDS = (1, 2, 3, 4)

# Our internal backtest models, already scored row-by-row (one file per family).
INTERNAL_SCORED = {
    "dengue": ["final_scored.csv"],
    "chikungunya": ["chik_final_scored.csv"],
}
# Extended / experimental models (Sec. "Nine further models" + FUTURE_WORK.md 6b-6e), labelled
# internal:<file stem>:<model> so variants of the same model name stay distinct.
EXTRA_SCORED = {
    "dengue": ["surge_template_scored.csv", "climate_surge_scored.csv", "prophet_v2_scored.csv",
               "sarimax_v2_scored.csv", "sarimax_v2_perfold_scored.csv", "gru_wis_cpu_scored.csv",
               "hierarchical_scored.csv", "mechanistic_nsub_scored.csv", "xgb_recency_only_scored.csv",
               "ablation_xgb_no_climate_scored.csv", "ablation_xgb_holdout_tuned_scored.csv",
               "ablation_xgb_combined_best_scored.csv", "prophet_scored.csv", "sarimax_scored.csv"],
}


def _observed(disease: str) -> pd.DataFrame:
    frames = []
    for fold in get_folds(disease):
        obs = build_state_observed_frame(fold, disease).rename(columns={"casos": "observed_value"})
        obs["fold_id"] = fold.id
        frames.append(obs)
    return pd.concat(frames, ignore_index=True)


def _fold_from_dates(df: pd.DataFrame) -> pd.Series:
    """Season of each upload, from its first forecast date (teams' free-text descriptions are
    not parseable consistently). A prediction belongs to fold k if it starts within 13 weeks of
    fold k's official target start (EW41); anything else (e.g. the 2026-27 forecast) -> NaN."""
    starts = {f.id: f.target_start for f in get_folds("dengue")}
    first = df.groupby("prediction_id")["date"].transform("min")
    fold = pd.Series(np.nan, index=df.index)
    for k, start in starts.items():
        fold[(first >= start - pd.Timedelta(weeks=2)) & (first < start + pd.Timedelta(weeks=13))] = k
    return fold


def _score_platform() -> pd.DataFrame:
    """Row-level WIS for every platform model's validation predictions."""
    observed = {d: _observed(d) for d in DISEASES.values()}
    out = []
    for path in sorted(PLATFORM.glob("model_*.csv.gz")):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except pd.errors.EmptyDataError:
            continue  # model has no state-level predictions (city-only or none)
        if df.empty:
            continue
        df = df[df["disease"].isin(DISEASES)]
        df["fold_id"] = _fold_from_dates(df)
        df = df[df["fold_id"].isin(VALIDATION_FOLDS)].copy()
        df["fold_id"] = df["fold_id"].astype(int)
        if df.empty:
            continue
        df["disease"] = df["disease"].map(DISEASES)
        df["uf"] = df["adm_1"].map(ADM1_TO_UF)
        # Some teams uploaded a season more than once; keep each (UF, season)'s latest upload.
        latest = df.groupby(["disease", "uf", "fold_id"])["prediction_id"].transform("max")
        df = df[df["prediction_id"] == latest]
        for disease, part in df.groupby("disease"):
            s = part.merge(observed[disease], on=["fold_id", "uf", "date"], how="inner")
            if s.empty:
                continue
            s["wis"] = wis_from_intervals(s, s["observed_value"].to_numpy())
            out.append(s[["model_id", "repository", "disease", "fold_id", "uf", "date", "wis"]])
    return pd.concat(out, ignore_index=True)


def _internal_scored() -> pd.DataFrame:
    out = []
    for disease, files in INTERNAL_SCORED.items():
        for f in files:
            s = pd.read_csv(METRICS_DIR / f, parse_dates=["date"])
            s = s[s["fold_id"].isin(VALIDATION_FOLDS)]
            s = s.assign(disease=disease, model_id=-1, repository="internal:" + s["model"])
            out.append(s[["model_id", "repository", "disease", "fold_id", "uf", "date", "wis"]])
    for disease, files in EXTRA_SCORED.items():
        for f in files:
            s = pd.read_csv(METRICS_DIR / f, parse_dates=["date"])
            s = s[s["fold_id"].isin(VALIDATION_FOLDS)]
            label = "internal:" + f.removesuffix("_scored.csv") + ":" + s["model"]
            s = s.assign(disease=disease, model_id=-1, repository=label)
            out.append(s[["model_id", "repository", "disease", "fold_id", "uf", "date", "wis"]])
    return pd.concat(out, ignore_index=True)


def _rv_by_uf_fold(rows: pd.DataFrame) -> pd.DataFrame:
    ref = rows[rows["model_id"] == REFERENCE_ID][["disease", "fold_id", "uf", "date", "wis"]]
    ref = ref.rename(columns={"wis": "wis_ref"})
    m = rows[rows["model_id"] != REFERENCE_ID].merge(ref, on=["disease", "fold_id", "uf", "date"])
    g = (m.groupby(["model_id", "repository", "disease", "fold_id", "uf"])
         .agg(wis_model=("wis", "mean"), wis_ref=("wis_ref", "mean"), n_weeks=("wis", "size"))
         .reset_index())
    g["rv"] = g["wis_model"] / g["wis_ref"]
    g["log_rv"] = np.log(g["rv"].clip(lower=1e-9))
    return g


def _summary(rv: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "repository", "disease"]
    per_uf = rv.groupby(keys + ["uf"])["log_rv"].mean().reset_index()
    per_fold = rv.groupby(keys + ["fold_id"])["log_rv"].mean().reset_index()
    s = rv.groupby(keys).agg(geo_rv=("log_rv", lambda x: float(np.exp(x.mean()))),
                             mean_log_rv=("log_rv", "mean"),
                             sd_log_rv=("log_rv", "std"),
                             n_pairs=("log_rv", "size"),
                             n_ufs=("uf", "nunique"),
                             n_folds=("fold_id", "nunique")).reset_index()
    s = s.merge(per_uf.assign(below=per_uf["log_rv"] < 0).groupby(keys)["below"].sum()
                .rename("ufs_below_1").reset_index(), on=keys)
    s = s.merge(per_fold.groupby(keys)["log_rv"].agg(sd_log_rv_seasons="std")
                .reset_index(), on=keys)
    wide = per_fold.pivot_table(index=keys, columns="fold_id", values="log_rv")
    wide.columns = [f"geo_rv_fold{c}" for c in wide.columns]
    s = s.merge(np.exp(wide).reset_index(), on=keys, how="left")
    s["ours"] = (s["model_id"] == OUR_ID) | s["repository"].str.startswith("internal:")
    return s.sort_values(["disease", "geo_rv"]).reset_index(drop=True)


def main():
    rows = pd.concat([_score_platform(), _internal_scored()], ignore_index=True)
    rv = _rv_by_uf_fold(rows)
    rv.to_csv(METRICS_DIR / "official_rv_by_uf_fold.csv", index=False)
    summary = _summary(rv)
    summary.to_csv(METRICS_DIR / "official_rv_summary.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["model_id", "repository", "geo_rv", "ufs_below_1", "n_ufs", "n_folds",
            "mean_log_rv", "sd_log_rv", "sd_log_rv_seasons"]
    for disease, part in summary.groupby("disease"):
        print(f"\n=== {disease} (reference = model {REFERENCE_ID}) ===")
        print(part[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
