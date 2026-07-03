"""Synthetic-origin panel construction for the classical-ML models.

Only 4 true forecast origins exist per fold - far too few for gradient
boosting. This module turns each fold into a rolling panel of synthetic
weekly origins x forecast horizons, with strictly leakage-safe, origin-anchored
features plus a target-relative seasonal anchor.

Two leakage disciplines are enforced here (the single place cutoff logic
lives for ML features, reusing imdc.data.folds / imdc.data.validate):

1. Everything is filtered to `date <= fold.train_cutoff` via cutoff_filter,
   asserted with assert_no_leakage. This covers the confirmed 15-week-gap trap.
2. Lag/rolling features are computed from the FULL continuous cutoff-filtered
   history, never from the gapped target window - the gap is a withheld-
   evaluation gap (simulating reporting lag), not a missing-observation gap,
   so the underlying weekly series is continuous up to the cutoff. Using the
   target window to build lags would silently produce wrong/NaN lags; see the
   regression test in tests/test_panel_leakage.py.

Heavy per-history tables (state cases, population-weighted state climate,
static features) are cached as parquet in data/processed/ since they are
identical across folds up to a date filter.
"""
import numpy as np
import pandas as pd
from epiweeks import Week

from imdc.config import DATA_PROCESSED, MANDATORY_UFS
from imdc.data.aggregate import aggregate_cases_to_state, population_weighted_state_climate
from imdc.data.folds import Fold, cutoff_filter
from imdc.data.loaders import (
    load_cases,
    load_climate,
    load_environ_vars,
    load_forecasting_climate,
    load_ocean_indices,
    load_population,
)
from imdc.data.validate import assert_no_leakage

INCIDENCE_SCALE = 1e5
MAX_HORIZON = 67  # 15-week gap + 52-week season
_LAGS = [1, 2, 3, 4, 8, 52]
_OCEAN_LAGS = [0, 4, 8, 12, 26, 52]
ECMWF_COLS = ["ecmwf_temp", "ecmwf_humid", "ecmwf_precip"]


# --------------------------------------------------------------------------
# Cached per-history tables (identical across folds up to a date filter)
# --------------------------------------------------------------------------
def _cache_path(name: str) -> "Path":  # noqa: F821
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED / f"{name}.parquet"


def _cached(name: str, builder):
    path = _cache_path(name)
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    return df


def state_population() -> pd.DataFrame:
    """uf, year, population (municipal populations summed to state)."""
    def build():
        pop = load_population()
        geo_uf = load_cases("dengue")[["geocode", "uf"]].drop_duplicates()
        merged = pop.merge(geo_uf, on="geocode", how="inner")
        return merged.groupby(["uf", "year"], as_index=False)["population"].sum()

    return _cached("state_population", build)


def state_climate_full() -> pd.DataFrame:
    """Population-weighted state climate for the full history (all dates)."""
    def build():
        climate = load_climate()
        pop = load_population()
        geo_uf = load_cases("dengue")[["geocode", "uf"]].drop_duplicates()
        return population_weighted_state_climate(climate, pop, geo_uf)

    return _cached("state_climate_full", build)


def state_ecmwf_full() -> pd.DataFrame:
    """Population-weighted state ECMWF seasonal-climate forecast, indexed by (uf,
    reference_month, target_month). The genuine *future* climate covariate: a forecast
    issued at `reference_month` for `target_month = reference_month + months_ahead` (<=6).
    """
    def build():
        fc = load_forecasting_climate()
        geo_uf = load_cases("dengue")[["geocode", "uf"]].drop_duplicates()
        pop24 = load_population()
        pop24 = pop24[pop24["year"] == 2024][["geocode", "population"]]
        fc = fc.merge(geo_uf, on="geocode", how="inner").merge(pop24, on="geocode", how="left")
        fc["population"] = fc["population"].fillna(0.0)
        cols = ["temp_med", "umid_med", "precip_tot"]
        for c in cols:
            fc[c] = fc[c] * fc["population"]
        g = fc.groupby(["uf", "reference_month", "forecast_months_ahead"], as_index=False)[cols + ["population"]].sum()
        for c in cols:
            g[c] = g[c] / g["population"].replace(0, np.nan)
        g = g.drop(columns="population")
        g["target_month"] = [rm + pd.DateOffset(months=int(a))
                             for rm, a in zip(g["reference_month"], g["forecast_months_ahead"])]
        return g.rename(columns={"temp_med": "ecmwf_temp", "umid_med": "ecmwf_humid",
                                 "precip_tot": "ecmwf_precip"})[
            ["uf", "reference_month", "target_month"] + ECMWF_COLS]

    return _cached("state_ecmwf_full", build)


def _attach_ecmwf(feats: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe ECMWF features: the latest forecast issued at reference_month <= origin_month
    for each row's target month. Rows whose target is >6 months past any valid origin get NaN
    (handled natively by LightGBM)."""
    ecmwf = state_ecmwf_full().sort_values("reference_month")
    f = feats.copy()
    f["origin_month"] = pd.to_datetime(f["origin_date"]).values.astype("datetime64[M]").astype("datetime64[ns]")
    f["target_month"] = pd.to_datetime(f["target_date"]).values.astype("datetime64[M]").astype("datetime64[ns]")
    f = f.sort_values("origin_month")
    merged = pd.merge_asof(
        f, ecmwf, left_on="origin_month", right_on="reference_month",
        by=["uf", "target_month"], direction="backward",
    )
    return merged.drop(columns=["origin_month", "target_month", "reference_month"], errors="ignore")


def state_static_features() -> pd.DataFrame:
    """Time-invariant per-state features: population-weighted Koppen/biome fractions.

    Uses a fixed recent population year (2024) as weights - the fractions are
    structural (which climate zones a state's people live in) and effectively
    constant, so this carries no leakage risk.
    """
    def build():
        env = load_environ_vars()
        pop = load_population()
        geo_uf = load_cases("dengue")[["geocode", "uf"]].drop_duplicates()
        pop24 = pop[pop["year"] == 2024][["geocode", "population"]]
        df = env.merge(geo_uf, on="geocode", how="inner").merge(pop24, on="geocode", how="left")
        df["population"] = df["population"].fillna(0.0)

        rows = []
        for uf, g in df.groupby("uf"):
            total = g["population"].sum()
            row = {"uf": uf}
            for kop, gg in g.groupby("koppen"):
                row[f"koppen_frac_{kop}"] = gg["population"].sum() / total if total else 0.0
            for bio, gg in g.groupby("biome"):
                clean = bio.replace(" ", "_").replace("â", "a").replace(" â", "a")
                row[f"biome_frac_{clean}"] = gg["population"].sum() / total if total else 0.0
            rows.append(row)
        out = pd.DataFrame(rows).fillna(0.0)
        return out

    return _cached("state_static_features", build)


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------
def _epiweek(dates) -> np.ndarray:
    return np.array([Week.fromdate(d).week for d in pd.to_datetime(dates)])


def _add_incidence(state_cases: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Attach incidence (per 100k) and log1p(incidence) using each date's year population."""
    df = state_cases.copy()
    df["year"] = df["date"].dt.year
    year_min, year_max = pop["year"].min(), pop["year"].max()
    df["pop_year"] = df["year"].clip(lower=year_min, upper=year_max)
    df = df.merge(pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
    df["incidence"] = df["casos"] / df["population"] * INCIDENCE_SCALE
    df["log_inc"] = np.log1p(df["incidence"])
    return df


def _origin_anchored_series(fold: Fold, disease: str) -> pd.DataFrame:
    """Per (uf, date) origin-anchored features from the continuous cutoff-filtered history."""
    cases = cutoff_filter(load_cases(disease), fold.train_cutoff)
    assert_no_leakage(cases, fold.train_cutoff, name=f"fold{fold.id} panel cases")
    state = aggregate_cases_to_state(cases)
    pop = state_population()
    df = _add_incidence(state, pop).sort_values(["uf", "date"]).reset_index(drop=True)

    g = df.groupby("uf", group_keys=False)
    for lag in _LAGS:
        df[f"lag_{lag}"] = g["log_inc"].shift(lag - 1)  # lag_1 = current observed value at origin
    df["roll_mean_4"] = g["log_inc"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["roll_std_4"] = g["log_inc"].transform(lambda s: s.rolling(4, min_periods=2).std())
    df["roll_mean_8"] = g["log_inc"].transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["roll_max_12"] = g["log_inc"].transform(lambda s: s.rolling(12, min_periods=1).max())
    df["epiweek"] = _epiweek(df["date"])

    # origin-anchored climate (recent 4-week means + temp anomaly vs epiweek climatology)
    climate = cutoff_filter(state_climate_full(), fold.train_cutoff)
    climate = climate.sort_values(["uf", "date"]).reset_index(drop=True)
    cg = climate.groupby("uf", group_keys=False)
    for col in ["temp_med", "precip_med", "rel_humid_med"]:
        climate[f"{col}_roll4"] = cg[col].transform(lambda s: s.rolling(4, min_periods=1).mean())
    climate["epiweek"] = _epiweek(climate["date"])
    temp_clim = climate.groupby(["uf", "epiweek"])["temp_med"].transform("mean")
    climate["temp_anomaly"] = climate["temp_med"] - temp_clim
    climate_feats = climate[["uf", "date", "temp_med_roll4", "precip_med_roll4",
                             "rel_humid_med_roll4", "temp_anomaly"]]
    df = df.merge(climate_feats, on=["uf", "date"], how="left")

    # origin-anchored ocean indices (national, merged by date) + lags
    ocean = cutoff_filter(load_ocean_indices(), fold.train_cutoff).sort_values("date").reset_index(drop=True)
    for idx in ["enso", "iod", "pdo"]:
        for lag in _OCEAN_LAGS:
            ocean[f"{idx}_lag{lag}"] = ocean[idx].shift(lag)
    ocean_cols = ["date"] + [f"{i}_lag{l}" for i in ["enso", "iod", "pdo"] for l in _OCEAN_LAGS]
    df = df.merge(ocean[ocean_cols], on="date", how="left")

    return df


def _seasonal_anchor_lookup(origin_df: pd.DataFrame) -> dict:
    """(uf, epiweek) -> (sorted date array, log_inc array) for most-recent-same-epiweek anchor."""
    lookup = {}
    for (uf, ew), g in origin_df.groupby(["uf", "epiweek"]):
        g = g.sort_values("date")
        lookup[(uf, int(ew))] = (g["date"].to_numpy(), g["log_inc"].to_numpy())
    return lookup


def _assemble_rows(
    origin_df: pd.DataFrame, pairs: pd.DataFrame, anchor_lookup: dict, with_label: bool
) -> pd.DataFrame:
    """Join origin features onto (uf, origin_date, horizon, target_date) pairs; add target features."""
    feats = pairs.merge(
        origin_df.drop(columns=["casos", "incidence", "year", "pop_year", "population"], errors="ignore"),
        left_on=["uf", "origin_date"], right_on=["uf", "date"], how="left", suffixes=("", "_origin"),
    ).drop(columns=["date"], errors="ignore")

    feats["target_epiweek"] = _epiweek(feats["target_date"])
    w = feats["target_epiweek"].to_numpy()
    feats["sin1"] = np.sin(2 * np.pi * w / 52)
    feats["cos1"] = np.cos(2 * np.pi * w / 52)
    feats["sin2"] = np.sin(4 * np.pi * w / 52)
    feats["cos2"] = np.cos(4 * np.pi * w / 52)

    # target-relative seasonal anchor: most recent same-target-epiweek log_inc at or before origin
    anchor = np.full(len(feats), np.nan)
    for (uf, tew), grp in feats.groupby(["uf", "target_epiweek"]):
        dates, vals = anchor_lookup.get((uf, int(tew)), (np.array([]), np.array([])))
        if len(dates) == 0:
            continue
        origins = grp["origin_date"].to_numpy()
        idx = np.searchsorted(dates, origins, side="right") - 1
        valid = idx >= 0
        a = np.full(len(grp), np.nan)
        a[valid] = vals[idx[valid]]
        anchor[grp.index.to_numpy()] = a
    feats["seasonal_anchor"] = anchor

    if with_label:
        label_src = origin_df[["uf", "date", "log_inc"]].rename(
            columns={"date": "target_date", "log_inc": "label"}
        )
        feats = feats.merge(label_src, on=["uf", "target_date"], how="left")
    return feats


FEATURE_COLS = (
    [f"lag_{l}" for l in _LAGS]
    + ["roll_mean_4", "roll_std_4", "roll_mean_8", "roll_max_12"]
    + ["temp_med_roll4", "precip_med_roll4", "rel_humid_med_roll4", "temp_anomaly"]
    + [f"{i}_lag{l}" for i in ["enso", "iod", "pdo"] for l in _OCEAN_LAGS]
    + ["horizon_weeks", "target_epiweek", "sin1", "cos1", "sin2", "cos2", "seasonal_anchor", "log_pop"]
    + ECMWF_COLS
)


def _attach_static(feats: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    static = state_static_features()
    feats = feats.merge(static, on="uf", how="left")
    static_cols = [c for c in static.columns if c != "uf"]
    feats[static_cols] = feats[static_cols].fillna(0.0)

    feats["target_year"] = pd.to_datetime(feats["target_date"]).dt.year
    year_min, year_max = pop["year"].min(), pop["year"].max()
    feats["pop_year"] = feats["target_year"].clip(lower=year_min, upper=year_max)
    feats = feats.merge(pop.rename(columns={"year": "pop_year"}), on=["uf", "pop_year"], how="left")
    feats["log_pop"] = np.log(feats["population"])
    return feats, static_cols


def build_panel(fold: Fold, disease: str = "dengue", ufs: list = MANDATORY_UFS,
                min_origin: str = "2014-01-01", max_horizon: int = MAX_HORIZON):
    """Training panel for one fold: every (uf, origin, horizon) with an observed label <= cutoff."""
    origin_df = _origin_anchored_series(fold, disease)
    origin_df = origin_df[origin_df["uf"].isin(ufs)]
    anchor_lookup = _seasonal_anchor_lookup(origin_df)
    pop = state_population()

    min_origin = pd.Timestamp(min_origin)
    origins = origin_df[origin_df["date"] >= min_origin][["uf", "date"]].rename(columns={"date": "origin_date"})

    pairs = []
    for h in range(1, max_horizon + 1):
        p = origins.copy()
        p["horizon_weeks"] = h
        p["target_date"] = p["origin_date"] + pd.Timedelta(weeks=h)
        pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs = pairs[pairs["target_date"] <= fold.train_cutoff]  # label must be observed

    feats = _assemble_rows(origin_df, pairs, anchor_lookup, with_label=True)
    feats, static_cols = _attach_static(feats, pop)
    feats = _attach_ecmwf(feats)
    feats = feats.dropna(subset=["label"])
    return feats, FEATURE_COLS + static_cols


def build_prediction_features(fold: Fold, target_grid: pd.DataFrame, disease: str = "dengue"):
    """Features for the real target: origin = train_cutoff, horizons/dates from target_grid."""
    origin_df = _origin_anchored_series(fold, disease)
    anchor_lookup = _seasonal_anchor_lookup(origin_df)
    pop = state_population()

    pairs = target_grid.copy()
    pairs["origin_date"] = fold.train_cutoff
    pairs = pairs.rename(columns={"date": "target_date"})

    feats = _assemble_rows(origin_df, pairs, anchor_lookup, with_label=False)
    feats, static_cols = _attach_static(feats, pop)
    feats = _attach_ecmwf(feats)
    return feats, FEATURE_COLS + static_cols
