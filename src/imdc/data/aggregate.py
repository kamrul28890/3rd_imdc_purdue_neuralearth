"""Municipality -> state (UF) aggregation.

Top-down design: sum case counts across municipalities into 26 state-level
weekly series (mandatory target excludes ES), and population-weight climate
covariates so state-level exposure reflects where people actually live.
"""
import pandas as pd

from imdc.config import EXCLUDED_UF


def aggregate_cases_to_state(df: pd.DataFrame, exclude_uf: str = EXCLUDED_UF) -> pd.DataFrame:
    """Sum weekly case counts across municipalities within each UF."""
    df = df[df["uf"] != exclude_uf] if exclude_uf else df
    return (
        df.groupby(["uf", "date"], as_index=False)["casos"]
        .sum()
        .sort_values(["uf", "date"])
        .reset_index(drop=True)
    )


def _population_weighted_aggregate(
    climate: pd.DataFrame, population: pd.DataFrame, geocode_to_uf: pd.DataFrame,
    group_cols: list[str], exclude_uf: str = EXCLUDED_UF,
) -> pd.DataFrame:
    """Population-weighted mean of municipal climate variables, aggregated by group_cols.

    `geocode_to_uf` must have columns ['geocode', 'uf']. Population is yearly;
    each climate row's year is used to select the matching population weight,
    falling back to the nearest available year in `population` outside its range.
    """
    value_cols = [c for c in climate.columns if c not in ("date", "epiweek", "geocode")]

    climate = climate.merge(geocode_to_uf[["geocode", "uf"]].drop_duplicates(), on="geocode", how="inner")
    if exclude_uf:
        climate = climate[climate["uf"] != exclude_uf]

    climate = climate.copy()
    climate["year"] = climate["date"].dt.year
    pop_years = population["year"]
    year_min, year_max = pop_years.min(), pop_years.max()
    climate["pop_year"] = climate["year"].clip(lower=year_min, upper=year_max)

    climate = climate.merge(
        population.rename(columns={"year": "pop_year", "population": "weight"}),
        on=["geocode", "pop_year"], how="left",
    )
    climate["weight"] = climate["weight"].fillna(1.0)

    weighted = climate.copy()
    for col in value_cols:
        weighted[col] = weighted[col] * weighted["weight"]

    grouped = weighted.groupby(group_cols, as_index=False)[value_cols + ["weight"]].sum()
    for col in value_cols:
        grouped[col] = grouped[col] / grouped["weight"]
    return grouped.drop(columns="weight").sort_values(group_cols).reset_index(drop=True)


def population_weighted_state_climate(
    climate: pd.DataFrame, population: pd.DataFrame, geocode_to_uf: pd.DataFrame,
    exclude_uf: str = EXCLUDED_UF,
) -> pd.DataFrame:
    """Population-weighted mean of municipal climate variables, aggregated to UF-week."""
    return _population_weighted_aggregate(climate, population, geocode_to_uf, ["uf", "date"], exclude_uf)


def population_weighted_national_climate(
    climate: pd.DataFrame, population: pd.DataFrame, geocode_to_uf: pd.DataFrame,
    exclude_uf: str = EXCLUDED_UF,
) -> pd.DataFrame:
    """Population-weighted mean of municipal climate variables, aggregated to national-week."""
    return _population_weighted_aggregate(climate, population, geocode_to_uf, ["date"], exclude_uf)
