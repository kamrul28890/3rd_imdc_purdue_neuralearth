"""Central paths and reference constants for the IMDC 2026 pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "data_imdc_2026"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

RAW_FILES = {
    "dengue": DATA_RAW / "dengue.csv.gz",
    "chikungunya": DATA_RAW / "chikungunya.csv.gz",
    "climate": DATA_RAW / "climate.csv.gz",
    "forecasting_climate": DATA_RAW / "forecasting_climate.csv.gz",
    "ocean": DATA_RAW / "ocean_climate_oscillations.csv.gz",
    "environ_vars": DATA_RAW / "environ_vars.csv.gz",
    "population": DATA_RAW / "datasus_population_2001_2025.csv.gz",
    "access_afya": DATA_RAW / "access_afya_dengue_2021_2026.csv.gz",
    "map_regional_health": DATA_RAW / "map_regional_health.csv",
    "shape_muni": DATA_RAW / "shape_muni.gpkg",
    "shape_regional_health": DATA_RAW / "shape_regional_health.gpkg",
    "shape_macroregional_health": DATA_RAW / "shape_macroregional_health.gpkg",
}

# All 27 Brazilian UF codes as they appear in the raw data.
ALL_UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
]

# Espirito Santo is excluded from the mandatory state-level target.
EXCLUDED_UF = "ES"
MANDATORY_UFS = [uf for uf in ALL_UFS if uf != EXCLUDED_UF]

# Optional-challenge target cities (IBGE geocodes), confirmed from target_city
# flag in the raw data: 15 for dengue, 10 for chikungunya (different sets).
DENGUE_TARGET_CITIES = [
    1200203, 1200401, 1716109, 2302503, 2931350, 2933307, 3119401, 3541406,
    3549805, 4103701, 4104808, 4113700, 5102637, 5201405, 5215231,
]
CHIKUNGUNYA_TARGET_CITIES = [
    1716109, 1721000, 2211001, 2931350, 3119401, 3143302, 4104808, 4219507,
    5102637, 5103403,
]

# Required quantile levels for submissions: median + 50/80/90/95% intervals.
QUANTILE_LEVELS = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
INTERVAL_LEVELS = [50, 80, 90, 95]

# Wide submission-format column name for each quantile level, in ascending order.
QUANTILE_COLUMNS = [
    "lower_95", "lower_90", "lower_80", "lower_50",
    "pred",
    "upper_50", "upper_80", "upper_90", "upper_95",
]
QUANTILE_LEVEL_TO_COLUMN = dict(zip(QUANTILE_LEVELS, QUANTILE_COLUMNS))
QUANTILE_COLUMN_TO_LEVEL = dict(zip(QUANTILE_COLUMNS, QUANTILE_LEVELS))

N_FOLDS = 4
