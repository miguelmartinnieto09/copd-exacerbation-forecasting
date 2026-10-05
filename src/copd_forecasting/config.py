from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FEATURE_SELECTION_FILE = (
    REPORTS_DIR
    / "feature_selection_results.json"
)

CV_RESULTS_FILE = (
    REPORTS_DIR
    / "temporal_cv_results.csv"
)

# ---------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------

DATE_COLUMN = "fecha_asistencia"

STUDY_START = "2010-01-01"
STUDY_END = "2023-12-31"

TEST_YEAR = 2023

MAX_LAG = 7


# ---------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------

DAILY_SHEET_NAME = "EPOC diario"

RAW_DATA_FILE = RAW_DATA_DIR / "copd_daily_data.xlsx"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "copd_daily_data_cleaned.xlsx"