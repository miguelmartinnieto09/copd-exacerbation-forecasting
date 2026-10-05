import numpy as np
import pandas as pd

from .config import DATE_COLUMN, MAX_LAG


POLLUTION_VARS = [
    "PM10",
    "PM2p5",
    "NO",
    "NO2",
    "CO",
    "O3",
]

METEOROLOGICAL_VARS = [
    "temp_min",
    "temp_med",
    "temp_max",
    "temp_sens",
    "pres_min",
    "pres_med",
    "pres_max",
    "prec_24",
    "sol_24",
    "vel_med",
    "vel_max",
    "hume_med",
    "hume_max",
]

TEMPORAL_COLS = [
    "dia_semana",
    "estac_sin",
    "estac_cos",
]

# Mean environmental measurements excluded from subset A
# to obtain a more compact representation.
SUBSET_A_EXCLUDED_BASES = {
    "temp_med",
    "pres_med",
    "vel_med",
    "hume_med",
}


def add_temporal_features(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
) -> pd.DataFrame:
    """
    Add weekday and cyclical annual-seasonality features.

    Weekday is encoded from 0 (Monday) to 6 (Sunday).
    Annual seasonality is represented using sine and cosine
    components, accounting for leap years.
    """
    df = df.copy()

    dates = pd.to_datetime(
        df[date_column]
    )

    df["dia_semana"] = dates.dt.dayofweek

    day_of_year = dates.dt.dayofyear

    days_in_year = np.where(
        dates.dt.is_leap_year,
        366,
        365,
    )

    angle = (
        2
        * np.pi
        * (day_of_year - 1)
        / days_in_year
    )

    df["estac_sin"] = np.sin(angle)
    df["estac_cos"] = np.cos(angle)

    return df


def create_binary_target(
    df: pd.DataFrame,
    count_column: str = "n_urgencias",
) -> pd.Series:
    """
    Create the binary forecasting target.

    0 = no COPD-related emergency visits
    1 = at least one COPD-related emergency visit
    """
    return (
        df[count_column] >= 1
    ).astype(int)


def lag_column_name(
    base_variable: str,
    lag: int,
) -> str:
    """
    Return the column name used by the original dataset
    for a given temporal lag.
    """
    if lag == 0:
        return f"{base_variable}_LAG0"

    return f"{base_variable}_LAG-{lag}"


def get_environmental_lag_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return environmental predictor columns from LAG0 to LAG-7.

    Healthcare variables and air-quality indices are excluded.
    """
    columns = []

    environmental_vars = (
        POLLUTION_VARS
        + METEOROLOGICAL_VARS
    )

    for base in environmental_vars:
        for lag in range(
            MAX_LAG + 1
        ):
            col = lag_column_name(
                base,
                lag,
            )

            if col in df.columns:
                columns.append(col)

    return columns


def build_subset_a(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build subset A using immediate and accumulated exposures.

    For each retained environmental variable:
    - IMM: same-day exposure (LAG0)
    - ACUM_0_3: mean exposure from LAG0 to LAG-3
    - ACUM_0_7: mean exposure from LAG0 to LAG-7

    Mean environmental measurements are excluded to obtain
    the compact representation used in the thesis.
    """
    subset = pd.DataFrame(
        index=df.index
    )

    environmental_vars = [
        var
        for var in (
            POLLUTION_VARS
            + METEOROLOGICAL_VARS
        )
        if var
        not in SUBSET_A_EXCLUDED_BASES
    ]

    for base in environmental_vars:
        lag_cols = {
            lag: lag_column_name(
                base,
                lag,
            )
            for lag in range(
                MAX_LAG + 1
            )
        }

        missing = [
            col
            for col in lag_cols.values()
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing lag columns for "
                f"'{base}': {missing}"
            )

        subset[
            f"{base}_IMM"
        ] = df[lag_cols[0]]

        subset[
            f"{base}_ACUM_0_3"
        ] = df[
            [
                lag_cols[lag]
                for lag in range(4)
            ]
        ].mean(axis=1)

        subset[
            f"{base}_ACUM_0_7"
        ] = df[
            [
                lag_cols[lag]
                for lag in range(8)
            ]
        ].mean(axis=1)

    for col in TEMPORAL_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Temporal feature "
                f"'{col}' is missing."
            )

        subset[col] = df[col]

    return subset


def build_subset_b(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build subset B using all individual environmental lags
    together with the temporal features.
    """
    lag_columns = (
        get_environmental_lag_columns(
            df
        )
    )

    columns = (
        lag_columns
        + TEMPORAL_COLS
    )

    missing = [
        col
        for col in TEMPORAL_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing temporal features: "
            f"{missing}"
        )

    return df[columns].copy()