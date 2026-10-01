from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DAILY_SHEET_NAME,
    DATE_COLUMN,
    MAX_LAG,
    STUDY_END,
    STUDY_START,
)


def load_daily_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the aggregated daily COPD dataset from the original Excel file.

    Only the aggregated daily sheet is loaded. Patient-level and
    attendance-level sheets are not required by the public pipeline.
    """
    df = pd.read_excel(
        file_path,
        sheet_name=DAILY_SHEET_NAME,
    )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Expected date column '{DATE_COLUMN}' was not found."
        )

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    return df


def filter_study_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to the study period.
    """
    df = df.copy()

    mask = df[DATE_COLUMN].between(
        pd.Timestamp(STUDY_START),
        pd.Timestamp(STUDY_END),
        inclusive="both",
    )

    return (
        df.loc[mask]
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )


def validate_daily_timeline(df: pd.DataFrame) -> None:
    """
    Check that dates are unique, ordered and consecutive.

    Raises
    ------
    ValueError
        If duplicated dates or gaps in the daily time series are found.
    """
    dates = pd.to_datetime(df[DATE_COLUMN])

    if dates.duplicated().any():
        duplicated_dates = dates[dates.duplicated()].dt.date.tolist()

        raise ValueError(
            f"Duplicated dates found: {duplicated_dates[:10]}"
        )

    expected_dates = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        freq="D",
    )

    missing_dates = expected_dates.difference(dates)

    if len(missing_dates) > 0:
        raise ValueError(
            f"{len(missing_dates)} dates are missing from the daily timeline."
        )


def validate_lag_boundary_missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether the additional missing values in lagged variables
    are consistent with the expected temporal boundary effect.

    For each variable:

        missing(LAGk) = missing(LAG0) + k

    Returns
    -------
    pd.DataFrame
        Validation summary for every lagged variable found.
    """
    results = []

    lag0_columns = [
        col
        for col in df.columns
        if col.endswith("_LAG0")
    ]

    for lag0_col in lag0_columns:
        base_name = lag0_col.removesuffix("_LAG0")
        missing_lag0 = df[lag0_col].isna().sum()

        for lag in range(1, MAX_LAG + 1):
            lag_col = f"{base_name}_LAG-{lag}"

            if lag_col not in df.columns:
                continue

            observed = df[lag_col].isna().sum()
            expected = missing_lag0 + lag

            results.append(
                {
                    "variable": base_name,
                    "lag": lag,
                    "observed_missing": observed,
                    "expected_missing": expected,
                    "valid": observed == expected,
                }
            )

    return pd.DataFrame(results)


def replace_invalid_pressure_values(
    df: pd.DataFrame,
    minimum_pressure: float = 800.0,
) -> pd.DataFrame:
    """
    Replace physically implausible atmospheric pressure values with NaN.

    The original study considered pressure values below 800 hPa
    incompatible with the study context.
    """
    df = df.copy()

    pressure_columns = [
        col
        for col in df.columns
        if col.lower().startswith("pres_")
        and "_LAG" in col
    ]

    for col in pressure_columns:
        df.loc[df[col] < minimum_pressure, col] = np.nan

    return df


def remove_lag_boundary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove the initial observations affected by the maximum lag.
    """
    return (
        df.iloc[MAX_LAG:]
        .copy()
        .reset_index(drop=True)
    )


def impute_lagged_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Impute missing numerical lagged variables using linear interpolation.

    This reproduces the imputation strategy used in the thesis.
    """
    df = df.copy()

    columns_to_impute = [
        col
        for col in df.columns
        if "_LAG" in col
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].isna().any()
    ]

    for col in columns_to_impute:
        df[col] = df[col].interpolate(
            method="linear",
            limit_direction="both",
        )

    return df


def check_remaining_missing_values(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return columns that still contain missing values.
    """
    missing = df.isna().sum()

    return missing[missing > 0].sort_values(
        ascending=False
    )