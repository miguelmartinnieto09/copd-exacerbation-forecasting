import pandas as pd

from copd_forecasting.validation import (
    get_year_based_splits,
)


def test_sliding_splits():
    dates = pd.Series(
        pd.date_range(
            start="2010-01-01",
            end="2022-12-31",
            freq="D",
        )
    )

    splits = get_year_based_splits(
        dates,
        strategy="sliding",
    )

    assert len(splits) == 10

    train_idx, validation_idx = (
        splits[0]
    )

    train_years = sorted(
        dates.iloc[
            train_idx
        ].dt.year.unique()
    )

    validation_years = sorted(
        dates.iloc[
            validation_idx
        ].dt.year.unique()
    )

    assert train_years == [
        2010,
        2011,
    ]

    assert validation_years == [
        2012,
        2013,
    ]


def test_expanding_splits():
    dates = pd.Series(
        pd.date_range(
            start="2010-01-01",
            end="2022-12-31",
            freq="D",
        )
    )

    splits = get_year_based_splits(
        dates,
        strategy="expanding",
    )

    assert len(splits) == 5

    train_idx, validation_idx = (
        splits[-1]
    )

    train_years = sorted(
        dates.iloc[
            train_idx
        ].dt.year.unique()
    )

    validation_years = sorted(
        dates.iloc[
            validation_idx
        ].dt.year.unique()
    )

    assert train_years == list(
        range(
            2010,
            2021,
        )
    )

    assert validation_years == [
        2021,
        2022,
    ]