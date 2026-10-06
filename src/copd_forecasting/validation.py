import numpy as np
import pandas as pd


def get_year_based_splits(
    dates: pd.Series,
    strategy: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate chronological train-validation splits.

    sliding
        - 2 years training
        - 2 subsequent years validation
        - 1-year shift
        - 10 folds

    expanding
        - 3 initial training years
        - 2 subsequent years validation
        - training window expands by 2 years
        - 5 folds
    """
    dates = (
        pd.to_datetime(dates)
        .reset_index(drop=True)
    )

    start_year = int(
        dates.dt.year.min()
    )

    def year_indices(
        year_start: int,
        year_end: int,
    ) -> np.ndarray:
        mask = (
            (dates.dt.year >= year_start)
            & (dates.dt.year < year_end)
        )

        return np.where(mask)[0]

    splits = []

    if strategy == "sliding":
        train_years = 2
        validation_years = 2
        n_folds = 10

        for fold in range(n_folds):
            train_start = (
                start_year + fold
            )

            train_end = (
                train_start
                + train_years
            )

            validation_start = train_end

            validation_end = (
                validation_start
                + validation_years
            )

            splits.append(
                (
                    year_indices(
                        train_start,
                        train_end,
                    ),
                    year_indices(
                        validation_start,
                        validation_end,
                    ),
                )
            )

    elif strategy == "expanding":
        initial_train_years = 3
        validation_years = 2
        step = 2
        n_folds = 5

        for fold in range(n_folds):
            train_start = start_year

            train_end = (
                start_year
                + initial_train_years
                + fold * step
            )

            validation_start = train_end

            validation_end = (
                validation_start
                + validation_years
            )

            splits.append(
                (
                    year_indices(
                        train_start,
                        train_end,
                    ),
                    year_indices(
                        validation_start,
                        validation_end,
                    ),
                )
            )

    else:
        raise ValueError(
            f"Unknown validation strategy: "
            f"{strategy}"
        )

    return splits


def summarize_temporal_splits(
    dates: pd.Series,
    splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> pd.DataFrame:
    """
    Return a readable summary of temporal CV folds.
    """
    dates = (
        pd.to_datetime(dates)
        .reset_index(drop=True)
    )

    records = []

    for fold, (
        train_idx,
        validation_idx,
    ) in enumerate(
        splits,
        start=1,
    ):
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

        records.append(
            {
                "Fold": fold,
                "Training": (
                    f"{train_years[0]}"
                    f"–{train_years[-1]}"
                ),
                "Validation": (
                    f"{validation_years[0]}"
                    f"–{validation_years[-1]}"
                ),
                "N_train": len(train_idx),
                "N_validation": len(
                    validation_idx
                ),
            }
        )

    return pd.DataFrame(records)