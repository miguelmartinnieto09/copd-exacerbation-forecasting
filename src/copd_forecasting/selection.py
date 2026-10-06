import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def filter_correlated_features(
    df_features: pd.DataFrame,
    threshold: float = 0.8,
) -> tuple[list[str], list[str]]:
    """
    Remove highly redundant predictors using absolute
    Pearson correlation.

    When two variables have an absolute correlation above
    the specified threshold, the variable with the higher
    mean absolute correlation with the remaining predictors
    is removed.

    Returns
    -------
    kept_columns : list[str]
        Predictor names retained after filtering.

    removed_columns : list[str]
        Predictor names removed because of redundancy.
    """
    corr = (
        df_features
        .corr(method="pearson")
        .abs()
    )

    mean_corr = corr.mean(axis=1)

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)
    )

    removed = set()

    for col in upper.columns:
        correlated = upper.index[
            upper[col] > threshold
        ].tolist()

        for other_col in correlated:
            if (
                col in removed
                or other_col in removed
            ):
                continue

            if (
                mean_corr[col]
                >= mean_corr[other_col]
            ):
                removed.add(col)

            else:
                removed.add(other_col)

    kept = [
        col
        for col in df_features.columns
        if col not in removed
    ]

    return (
        kept,
        sorted(removed),
    )


def prepare_feature_subset(
    feature_df: pd.DataFrame,
    development_mask: np.ndarray,
    test_mask: np.ndarray,
    correlation_threshold: float = 0.8,
) -> dict:
    """
    Apply redundancy filtering and Min-Max scaling.

    Feature filtering and scaler fitting are performed using
    development data only. The learned transformation is then
    applied to the independent test set.
    """
    feature_names = (
        feature_df.columns.tolist()
    )

    X_all = feature_df.to_numpy()

    X_development_raw = X_all[
        development_mask
    ]

    X_test_raw = X_all[
        test_mask
    ]

    development_df = pd.DataFrame(
        X_development_raw,
        columns=feature_names,
    )

    (
        kept_columns,
        removed_columns,
    ) = filter_correlated_features(
        development_df,
        threshold=correlation_threshold,
    )

    kept_indices = [
        feature_names.index(col)
        for col in kept_columns
    ]

    scaler = MinMaxScaler()

    X_development = scaler.fit_transform(
        X_development_raw[
            :,
            kept_indices,
        ]
    )

    X_test = scaler.transform(
        X_test_raw[
            :,
            kept_indices,
        ]
    )

    return {
        "X_development": X_development,
        "X_test": X_test,
        "features": kept_columns,
        "removed_features": removed_columns,
        "scaler": scaler,
    }

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

def compute_mi_matrix(
    X: np.ndarray,
    feature_names: list[str],
    n_neighbors: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute the mutual-information redundancy matrix
    between predictors.

    The matrix is calculated once before the bootstrap
    procedure and reused in every replicate.
    """
    n_features = X.shape[1]

    mi_matrix = np.zeros(
        (n_features, n_features)
    )

    for i in range(n_features):
        mi_row = mutual_info_regression(
            X,
            X[:, i],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        # Preserve the symmetric construction used
        # in the original thesis implementation.
        mi_matrix[i, :] = mi_row
        mi_matrix[:, i] = mi_row

    np.fill_diagonal(
        mi_matrix,
        0.0,
    )

    return mi_matrix

def mrmr_scores(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    mi_matrix_full: np.ndarray,
    full_feature_names: list[str],
    max_features: int | None = None,
    random_state: int = 42,
) -> tuple[list[str], list[float]]:
    """
    Greedy minimum-redundancy maximum-relevance ranking.

    Relevance:
        Mutual information between a predictor and the
        binary target.

    Redundancy:
        Mean mutual information between a candidate and
        the predictors already selected.
    """
    n_features = X.shape[1]

    if max_features is None:
        max_features = n_features

    max_features = min(
        max_features,
        n_features,
    )

    # Relevance with respect to the target
    mi_target = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        n_neighbors=3,
        random_state=random_state,
    )

    name_to_full_idx = {
        name: i
        for i, name
        in enumerate(full_feature_names)
    }

    global_idx = [
        name_to_full_idx[name]
        for name in feature_names
    ]

    def mean_redundancy(
        candidate_local_idx: int,
        selected_local_idx: list[int],
    ) -> float:
        if not selected_local_idx:
            return 0.0

        candidate_global_idx = (
            global_idx[candidate_local_idx]
        )

        selected_global_idx = [
            global_idx[idx]
            for idx in selected_local_idx
        ]

        return float(
            mi_matrix_full[
                candidate_global_idx,
                selected_global_idx,
            ].mean()
        )

    selected = []
    unselected = list(
        range(n_features)
    )

    ranking = []
    scores = []

    while (
        unselected
        and len(ranking) < max_features
    ):
        if not selected:
            best_idx = int(
                np.argmax(mi_target)
            )

            best_score = float(
                mi_target[best_idx]
            )

        else:
            candidate_scores = {
                idx: (
                    mi_target[idx]
                    - mean_redundancy(
                        idx,
                        selected,
                    )
                )
                for idx in unselected
            }

            best_idx = max(
                candidate_scores,
                key=candidate_scores.get,
            )

            best_score = float(
                candidate_scores[best_idx]
            )

        selected.append(best_idx)
        unselected.remove(best_idx)

        ranking.append(
            feature_names[best_idx]
        )

        scores.append(best_score)

    return ranking, scores

def temporal_block_bootstrap_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_bootstrap: int = 1000,
    block_size: int = 7,
    random_state: int = 42,
) -> dict:
    """
    Perform mRMR feature selection with temporal block
    bootstrap.

    Procedure
    ---------
    1. Divide the development series into contiguous blocks.
    2. Sample complete blocks with replacement.
    3. Run greedy mRMR within every replicate.
    4. Count predictors with positive mRMR scores.
    5. Estimate selection frequency.
    6. Retain predictors whose frequency exceeds the
       mean selection frequency.
    """
    rng = np.random.default_rng(
        random_state
    )

    n_samples = len(X)

    print(
        "Computing predictor mutual-information matrix..."
    )

    mi_matrix = compute_mi_matrix(
        X,
        feature_names,
        random_state=random_state,
    )

    # Contiguous temporal blocks
    blocks = []

    start = 0

    while start < n_samples:
        blocks.append(
            np.arange(
                start,
                min(
                    start + block_size,
                    n_samples,
                ),
            )
        )

        start += block_size

    n_blocks = len(blocks)

    selection_count = np.zeros(
        len(feature_names),
        dtype=int,
    )

    name_to_index = {
        name: i
        for i, name
        in enumerate(feature_names)
    }

    for replicate in range(
        n_bootstrap
    ):
        block_indices = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        sample_indices = np.concatenate(
            [
                blocks[i]
                for i in block_indices
            ]
        )

        sample_indices = sample_indices[
            :n_samples
        ]

        X_boot = X[
            sample_indices
        ]

        y_boot = y[
            sample_indices
        ]

        # Some variables may become effectively
        # constant in an individual bootstrap sample.
        valid_mask = (
            X_boot.std(axis=0)
            > 1e-8
        )

        if valid_mask.sum() < 2:
            continue

        X_valid = X_boot[
            :,
            valid_mask,
        ]

        valid_names = [
            feature_names[i]
            for i, valid
            in enumerate(valid_mask)
            if valid
        ]

        ranking, scores = mrmr_scores(
            X_valid,
            y_boot,
            valid_names,
            mi_matrix_full=mi_matrix,
            full_feature_names=feature_names,
            max_features=len(
                valid_names
            ),
            random_state=random_state,
        )

        selected_in_replicate = [
            variable
            for variable, score
            in zip(
                ranking,
                scores,
            )
            if score > 0
        ]

        for variable in (
            selected_in_replicate
        ):
            selection_count[
                name_to_index[variable]
            ] += 1

        if (
            replicate + 1
        ) % 100 == 0:
            print(
                f"Bootstrap replicate "
                f"{replicate + 1}/"
                f"{n_bootstrap}"
            )

    frequencies = (
        selection_count
        / n_bootstrap
    )

    threshold = float(
        frequencies.mean()
    )

    selected_features = [
        feature_names[i]
        for i
        in range(
            len(feature_names)
        )
        if frequencies[i] > threshold
    ]

    return {
        "frequencies": frequencies,
        "selected_features": selected_features,
        "threshold": threshold,
        "feature_names": feature_names,
    }