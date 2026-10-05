import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from lightgbm import (
    LGBMClassifier,
    early_stopping as lgbm_early_stopping,
)

from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Final hyperparameter configurations
# ---------------------------------------------------------------------

OPTIMAL_PARAMS = {
    "sliding": {
        "LightGBM": {
            "a": {
                "max_depth": -1,
                "num_leaves": 3,
                "min_child_samples": 8,
                "learning_rate": 0.13,
            },
            "b": {
                "max_depth": -1,
                "num_leaves": 3,
                "min_child_samples": 100,
                "learning_rate": 0.13,
            },
        },
        "CatBoost": {
            "a": {
                "depth": 2,
                "l2_leaf_reg": 9,
                "learning_rate": 0.05,
            },
            "b": {
                "depth": 2,
                "l2_leaf_reg": 11,
                "learning_rate": 0.1,
            },
        },
        "MLP": {
            "a": {
                "hidden_layer_sizes": (
                    128,
                    64,
                ),
                "alpha": 1e-3,
                "learning_rate_init": 0.01,
            },
            "b": {
                "hidden_layer_sizes": (
                    256,
                    128,
                    64,
                ),
                "alpha": 1e-4,
                "learning_rate_init": 0.005,
            },
        },
    },

    "expanding": {
        "LightGBM": {
            "a": {
                "max_depth": 3,
                "num_leaves": 8,
                "min_child_samples": 20,
                "learning_rate": 0.1,
            },
            "b": {
                "max_depth": -1,
                "num_leaves": 2,
                "min_child_samples": 100,
                "learning_rate": 0.25,
            },
        },
        "CatBoost": {
            "a": {
                "depth": 2,
                "l2_leaf_reg": 5,
                "learning_rate": 0.05,
            },
            "b": {
                "depth": 2,
                "l2_leaf_reg": 12,
                "learning_rate": 0.2,
            },
        },
        "MLP": {
            "a": {
                "hidden_layer_sizes": (
                    64,
                    32,
                ),
                "alpha": 1e-6,
                "learning_rate_init": 0.005,
            },
            "b": {
                "hidden_layer_sizes": (
                    256,
                    128,
                    64,
                ),
                "alpha": 1e-5,
                "learning_rate_init": 0.001,
            },
        },
    },
}


# ---------------------------------------------------------------------
# Temporal cross-validation evaluation
# ---------------------------------------------------------------------

def evaluate_temporal_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
    params: dict,
) -> dict:
    """
    Evaluate one model configuration using chronological
    cross-validation folds.

    Parameters
    ----------
    model_name
        Name of the model:
        "LightGBM", "CatBoost" or "MLP".

    X
        Predictor matrix corresponding to the development
        period.

    y
        Binary response corresponding to the development
        period.

    splits
        Chronological train-validation index pairs.

    params
        Hyperparameters for the selected model.

    Returns
    -------
    dict
        AUC-ROC values for every fold and their mean.
    """
    fold_aucs = []

    for (
        train_idx,
        validation_idx,
    ) in splits:
        X_train = X[
            train_idx
        ]

        X_validation = X[
            validation_idx
        ]

        y_train = y[
            train_idx
        ]

        y_validation = y[
            validation_idx
        ]

        # -------------------------------------------------------------
        # LightGBM
        # -------------------------------------------------------------

        if model_name == "LightGBM":
            feature_names = [
                f"feature_{i}"
                for i in range(
                    X.shape[1]
                )
            ]

            X_train_model = pd.DataFrame(
                X_train,
                columns=feature_names,
            )

            X_validation_model = pd.DataFrame(
                X_validation,
                columns=feature_names,
            )

            model = LGBMClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
                **params,
            )

            model.fit(
                X_train_model,
                y_train,
                eval_set=[
                    (
                        X_validation_model,
                        y_validation,
                    )
                ],
                callbacks=[
                    lgbm_early_stopping(
                        30,
                        verbose=False,
                    )
                ],
            )

            prediction_data = (
                X_validation_model
            )

        # -------------------------------------------------------------
        # CatBoost
        # -------------------------------------------------------------

        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                iterations=400,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
                **params,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(
                    X_validation,
                    y_validation,
                ),
                early_stopping_rounds=30,
            )

            prediction_data = (
                X_validation
            )

        # -------------------------------------------------------------
        # MLP
        # -------------------------------------------------------------

        elif model_name == "MLP":
            model = MLPClassifier(
                activation="relu",
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
                **params,
            )

            model.fit(
                X_train,
                y_train,
            )

            prediction_data = (
                X_validation
            )

        else:
            raise ValueError(
                f"Unknown model: "
                f"{model_name}"
            )

        probabilities = (
            model.predict_proba(
                prediction_data
            )[:, 1]
        )

        auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        fold_aucs.append(
            float(auc)
        )

    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(
            np.mean(
                fold_aucs
            )
        ),
    }


# ---------------------------------------------------------------------
# Final model training
# ---------------------------------------------------------------------

def train_final_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
):
    """
    Train a final model using the complete development
    period (2010-2022).

    The trained model can subsequently be evaluated on the
    independent 2023 test set.
    """

    if model_name == "LightGBM":
        feature_names = [
            f"feature_{i}"
            for i in range(
                X_train.shape[1]
            )
        ]

        X_train_model = pd.DataFrame(
            X_train,
            columns=feature_names,
        )

        model = LGBMClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            **params,
        )

        model.fit(
            X_train_model,
            y_train,
        )

    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=400,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
            **params,
        )

        model.fit(
            X_train,
            y_train,
        )

    elif model_name == "MLP":
        model = MLPClassifier(
            activation="relu",
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            **params,
        )

        model.fit(
            X_train,
            y_train,
        )

    else:
        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )

    return model