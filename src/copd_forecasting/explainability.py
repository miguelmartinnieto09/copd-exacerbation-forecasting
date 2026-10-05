import numpy as np
import pandas as pd
import shap


RANDOM_STATE = 42


def _extract_positive_class_shap_values(
    shap_values,
) -> np.ndarray:
    """
    Convert different SHAP output formats to a 2D array
    corresponding to the positive class.
    """
    if isinstance(
        shap_values,
        list,
    ):
        values = np.asarray(
            shap_values[1]
        )

    else:
        values = np.asarray(
            shap_values
        )

    if values.ndim == 3:
        values = values[
            :,
            :,
            1,
        ]

    return values


def compute_shap_values(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    background_size: int = 100,
    kernel_nsamples: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Calculate SHAP values on the independent test set.

    TreeExplainer is used for LightGBM and CatBoost.
    KernelExplainer is used for the MLP.

    SHAP contributions correspond to the positive class.
    """
    background = shap.sample(
        X_train,
        min(
            background_size,
            X_train.shape[0],
        ),
        random_state=random_state,
    )

    # -------------------------------------------------------------
    # Tree-based models
    # -------------------------------------------------------------

    if model_name in [
        "LightGBM",
        "CatBoost",
    ]:
        if model_name == "LightGBM":
            feature_names = [
                f"feature_{i}"
                for i in range(
                    X_train.shape[1]
                )
            ]

            background_model = (
                pd.DataFrame(
                    background,
                    columns=feature_names,
                )
            )

            X_test_model = (
                pd.DataFrame(
                    X_test,
                    columns=feature_names,
                )
            )

        else:
            background_model = background
            X_test_model = X_test

        explainer = shap.TreeExplainer(
            model,
            data=background_model,
            model_output="probability",
        )

        shap_values = (
            explainer.shap_values(
                X_test_model
            )
        )

    # -------------------------------------------------------------
    # Multilayer perceptron
    # -------------------------------------------------------------

    elif model_name == "MLP":
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background,
        )

        shap_values = (
            explainer.shap_values(
                X_test,
                nsamples=kernel_nsamples,
            )
        )

    else:
        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )

    shap_values = (
        _extract_positive_class_shap_values(
            shap_values
        )
    )

    return {
        "shap_values": shap_values,
        "explainer": explainer,
    }