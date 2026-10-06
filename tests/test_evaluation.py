import numpy as np

from copd_forecasting.evaluation import (
    calculate_classification_metrics,
)


def test_classification_metrics():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            1,
        ]
    )

    y_prob = np.array(
        [
            0.1,
            0.7,
            0.8,
            0.9,
        ]
    )

    metrics = (
        calculate_classification_metrics(
            y_true,
            y_pred,
            y_prob,
        )
    )

    assert metrics["TP"] == 2
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0

    assert (
        metrics["Sensitivity"]
        == 1.0
    )

    assert (
        metrics["Specificity"]
        == 0.5
    )

    assert (
        metrics["Accuracy"]
        == 0.75
    )