import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Calculate classification and discrimination metrics.

    Parameters
    ----------
    y_true
        True binary labels.

    y_pred
        Predicted binary labels.

    y_prob
        Predicted probability of the positive class.

    Returns
    -------
    dict
        Classification and discrimination metrics.
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    lr_positive = (
        sensitivity
        / (1 - specificity)
        if (1 - specificity) > 0
        else np.inf
    )

    lr_negative = (
        (1 - sensitivity)
        / specificity
        if specificity > 0
        else np.inf
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    auc_roc = roc_auc_score(
        y_true,
        y_prob,
    )

    auc_pr = average_precision_score(
        y_true,
        y_prob,
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "LR+": float(lr_positive),
        "LR-": float(lr_negative),
        "Accuracy": float(accuracy),
        "F1": float(f1),
        "Balanced_Accuracy": float(
            balanced_accuracy
        ),
        "AUC_ROC": float(auc_roc),
        "AUC_PR": float(auc_pr),
    }