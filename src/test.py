"""
test.py — Evaluation and metrics for Waste Segregation models

Computes binary classification metrics including:
Accuracy, Precision, Recall, F1, Specificity, ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def probability_predictions(model, test_ds):
    """
    Get probability predictions for the positive class.

    Args:
        model: Trained Keras model
        test_ds: Test tf.data.Dataset

    Returns:
        1D array of probabilities for the positive class
    """
    probs = model.predict(test_ds, verbose=0)
    if probs.ndim == 2:
        return probs[:, 1]
    return probs.reshape(-1)


def binary_metrics(y_true, probabilities, model_name,
                   parameter_count, train_seconds):
    """
    Compute a full set of binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        probabilities: Predicted probabilities for class 1
        model_name: Name of the model
        parameter_count: Total parameter count of the model
        train_seconds: Training time in seconds

    Returns:
        dict with all metrics
    """
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1],
    ).ravel()

    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, predicted),
        'Precision': precision_score(y_true, predicted, zero_division=0),
        'Recall': recall_score(y_true, predicted, zero_division=0),
        'F1': f1_score(y_true, predicted, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        'ROC-AUC': roc_auc_score(y_true, probabilities),
        'Parameters': int(parameter_count),
        'Training Time (s)': train_seconds,
    }
