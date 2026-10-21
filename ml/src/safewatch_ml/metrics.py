"""Threshold selection is restricted to validation subjects."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def select_threshold(labels, scores, minimum_specificity):
    """Maximize validation sensitivity subject to a specificity constraint.

    Ties prefer specificity, then proximity to 0.5. A threshold above every
    score ensures that a feasible all-negative operating point exists.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    candidates = np.unique(np.r_[0.0, 0.5, scores, np.nextafter(scores.max(), np.inf)])
    best = None
    for threshold in candidates:
        prediction = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp)
        if specificity + 1e-12 < minimum_specificity:
            continue
        key = (tp / (tp + fn), specificity, -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = key, float(threshold)
    return best[1]


def classification_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
