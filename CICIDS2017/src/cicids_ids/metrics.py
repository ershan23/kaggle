"""Evaluation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_metrics(y_true, y_pred) -> dict:
    """Compute IDS binary metrics with attack as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def multiclass_report(y_true, y_pred) -> dict:
    """Return a sklearn classification report as a dict."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def metrics_table(records: list[dict]) -> pd.DataFrame:
    """Build a sorted metrics table for model comparison."""
    table = pd.DataFrame(records)
    sort_col = "f1_attack" if "f1_attack" in table.columns else "accuracy"
    return table.sort_values(sort_col, ascending=False).reset_index(drop=True)
