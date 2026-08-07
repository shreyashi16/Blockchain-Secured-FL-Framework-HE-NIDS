"""
utils/metrics.py
=================
Centralised evaluation logic shared by client-side local evaluation and
server-side global evaluation, so every reported number in the pipeline is
computed the exact same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for a single evaluation's results."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: Optional[float]
    loss: float
    confusion_matrix: List[List[int]]
    n_samples: int
    num_classes: int

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "roc_auc": self.roc_auc,
            "loss": self.loss,
            "confusion_matrix": self.confusion_matrix,
            "n_samples": self.n_samples,
            "num_classes": self.num_classes,
        }

    def summary(self) -> str:
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        return (
            f"acc={self.accuracy:.4f} "
            f"prec={self.precision_macro:.4f} "
            f"rec={self.recall_macro:.4f} "
            f"f1={self.f1_macro:.4f} "
            f"roc_auc={auc_str} "
            f"loss={self.loss:.4f}"
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    all_classes: List[int],
) -> ClassificationMetrics:
    """
    Compute the full metrics suite required by the project spec: accuracy,
    macro precision/recall/F1, ROC-AUC, log-loss and the confusion matrix.

    ``all_classes`` should be the full label set known globally (not just
    the labels present in this particular batch), so per-round confusion
    matrices stay a consistent shape and ROC-AUC does not blow up on a
    client that happens not to see every class in a given split.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    num_classes = len(all_classes)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(
        y_true, y_pred, labels=all_classes, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, labels=all_classes, average="macro", zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    try:
        loss = log_loss(y_true, y_proba, labels=all_classes)
    except ValueError:
        loss = float("nan")

    roc_auc: Optional[float]
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_true, y_proba, labels=all_classes, multi_class="ovr", average="macro"
            )
    except ValueError:
        # Happens if a split does not contain every class - not fatal.
        roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        loss=float(loss),
        confusion_matrix=cm.tolist(),
        n_samples=int(len(y_true)),
        num_classes=num_classes,
    )
