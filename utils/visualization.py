"""
utils/visualization.py
=======================
All plots required by the project spec:
  1. Communication Round vs Accuracy
  2. Communication Round vs F1 Score
  3. Communication Round vs Loss
  4. Client-wise Performance Comparison
  5. ROC Curve
  6. Confusion Matrix

Every function saves its figure to ``config.PLOTS_DIR`` and returns the path
written, so callers (train.py / evaluate.py / the notebook) can log or
display it without duplicating file-naming logic.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(config.PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_round_metric(
    rounds: Sequence[int],
    global_values: Sequence[float],
    metric_name: str,
    filename: str,
    client_values: Dict[int, Sequence[float]] | None = None,
) -> str:
    """Communication Round vs <metric_name>, global line plus optional per-client lines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, global_values, marker="o", linewidth=2.5, label="Global model", color="#1f77b4")

    if client_values:
        for client_id, values in client_values.items():
            ax.plot(rounds, values, marker=".", linestyle="--", alpha=0.6, label=f"Client {client_id}")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Communication Round vs {metric_name}")
    ax.set_xticks(list(rounds))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_client_comparison(
    client_ids: Sequence[int],
    metric_values: Sequence[float],
    metric_name: str,
    round_num: int,
    filename: str,
) -> str:
    """Bar chart comparing clients on a single metric for a given round."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_ids)))
    bars = ax.bar([f"Client {c}" for c in client_ids], metric_values, color=colors)
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Client-wise {metric_name} Comparison (Round {round_num})")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[int],
    filename: str,
    title: str = "ROC Curve (One-vs-Rest, macro average)",
) -> str:
    """Multi-class (or binary) ROC curve, one-vs-rest, plus macro-average."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="#d62728", linewidth=2)
    else:
        y_bin = label_binarize(y_true, classes=classes)
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        plotted = 0
        for i, cls in enumerate(classes):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            plotted += 1
            ax.plot(fpr, tpr, alpha=0.25, linewidth=1)
        if plotted > 0:
            mean_tpr /= plotted
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color="navy", linewidth=2.5,
                     label=f"Macro-average (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    filename: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> str:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_display, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90 if len(classes) > 8 else 0)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm_display.max() / 2.0 if cm_display.size else 0
    if len(classes) <= 15:
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                ax.text(j, i, format(cm_display[i, j], fmt),
                         ha="center", va="center", fontsize=7,
                         color="white" if cm_display[i, j] > threshold else "black")
    return _save(fig, filename)
