"""
evaluate.py
===========
Loads a previously trained + saved global model (``outputs/models/global_model.pkl``,
produced by ``train.py``) and evaluates it on the shared global test set,
printing every required metric and regenerating the ROC/confusion-matrix
plots.

Usage
-----
    python evaluate.py
    python evaluate.py --model-path outputs/models/global_model.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

import config
from federated.dataset_partition import DatasetPartitioner
from utils.logger import get_logger
from utils.metrics import compute_classification_metrics
from utils.visualization import plot_confusion_matrix, plot_roc_curve

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved global federated model.")
    parser.add_argument(
        "--model-path", type=str,
        default=os.path.join(config.MODELS_DIR, "global_model.pkl"),
    )
    return parser.parse_args()


def weighted_predict_proba(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    total = None
    for client_id, booster in model_bundle["client_models"].items():
        weight = model_bundle["client_weights"].get(client_id, 0.0)
        if weight == 0.0:
            continue
        weighted = weight * booster.predict_proba(X)
        total = weighted if total is None else total + weighted
    return total


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No saved model found at {args.model_path}. Run train.py first."
        )

    with open(args.model_path, "rb") as f:
        model_bundle = pickle.load(f)

    logger.info("Loaded global model (%s) with %d client members",
                model_bundle["model_version"], len(model_bundle["client_models"]))

    partitioner = DatasetPartitioner()
    X_test, y_test, _ = partitioner.load_global_test()
    classes = model_bundle["classes"]

    y_proba = weighted_predict_proba(model_bundle, X_test)
    y_pred = np.array(classes)[np.argmax(y_proba, axis=1)]

    metrics = compute_classification_metrics(y_test, y_pred, y_proba, classes)
    logger.info("Evaluation results: %s", metrics.summary())
    print(json.dumps(metrics.to_dict(), indent=2))

    plot_confusion_matrix(
        np.array(metrics.confusion_matrix), classes=classes,
        filename="evaluate_confusion_matrix.png", title="Global Model Confusion Matrix (evaluate.py)",
    )
    plot_roc_curve(
        y_test, y_proba, classes=classes,
        filename="evaluate_roc_curve.png", title="Global Model ROC Curve (evaluate.py)",
    )

    metrics_path = os.path.join(config.METADATA_DIR, "evaluate_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Saved evaluation metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
