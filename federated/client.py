"""
federated/client.py
====================
Represents a single organization ("Organization A/B/C"). Each client:

  * keeps its own data local at all times (only ``ClientDataset`` objects
    ever touch raw feature values, and those never leave this object)
  * trains its own XGBoost model for a configurable number of local epochs
    (boosting rounds), warm-started from its own model of the previous round
  * evaluates locally
  * produces a compact, numeric "local update" (feature-importance vector +
    performance stats) that is what actually gets homomorphically encrypted
    and shared with the server - raw traffic and raw model internals never
    leave the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

import config
from federated.dataset_partition import ClientDataset
from models.xgboost_model import XGBoostModel
from utils.logger import get_training_logger
from utils.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass
class LocalUpdate:
    """What a client hands to the server each round (pre-encryption)."""

    client_id: int
    round_num: int
    n_samples: int
    update_vector: np.ndarray  # [feature_importance (P dims), accuracy, f1_macro]
    metrics: ClassificationMetrics


class FederatedClient:
    """One simulated organization participating in federated training."""

    def __init__(self, client_id: int, dataset: ClientDataset, global_classes: List[int]) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.global_classes = global_classes
        self.logger = get_training_logger(f"federated.client.{client_id}")
        self.model = XGBoostModel(
            num_classes=len(global_classes),
            feature_names=dataset.feature_names,
        )
        self.round_history: List[LocalUpdate] = []

    def train_local(self, round_num: int, local_epochs: int = config.LOCAL_EPOCHS) -> LocalUpdate:
        """Run local training for this round and package the resulting update."""
        self.logger.info(
            "Round %d | client %d | local training start (local_epochs=%d, samples=%d)",
            round_num, self.client_id, local_epochs, self.dataset.n_train_samples,
        )
        self.model.fit(
            self.dataset.X_train, self.dataset.y_train,
            n_estimators=local_epochs, warm_start=True,
        )

        metrics = self.evaluate_local()
        self.logger.info(
            "Round %d | client %d | local eval -> %s",
            round_num, self.client_id, metrics.summary(),
        )

        importance = self.model.get_feature_importance()
        update_vector = np.concatenate([importance, [metrics.accuracy, metrics.f1_macro]])

        update = LocalUpdate(
            client_id=self.client_id,
            round_num=round_num,
            n_samples=self.dataset.n_train_samples,
            update_vector=update_vector,
            metrics=metrics,
        )
        self.round_history.append(update)
        return update

    def evaluate_local(self) -> ClassificationMetrics:
        y_pred = self.model.predict(self.dataset.X_val)
        y_proba = self.model.predict_proba(self.dataset.X_val)
        return compute_classification_metrics(
            self.dataset.y_val, y_pred, y_proba, self.global_classes
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Used by the server's weighted-ensemble global model."""
        return self.model.predict_proba(X)
