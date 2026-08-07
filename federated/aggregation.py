"""
federated/aggregation.py
=========================
Builds and maintains the **global XGBoost model** from participating
clients' locally-trained boosters.

Why a weighted ensemble instead of literal weight-averaging
-------------------------------------------------------------
Unlike neural-network weights, gradient-boosted tree leaf values cannot be
meaningfully averaged element-wise (trees from different clients have
different structures). We therefore use the standard, well-established
strategy for federated tree ensembles ("bagging aggregation"): the global
model is a **sample-count-weighted soft-voting ensemble** of every
participating client's booster,

    P_global(y | x) = sum_c  w_c * P_client_c(y | x),   sum_c w_c = 1

which is exactly the FedAvg weighting rule applied at the level of
prediction probabilities rather than raw weights, and is well-defined for
any number of trees/any tree structure. The numeric aggregation that *is*
homomorphically encrypted (feature-importance + performance-summary
vectors, see ``federated/client.py`` and ``encryption/secure_aggregation.py``)
is what is protected under CKKS and cryptographically validated each round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from federated.client import FederatedClient
from utils.logger import get_aggregation_logger

logger = get_aggregation_logger("federated.aggregation")


@dataclass
class GlobalModel:
    """Weighted ensemble of participating clients' local XGBoost models."""

    classes: List[int]
    client_weights: Dict[int, float] = field(default_factory=dict)
    _clients: Dict[int, FederatedClient] = field(default_factory=dict, repr=False)
    version: int = 0

    def update(self, participating_clients: Dict[int, FederatedClient], weights: Dict[int, float]) -> None:
        self._clients = participating_clients
        self.client_weights = weights
        self.version += 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._clients:
            raise RuntimeError("GlobalModel has not been updated with any client yet.")
        total = None
        for client_id, client in self._clients.items():
            weighted = self.client_weights[client_id] * client.predict_proba(X)
            total = weighted if total is None else total + weighted
        return total

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        class_arr = np.array(self.classes)
        return class_arr[np.argmax(proba, axis=1)]

    @property
    def model_version_tag(self) -> str:
        return f"global-v{self.version}"


class FedAvgAggregator:
    """Computes sample-count-proportional (or uniform) aggregation weights."""

    def __init__(self, weighting: str = "sample_count") -> None:
        self.weighting = weighting

    def compute_weights(self, client_sample_counts: Dict[int, int]) -> Dict[int, float]:
        client_ids = sorted(client_sample_counts.keys())
        if self.weighting == "uniform":
            weight = 1.0 / len(client_ids)
            weights = {c: weight for c in client_ids}
        else:  # "sample_count" - standard FedAvg weighting
            total = sum(client_sample_counts.values())
            weights = {c: client_sample_counts[c] / total for c in client_ids}

        logger.info("Computed aggregation weights (%s): %s", self.weighting, {
            k: round(v, 4) for k, v in weights.items()
        })
        return weights

    @staticmethod
    def aggregate_plaintext_vectors(
        vectors: Dict[int, np.ndarray], weights: Dict[int, float]
    ) -> np.ndarray:
        """Plaintext weighted sum - used only as the expected value to
        validate the homomorphically-decrypted aggregate against."""
        client_ids = sorted(vectors.keys())
        total = sum(weights[c] for c in client_ids)
        return sum((weights[c] / total) * np.asarray(vectors[c]) for c in client_ids)
