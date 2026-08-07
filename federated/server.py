"""
federated/server.py
====================
The coordinator. Per communication round it:

  1. Selects a subset of clients (``config.PARTICIPATION_RATIO``)
  2. Triggers local training on each selected client
  3. Encrypts + homomorphically aggregates their update vectors
     (``encryption/secure_aggregation.py``)
  4. Decrypts + validates the aggregate
  5. Rebuilds the global model (weighted ensemble, ``federated/aggregation.py``)
  6. Evaluates the global model on the held-out global test set
  7. "Broadcasts" a global-model snapshot back to clients (metadata only -
     see module docstring in federated/aggregation.py for why tree-based
     global weights aren't literally shipped back)
  8. Emits blockchain-ready metadata for the round
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import config
from encryption.secure_aggregation import BlockchainMetadataRecord, SecureAggregationPipeline
from federated.aggregation import FedAvgAggregator, GlobalModel
from federated.client import FederatedClient, LocalUpdate
from utils.logger import get_aggregation_logger
from utils.metrics import ClassificationMetrics, compute_classification_metrics

logger = get_aggregation_logger("federated.server")


@dataclass
class GlobalBroadcast:
    """What clients conceptually receive back at the end of a round."""

    round_num: int
    model_version: str
    participating_clients: List[int]
    aggregated_feature_importance: np.ndarray
    validation_passed: bool


@dataclass
class RoundResult:
    round_num: int
    participating_clients: List[int]
    local_updates: Dict[int, LocalUpdate]
    global_metrics: ClassificationMetrics
    validation_passed: bool
    validation_max_error: float
    blockchain_records: List[BlockchainMetadataRecord]
    model_version: str
    global_proba: Optional[np.ndarray] = field(default=None, repr=False)


class FederatedServer:
    def __init__(
        self,
        clients: Dict[int, FederatedClient],
        global_classes: List[int],
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.clients = clients
        self.global_classes = global_classes
        self.X_test = X_test
        self.y_test = y_test
        self._rng = random.Random(seed)

        self.fedavg = FedAvgAggregator(weighting=config.AGGREGATION_WEIGHTING)
        self.global_model = GlobalModel(classes=global_classes)
        self.secure_aggregation = SecureAggregationPipeline()

        self.history: List[RoundResult] = []

    def _select_participants(self) -> List[int]:
        client_ids = sorted(self.clients.keys())
        k = max(1, round(config.PARTICIPATION_RATIO * len(client_ids)))
        selected = sorted(self._rng.sample(client_ids, k))
        logger.info("Selected participants for this round: %s", selected)
        return selected

    def run_round(self, round_num: int, local_epochs: int = config.LOCAL_EPOCHS) -> RoundResult:
        participants = self._select_participants()

        # 1) Local training on each participating client.
        local_updates: Dict[int, LocalUpdate] = {}
        for client_id in participants:
            local_updates[client_id] = self.clients[client_id].train_local(round_num, local_epochs)

        # 2) FedAvg-style weighting (proportional to local sample counts).
        sample_counts = {cid: upd.n_samples for cid, upd in local_updates.items()}
        weights = self.fedavg.compute_weights(sample_counts)

        # 3) Secure aggregation: encrypt -> homomorphic weighted sum -> decrypt -> validate.
        update_vectors = {cid: upd.update_vector for cid, upd in local_updates.items()}
        agg_result = self.secure_aggregation.run(
            round_num=round_num,
            client_vectors=update_vectors,
            client_weights=weights,
            model_version=f"global-v{self.global_model.version + 1}",
        )

        # 4) Rebuild the global model as the weighted ensemble of participants.
        participating_clients = {cid: self.clients[cid] for cid in participants}
        self.global_model.update(participating_clients, weights)

        # 5) Evaluate the global model on the shared held-out test set.
        global_proba = self.global_model.predict_proba(self.X_test)
        global_pred = self.global_model.classes and np.array(self.global_model.classes)[
            np.argmax(global_proba, axis=1)
        ]
        global_metrics = compute_classification_metrics(
            self.y_test, global_pred, global_proba, self.global_classes
        )
        logger.info("Round %d | GLOBAL model eval -> %s", round_num, global_metrics.summary())

        result = RoundResult(
            round_num=round_num,
            participating_clients=participants,
            local_updates=local_updates,
            global_metrics=global_metrics,
            validation_passed=agg_result.validation.is_valid,
            validation_max_error=agg_result.validation.max_abs_error,
            blockchain_records=agg_result.blockchain_records,
            model_version=self.global_model.model_version_tag,
            global_proba=global_proba,
        )
        self.history.append(result)
        return result

    def broadcast(self, result: RoundResult) -> GlobalBroadcast:
        """Metadata-only broadcast back to clients (see aggregation.py docstring)."""
        weighted_importance = None
        for cid in result.participating_clients:
            imp = result.local_updates[cid].update_vector[:-2]  # strip trailing acc/f1 stats
            w = self.global_model.client_weights[cid]
            weighted_importance = w * imp if weighted_importance is None else weighted_importance + w * imp

        return GlobalBroadcast(
            round_num=result.round_num,
            model_version=result.model_version,
            participating_clients=result.participating_clients,
            aggregated_feature_importance=weighted_importance,
            validation_passed=result.validation_passed,
        )
