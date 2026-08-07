"""
federated/trainer.py
=====================
Top-level orchestrator: loads data, builds clients + server, runs
``config.COMMUNICATION_ROUNDS`` rounds, and returns the full round-by-round
history for ``train.py`` / the notebook to persist, plot and report on.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List

import numpy as np

import config
from federated.client import FederatedClient
from federated.dataset_partition import DatasetPartitioner
from federated.server import FederatedServer, RoundResult
from utils.logger import get_training_logger

logger = get_training_logger("federated.trainer")


class FederatedTrainer:
    def __init__(self, seed: int = config.RANDOM_SEED) -> None:
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.partitioner = DatasetPartitioner()
        self.clients: Dict[int, FederatedClient] = {}
        self.server: FederatedServer | None = None
        self.global_classes: List[int] = []

    def setup(self) -> None:
        logger.info("Loading and partitioning dataset (strategy=%s)...", config.PARTITION_STRATEGY)
        client_datasets, global_classes = self.partitioner.load()
        X_test, y_test, test_features = self.partitioner.load_global_test()

        self.global_classes = global_classes
        self.clients = {
            cid: FederatedClient(client_id=cid, dataset=ds, global_classes=global_classes)
            for cid, ds in client_datasets.items()
        }
        self.server = FederatedServer(
            clients=self.clients,
            global_classes=global_classes,
            X_test=X_test,
            y_test=y_test,
            seed=self.seed,
        )
        logger.info(
            "Setup complete | clients=%d | classes=%d | global_test_samples=%d",
            len(self.clients), len(global_classes), len(y_test),
        )

    def run(
        self,
        rounds: int = config.COMMUNICATION_ROUNDS,
        local_epochs: int = config.LOCAL_EPOCHS,
    ) -> List[RoundResult]:
        if self.server is None:
            self.setup()

        history: List[RoundResult] = []
        for round_num in range(1, rounds + 1):
            logger.info("=" * 70)
            logger.info("COMMUNICATION ROUND %d / %d", round_num, rounds)
            logger.info("=" * 70)

            result = self.server.run_round(round_num=round_num, local_epochs=local_epochs)
            broadcast = self.server.broadcast(result)

            logger.info(
                "Round %d broadcast -> model_version=%s validation_passed=%s",
                round_num, broadcast.model_version, broadcast.validation_passed,
            )
            history.append(result)

        return history
