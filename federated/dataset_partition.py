"""
federated/dataset_partition.py
===============================
Loads the already-preprocessed dataset and produces the per-organization
splits used by the federated pipeline.

IMPORTANT: this module performs NO preprocessing (no scaling, no encoding,
no imputation, no balancing). The uploaded CSVs are already fully
preprocessed by the data team - this module only reads them, optionally
partitions a single combined file across simulated clients, and carves out
each client's local train/validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils.logger import get_training_logger

logger = get_training_logger("federated.dataset_partition")


@dataclass
class ClientDataset:
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]

    @property
    def n_train_samples(self) -> int:
        return len(self.y_train)


class DatasetPartitioner:
    """
    Produces ``{client_id: ClientDataset}`` plus the shared global test set,
    according to ``config.PARTITION_STRATEGY``:

      * "preassigned" (default): load ``config.CLIENT_FILES`` directly - this
        matches the dataset as delivered (already split per organization).
      * "iid": load a single combined CSV and split it uniformly at random
        across ``config.NUM_CLIENTS`` clients.
      * "noniid": load a single combined CSV and split it with a
        Dirichlet-distribution label skew across clients (label heterogeneity
        simulation), controlled by ``config.NON_IID_DIRICHLET_ALPHA``.
    """

    def __init__(
        self,
        label_column: str = config.LABEL_COLUMN,
        val_split: float = config.LOCAL_VALIDATION_SPLIT,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.label_column = label_column
        self.val_split = val_split
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> Tuple[Dict[int, ClientDataset], List[int]]:
        """Returns ``(client_datasets, sorted_global_class_list)``."""
        strategy = config.PARTITION_STRATEGY
        if strategy == "preassigned":
            raw_clients = self._load_preassigned()
        elif strategy == "iid":
            raw_clients = self._load_and_partition(strategy="iid")
        elif strategy == "noniid":
            raw_clients = self._load_and_partition(strategy="noniid")
        else:
            raise ValueError(f"Unknown PARTITION_STRATEGY: {strategy!r}")

        client_datasets: Dict[int, ClientDataset] = {}
        all_labels: set = set()
        for client_id, (X_df, y_series) in raw_clients.items():
            feature_names = list(X_df.columns)
            X = X_df.to_numpy(dtype=np.float64)
            y = y_series.to_numpy().ravel()
            all_labels.update(np.unique(y).tolist())

            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.val_split,
                random_state=self.seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            client_datasets[client_id] = ClientDataset(
                client_id=client_id,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                feature_names=feature_names,
            )
            logger.info(
                "Client %d | train=%d val=%d classes=%d",
                client_id, len(y_train), len(y_val), len(np.unique(y)),
            )

        return client_datasets, sorted(int(c) for c in all_labels)

    def load_global_test(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_df = pd.read_csv(config.GLOBAL_TEST_X)
        y_df = pd.read_csv(config.GLOBAL_TEST_Y)
        feature_names = list(X_df.columns)
        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy().ravel()
        logger.info("Loaded global held-out test set | samples=%d classes=%d", len(y), len(np.unique(y)))
        return X, y, feature_names

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #
    def _load_preassigned(self) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        clients: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for client_id, paths in config.CLIENT_FILES.items():
            X_df = pd.read_csv(paths["X"])
            y_df = pd.read_csv(paths["y"])
            if self.label_column in y_df.columns:
                y_df = y_df[[self.label_column]]
            clients[client_id] = (X_df, y_df)
        return clients

    def _load_and_partition(self, strategy: str) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        df = pd.read_csv(config.COMBINED_DATASET_PATH)
        y_full = df[self.label_column]
        X_full = df.drop(columns=[self.label_column])
        num_clients = config.NUM_CLIENTS
        rng = np.random.default_rng(self.seed)

        if strategy == "iid":
            indices = rng.permutation(len(df))
            splits = np.array_split(indices, num_clients)
        else:  # "noniid" - Dirichlet label-skew partition
            splits = self._dirichlet_partition(y_full.to_numpy(), num_clients, rng)

        clients: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for i, idx in enumerate(splits, start=1):
            clients[i] = (
                X_full.iloc[idx].reset_index(drop=True),
                y_full.iloc[idx].to_frame(name=self.label_column).reset_index(drop=True),
            )
        return clients

    @staticmethod
    def _dirichlet_partition(
        y: np.ndarray, num_clients: int, rng: np.random.Generator
    ) -> List[np.ndarray]:
        """Standard Dirichlet(alpha) label-skew non-IID partition."""
        alpha = config.NON_IID_DIRICHLET_ALPHA
        classes = np.unique(y)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(cls_idx, proportions)):
                client_indices[client_id].extend(split.tolist())

        return [np.array(sorted(idx)) for idx in client_indices]
