"""
config.py
=========
Single source of truth for every hyperparameter and path used across the
Federated Learning + Homomorphic Encryption + XGBoost pipeline.

Nothing in this project reads a hyperparameter from anywhere else - if you
need to change how the system behaves, change it here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
LOG_DIR: str = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR: str = os.path.join(OUTPUT_DIR, "models")
METADATA_DIR: str = os.path.join(OUTPUT_DIR, "metadata")

for _dir in (DATASET_DIR, LOG_DIR, OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, METADATA_DIR):
    os.makedirs(_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset configuration
# --------------------------------------------------------------------------- #
# The preprocessing team has already produced a fully preprocessed,
# feature-scaled, per-organization split of the CIC-IDS2018 flow dataset.
# We consume it as-is (LABEL_COLUMN holds an already-encoded integer class id).
LABEL_COLUMN: str = "Label"

# Pre-assigned client files (one organization = one CSV pair). This is the
# "preassigned" partition strategy and is the DEFAULT because the uploaded
# dataset already arrives split per organization.
CLIENT_FILES: Dict[int, Dict[str, str]] = {
    1: {
        "X": os.path.join(DATASET_DIR, "client_1_X.csv"),
        "y": os.path.join(DATASET_DIR, "client_1_y.csv"),
    },
    2: {
        "X": os.path.join(DATASET_DIR, "client_2_X.csv"),
        "y": os.path.join(DATASET_DIR, "client_2_y.csv"),
    },
    3: {
        "X": os.path.join(DATASET_DIR, "client_3_X.csv"),
        "y": os.path.join(DATASET_DIR, "client_3_y.csv"),
    },
}

# Global, held-out evaluation set (shared by all organizations to score the
# aggregated global model - it is NEVER used for local client training).
GLOBAL_TEST_X: str = os.path.join(DATASET_DIR, "X_test.csv")
GLOBAL_TEST_Y: str = os.path.join(DATASET_DIR, "y_test.csv")

# Partition strategy: "preassigned" uses CLIENT_FILES directly (default, and
# what matches the uploaded dataset). "iid" / "noniid" are also implemented
# in federated/dataset_partition.py in case a single combined CSV is used
# instead (e.g. for future datasets that are not pre-split per organization).
PARTITION_STRATEGY: str = "preassigned"  # "preassigned" | "iid" | "noniid"
COMBINED_DATASET_PATH: str = os.path.join(DATASET_DIR, "combined.csv")
NON_IID_DIRICHLET_ALPHA: float = 0.5  # lower = more label-skewed clients

# Each client carves out a local validation split from its own data (the
# uploaded dataset does not ship a per-client test split, only a shared
# global test set, so this is the "train-test split performed within the FL
# pipeline" required when one is absent).
LOCAL_VALIDATION_SPLIT: float = 0.15


# --------------------------------------------------------------------------- #
# Federated learning configuration
# --------------------------------------------------------------------------- #
NUM_CLIENTS: int = 3
COMMUNICATION_ROUNDS: int = 10
LOCAL_EPOCHS: int = 20          # boosting rounds (new trees) trained locally per FL round
PARTICIPATION_RATIO: float = 1.0  # fraction of clients sampled each round (0 < r <= 1)
RANDOM_SEED: int = 42


# --------------------------------------------------------------------------- #
# XGBoost configuration
# --------------------------------------------------------------------------- #
@dataclass
class XGBConfig:
    learning_rate: float = 0.15
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_jobs: int = -1
    tree_method: str = "hist"
    use_class_balancing: bool = True  # sample_weight (multiclass) / scale_pos_weight (binary)


XGB_CONFIG: XGBConfig = XGBConfig()


# --------------------------------------------------------------------------- #
# Homomorphic Encryption (CKKS via TenSEAL) configuration
# --------------------------------------------------------------------------- #
@dataclass
class HEConfig:
    scheme: str = "CKKS"
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    global_scale_power: int = 40  # global_scale = 2 ** global_scale_power
    validation_atol: float = 1e-2  # tolerance for decrypted-vs-plaintext comparison


HE_CONFIG: HEConfig = HEConfig()


# --------------------------------------------------------------------------- #
# Global ensemble aggregation
# --------------------------------------------------------------------------- #
# Weighting scheme used to combine per-client local models into the global
# model each round. "sample_count" (FedAvg-style, weight proportional to the
# number of local training samples) is the default.
AGGREGATION_WEIGHTING: str = "sample_count"  # "sample_count" | "uniform"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = "INFO"
MAIN_LOG_FILE: str = os.path.join(LOG_DIR, "main.log")
TRAINING_LOG_FILE: str = os.path.join(LOG_DIR, "training.log")
ENCRYPTION_LOG_FILE: str = os.path.join(LOG_DIR, "encryption.log")
AGGREGATION_LOG_FILE: str = os.path.join(LOG_DIR, "aggregation.log")
