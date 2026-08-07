"""
train.py
========
Entry point: Dataset Upload -> Federated Learning -> Homomorphic Encryption
-> Secure Aggregation -> Global XGBoost Model.

Usage
-----
    python train.py
    python train.py --rounds 15 --local-epochs 30 --participation-ratio 0.67
    python train.py --partition noniid

Produces
--------
    logs/*.log                          structured training/encryption/aggregation logs
    outputs/plots/*.png                 all required visualizations
    outputs/metadata/round_history.json round-by-round metrics + blockchain metadata
    outputs/models/global_model.json    persisted global ensemble (client boosters + weights)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict

import numpy as np

import config
from federated.trainer import FederatedTrainer
from utils.logger import get_logger
from utils.visualization import (
    plot_client_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_round_metric,
)

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated NIDS training with HE-secured aggregation.")
    parser.add_argument("--rounds", type=int, default=config.COMMUNICATION_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    parser.add_argument("--participation-ratio", type=float, default=config.PARTICIPATION_RATIO)
    parser.add_argument(
        "--partition", type=str, default=config.PARTITION_STRATEGY,
        choices=["preassigned", "iid", "noniid"],
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def _round_history_to_json(history) -> list:
    payload = []
    for r in history:
        payload.append({
            "round_num": r.round_num,
            "participating_clients": r.participating_clients,
            "model_version": r.model_version,
            "validation_passed": r.validation_passed,
            "validation_max_error": r.validation_max_error,
            "global_metrics": r.global_metrics.to_dict(),
            "local_metrics": {
                cid: upd.metrics.to_dict() for cid, upd in r.local_updates.items()
            },
            "blockchain_records": [asdict(rec) for rec in r.blockchain_records],
        })
    return payload


def main() -> None:
    args = parse_args()
    config.COMMUNICATION_ROUNDS = args.rounds
    config.LOCAL_EPOCHS = args.local_epochs
    config.PARTICIPATION_RATIO = args.participation_ratio
    config.PARTITION_STRATEGY = args.partition
    config.RANDOM_SEED = args.seed

    logger.info("Starting federated training | rounds=%d local_epochs=%d participation=%.2f partition=%s",
                args.rounds, args.local_epochs, args.participation_ratio, args.partition)

    start = time.time()
    trainer = FederatedTrainer(seed=args.seed)
    trainer.setup()
    history = trainer.run(rounds=args.rounds, local_epochs=args.local_epochs)
    elapsed = time.time() - start
    logger.info("Federated training complete in %.1fs", elapsed)

    # ---------------------------------------------------------------- #
    # Persist round history + blockchain-ready metadata
    # ---------------------------------------------------------------- #
    history_json = _round_history_to_json(history)
    history_path = os.path.join(config.METADATA_DIR, "round_history.json")
    with open(history_path, "w") as f:
        json.dump(history_json, f, indent=2)
    logger.info("Saved round history + blockchain metadata -> %s", history_path)

    # ---------------------------------------------------------------- #
    # Persist the final global model (client boosters + ensemble weights)
    # ---------------------------------------------------------------- #
    model_bundle = {
        "classes": trainer.global_classes,
        "client_weights": trainer.server.global_model.client_weights,
        "client_models": {
            cid: client.model.booster_estimator for cid, client in trainer.clients.items()
        },
        "model_version": trainer.server.global_model.model_version_tag,
    }
    model_path = os.path.join(config.MODELS_DIR, "global_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info("Saved global ensemble model -> %s", model_path)

    # ---------------------------------------------------------------- #
    # Visualizations
    # ---------------------------------------------------------------- #
    rounds = [r.round_num for r in history]
    client_ids = sorted(trainer.clients.keys())

    global_acc = [r.global_metrics.accuracy for r in history]
    global_f1 = [r.global_metrics.f1_macro for r in history]
    global_loss = [r.global_metrics.loss for r in history]

    client_acc = {cid: [] for cid in client_ids}
    client_f1 = {cid: [] for cid in client_ids}
    for r in history:
        for cid in client_ids:
            upd = r.local_updates.get(cid)
            client_acc[cid].append(upd.metrics.accuracy if upd else np.nan)
            client_f1[cid].append(upd.metrics.f1_macro if upd else np.nan)

    plot_round_metric(rounds, global_acc, "Accuracy", "round_vs_accuracy.png", client_acc)
    plot_round_metric(rounds, global_f1, "F1 Score (macro)", "round_vs_f1.png", client_f1)
    plot_round_metric(rounds, global_loss, "Loss (log-loss)", "round_vs_loss.png")

    last_round = history[-1]
    plot_client_comparison(
        client_ids=last_round.participating_clients,
        metric_values=[last_round.local_updates[c].metrics.f1_macro for c in last_round.participating_clients],
        metric_name="F1 Score (macro)",
        round_num=last_round.round_num,
        filename="client_comparison_final_round.png",
    )

    plot_confusion_matrix(
        np.array(last_round.global_metrics.confusion_matrix),
        classes=trainer.global_classes,
        filename="global_confusion_matrix_final_round.png",
        title=f"Global Model Confusion Matrix (Round {last_round.round_num})",
    )

    X_test, y_test, _ = trainer.partitioner.load_global_test()
    plot_roc_curve(
        y_test, last_round.global_proba, classes=trainer.global_classes,
        filename="global_roc_curve_final_round.png",
    )

    logger.info("All plots saved to %s", config.PLOTS_DIR)
    logger.info(
        "FINAL GLOBAL MODEL (round %d) -> %s",
        last_round.round_num, last_round.global_metrics.summary(),
    )


if __name__ == "__main__":
    main()
