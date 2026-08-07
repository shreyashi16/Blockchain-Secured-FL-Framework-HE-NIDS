"""
encryption/secure_aggregation.py
=================================
The single entry point the FL server uses for privacy-preserving
aggregation. It wires together context/keys/encrypt/decrypt into one
round-level pipeline and returns clean, serializable records that a future
blockchain module can consume unmodified (round number, client id,
encrypted-update hash, aggregation metadata, model version).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from encryption.decrypt import UpdateDecryptor, ValidationReport
from encryption.encrypt import EncryptedUpdate, UpdateEncryptor
from encryption.keys import KeyManager
from utils.logger import get_encryption_logger

logger = get_encryption_logger("encryption.secure_aggregation")


@dataclass
class BlockchainMetadataRecord:
    """
    Exactly the fields the blockchain team asked for: everything needed to
    write an immutable audit trail of a training round, with no raw data or
    plaintext model internals in sight.
    """

    round_num: int
    client_id: int
    encrypted_update_hash: str
    ciphertext_size_bytes: int
    aggregation_weight: float
    model_version: str


@dataclass
class SecureAggregationResult:
    round_num: int
    aggregated_vector: np.ndarray
    validation: ValidationReport
    blockchain_records: List[BlockchainMetadataRecord]
    model_version: str


class SecureAggregationPipeline:
    """
    Round-level secure aggregation: encrypt each client's plaintext update
    vector, homomorphically compute the weighted sum on ciphertexts only,
    decrypt the *aggregate* (never an individual client's update) and
    validate it against the plaintext weighted sum.
    """

    def __init__(self, key_manager: KeyManager | None = None) -> None:
        self._key_manager = key_manager or KeyManager()
        self._encryptor = UpdateEncryptor(self._key_manager.distribute_public_key_material().public_context)
        self._decryptor = UpdateDecryptor(self._key_manager.get_decryption_context())

    def run(
        self,
        round_num: int,
        client_vectors: Dict[int, np.ndarray],
        client_weights: Dict[int, float],
        model_version: str,
    ) -> SecureAggregationResult:
        """
        Parameters
        ----------
        client_vectors:
            ``{client_id: plaintext_update_vector}`` - e.g. each client's
            normalized feature-importance vector concatenated with its local
            performance statistics.
        client_weights:
            ``{client_id: aggregation_weight}`` - normalized weights (sum to
            1.0), typically proportional to local sample count (FedAvg).
        """
        client_ids = sorted(client_vectors.keys())
        logger.info("Round %d | starting secure aggregation for %d clients", round_num, len(client_ids))

        total_weight = sum(client_weights[c] for c in client_ids)
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            logger.warning("Client weights do not sum to 1.0 (sum=%.6f) - normalizing.", total_weight)

        encrypted_updates: Dict[int, EncryptedUpdate] = {}
        weighted_ciphertext = None

        for client_id in client_ids:
            normalized_weight = client_weights[client_id] / total_weight
            enc_update = self._encryptor.encrypt_vector(
                plain_vector=client_vectors[client_id],
                client_id=client_id,
                round_num=round_num,
                weight=normalized_weight,
            )
            encrypted_updates[client_id] = enc_update

            # Homomorphic scalar multiplication (plaintext weight * ciphertext)
            # followed by homomorphic addition - the server never sees a
            # decrypted individual client update at any point in this loop.
            scaled = enc_update.ciphertext * normalized_weight
            weighted_ciphertext = scaled if weighted_ciphertext is None else weighted_ciphertext + scaled

        vector_length = next(iter(encrypted_updates.values())).vector_length
        aggregated_bytes = weighted_ciphertext.serialize()
        aggregated_vector = self._decryptor.decrypt_vector(aggregated_bytes, vector_length)

        expected_plaintext = sum(
            (client_weights[c] / total_weight) * np.asarray(client_vectors[c], dtype=np.float64)
            for c in client_ids
        )
        validation = self._decryptor.validate_against_plaintext(aggregated_vector, expected_plaintext)

        blockchain_records = [
            BlockchainMetadataRecord(
                round_num=round_num,
                client_id=client_id,
                encrypted_update_hash=enc.update_hash,
                ciphertext_size_bytes=len(enc.ciphertext_bytes),
                aggregation_weight=enc.weight,
                model_version=model_version,
            )
            for client_id, enc in encrypted_updates.items()
        ]

        logger.info(
            "Round %d | secure aggregation complete | validation=%s",
            round_num, "PASSED" if validation.is_valid else "FAILED",
        )

        return SecureAggregationResult(
            round_num=round_num,
            aggregated_vector=aggregated_vector,
            validation=validation,
            blockchain_records=blockchain_records,
            model_version=model_version,
        )
