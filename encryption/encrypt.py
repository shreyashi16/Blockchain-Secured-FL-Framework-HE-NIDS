"""
encryption/encrypt.py
======================
Encrypts a client's plaintext model-update vector into a CKKS ciphertext,
and provides the hashing utility used to produce a tamper-evident fingerprint
of that ciphertext for the (future) blockchain module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import tenseal as ts

from utils.logger import get_encryption_logger

logger = get_encryption_logger("encryption.encrypt")


@dataclass
class EncryptedUpdate:
    """A client's encrypted update, ready for "transmission" to the server."""

    client_id: int
    round_num: int
    ciphertext: ts.CKKSVector
    ciphertext_bytes: bytes
    update_hash: str
    vector_length: int
    weight: float  # aggregation weight (e.g. proportional to sample count)


class UpdateEncryptor:
    """Encrypts numeric model-update vectors under a client's public context."""

    def __init__(self, public_context: ts.Context) -> None:
        self._context = public_context

    def encrypt_vector(
        self,
        plain_vector: np.ndarray,
        client_id: int,
        round_num: int,
        weight: float,
    ) -> EncryptedUpdate:
        """Encrypt ``plain_vector`` and package it with blockchain-ready metadata."""
        vector = np.asarray(plain_vector, dtype=np.float64).ravel()
        ciphertext = ts.ckks_vector(self._context, vector.tolist())
        ciphertext_bytes = ciphertext.serialize()
        update_hash = self.compute_update_hash(ciphertext_bytes)

        logger.info(
            "Encrypted update | client=%d round=%d dim=%d bytes=%d hash=%s",
            client_id, round_num, len(vector), len(ciphertext_bytes), update_hash[:16],
        )

        return EncryptedUpdate(
            client_id=client_id,
            round_num=round_num,
            ciphertext=ciphertext,
            ciphertext_bytes=ciphertext_bytes,
            update_hash=update_hash,
            vector_length=len(vector),
            weight=weight,
        )

    @staticmethod
    def compute_update_hash(ciphertext_bytes: bytes) -> str:
        """SHA-256 fingerprint of the serialized ciphertext (blockchain-ready)."""
        return hashlib.sha256(ciphertext_bytes).hexdigest()
