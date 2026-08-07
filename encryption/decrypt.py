"""
encryption/decrypt.py
======================
Decrypts the homomorphically-aggregated update and validates it against the
plaintext-computed expected result, so every round produces hard evidence
that secure aggregation did not silently corrupt anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import tenseal as ts

import config
from utils.logger import get_encryption_logger

logger = get_encryption_logger("encryption.decrypt")


@dataclass
class ValidationReport:
    is_valid: bool
    max_abs_error: float
    mean_abs_error: float
    tolerance: float


class UpdateDecryptor:
    """Decrypts CKKS ciphertexts using the secret-key-bearing context."""

    def __init__(self, secret_context: ts.Context) -> None:
        self._context = secret_context

    def decrypt_vector(self, ciphertext_bytes: bytes, vector_length: int) -> np.ndarray:
        """Deserialize + decrypt a ciphertext back into a plaintext numpy vector."""
        ckks_vector = ts.ckks_vector_from(self._context, ciphertext_bytes)
        decrypted = np.array(ckks_vector.decrypt(), dtype=np.float64)
        return decrypted[:vector_length]

    @staticmethod
    def validate_against_plaintext(
        decrypted: np.ndarray,
        expected_plaintext: np.ndarray,
        tolerance: float | None = None,
    ) -> ValidationReport:
        """
        Compare the decrypted, homomorphically-aggregated vector against the
        value the server would have gotten by summing the plaintext updates
        directly. CKKS is an *approximate* scheme, so we check closeness
        within ``config.HE_CONFIG.validation_atol`` rather than bit-exactness.
        """
        tol = tolerance if tolerance is not None else config.HE_CONFIG.validation_atol
        decrypted = np.asarray(decrypted, dtype=np.float64)
        expected = np.asarray(expected_plaintext, dtype=np.float64)

        abs_error = np.abs(decrypted - expected)
        max_err = float(abs_error.max()) if abs_error.size else 0.0
        mean_err = float(abs_error.mean()) if abs_error.size else 0.0
        is_valid = bool(max_err <= tol)

        log_fn = logger.info if is_valid else logger.warning
        log_fn(
            "Validation %s | max_abs_error=%.6f mean_abs_error=%.6f tolerance=%.6f",
            "PASSED" if is_valid else "FAILED", max_err, mean_err, tol,
        )

        return ValidationReport(
            is_valid=is_valid, max_abs_error=max_err, mean_abs_error=mean_err, tolerance=tol
        )
