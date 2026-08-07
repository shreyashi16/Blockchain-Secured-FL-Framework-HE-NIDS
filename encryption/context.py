"""
encryption/context.py
======================
Creates and manages the TenSEAL CKKS homomorphic-encryption context that
every client and the server share.

Trust model
-----------
This module implements *secure aggregation via CKKS* under the standard
"honest-but-curious server" simplification used throughout the FL
literature for a single-aggregator setup:

  * The **secret key** is generated once and kept only by the entity
    performing decryption (in this simulation: the coordinator that
    validates the aggregated result - see ``encryption/decrypt.py``). It is
    NEVER shipped inside a "public" context.
  * Clients only ever receive a **public context** (public key + relin keys
    + Galois keys, secret key stripped) - they can encrypt but cannot
    decrypt anything, including each other's updates.
  * The server can homomorphically add/scale ciphertexts without the secret
    key at all (that is the whole point of CKKS) - see
    ``encryption/secure_aggregation.py``.

A production multi-organization deployment would replace the single secret
key with a threshold/multi-key HE scheme so that no single party can decrypt
individual updates; that upgrade only touches this module and
``encryption/keys.py`` - the rest of the FL pipeline is unaffected, which is
exactly the "clean interface" the blockchain team was asked to be able to
build on top of.
"""

from __future__ import annotations

from dataclasses import dataclass

import tenseal as ts

import config
from utils.logger import get_encryption_logger

logger = get_encryption_logger("encryption.context")


@dataclass
class HEContextBundle:
    """Holds both the full (secret-key-bearing) and public serialized contexts."""

    full_context: ts.Context
    public_context_bytes: bytes


class HEContextManager:
    """Builds and serializes the CKKS context used across the pipeline."""

    def __init__(self) -> None:
        self._he_config = config.HE_CONFIG
        self._bundle: HEContextBundle | None = None

    def create_context(self) -> HEContextBundle:
        """Generate a fresh CKKS context with keys, matching config.HE_CONFIG."""
        logger.info(
            "Generating CKKS context (poly_modulus_degree=%d, coeff_mod_bit_sizes=%s)",
            self._he_config.poly_modulus_degree,
            self._he_config.coeff_mod_bit_sizes,
        )
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._he_config.poly_modulus_degree,
            coeff_mod_bit_sizes=self._he_config.coeff_mod_bit_sizes,
        )
        context.generate_galois_keys()
        context.global_scale = 2 ** self._he_config.global_scale_power

        # Serialize a PUBLIC copy (secret key stripped) before we do anything
        # else - this is what gets "transmitted" to clients.
        public_bytes = context.serialize(save_secret_key=False)

        self._bundle = HEContextBundle(full_context=context, public_context_bytes=public_bytes)
        logger.info("CKKS context ready. Public context size: %d bytes", len(public_bytes))
        return self._bundle

    @property
    def bundle(self) -> HEContextBundle:
        if self._bundle is None:
            return self.create_context()
        return self._bundle

    def get_public_context(self) -> ts.Context:
        """Deserialize and return a secret-key-free context, as a client would receive it."""
        return ts.context_from(self.bundle.public_context_bytes)

    def get_full_context(self) -> ts.Context:
        """Return the full context (secret key included) - decryption side only."""
        return self.bundle.full_context
