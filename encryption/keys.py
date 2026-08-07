"""
encryption/keys.py
===================
Thin key-management layer on top of ``encryption/context.py``. Kept as a
separate module (rather than folded into context.py) so the blockchain team
- or a future multi-party/threshold-HE upgrade - has a single, obvious place
to plug in real key distribution/rotation without touching the FL code.
"""

from __future__ import annotations

from dataclasses import dataclass

import tenseal as ts

from encryption.context import HEContextManager
from utils.logger import get_encryption_logger

logger = get_encryption_logger("encryption.keys")


@dataclass
class KeyMaterial:
    """Everything a *client* needs to encrypt (no secret key)."""

    public_context: ts.Context
    context_version: str


class KeyManager:
    """
    Owns the single HE context for a training run and hands out public
    (encrypt-only) key material to clients, while keeping the secret key
    confined to the coordinator that performs decryption.
    """

    def __init__(self, context_manager: HEContextManager | None = None) -> None:
        self._context_manager = context_manager or HEContextManager()
        self._context_manager.create_context()
        self._version = "he-ctx-v1"
        logger.info("KeyManager initialised with context version '%s'", self._version)

    def distribute_public_key_material(self) -> KeyMaterial:
        """What every client receives at the start of training."""
        return KeyMaterial(
            public_context=self._context_manager.get_public_context(),
            context_version=self._version,
        )

    def get_decryption_context(self) -> ts.Context:
        """Secret-key-bearing context - only ever used server-side for validation."""
        return self._context_manager.get_full_context()

    @property
    def context_manager(self) -> HEContextManager:
        return self._context_manager
