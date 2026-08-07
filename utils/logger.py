"""
utils/logger.py
================
Structured logging setup. Every subsystem (training, encryption,
aggregation) gets its own rotating log file under ``logs/`` plus a shared
console + main-log handler, so a reviewer can trace exactly what happened in
a given round from the log files alone.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import config

_CONFIGURED_LOGGERS: dict = {}

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, e.g. ``"federated.client.1"``.
    log_file:
        Optional dedicated log file. The logger always also writes to
        ``config.MAIN_LOG_FILE`` and to stdout.
    """
    cache_key = f"{name}:{log_file}"
    if cache_key in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[cache_key]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        os.makedirs(config.LOG_DIR, exist_ok=True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_FORMATTER)
        logger.addHandler(console_handler)

        main_handler = logging.FileHandler(config.MAIN_LOG_FILE)
        main_handler.setFormatter(_FORMATTER)
        logger.addHandler(main_handler)

        if log_file is not None:
            dedicated_handler = logging.FileHandler(log_file)
            dedicated_handler.setFormatter(_FORMATTER)
            logger.addHandler(dedicated_handler)

    _CONFIGURED_LOGGERS[cache_key] = logger
    return logger


def get_training_logger(name: str) -> logging.Logger:
    return get_logger(name, config.TRAINING_LOG_FILE)


def get_encryption_logger(name: str) -> logging.Logger:
    return get_logger(name, config.ENCRYPTION_LOG_FILE)


def get_aggregation_logger(name: str) -> logging.Logger:
    return get_logger(name, config.AGGREGATION_LOG_FILE)
