from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a simple console logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
