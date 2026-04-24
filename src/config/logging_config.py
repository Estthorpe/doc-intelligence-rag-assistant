# src/config/logging_config.py
"""
Structured logging using loguru.

Usage anywhere in the codebase:
    from src.config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document: {path}", path=file_path)
"""

from __future__ import annotations

import sys

from loguru import logger
from loguru._logger import Logger


def configure_logging(log_level: str = "INFO") -> None:
    """Configure loguru. Called once at startup."""
    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="gz",
    )


def get_logger(name: str) -> Logger:
    """Get a named logger. Pass __name__ from the calling module."""
    return logger.bind(module=name)


configure_logging()
