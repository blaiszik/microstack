"""Logging configuration for ATOMIC."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from atomic_materials import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to use config default)
        log_to_console: Enable console logging (None to use config default)
        log_to_file: Enable file logging (None to use config default)

    Returns:
        Configured logger instance
    """
    # Use config defaults if not provided
    log_level = log_level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE
    log_to_console = (
        log_to_console if log_to_console is not None else config.LOG_TO_CONSOLE
    )
    log_to_file = log_to_file if log_to_file is not None else config.LOG_TO_FILE

    # Create logger
    logger = logging.getLogger("atomic")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with Rich formatting
    if log_to_console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=config.DEBUG_MODE,
            rich_tracebacks=True,
            tracebacks_show_locals=config.DEBUG_MODE,
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with detailed formatting
    if log_to_file and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (uses atomic if None)

    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("atomic")
    return logging.getLogger(f"atomic.{name}")


# Initialize default logger
logger = setup_logging()
