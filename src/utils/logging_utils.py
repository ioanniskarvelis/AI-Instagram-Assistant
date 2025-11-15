"""
Structured logging utilities using structlog
Provides JSON-formatted logging for better observability
"""

import os
import sys
import structlog
from datetime import datetime


def configure_logging(debug: bool = False):
    """
    Configure structured logging for the application

    Args:
        debug: Enable debug mode with pretty console output
    """
    if debug:
        # Development: Pretty console output
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging_level=10),  # DEBUG
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False
        )
    else:
        # Production: JSON output for log aggregation
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging_level=20),  # INFO
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True
        )


def get_logger(name: str = "app"):
    """
    Get a configured logger instance

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """
    Adapter to bridge old file-based logging to structured logging
    Maintains backwards compatibility while improving observability
    """

    def __init__(self, logger_name: str = "app"):
        self.logger = get_logger(logger_name)
        self.closed = False

    def write(self, message: str):
        """Write method to maintain file-like interface"""
        if message and message.strip():
            self.logger.info(message.strip())

    def flush(self):
        """Flush method to maintain file-like interface (no-op for structlog)"""
        pass

    def close(self):
        """Close method to maintain file-like interface"""
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
