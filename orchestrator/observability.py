from __future__ import annotations

import logging
import sys

import structlog

from orchestrator.config import settings


def configure_logging() -> None:
    """Set up structlog with JSON output and shared processors."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def bind_request_context(*, session_id: str, client_id: str) -> None:
    """Bind session_id and client_id into every subsequent log call in this task."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        client_id=client_id,
    )
