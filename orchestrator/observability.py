from __future__ import annotations

import logging
import sys

import structlog

from orchestrator.config import settings


def configure_logging() -> None:
    """Set up structlog with JSON output and shared processors."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_request_context(*, session_id: str, client_id: str) -> None:
    """Bind session_id and client_id into every subsequent log call in this task."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        client_id=client_id,
    )
