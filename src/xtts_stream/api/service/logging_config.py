"""Shared logging configuration for balancer and worker processes."""

"""Unified console logging for balancer and workers."""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any, Dict

RESET = "\033[0m"
LEVEL_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}


class RequestAwareFormatter(logging.Formatter):
    """Formatter that adds color and optional request IDs."""

    def __init__(self, service_name: str, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "service"):
            record.service = self.service_name

        color = LEVEL_COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname}{RESET}"
        return super().format(record)


def _resolve_level(default_level: str | None = None) -> str:
    env_level = os.environ.get("XTTS_LOG_LEVEL")
    level = (env_level or default_level or "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    return level


def build_logging_config(service_name: str, default_level: str | None = None) -> Dict[str, Any]:
    level = _resolve_level(default_level)
    fmt = "%(asctime)s | %(levelname_colored)s | %(service)s | %(name)s | %(message)s | req=%(request_id)s"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "xtts_stream.api.service.logging_config.RequestAwareFormatter",
                "service_name": service_name,
                "fmt": fmt,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "colored",
            }
        },
        "loggers": {
            "": {"handlers": ["console"], "level": level},
            "uvicorn": {"handlers": ["console"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": level, "propagate": False},
            "uvicorn.access": {"handlers": ["console"], "level": level, "propagate": False},
        },
    }


def configure_logging(service_name: str, default_level: str | None = None) -> Dict[str, Any]:
    config = build_logging_config(service_name, default_level=default_level)
    logging.config.dictConfig(config)
    logging.getLogger(__name__).debug(
        "Logging configured for %s with level %s", service_name, config["loggers"][""]["level"]
    )
    return config
