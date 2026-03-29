"""
app/utils/logging_config.py
============================
Centralised logging configuration for the research API.

Two output formats, selected by the LOG_FORMAT env var:
  pretty — colourised human-readable output (default, for development)
  json   — structured JSON lines (for production log aggregators)

Call configure_logging() once at application startup (in main.py).
All subsequent getLogger() calls anywhere in the codebase will inherit
the configured handler and formatter automatically.
"""
from __future__ import annotations
import json
import logging
import sys
import time


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line — easy to pipe to log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class _PrettyFormatter(logging.Formatter):
    """Colourised, human-readable format for development use."""

    _COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{color}[{ts}] {record.levelname:<8}{self._RESET} {record.name}: {msg}"


def configure_logging(log_format: str | None = None, log_level: str | None = None) -> None:
    """
    Configure the root logger with the chosen format and level.

    Safe to call multiple times — existing handlers are cleared first to
    prevent duplicate output during uvicorn --reload cycles.

    Args:
        log_format: "pretty" or "json". Defaults to LOG_FORMAT from config.
        log_level:  "DEBUG", "INFO", "WARNING", "ERROR". Defaults to LOG_LEVEL.
    """
    from app.config import LOG_FORMAT, LOG_LEVEL

    fmt = log_format or LOG_FORMAT
    lvl_str = log_level or LOG_LEVEL
    level = getattr(logging, lvl_str.upper(), logging.INFO)

    formatter: logging.Formatter = (
        _JsonFormatter() if fmt == "json" else _PrettyFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
