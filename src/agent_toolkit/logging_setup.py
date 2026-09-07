"""
Structured (JSON) logging.

Every tool call and API round-trip should be traceable by request id
without leaking secrets (API keys, raw user PII) into logs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

# Standard attributes every LogRecord carries — anything NOT in this set
# was passed in via `extra=` and should be surfaced in the JSON payload.
_STANDARD_RECORD_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Structured extras (e.g. request_id, tool_name) attached via `extra=`
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS:
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = str(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
