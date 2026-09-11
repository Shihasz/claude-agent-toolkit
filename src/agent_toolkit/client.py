"""
Thin wrapper around the Anthropic SDK.

Rate limits and transient 5xx/overload errors are expected in
production, not exceptional — this wraps every call in bounded
exponential-backoff retry, a request timeout, and structured logging
of latency/token usage, without leaking the API key into logs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agent_toolkit.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class ClaudeClient:
    """Wraps anthropic.Anthropic with retries, timeouts, and logging."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.agent_request_timeout,
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> anthropic.types.Message:
        start = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self._settings.agent_model,
                max_tokens=self._settings.agent_max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=messages,
                tools=tools or anthropic.NOT_GIVEN,
            )
        except anthropic.APIStatusError as exc:
            logger.error(
                "anthropic_api_error",
                extra={
                    "status_code": exc.status_code,
                    "duration_ms": round((time.monotonic() - start) * 1000, 1),
                },
            )
            raise
        else:
            logger.info(
                "anthropic_api_call",
                extra={
                    "duration_ms": round((time.monotonic() - start) * 1000, 1),
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
            return response
