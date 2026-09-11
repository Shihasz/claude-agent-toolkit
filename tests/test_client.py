from unittest.mock import MagicMock, patch

import anthropic
import pytest

from agent_toolkit.client import ClaudeClient


def make_settings():
    from agent_toolkit.config import Settings

    return Settings(
        ANTHROPIC_API_KEY="sk-ant-test-key-not-real",
        AGENT_MODEL="claude-sonnet-5",
        AGENT_MAX_TOKENS=1024,
        AGENT_REQUEST_TIMEOUT=30,
    )


class TestClaudeClient:
    def test_successful_call_returns_response(self):
        fake_response = MagicMock()
        fake_response.stop_reason = "end_turn"
        fake_response.usage.input_tokens = 10
        fake_response.usage.output_tokens = 5

        client = ClaudeClient(make_settings())
        with patch.object(
            client._client.messages, "create", return_value=fake_response
        ):
            result = client.create_message(messages=[{"role": "user", "content": "hi"}])

        assert result is fake_response

    def test_retries_on_rate_limit_then_succeeds(self):
        fake_response = MagicMock()
        fake_response.stop_reason = "end_turn"
        fake_response.usage.input_tokens = 1
        fake_response.usage.output_tokens = 1

        rate_limit_error = anthropic.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None
        )

        client = ClaudeClient(make_settings())
        with patch.object(
            client._client.messages,
            "create",
            side_effect=[rate_limit_error, fake_response],
        ) as mock_create:
            result = client.create_message(messages=[{"role": "user", "content": "hi"}])

        assert result is fake_response
        assert mock_create.call_count == 2

    def test_gives_up_after_max_attempts(self):
        rate_limit_error = anthropic.RateLimitError(
            "rate limited", response=MagicMock(status_code=429), body=None
        )
        client = ClaudeClient(make_settings())
        with (
            patch.object(
                client._client.messages, "create", side_effect=rate_limit_error
            ) as mock_create,
            pytest.raises(anthropic.RateLimitError),
        ):
            client.create_message(messages=[{"role": "user", "content": "hi"}])

        assert mock_create.call_count == 4  # stop_after_attempt(4)

    def test_non_retryable_error_raises_immediately(self):
        bad_request = anthropic.BadRequestError(
            "bad request", response=MagicMock(status_code=400), body=None
        )
        client = ClaudeClient(make_settings())
        with (
            patch.object(
                client._client.messages, "create", side_effect=bad_request
            ) as mock_create,
            pytest.raises(anthropic.BadRequestError),
        ):
            client.create_message(messages=[{"role": "user", "content": "hi"}])

        assert mock_create.call_count == 1
