"""
Centralized, validated configuration.

A single validated Settings object gives you one place to reason about
secrets, timeouts, and tunables — and makes the app trivially testable
(override Settings in tests instead of monkeypatching env vars).
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")

    agent_model: str = Field(default="claude-sonnet-5", alias="AGENT_MODEL")
    agent_max_turns: int = Field(default=8, ge=1, le=25, alias="AGENT_MAX_TURNS")
    agent_request_timeout: int = Field(
        default=60, ge=1, le=600, alias="AGENT_REQUEST_TIMEOUT"
    )
    agent_max_tokens: int = Field(
        default=2048, ge=1, le=64000, alias="AGENT_MAX_TOKENS"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_rate_limit_per_minute: int = Field(
        default=30, ge=1, alias="API_RATE_LIMIT_PER_MINUTE"
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v_upper


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so Settings is parsed once per process."""
    return Settings()  # type: ignore[call-arg]
