"""
Tool protocol + registry.

Each tool declares its own input schema (validated with Pydantic,
exported as JSON Schema for the Anthropic `tools` parameter) and its
own execution logic, isolated from the agent loop. That separation is
what makes tools independently unit-testable and safe to sandbox.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, ClassVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Raised by a tool when it fails in an expected, user-facing way."""


class BaseTool(abc.ABC):
    """Subclass this for every tool the agent can call."""

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Export this tool as an Anthropic Messages API tool definition."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Raise ToolError for expected failures."""

    def __call__(self, raw_input: dict[str, Any]) -> Any:
        """Validate raw model-provided arguments, then execute."""
        try:
            validated = self.args_model.model_validate(raw_input)
        except Exception as exc:  # pydantic ValidationError, TypeError, etc.
            logger.warning(
                "tool_input_validation_failed",
                extra={"tool_name": self.name, "error": str(exc)},
            )
            raise ToolError(f"Invalid arguments for tool '{self.name}': {exc}") from exc

        logger.info("tool_call_started", extra={"tool_name": self.name})
        try:
            result = self.run(**validated.model_dump())
        except ToolError:
            raise
        except Exception as exc:  # unexpected internal error
            logger.exception(
                "tool_call_failed_unexpectedly", extra={"tool_name": self.name}
            )
            raise ToolError(f"Tool '{self.name}' failed internally: {exc}") from exc
        logger.info("tool_call_succeeded", extra={"tool_name": self.name})
        return result


class ToolRegistry:
    """Holds the set of tools available to the agent for one run."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
