"""
Schema for the agent's final structured answer.

Rather than parsing free text at the end of a tool-use loop, the agent
is forced to call a `final_answer` tool whose schema is this model —
so the response callers get back is always machine-parseable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str = Field(..., description="Identifier of the source used")
    note: str = Field(..., description="What this source supported")


class FinalAnswer(BaseModel):
    answer: str = Field(..., description="The direct answer to the user's question")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Self-reported confidence, 0-1"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Any knowledge-base or tool sources the answer relied on",
    )


class FinalAnswerArgs(BaseModel):
    """Wraps FinalAnswer so it can be exposed as an Anthropic tool."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)


FINAL_ANSWER_TOOL_NAME = "final_answer"


def final_answer_tool_schema() -> dict:
    """Anthropic tool definition that forces a structured final response."""
    schema = FinalAnswerArgs.model_json_schema()
    schema.pop("title", None)
    return {
        "name": FINAL_ANSWER_TOOL_NAME,
        "description": (
            "Call this exactly once, as your last action, to deliver the "
            "final answer to the user in a structured, machine-readable form."
        ),
        "input_schema": schema,
    }
