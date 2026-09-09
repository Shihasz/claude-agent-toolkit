"""
A tiny local knowledge-base search tool.

Stands in for a retrieval step in a RAG pipeline: swap `_DOCS` for a
real vector store (embeddings + a vector DB) without touching the
agent loop or the Claude API tool-calling contract at all — that
boundary is the point.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from agent_toolkit.tools.base import BaseTool

_DOCS: dict[str, str] = {
    "python-context-managers": (
        "Context managers (the `with` statement) guarantee cleanup code "
        "runs even if an exception is raised, by implementing __enter__ "
        "and __exit__ (or using contextlib.contextmanager)."
    ),
    "rest-vs-rpc": (
        "REST models resources with nouns and standard HTTP verbs, while "
        "RPC-style APIs model actions with verbs directly, which often "
        "fits agent tool calls more naturally than CRUD resources do."
    ),
    "idempotency": (
        "An idempotent operation produces the same result no matter how "
        "many times it's applied — critical for retryable operations like "
        "the ones a tool-calling agent might repeat after a timeout."
    ),
    "structured-outputs": (
        "Structured outputs guarantee that a model's response conforms to "
        "a JSON schema or tool definition, eliminating schema-parsing "
        "failures in production pipelines."
    ),
}


class KnowledgeBaseArgs(BaseModel):
    query: str = Field(..., description="Free-text search query")
    top_k: int = Field(default=2, ge=1, le=5)


class KnowledgeBaseTool(BaseTool):
    name: ClassVar[str] = "knowledge_base_search"
    description: ClassVar[str] = (
        "Search a small internal knowledge base of engineering notes. "
        "Returns the most relevant snippets."
    )
    args_model: ClassVar[type[BaseModel]] = KnowledgeBaseArgs

    def run(self, query: str, top_k: int = 2) -> dict:
        query_terms = set(query.lower().split())
        scored = []
        for key, text in _DOCS.items():
            haystack = (key + " " + text).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0:
                scored.append((score, key, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]
        return {
            "query": query,
            "results": [{"id": key, "text": text} for _, key, text in top],
        }
