"""Structured web search backed by an operator-configured endpoint."""

from __future__ import annotations

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata
from symphonai_api.web_search import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT,
    SearchBackend,
    SearchBackendError,
)


class WebSearchTool(LocalTool):
    def __init__(self, backend: SearchBackend) -> None:
        self._backend = backend

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web through the configured search endpoint."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results, clamped from 1 to 10.",
                    "default": DEFAULT_SEARCH_LIMIT,
                },
            },
            "required": ["query"],
        }

    def validate(self, arguments: dict) -> str | None:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            return "query must be a non-empty string"
        limit = arguments.get("limit", DEFAULT_SEARCH_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int):
            return "limit must be an integer"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
            result_hint=ResultHint.TEXT,
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        limit = max(
            MIN_SEARCH_LIMIT,
            min(MAX_SEARCH_LIMIT, tool_call.arguments.get("limit", DEFAULT_SEARCH_LIMIT)),
        )
        try:
            hits = self._backend.search(
                tool_call.arguments["query"], limit=limit, cancel=cancel
            )
        except SearchBackendError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        rendered: list[str] = []
        payload_hits: list[dict[str, str]] = []
        for index, hit in enumerate(hits, start=1):
            rendered.append(f"{index}. {hit.title} — {hit.url}")
            if hit.snippet:
                rendered.append(f"   {hit.snippet}")
            payload_hits.append(
                {"title": hit.title, "url": hit.url, "snippet": hit.snippet}
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content="\n".join(rendered),
            payload={"hits": payload_hits},
        )
