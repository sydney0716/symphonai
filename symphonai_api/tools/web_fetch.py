"""GET-only web fetching under per-domain permission checks."""

from __future__ import annotations

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_api.web import WebFetchError, fetch_url


class WebFetchTool(LocalTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch an approved domain with an HTTP GET request only."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The absolute HTTP or HTTPS URL to fetch.",
                }
            },
            "required": ["url"],
        }

    def validate(self, arguments: dict) -> str | None:
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            return "url must be a non-empty string"
        return None

    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        try:
            page = fetch_url(tool_call.arguments["url"], policy, cancel=cancel)
        except WebFetchError as exc:
            return ToolResult(tool_call_id=tool_call.id, ok=False, error=str(exc))
        return ToolResult(
            tool_call_id=tool_call.id,
            ok=True,
            content=f"URL: {page.url}\nStatus: {page.status}\n\n{page.text}",
        )
