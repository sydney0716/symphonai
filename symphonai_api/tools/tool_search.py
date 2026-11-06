"""Per-agent discovery and loading for explicitly marked MCP tools."""

from __future__ import annotations

import json

from symphonai_api.compaction import estimate_text_tokens
from symphonai_api.models import ToolResult
from symphonai_api.session import read_records
from symphonai_api.tool_schema import to_provider_tool_schema
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolMetadata


def schema_name(schema):
    return schema.get("function", schema).get("name")


class DeferredTool(LocalTool):
    def __init__(self, tool, search):
        self.tool, self.search = tool, search

    @property
    def name(self):
        return self.tool.name

    @property
    def description(self):
        return self.tool.description

    @property
    def parameters(self):
        return self.tool.parameters

    def validate(self, arguments):
        return self.tool.validate(arguments)

    def metadata(self, arguments):
        return self.tool.metadata(arguments)

    def _execute(self, tool_call, policy, cancel=None):
        if self.search.active and self.name not in self.search.loaded:
            return ToolResult(tool_call.id, ok=False, error=f"Load {self.name} with search_tools before calling it")
        return self.tool.execute(tool_call, policy, cancel=cancel)


class SearchTools(LocalTool):
    name = "search_tools"
    description = "Find and load additional MCP tools by keywords in their names and descriptions. Loaded tools stay available."
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
    }, "required": ["query"]}

    def __init__(self, tools, history=None):
        self.tools = dict(tools)
        self.loaded = set()
        self.active = False
        self.available = set(tools)
        self._restored = False
        self._cache_key = None
        self._cache_source = None
        self._cached_schemas = []
        if history is not None and history.is_file():
            records, _ = read_records(history)
            for record in records:
                data = record.get("data", {})
                if record.get("type") == "message":
                    data = data.get("tool_result") or {}
                elif record.get("type") != "tool_result":
                    continue
                if data.get("ok") is True:
                    self._restore_payload(data.get("payload"))

            self._restored = True

    def _restore_payload(self, payload):
        if not isinstance(payload, dict) or payload.get("kind") != "tool_search":
            return
        for item in payload.get("tools", []):
            if isinstance(item, dict) and item.get("name") in self.tools:
                self.loaded.add(item["name"])

    def restore_messages(self, messages):
        if self._restored:
            return
        self._restored = True
        for message in messages:
            if message.tool_result is not None and message.tool_result.ok:
                self._restore_payload(message.tool_result.payload)

    def request_schemas(self, schemas, wire_format, budget):
        key = (wire_format, budget, frozenset(self.loaded))
        if self._cache_source is schemas and self._cache_key == key:
            return self._cached_schemas
        self._cache_source, self._cache_key = schemas, key
        schemas = [schema for schema in schemas if schema_name(schema) != self.name]
        deferred = [schema for schema in schemas if schema_name(schema) in self.tools]
        self.available = {schema_name(schema) for schema in deferred}
        self.active = bool(deferred) and (len(deferred) > 40 or estimate_text_tokens(json.dumps(deferred, ensure_ascii=False)) > budget * .1)
        self._cached_schemas = schemas if not self.active else [
            schema for schema in schemas if schema_name(schema) not in self.tools or schema_name(schema) in self.loaded
        ] + [to_provider_tool_schema(self, wire_format)]
        return self._cached_schemas

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=False)

    def validate(self, arguments):
        return None if arguments["query"].strip() else "query must be non-blank"

    def _execute(self, tool_call, policy, cancel=None):
        terms = tool_call.arguments["query"].casefold().split()
        matches = []
        for name, tool in sorted(self.tools.items()):
            if name not in self.available:
                continue
            searchable = f"{name} {tool.description}".casefold()
            if all(term in searchable for term in terms):
                matches.append({
                    "name": name,
                    "server": name[5:].split("__", 1)[0] if name.startswith("mcp__") else name.split("_", 1)[0],
                    "description": " ".join(tool.description.splitlines())[:200],
                })
            if len(matches) >= tool_call.arguments.get("limit", 10):
                break
        self.loaded.update(item["name"] for item in matches)
        return ToolResult(tool_call.id, ok=True, content=json.dumps(matches, ensure_ascii=False),
                          payload={"kind": "tool_search", "tools": matches})
