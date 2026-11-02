"""Read-only conversation search for a project's leader."""

import json
from pathlib import Path

from symphonai_api.models import Role, ToolResult
from symphonai_api.secrets import mask_secrets
from symphonai_api.serialization import message_from_json
from symphonai_api.session import SessionStore, TranscriptError, read_records
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata
from symphonai_host.sessions import list_sessions, matching_snippets


class SessionTool(LocalTool):
    def __init__(self, root, repo_root, current_session):
        self.root = Path(root).resolve()
        self.repo_root = str(repo_root)
        self.current_session = current_session

    def metadata(self, arguments):
        return ToolMetadata(ToolEffect.READ_ONLY, concurrency_safe=True, paths=None)

    def store(self, session_id):
        directory = (self.root / session_id).resolve()
        if directory.parent != self.root or directory.name != session_id:
            raise ValueError("Session is unavailable in this project")
        store = SessionStore.open(self.root, session_id)
        if store.read_meta().get("repo_root") != self.repo_root:
            raise ValueError("Session is unavailable in this project")
        return store

    def _execute(self, tool_call, policy, cancel=None):
        try:
            value = self.read(tool_call.arguments)
        except (OSError, TranscriptError, ValueError, KeyError, TypeError):
            return ToolResult(tool_call.id, False, error="Session is unavailable in this project")
        text = mask_secrets(json.dumps(value, ensure_ascii=False))
        limit = policy.shell_output_limit_chars
        if len(text) > limit:
            text = text[:limit] + "\n[output truncated]"
        return ToolResult(tool_call.id, True, content=text)


class SearchSessions(SessionTool):
    name = "search_sessions"
    description = "Search this project's past conversation titles and messages."
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
    }, "required": ["query"], "additionalProperties": False}

    def read(self, arguments):
        found = []
        items = list_sessions(self.root) + list_sessions(self.root, archived=True)
        items.sort(key=lambda item: item["updated_at"] or "", reverse=True)
        for item in items:
            session_id = item["run_id"]
            if item["repo_root"] != self.repo_root or session_id == self.current_session:
                continue
            snippets = matching_snippets(item["title"], self.store(session_id), arguments["query"])
            if snippets:
                found.append({"session_id": session_id, "title": item["title"],
                              "date": item["updated_at"], "snippets": snippets})
                if len(found) >= arguments.get("limit", 10):
                    break
        return {"sessions": found}


class ReadSession(SessionTool):
    name = "read_session"
    description = "Read a page of this project's conversation, without tool output."
    parameters = {"type": "object", "properties": {
        "session_id": {"type": "string", "minLength": 1},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "default": 40},
    }, "required": ["session_id"], "additionalProperties": False}

    def read(self, arguments):
        store = self.store(arguments["session_id"])
        records, _ = read_records(store.directory / "run.jsonl")
        messages = []
        for record in records:
            if record.get("type") != "message":
                continue
            message = message_from_json(record["data"])
            if message.role not in (Role.USER, Role.ASSISTANT):
                continue
            messages.append({"role": message.role.value, "text": message.text,
                             "tool_calls": [call.name for call in message.tool_calls]})
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 40)
        return {"messages": messages[offset:offset + limit], "total": len(messages),
                "next_offset": offset + limit if offset + limit < len(messages) else None}


def session_tools(root, repo_root, current_session):
    return {tool.name: tool for tool in (SearchSessions(root, repo_root, current_session),
                                        ReadSession(root, repo_root, current_session))}
