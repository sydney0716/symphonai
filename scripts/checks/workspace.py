"""Per-check temporary workspaces for tool and agent checks."""

from __future__ import annotations

import sys
import io
import json
import os
import subprocess
from unittest import mock
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from symphonai_api.permissions import PermissionPolicy  # noqa: E402
from symphonai_api.runner import standard_tool_registry  # noqa: E402
from symphonai_api.tools.base import LocalTool  # noqa: E402
from symphonai_api.models import Message, ModelResponse, Role, ToolCall, Usage
from symphonai_api.providers.fake import FakeModelProvider


@dataclass(frozen=True)
class Workspace:
    root: Path
    outside: Path
    policy: PermissionPolicy
    tools: dict[str, LocalTool]


def handoff_spec(goal: str, command: str = "true") -> str:
    return (
        f"## Goal\n{goal}\n\n## Scope\nCheck fixture\n\n"
        f"## Validation\n```sh\n{command}\n```"
    )


@contextmanager
def workspace() -> Iterator[Workspace]:
    """A seeded temp root, an unrelated outside dir, a policy, and a registry."""
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
        root = Path(tmp).resolve()
        outside = Path(outside_tmp).resolve()
        (root / "existing.txt").write_text("hello from disk")
        (root / ".env").write_text("SECRET=do-not-read-me")
        policy = PermissionPolicy(repo_root=root, allowed_write_scope=[root])
        yield Workspace(
            root=root,
            outside=outside,
            policy=policy,
            tools=standard_tool_registry(),
        )



KEY = "core-vendor-inert-key"
ENV = {name: KEY for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CORE_CHAT_KEY")}


class Script(FakeModelProvider):
    def __init__(self, responses=None, before=None):
        super().__init__(responses)
        self.requests = []
        self.before = before

    def create_response(self, request, *, cancel=None):
        self.requests.append(request)
        if self.before is not None:
            self.before(request, len(self.requests))
        return super().create_response(request, cancel=cancel)


def answer(text="finished", *calls, usage=None):
    if len(calls) == 1 and isinstance(calls[0], (list, tuple)):
        calls = calls[0]
    return ModelResponse(Message(Role.ASSISTANT, text, tool_calls=list(calls)), usage=usage or Usage())


def call(name, /, **arguments):
    return ToolCall(f"core-{os.urandom(5).hex()}", name, arguments)


class Response(io.BytesIO):
    status = 200
    headers = {}

    def __init__(self, value):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())


@contextmanager
def exchange(*responses, environment=None):
    bodies, requests, timeouts = [], [], []
    pending = iter(responses)

    def send(request, **options):
        requests.append(request)
        bodies.append(json.loads(request.data))
        timeouts.append(options["timeout"])
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return Response(response)

    with mock.patch.dict(os.environ, ENV if environment is None else environment), mock.patch("urllib.request.urlopen", side_effect=send), mock.patch("time.sleep"):
        yield bodies, requests, timeouts


def git(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout


def init_repository(root):
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "core@example.invalid")
    git(root, "config", "user.name", "Core checks")
    git(root, "add", "existing.txt")
    git(root, "commit", "--quiet", "-m", "fixture")
