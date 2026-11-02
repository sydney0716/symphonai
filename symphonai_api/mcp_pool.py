"""Own a set of enabled MCP clients for one explicit lifetime."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
import threading

from symphonai_api.mcp import McpClient, McpError, McpServerSpec
from symphonai_api.tools.base import LocalTool


class McpPool:
    """Every enabled server, started together and closed together."""

    def __init__(
        self,
        specs: Sequence[McpServerSpec],
        *,
        cwd: Path,
        reserved_names: Collection[str] = (),
    ) -> None:
        self._specs = tuple(specs)
        self._cwd = Path(cwd)
        self._reserved_names = frozenset(reserved_names)
        self._clients: list[McpClient] = []
        self._tools: dict[str, LocalTool] = {}
        self._instructions: tuple[tuple[str, str], ...] = ()
        self._started = False
        self._start_lock = threading.RLock()
        self._retired_clients = []
        self.clients: dict[str, McpClient] = {}
        self.states: dict[str, dict] = {}
        self._server_tools: dict[str, tuple[LocalTool, ...]] = {}

    def start(self, server: str | None = None) -> dict[str, LocalTool]:
        """Start servers independently, or retry one without replacing its peers."""
        with self._start_lock:
            import sys
            if self._started and server is None:
                return self.tools
            if server is not None and server not in {spec.name for spec in self._specs if spec.enabled}:
                raise McpError('Unknown or disabled MCP server')
            for spec in self._specs:
                if not spec.enabled or server is not None and spec.name != server:
                    continue
                old = self.clients.get(spec.name)
                if old is not None:
                    if spec.name in self._server_tools and self.status(spec.name)['state'] == 'started':
                        self._retired_clients.append(old)
                    else:
                        old.close()
                self._server_tools.pop(spec.name, None)
                reserved = set(self._reserved_names)
                for tools in self._server_tools.values():
                    reserved.update(tool.name for tool in tools)
                client = McpClient(spec, cwd=self._cwd, reserved_names=reserved)
                self.clients[spec.name] = client
                try:
                    client.start()
                    listed = client.list_tools() if not client.capabilities or "tools" in client.capabilities else ()
                    self._server_tools[spec.name] = listed
                    self.states[spec.name] = {'state': 'started', 'error': ''}
                except McpError as exc:
                    client.close()
                    state = getattr(client, '_auth_state', 'failed')
                    if state != 'needs_sign_in':
                        state = 'failed'
                    self.states[spec.name] = {'state': state, 'error': str(exc)}
                    message = 'needs sign-in; sign in from the app' if state == 'needs_sign_in' else str(exc)
                    print(f'MCP server {spec.name!r} skipped: {message}', file=sys.stderr)
            self._clients = [*self._retired_clients, *self.clients.values()]
            self._tools = {tool.name: tool for tools in self._server_tools.values() for tool in tools}
            self._instructions = tuple((name, client.instructions) for name, client in self.clients.items()
                                       if name in self._server_tools and client.instructions)
            self._started = True
            return self.tools

    def status(self, name: str) -> dict:
        value = dict(self.states.get(name, {'state': 'disabled', 'error': ''}))
        client = self.clients.get(name)
        if getattr(client, '_auth_state', None) == 'failed':
            value = {'state': 'failed', 'error': 'MCP connection failed'}
        if getattr(client, '_auth_state', None) == 'needs_sign_in':
            value = {'state': 'needs_sign_in', 'error': 'Sign in from the app'}
        return value

    @property
    def resource_clients(self) -> dict[str, McpClient]:
        return {name: client for name, client in self.clients.items()
                if name in self._server_tools and self.status(name)['state'] == 'started' and 'resources' in client.capabilities}

    @property
    def prompt_clients(self) -> dict[str, McpClient]:
        return {name: client for name, client in self.clients.items()
                if name in self._server_tools and self.status(name)['state'] == 'started' and 'prompts' in client.capabilities}

    @property
    def resource_tools(self) -> dict[str, LocalTool]:
        from symphonai_api.tools.mcp_resources import ListMcpResourcesTool, ReadMcpResourceTool
        clients = self.resource_clients
        if not clients:
            return {}
        tools = (ListMcpResourcesTool(clients), ReadMcpResourceTool(clients))
        return {tool.name: tool for tool in tools}

    def prompts(self) -> list[dict]:
        prompts = []
        for name, client in self.prompt_clients.items():
            cursor, seen = None, set()
            while True:
                page = client.list_prompts(cursor)
                prompts.extend({**prompt, 'server': name} for prompt in page['prompts'])
                cursor = page.get('nextCursor')
                if cursor is None:
                    break
                if cursor in seen:
                    raise McpError('MCP prompt pagination repeated a cursor')
                seen.add(cursor)
        return prompts

    def close(self) -> None:
        """Close every constructed client and re-raise the first close failure."""
        clients = self._clients
        self._clients = []
        self._tools = {}
        self._instructions = ()
        self._started = False
        self._retired_clients = []
        self.clients = {}
        self._server_tools = {}
        self.states = {}
        first_error: Exception | None = None
        for client in reversed(clients):
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "McpPool":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def tools(self) -> dict[str, LocalTool]:
        return self._tools.copy()

    @property
    def instructions(self) -> tuple[tuple[str, str], ...]:
        return self._instructions
