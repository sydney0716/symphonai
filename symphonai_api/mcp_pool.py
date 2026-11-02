"""Own a set of enabled MCP clients for one explicit lifetime."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
import threading
import sys
from concurrent.futures import ThreadPoolExecutor

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
        self._tools: dict[str, LocalTool] = {}
        self._instructions: tuple[tuple[str, str], ...] = ()
        self._started = False
        self._start_lock = threading.RLock()
        self.clients: dict[str, McpClient] = {}
        self.states: dict[str, dict] = {}
        self._server_tools: dict[str, tuple[LocalTool, ...]] = {}

    def start(self, server: str | None = None) -> dict[str, LocalTool]:
        """Connect in parallel, then merge tools and failures in declared order."""
        with self._start_lock:
            if self._started and server is None:
                return self.tools
            selected = [spec for spec in self._specs if spec.enabled and (server is None or spec.name == server)]
            if server is not None and not selected:
                raise McpError('Unknown or disabled MCP server')

            def connect(spec):
                client = McpClient(spec, cwd=self._cwd, reserved_names=self._reserved_names)
                try:
                    client.start()
                    tools = client.list_tools() if not client.capabilities or "tools" in client.capabilities else ()
                    return client, tools, None
                except McpError as exc:
                    client.close()
                    return client, (), exc

            incoming = {}
            if selected:
                with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix="mcp-start") as workers:
                    futures = {spec.name: workers.submit(connect, spec) for spec in selected}
                    incoming = {spec.name: futures[spec.name].result() for spec in selected}
            reserved = set(self._reserved_names)
            for spec in self._specs:
                if spec.name in incoming:
                    client, tools, error = incoming[spec.name]
                    old = self.clients.get(spec.name)
                    if old is not None:
                        old.close()
                    self.clients[spec.name] = client
                    self._server_tools.pop(spec.name, None)
                    if error is None:
                        occupied = reserved | ({tool.name for peer in self._server_tools.values() for tool in peer}
                                               if server is not None else set())
                        collision = next((tool.name for tool in tools if tool.name in occupied), None)
                        if collision is not None:
                            name = collision.removeprefix(f"mcp__{spec.name}__")
                            error = McpError(f"MCP server {spec.name!r} tool {name!r} collides with reserved name {collision!r}")
                            client.close()
                    if error is None:
                        self._server_tools[spec.name] = tools
                        self.states[spec.name] = {'state': 'started', 'error': ''}
                    else:
                        state = 'needs_sign_in' if client.auth_state == 'needs_sign_in' else 'failed'
                        self.states[spec.name] = {'state': state, 'error': str(error)}
                        message = 'needs sign-in; sign in from the app' if state == 'needs_sign_in' else str(error)
                        print(f'MCP server {spec.name!r} skipped: {message}', file=sys.stderr)
                reserved.update(tool.name for tool in self._server_tools.get(spec.name, ()))
            self._tools = {tool.name: tool for spec in self._specs for tool in self._server_tools.get(spec.name, ())}
            self._instructions = tuple((spec.name, self.clients[spec.name].instructions) for spec in self._specs
                                       if spec.name in self._server_tools and self.clients[spec.name].instructions)
            self._started = True
            return self.tools

    def status(self, name: str) -> dict:
        client = self.clients.get(name)
        current = self.states.get(name, {'state': 'disabled', 'error': ''})
        if client is not None and client.auth_state in ('failed', 'needs_sign_in') and current['state'] != client.auth_state:
            self.states[name] = {'state': client.auth_state, 'error':
                                 'MCP connection failed' if client.auth_state == 'failed' else 'Sign in from the app'}
        return dict(self.states.get(name, current))

    def _started_clients(self, capability: str) -> dict[str, McpClient]:
        return {name: client for name, client in tuple(self.clients.items())
                if self.status(name)['state'] == 'started' and capability in client.capabilities}

    @property
    def resource_tools(self) -> dict[str, LocalTool]:
        from symphonai_api.tools.mcp_resources import ListMcpResourcesTool, ReadMcpResourceTool
        clients = self._started_clients("resources")
        if not clients:
            return {}
        tools = (ListMcpResourcesTool(clients), ReadMcpResourceTool(clients))
        return {tool.name: tool for tool in tools}

    def prompts(self, server: str | None = None) -> list[dict]:
        prompts = []
        for name, client in self._started_clients("prompts").items():
            if server is not None and name != server:
                continue
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
        with self._start_lock:
            clients = list(self.clients.values())
            self._tools = {}
            self._instructions = ()
            self._started = False
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
