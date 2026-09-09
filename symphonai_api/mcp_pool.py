"""Own a set of enabled MCP clients for one explicit lifetime."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

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
        self._started = False

    def start(self) -> dict[str, LocalTool]:
        """Start every enabled server and return their combined tool registry."""
        if self._started:
            return self.tools

        tools: dict[str, LocalTool] = {}
        owners: dict[str, tuple[str, int]] = {}
        reserved = set(self._reserved_names)
        try:
            for index, spec in enumerate(self._specs):
                if not spec.enabled:
                    continue
                client = McpClient(
                    spec,
                    cwd=self._cwd,
                    reserved_names=reserved,
                )
                self._clients.append(client)
                client.start()
                try:
                    listed = client.list_tools()
                except McpError as exc:
                    duplicate = next(
                        (
                            name
                            for name in owners
                            if "collides with reserved name" in str(exc)
                            and repr(name) in str(exc)
                        ),
                        None,
                    )
                    if duplicate is None:
                        raise
                    first_name, first_index = owners[duplicate]
                    raise McpError(
                        f"MCP servers {first_name!r} and {spec.name!r} at "
                        f"indices {first_index} and {index} both provide tool "
                        f"{duplicate!r}"
                    ) from exc
                for tool in listed:
                    if tool.name in tools:
                        first_name, first_index = owners[tool.name]
                        raise McpError(
                            f"MCP servers {first_name!r} and {spec.name!r} at "
                            f"indices {first_index} and {index} both provide tool "
                            f"{tool.name!r}"
                        )
                    tools[tool.name] = tool
                    owners[tool.name] = (spec.name, index)
                    reserved.add(tool.name)
        except McpError:
            try:
                self.close()
            except Exception:  # noqa: BLE001
                # The startup/list failure is the useful cause; cleanup was best-effort.
                pass
            raise

        self._tools = tools
        self._started = True
        return self.tools

    def close(self) -> None:
        """Close every constructed client and re-raise the first close failure."""
        clients = self._clients
        self._clients = []
        self._tools = {}
        self._started = False
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
