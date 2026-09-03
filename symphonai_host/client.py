"""Small standard-library client for the loopback host boundary."""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from typing import Iterator

from symphonai_host.protocol import decode_frame


class HostClientError(RuntimeError):
    """A host endpoint could not be reached or returned an invalid response."""


@dataclass(frozen=True)
class HostAddress:
    port: int
    token: str

    @classmethod
    def from_handshake(cls, line: str) -> "HostAddress":
        try:
            value = json.loads(line)
            port = value["port"]
            token = value["token"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HostClientError(f"invalid host handshake: {exc}") from None
        if type(port) is not int or port < 1 or type(token) is not str or not token:
            raise HostClientError("invalid host handshake: port and token are required")
        return cls(port, token)


class HostClient:
    def __init__(self, address: HostAddress, *, timeout: float = 30.0) -> None:
        self.address = address
        self.timeout = timeout
        self._events_connection: http.client.HTTPConnection | None = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        connection = http.client.HTTPConnection("127.0.0.1", self.address.port, timeout=self.timeout)
        try:
            encoded = None if body is None else json.dumps(body)
            headers = {"Authorization": f"Bearer {self.address.token}"}
            if encoded is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            if response.status >= 400:
                raise HostClientError(f"{path} returned HTTP {response.status}")
            return json.loads(payload) if payload else {}
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            if isinstance(exc, HostClientError):
                raise
            raise HostClientError(f"{path} connection failed: {type(exc).__name__}") from None
        finally:
            connection.close()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def events(self) -> Iterator[tuple[str, dict]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.address.port, timeout=self.timeout)
        self._events_connection = connection
        try:
            connection.request("GET", "/events", headers={"Authorization": f"Bearer {self.address.token}"})
            response = connection.getresponse()
            if response.status != 200:
                raise HostClientError(f"/events returned HTTP {response.status}")
            lines: list[str] = []
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    lines.append(line.removeprefix("data:").lstrip())
                    continue
                if not line and lines:
                    yield decode_frame("\n".join(lines))
                    lines.clear()
        except (OSError, http.client.HTTPException, UnicodeDecodeError) as exc:
            raise HostClientError(f"/events connection failed: {type(exc).__name__}") from None
        finally:
            connection.close()
            self._events_connection = None

    def send_prompt(self, prompt: str) -> dict:
        return self._request("POST", "/prompt", {"prompt": prompt})

    def send_approval(self, approval_id: str, *, allowed: bool, reason: str = "") -> dict:
        return self._request("POST", "/approval", {"approval_id": approval_id, "allowed": allowed, "reason": reason})

    def stop(self, reason: str = "") -> dict:
        return self._request("POST", "/stop", {"reason": reason})

    def close(self) -> None:
        if self._events_connection is not None:
            self._events_connection.close()
