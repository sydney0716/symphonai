"""Private credential file and host write-route checks."""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import symphonai_host.__main__ as host_main
from symphonai_api.events import RunFinished
from symphonai_api.models import Message, ModelResponse, Role
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.providers.fake import FakeModelProvider
from symphonai_host.credentials import CredentialError, apply_to_environment, load, store
from symphonai_host.protocol import encode_event, encode_frame
from symphonai_host.server import HostServer
from scripts.checks.harness import check, fail


def _target(directory: str) -> Path:
    return Path(directory) / "private" / "credentials.json"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _request(
    host: HostServer,
    method: str,
    path: str,
    *,
    body: dict | None = None,
    authorized: bool = True,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", host.port, timeout=2)
    headers = {"Authorization": f"Bearer {host.token}"} if authorized else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    response = connection.getresponse()
    try:
        return response.status, response.read()
    finally:
        connection.close()


def _host(directory: str) -> HostServer:
    root = Path(directory)
    host = HostServer(
        FakeModelProvider([ModelResponse(Message(Role.ASSISTANT, "done"))]),
        PermissionPolicy(repo_root=root),
        sessions_root=root / "sessions",
    )
    host.start()
    return host


@check("credentials.load_and_mode")
def check_load_and_mode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = _target(directory)
        with mock.patch.dict(os.environ, {"SYMPHONAI_CREDENTIALS_FILE": str(target)}):
            _require(os.environ["SYMPHONAI_CREDENTIALS_FILE"] == str(target), "credentials override was not installed")
            _require(load() == {}, "missing credentials file was not empty")
            target.parent.mkdir(mode=0o700)
            target.write_text("not JSON", encoding="utf-8")
            target.chmod(0o600)
            _require(load() == {}, "malformed credentials file was not ignored")
            target.write_text('{"version": 1, "keys": {"OTHER_API_KEY": "fixture"}}', encoding="utf-8")
            _require(load() == {"OTHER_API_KEY": "fixture"}, "valid unknown variable did not round-trip")
            target.chmod(0o644)
            try:
                load()
            except CredentialError as exc:
                _require("0644" in str(exc), "insecure file error omitted its mode")
            else:
                fail("insecure credentials file was read")
            errors = io.StringIO()
            with mock.patch.object(host_main, "load_extensions") as extensions, contextlib.redirect_stderr(errors):
                try:
                    host_main.main(["--repo-root", directory])
                except SystemExit as exc:
                    _require(exc.code == 2, "insecure credentials did not stop startup")
                else:
                    fail("host started with an insecure credentials file")
                extensions.assert_not_called()
            _require("0644" in errors.getvalue(), "startup error omitted the insecure mode")


@check("credentials.store_and_environment")
def check_store_and_environment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = _target(directory)
        with mock.patch.dict(os.environ, {"SYMPHONAI_CREDENTIALS_FILE": str(target)}):
            store("OTHER_API_KEY", "first")
            _require(target.is_file(), "override did not receive the credentials file")
            _require(_mode(target.parent) == 0o700, "credentials directory is not 0700")
            _require(_mode(target) == 0o600, "credentials file is not 0600")
            store("OTHER_API_KEY", "second")
            _require(load() == {"OTHER_API_KEY": "second"}, "rewrite lost a key")
            _require(_mode(target.parent) == 0o700 and _mode(target) == 0o600, "rewrite widened permissions")
            env = {"EMPTY": "", "EXPORTED": "exported"}
            applied = apply_to_environment(env, {
                "ABSENT": "stored-a", "EMPTY": "stored-b", "EXPORTED": "stored-c",
            })
            _require(applied == ["ABSENT", "EMPTY"], "environment reported incorrect applied names")
            _require(env == {
                "ABSENT": "stored-a", "EMPTY": "stored-b", "EXPORTED": "exported",
            }, "environment precedence changed")


@check("credentials.route_auth_and_write")
def check_route_auth_and_write() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = _target(directory)
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_CREDENTIALS_FILE": str(target), "OPENAI_API_KEY": "",
        }):
            host = _host(directory)
            try:
                payload = {"name": "OPENAI_API_KEY", "value": "stored-fixture"}
                status, _ = _request(host, "POST", "/credentials", body=payload, authorized=False)
                _require(status == 401, "unauthorized credential write was accepted")
                status, _ = _request(host, "POST", f"/credentials?token={host.token}", body=payload, authorized=False)
                _require(status == 401, "query token authorized credential write")
                status, body = _request(host, "POST", "/credentials", body=payload)
                _require(status == 200, "credential write failed")
                _require(json.loads(body) == {"stored": True, "name": "OPENAI_API_KEY"}, "credential response shape changed")
                _require(os.environ["OPENAI_API_KEY"] == "stored-fixture", "write did not update host environment")
                _require(load() == {"OPENAI_API_KEY": "stored-fixture"}, "write did not persist key")
                status, _ = _request(host, "POST", "/credentials", body={"name": "UNKNOWN_API_KEY", "value": "x"})
                _require(status == 400, "unknown variable was accepted")
                status, body = _request(host, "POST", "/credentials", body={"name": "OPENAI_API_KEY", "value": ""})
                _require(status == 200 and json.loads(body)["stored"] is True, "credential deletion failed")
                _require(load() == {} and "OPENAI_API_KEY" not in os.environ, "deletion left key accessible")
                status, settings_body = _request(host, "GET", "/settings")
                providers = json.loads(settings_body)["settings"]["providers"]
                _require(status == 200 and not next(row for row in providers if row["name"] == "openai")["key_present"], "deleted key still appeared present")
            finally:
                host.close()


@check("credentials.secret_not_exposed")
def check_secret_not_exposed() -> None:
    secret = "recognisable-credential-fixture-3f72"
    with tempfile.TemporaryDirectory() as directory:
        target = _target(directory)
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_CREDENTIALS_FILE": str(target), "OPENAI_API_KEY": "",
        }):
            host = _host(directory)
            subscription = host.broker.subscribe()
            try:
                status, write_body = _request(host, "POST", "/credentials", body={
                    "name": "OPENAI_API_KEY", "value": secret,
                })
                _require(status == 200 and secret.encode() not in write_body, "credential response exposed the value")
                status, settings_body = _request(host, "GET", "/settings")
                _require(status == 200 and secret.encode() not in settings_body, "settings exposed the value")
                providers = json.loads(settings_body)["settings"]["providers"]
                _require(next(row for row in providers if row["name"] == "openai")["key_present"], "settings did not show key presence")
                status, _ = _request(host, "POST", "/prompt", body={"prompt": "hello"})
                _require(status == 200, "fixture run did not start")
                frames = []
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    event = subscription.get(timeout=0.1)
                    if event is not None:
                        frames.append(encode_frame("event", encode_event(event)))
                        if isinstance(event, RunFinished):
                            break
                else:
                    fail("fixture run did not finish")
                _require(frames and all(secret not in frame for frame in frames), "event stream exposed the value")
                while host.run.active and time.monotonic() < deadline:
                    time.sleep(0.01)
                _require(not host.run.active, "fixture session did not close")
                records = [path.read_bytes() for path in (Path(directory) / "sessions").rglob("*") if path.is_file()]
                _require(records and all(secret.encode() not in record for record in records), "session record exposed the value")
            finally:
                subscription.close()
                host.close()


@check("credentials.startup_environment_wins")
def check_startup_environment_wins() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = _target(directory)
        store("OPENAI_API_KEY", "stored-fixture", path=target)
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_CREDENTIALS_FILE": str(target),
            "OPENAI_API_KEY": "exported-fixture",
        }), mock.patch.object(host_main, "load_extensions", return_value=SimpleNamespace(mcp_servers=())), \
                mock.patch.object(host_main, "McpPool") as pool_class, \
                mock.patch.object(host_main, "HostServer") as host_class, \
                mock.patch.object(host_main, "_provider"), \
                mock.patch.object(host_main.signal, "signal"):
            host_main.main(["--repo-root", directory])
            _require(os.environ["OPENAI_API_KEY"] == "exported-fixture", "stored key replaced exported key")
            host_class.assert_called_once()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                host_main.main(["--repo-root", directory])
                _require(os.environ["OPENAI_API_KEY"] == "stored-fixture", "startup did not apply stored key")
            _require(host_class.call_count == 2, "host did not build after loading stored key")
            _require(pool_class.return_value.close.call_count == 2, "host pool was not closed")
