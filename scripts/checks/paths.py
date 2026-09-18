"""Checks for the user-level SymphonAI directory and its callers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

from symphonai_api import symphonai_home as exported_home
from symphonai_api.config import Scope, load_config
from symphonai_api.discovery import discover
from symphonai_api.instructions import InstructionScope, load_instructions
from symphonai_api.paths import symphonai_home
from symphonai_api.permissions import PermissionPolicy
from symphonai_api.session import default_sessions_root
from symphonai_host.credentials import _path as credential_path
from scripts.checks.harness import check, fail


@check("paths.explicit_home")
def explicit_home() -> None:
    with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": "/other"}):
        if symphonai_home(Path("/given")) != Path("/given/.symphonai"):
            fail("explicit operating-system home did not win")
        if exported_home is not symphonai_home:
            fail("symphonai_home is not exported from the package")


@check("paths.environment_root")
def environment_root() -> None:
    with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": "/custom/root"}):
        if symphonai_home() != Path("/custom/root"):
            fail("SYMPHONAI_HOME was treated as a parent directory")


@check("paths.blank_environment")
def blank_environment() -> None:
    for value in ("", " ", "\t\n"):
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": value}):
            if symphonai_home() != Path.home() / ".symphonai":
                fail(f"blank SYMPHONAI_HOME did not use the default: {value!r}")


@check("paths.expansion_and_no_write")
def expansion_and_no_write() -> None:
    with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": "~/elsewhere"}):
        if symphonai_home() != Path.home() / "elsewhere":
            fail("SYMPHONAI_HOME did not expand the tilde")
    with tempfile.TemporaryDirectory() as temporary:
        absent = Path(temporary) / "absent" / "home"
        with mock.patch.dict(os.environ, {"SYMPHONAI_HOME": str(absent)}):
            if symphonai_home() != absent or absent.exists():
                fail("resolving the app directory created it")


@check("paths.call_sites")
def call_sites() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        home = root / "app-home"
        repo = root / "repo"
        repo.mkdir()
        home.mkdir()
        (home / "config.toml").write_text(
            '[agents]\ndirectory = "from-app-home"\n', encoding="utf-8"
        )
        user_instructions = home / "CLAUDE.md"
        user_instructions.write_text("from app home", encoding="utf-8")
        with mock.patch.dict(os.environ, {
            "SYMPHONAI_HOME": str(home),
            "SYMPHONAI_SESSIONS_DIR": "",
            "SYMPHONAI_CREDENTIALS_FILE": "",
        }):
            if default_sessions_root() != home / "sessions":
                fail("sessions did not use SYMPHONAI_HOME")
            if credential_path(None) != home / "credentials.json":
                fail("credentials did not use SYMPHONAI_HOME")
            config = load_config(repo_root=repo)
            if config.get("agents.directory") != "from-app-home":
                fail("config did not read the SYMPHONAI_HOME user scope")
            if config.scope_of("agents.directory") is not Scope.USER:
                fail("config user scope provenance was lost")
            with mock.patch("symphonai_api.discovery.load_skill_directory", return_value={}) as skills:
                discover(repo_root=repo)
            skills.assert_called_once_with(home / "skills")
            loaded = load_instructions(PermissionPolicy(repo_root=repo))
            if [(entry.scope, entry.path) for entry in loaded.entries] != [
                (InstructionScope.USER, user_instructions.resolve())
            ]:
                fail("user instructions did not use SYMPHONAI_HOME")
