"""Private, user-scoped provider keys for the host process."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from collections.abc import Mapping, MutableMapping
from pathlib import Path


class CredentialError(ValueError):
    """The credentials file cannot be used safely."""


_write_lock = threading.Lock()


def _path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("SYMPHONAI_CREDENTIALS_FILE")
    return Path(override) if override else Path.home() / ".symphonai" / "credentials.json"


def load(path: Path | None = None) -> dict[str, str]:
    """Read valid keys without trusting a malformed or publicly readable file."""
    target = _path(path)
    try:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return {}
    except OSError:
        raise CredentialError("credentials file cannot be opened") from None
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise CredentialError("credentials path is not a regular file")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & ~0o600:
            raise CredentialError(f"credentials file mode {mode:04o} is wider than 0600")
        try:
            data = json.load(source)
        except (ValueError, UnicodeDecodeError):
            return {}
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("keys"), dict)
        or any(not isinstance(name, str) or not isinstance(value, str)
               for name, value in data["keys"].items())
    ):
        return {}
    return dict(data["keys"])


def store(name: str, value: str, path: Path | None = None) -> None:
    """Atomically replace the private file after changing one key."""
    target = _path(path)
    with _write_lock:
        try:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_metadata = target.parent.lstat()
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise CredentialError("credentials directory is not a directory")
            parent_mode = stat.S_IMODE(parent_metadata.st_mode)
            if parent_mode & ~0o700:
                raise CredentialError(f"credentials directory mode {parent_mode:04o} is wider than 0700")
            keys = load(target)
            if value:
                keys[name] = value
            else:
                keys.pop(name, None)
            descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".credentials-")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump({"version": 1, "keys": keys}, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError:
            raise CredentialError("credentials file cannot be written") from None


def apply_to_environment(env: MutableMapping[str, str], keys: Mapping[str, str]) -> list[str]:
    """Fill only missing or empty environment values."""
    applied = []
    for name, value in keys.items():
        if not env.get(name):
            env[name] = value
            applied.append(name)
    return applied
