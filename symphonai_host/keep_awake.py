"""Reference-counted idle-sleep inhibition shared by host and headless work."""
from __future__ import annotations

import atexit
import logging
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable

from symphonai_api.process import stop_process_group

_LOG = logging.getLogger(__name__)
_lock = threading.Lock()
_holders: set[object] = set()
_process: subprocess.Popen | None = None
_started = 0.0
_linux_failed: set[str] = set()
_missing_logged = False


def _stop_process() -> None:
    global _process
    process, _process = _process, None
    if process is None:
        return
    stop_process_group(process, process.pid, timeout=0.5)


def _commands() -> list[list[str]]:
    if sys.platform == 'darwin':
        command = shutil.which('caffeinate')
        return [[command, '-i', '-t', '300']] if command else []
    if sys.platform.startswith('linux'):
        result = []
        for name, arguments in (
            ('systemd-inhibit', ['--what=idle', '--who=SymphonAI', '--why=agents are working', 'sleep', 'infinity']),
            ('gnome-session-inhibit', ['--inhibit', 'idle', 'sleep', 'infinity']),
        ):
            command = shutil.which(name)
            if command and command not in _linux_failed:
                result.append([command, *arguments])
        return result
    return []


def _update(token: object, active: bool) -> None:
    global _process, _started, _missing_logged
    with _lock:
        if active:
            _holders.add(token)
        else:
            _holders.discard(token)
        if not _holders:
            _stop_process()
            return
        if _process is not None:
            status = _process.poll()
            if status is None and not (sys.platform == 'darwin' and time.monotonic() - _started >= 240):
                return
            if status not in (None, 0) and sys.platform.startswith('linux'):
                _linux_failed.add(_process.args[0])
            _stop_process()
        for command in _commands():
            try:
                _process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                _started = time.monotonic()
                return
            except OSError:
                if sys.platform.startswith('linux'):
                    _linux_failed.add(command[0])
        if sys.platform.startswith('linux') and not _missing_logged:
            _LOG.warning('No usable Linux idle-sleep inhibitor; continuing without keep-awake')
            _missing_logged = True


class KeepAwake:
    """Poll activity every ten seconds, or hold continuously when busy is omitted."""

    def __init__(self, *, enabled: bool = True, busy: Callable[[], bool] | None = None):
        self._busy = busy or (lambda: True)
        self._token = object()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        if enabled and (sys.platform == 'darwin' or sys.platform.startswith('linux')):
            self._tick()
            self._thread = threading.Thread(target=self._watch, name='symphonai-keep-awake', daemon=True)
            self._thread.start()

    def _tick(self) -> None:
        try:
            active = self._busy()
        except Exception:
            active = False
            _LOG.debug('Could not determine keep-awake activity', exc_info=True)
        with self._lifecycle:
            if not self._stop.is_set():
                _update(self._token, active)

    def _watch(self) -> None:
        while not self._stop.wait(10):
            self._tick()

    def close(self) -> None:
        self._stop.set()
        with self._lifecycle:
            _update(self._token, False)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)


def _cleanup() -> None:
    with _lock:
        _holders.clear()
        _stop_process()


atexit.register(_cleanup)
