"""The person's project shells, backed by pseudo-terminals."""

from __future__ import annotations

import codecs
import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import termios
import threading
import uuid
import weakref


class Terminal:
    def __init__(self, root: Path, publish):
        self.id = uuid.uuid4().hex
        self.exited = False
        self.exit_code = None
        self.output = ""
        self._publish = publish
        self._lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._stopping = threading.Event()
        self.pid, self._fd = pty.fork()
        if self.pid == 0:
            try:
                os.chdir(root)
                environment = dict(os.environ, TERM="dumb", PAGER="cat", GIT_PAGER="cat")
                shell = environment.get("SHELL") or "/bin/sh"
                os.execvpe(shell, [shell, "-l", "-i"], environment)
            except Exception:
                os._exit(127)
        os.set_blocking(self._fd, False)
        self.resize(80, 24)
        self._reader = threading.Thread(target=self._read, name=f"terminal-{self.id}", daemon=True)
        self._reader.start()

    def _read(self):
        import logging

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                try:
                    data = os.read(self._fd, 8192)
                except OSError:
                    break
                if not data:
                    break
                text = decoder.decode(data)
                with self._lock:
                    self.output = (self.output + text)[-200000:]
                try:
                    self._publish(self.id, text, False, None)
                except Exception:
                    logging.getLogger(__name__).exception("Could not publish terminal output")
        finally:
            # Close the master before reaping, including on a reader failure.
            # Closing it also hangs up a shell that is still alive.
            with self._lock:
                os.close(self._fd)
                self._fd = -1
            _, status = os.waitpid(self.pid, 0)
            tail = decoder.decode(b"", final=True)
            with self._lock:
                self.exited = True
                self.exit_code = os.waitstatus_to_exitcode(status)
                self.output = (self.output + tail)[-200000:]
            try:
                self._publish(self.id, tail, True, self.exit_code)
            except Exception:
                logging.getLogger(__name__).exception("Could not publish terminal exit")

    def input(self, data: str):
        encoded = data.encode("utf-8")
        with self._input_lock:
            while encoded:
                with self._lock:
                    if self.exited or self._stopping.is_set():
                        raise ValueError("terminal has exited")
                    try:
                        written = os.write(self._fd, encoded[:8192])
                    except BlockingIOError:
                        written = 0
                encoded = encoded[written:]
                if not written:
                    self._stopping.wait(0.01)

    def resize(self, cols: int, rows: int):
        if not 1 <= cols <= 1000 or not 1 <= rows <= 1000:
            raise ValueError("terminal size must be between 1 and 1000")
        with self._lock:
            if self._fd >= 0:
                fcntl.ioctl(self._fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def snapshot(self):
        with self._lock:
            return {"id": self.id, "exited": self.exited, "exit_code": self.exit_code, "output": self.output}

    def stop(self):
        if self.exited:
            self._reader.join(timeout=0.3)
            return
        if self._stopping.is_set():
            self._reader.join(timeout=0.3)
            return
        self._stopping.set()
        # Interactive jobs can have their own process groups. Capture descendants
        # before terminating the shell so background jobs are stopped as well.
        groups = {self.pid}
        try:
            listing = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid="],
                                     capture_output=True, text=True, timeout=0.2, check=True)
            processes = [tuple(map(int, line.split())) for line in listing.stdout.splitlines()]
            descendants = {self.pid}
            while True:
                found = {pid for pid, parent, _ in processes if parent in descendants}
                if found <= descendants:
                    break
                descendants.update(found)
            groups.update(group for pid, _, group in processes if pid in descendants)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        groups.discard(os.getpgrp())
        groups.discard(0)
        for number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(self.pid, number)
            except ProcessLookupError:
                pass
            for group in groups:
                try:
                    os.killpg(group, number)
                except ProcessLookupError:
                    pass
            if number == signal.SIGTERM:
                self._reader.join(timeout=0.3)
        self._reader.join(timeout=0.3)


class TerminalSessions:
    def __init__(self, root: Path, publish):
        self._root = root
        self._publish = publish
        self._lock = threading.Lock()
        self._sessions = {}
        self._closed = False

    def start(self):
        with _LOCK:
            running = sum(not item["exited"] for sessions in _HOSTS.values() for item in sessions.list())
            if running >= 4:
                raise ValueError("4 terminals are already open")
            with self._lock:
                if self._closed:
                    raise ValueError("terminal host is closing")
                self._sessions = {key: value for key, value in self._sessions.items() if not value.exited}
                terminal = Terminal(self._root, self._publish)
                self._sessions[terminal.id] = terminal
                return terminal.id

    def get(self, identifier):
        with self._lock:
            terminal = self._sessions.get(identifier)
        if terminal is None:
            raise KeyError("unknown terminal")
        return terminal

    def stop(self, identifier):
        terminal = self.get(identifier)
        terminal.stop()
        with self._lock:
            self._sessions.pop(identifier, None)

    def list(self):
        with self._lock:
            return [terminal.snapshot() for terminal in self._sessions.values()]

    def close(self):
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
        for terminal in sessions:
            terminal.stop()
        with self._lock:
            self._sessions.clear()


_LOCK = threading.Lock()
_HOSTS = {}
_CLOSED = weakref.WeakSet()


def sessions_for(owner, root: Path, publish) -> TerminalSessions:
    with _LOCK:
        if owner in _CLOSED:
            raise ValueError("terminal host is closing")
        if owner not in _HOSTS:
            _HOSTS[owner] = TerminalSessions(root, publish)
        return _HOSTS[owner]


def close_host(owner):
    with _LOCK:
        _CLOSED.add(owner)
        sessions = _HOSTS.get(owner)
    if sessions is not None:
        sessions.close()
    with _LOCK:
        _HOSTS.pop(owner, None)
