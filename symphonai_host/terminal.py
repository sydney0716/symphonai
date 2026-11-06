"""The person's project shells, backed by pseudo-terminals."""

from __future__ import annotations

from symphonai_api.process import stop_process_group

import codecs
from collections import deque
import logging
import fcntl
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import termios
import threading
import time
import uuid
import weakref


class Terminal:
    def __init__(self, root: Path, publish):
        self.id = uuid.uuid4().hex
        self.exited = False
        self.exit_code = None
        self._chunks = deque()
        self._length = 0
        self.end = 0
        self._pending = deque()
        self._pending_length = 0
        self._last_publish = 0.0
        self._publish = publish
        self._lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._stopping = threading.Event()
        self.pid, self._fd = pty.fork()
        if self.pid == 0:
            try:
                os.chdir(root)
                environment = dict(os.environ, TERM="xterm-256color", COLORTERM="truecolor")
                shell = environment.get("SHELL") or "/bin/sh"
                os.execvpe(shell, [shell, "-l", "-i"], environment)
            except Exception:
                os._exit(127)
        os.set_blocking(self._fd, False)
        self.resize(80, 24)
        self._reader = threading.Thread(target=self._read, name=f"terminal-{self.id}", daemon=True)
        self._reader.start()

    def _append(self, text):
        with self._lock:
            if not text:
                return
            self._chunks.append(text)
            self._length += len(text)
            self.end += len(text)
            while self._length > 200000:
                excess = self._length - 200000
                first = self._chunks.popleft()
                removed = min(excess, len(first))
                if removed < len(first):
                    self._chunks.appendleft(first[removed:])
                self._length -= removed
            self._pending.append(text)
            self._pending_length += len(text)
            while self._pending_length > 65536:
                first = self._pending.popleft()
                removed = min(self._pending_length - 65536, len(first))
                if removed < len(first):
                    self._pending.appendleft(first[removed:])
                self._pending_length -= removed

    def _flush(self, exited=False):
        delay = 0.03 - (time.monotonic() - self._last_publish)
        if delay > 0:
            time.sleep(delay)
        with self._lock:
            text = ''.join(self._pending)
            self._pending.clear()
            self._pending_length = 0
            offset = self.end - len(text)
            code = self.exit_code
        if not text and not exited:
            return
        self._last_publish = time.monotonic()
        try:
            # Even JSON's worst-case surrogate escaping stays below 64 KiB.
            for start in range(0, len(text), 4096) if text else (0,):
                last = start + 4096 >= len(text)
                self._publish(self.id, text[start:start + 4096], exited and last,
                              code if exited and last else None, offset + start)
        except Exception:
            logging.getLogger(__name__).exception("Could not publish terminal output")

    def _read(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        closed = False
        try:
            while not closed:
                # Idle readers sleep until output or PTY closure, without polling.
                select.select([self._fd], [], [])
                deadline = max(time.monotonic(), self._last_publish + 0.03)
                while True:
                    try:
                        data = os.read(self._fd, 8192)
                    except BlockingIOError:
                        break
                    except OSError:
                        closed = True
                        break
                    if not data:
                        closed = True
                        break
                    self._append(decoder.decode(data))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([self._fd], [], [], remaining)[0]:
                        break
                if not closed:
                    self._flush()
        finally:
            with self._lock:
                os.close(self._fd)
                self._fd = -1
            _, status = os.waitpid(self.pid, 0)
            self._append(decoder.decode(b"", final=True))
            with self._lock:
                self.exited = True
                self.exit_code = os.waitstatus_to_exitcode(status)
            self._flush(exited=True)

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
            return {"id": self.id, "exited": self.exited, "exit_code": self.exit_code, "output": "".join(self._chunks), "end": self.end}

    def stop(self, groups=None, *, wait=True):
        if self.exited or self._stopping.is_set():
            if wait:
                self._reader.join(timeout=0.3)
            return
        self._stopping.set()
        if groups is None:
            groups = _process_groups([self])[self.pid]
        stop_process_group(None, tuple(set(groups) | {self.pid}), grace=0.3 if wait else 0)
        if wait:
            self._reader.join(timeout=0.3)


def _process_groups(terminals):
    processes = []
    try:
        listing = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid="],
                                 capture_output=True, text=True, timeout=0.2, check=True)
        processes = [tuple(map(int, line.split())) for line in listing.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    result = {}
    for terminal in terminals:
        descendants = {terminal.pid}
        while True:
            found = {pid for pid, parent, _ in processes if parent in descendants}
            if found <= descendants:
                break
            descendants.update(found)
        groups = {terminal.pid} | {group for pid, _, group in processes if pid in descendants}
        result[terminal.pid] = groups - {os.getpgrp(), 0}
    return result


class TerminalSessions:
    def __init__(self, root: Path, publish):
        self._root = root
        self._publish = publish
        self._lock = threading.Lock()
        self._sessions = {}
        self._closed = False

    def start(self):
        with _LOCK:
            running = sum(sessions.running_count() for sessions in _HOSTS.values())
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

    def running_count(self):
        with self._lock:
            return sum(not terminal.exited for terminal in self._sessions.values())

    def list(self):
        with self._lock:
            return [terminal.snapshot() for terminal in self._sessions.values()]

    def close(self):
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
        groups = _process_groups(sessions) if sessions else {}
        for terminal in sessions:
            terminal.stop(groups[terminal.pid], wait=False)
        for terminal in sessions:
            terminal._reader.join(timeout=0.6)
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
