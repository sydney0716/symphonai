#!/usr/bin/env python3
"""Attach a plain-text client through a private handshake pipe."""
from __future__ import annotations
import argparse
import subprocess
import sys
from symphonai_host.client import HostAddress, HostClient, HostClientError
from symphonai_host.cli import run

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--spawn", action="store_true")
mode.add_argument("--attach", action="store_true")
parser.add_argument("--provider", default="openai")
parser.add_argument("--model")
parser.add_argument("--repo-root")
arguments = parser.parse_args()
child = None
try:
    if arguments.spawn:
        command = [sys.executable, "-m", "symphonai_host", "--provider", arguments.provider]
        if arguments.model:
            command.extend(["--model", arguments.model])
        if arguments.repo_root:
            command.extend(["--repo-root", arguments.repo_root])
        child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        if child.stdout is None:
            raise HostClientError("spawned host has no handshake pipe")
        line = child.stdout.readline()
    else:
        line = sys.stdin.readline()
    run(HostClient(HostAddress.from_handshake(line)))
except HostClientError as exc:
    raise SystemExit(str(exc)) from None
finally:
    if child is not None:
        child.terminate()
        child.wait(timeout=5)
