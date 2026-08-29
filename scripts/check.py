#!/usr/bin/env python3
"""Run registered repository checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.checks import shell_and_registry  # noqa: E402, F401
from scripts.checks.harness import names, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list registered checks")
    parser.add_argument("--only", metavar="SUBSTRING", help="run matching checks only")
    arguments = parser.parse_args()
    if arguments.list:
        for name in names():
            print(name)
        return
    sys.exit(run(arguments.only))


if __name__ == "__main__":
    main()
