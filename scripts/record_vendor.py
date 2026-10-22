#!/usr/bin/env python3
"""Record one real, streaming vendor exchange cassette."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.checks.vendor_replay import main


if __name__ == "__main__":
    main()
