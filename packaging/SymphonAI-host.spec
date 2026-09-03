# PyInstaller recipe for the stdio loopback-host sidecar.
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(SPECPATH).parent
excludes = ["symphonai_tui", "textual", "tkinter", "unittest", "test"]
binary_name = os.environ["SYMPHONAI_HOST_BINARY_NAME"]

a = Analysis(
    [str(ROOT / "symphonai_host" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=binary_name,
    console=True,
    upx=False,
)
