# Host sidecar

Build it with `python3 scripts/build_host.py` after installing the build extra:
`python3 -m pip install -e ".[build]"`. The command produces an onedir
sidecar bundle in `dist/SymphonAI-host-<target-triple>/`, whose executable has
the same target-suffixed name.

Copy the complete target-suffixed directory beside the Tauri app binary, with
the executable and its `_internal/` directory kept together. The suffix is
mandatory: Tauri uses it to select the host-platform sidecar. Sign the bundle
as part of signing the app; do not try to sign the sidecar separately.

On macOS arm64, the one-file form failed to hand back a startup handshake
within the three-second budget. The onedir form was therefore chosen to avoid
unpacking the Python archive at every app launch. The onedir form also exceeds
the three-second budget on the build host (the guard reports at least 3.00
seconds, and an uncapped diagnostic measurement observed 7.45 seconds), so
`scripts/build_host.py` fails loudly rather than silently shipping an
unmeasured regression.

The sidecar writes its JSON handshake as its first stdout line. The parent must
keep stdout free of wrapper logging until it has read that line. The parent also
owns the process lifetime and must terminate the sidecar on app exit: the
host's signal handler only handles signals that the parent actually sends.
