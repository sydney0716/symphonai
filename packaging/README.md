# Host sidecar

Build it with `python3 scripts/build_host.py` after installing the build extra:
`python3 -m pip install -e ".[build]"`. The command produces an onedir
sidecar bundle in `dist/SymphonAI-host-<target-triple>/`, whose executable has
the same target-suffixed name.

Copy the complete target-suffixed directory beside the Tauri app binary, with
the executable and its `_internal/` directory kept together. The suffix is
mandatory: Tauri uses it to select the host-platform sidecar. Sign the bundle
as part of signing the app; do not try to sign the sidecar separately.

On 2026-09-03, the macOS arm64 build host measured the freshly written onedir
bundle's first launch between 2.07 and 10.06 seconds, from launch to its
handshake. That cold range is first-read I/O and one-time OS verification of
the unsigned 33 MB bundle, not the start an installed, previously-run signed
app user pays. After a discarded warmup launch, the three warm launches have a
0.11-second median. Onefile was rejected because it adds archive extraction to
every launch; onedir avoids that work. `scripts/build_host.py` therefore gates
the three-sample warm median at one second, while still printing the discarded
cold launch for diagnosis.

The sidecar writes its JSON handshake as its first stdout line. The parent must
keep stdout free of wrapper logging until it has read that line. The parent also
owns the process lifetime and must terminate the sidecar on app exit: the
host's signal handler only handles signals that the parent actually sends.

The packaged sidecar defaults to `--permission-mode prompt`: side-effectful
tools park until the client answers their approval request (or it times out).
Clients that deliberately need another policy can pass `auto`, `plan`, or
`accept_edits` explicitly.
