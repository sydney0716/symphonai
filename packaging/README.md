# Host sidecar

Build it with `python3 scripts/build_host.py` after installing the build extra:
`python3 -m pip install -e ".[build]"`. The command produces exactly one
executable in `dist/`, named `SymphonAI-host-<target-triple>`.

Copy that target-suffixed executable to a Tauri project's
`src-tauri/binaries/` directory. The suffix is mandatory: Tauri uses it to
select the host-platform sidecar. Sign the executable as part of signing the
app; do not try to sign the sidecar separately.

The sidecar writes its JSON handshake as its first stdout line. The parent must
keep stdout free of wrapper logging until it has read that line. The parent also
owns the process lifetime and must terminate the sidecar on app exit: the
host's signal handler only handles signals that the parent actually sends.
