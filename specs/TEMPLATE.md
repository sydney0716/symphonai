# <task-id> — <short title>

<!-- Lives in `specs/<phase>/`; its report goes in `specs/report/<phase>/`. -->

## Goal

<One or two sentences: what should be true when this is done, and why it
matters. If the change is app-only or host-only, say so here.>

## Scope

Edit only:
- `path/to/file.py`

Do not touch: <anything adjacent that could plausibly be dragged in>.

<Registry delta: the checks the acceptance criteria need, usually one per
criterion that names new behaviour; `0` is fine. Never "at least +N": a minimum
invites checks written for the count.>

## Context

<Only what Codex would otherwise get wrong: traps such as a check asserting the
exact shape being changed, an interface two tasks share, or the current
behaviour when it is the bug.>

## Contract

<What the thing must do. Exact values only where a reasonable implementation
could differ.>

## Acceptance criteria

1. <A property, naming what would fail without the change.>
2. ...

## Tests

<What to add, and to which file.>

## Validation

- `python3 scripts/check.py --only <area>.` — the selectors for the checks
  this change adds or touches, each ending with a dot. Never the full suite:
  the reviewer runs it once on the final tree.
- `python3 scripts/checks/_selfcheck.py` — only when this change edits
  `scripts/check.py` or `scripts/checks/harness.py`. Once.
- `.venv/bin/python scripts/smoke_host.py` — only when the host changed.
- `git status --porcelain` and `git diff --check` — always. Paste the first
  verbatim.

## Report

`specs/report/<phase>/<task-id>-report.md`

## Notes

<Known traps. Prior art. Anything deliberately out of scope.>
