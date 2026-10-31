# <task-id> — <short title>

<!-- Lives in `specs/<phase>/`; its report goes in `specs/report/<phase>/`. -->

## Goal

<One or two sentences: what should be true when this is done, and why it
matters. If the change is app-only or host-only, say so here.>

## Scope

Edit only:
- `path/to/file.py`

Do not touch: <anything adjacent that could plausibly be dragged in>.

<Registry delta: usually 0. Add a check only for behaviour that is easy to
break by accident and would hurt if broken — permissions and secrets, saved
history, what a vendor needs sent back, the core loop, the host's safety — and
put it in its core area (`access.`, `history.`, `vendors.`, `turns.`, `host.`).
Never "at least +N".>

## Context

<Only what Codex would otherwise get wrong: traps such as a check asserting the
exact shape being changed, an interface two tasks share, or the current
behaviour when it is the bug.>

## Contract

<What the thing must do. Exact values only where a reasonable implementation
could differ.>

## Acceptance criteria

1. <A property of the finished change, stated so the reviewer can confirm it.>
2. ...

## Checks

<Only if Registry delta is not 0: the behaviour the new check protects, and its
core area. A check tests what a caller sees and fails only when that behaviour
breaks — no prose pins, no patching private names, no reading source or docs,
no snapshots of earlier output. Omit this section otherwise.>

## Validation

- `git status --porcelain` and `git diff --check` — always. Paste the first
  verbatim. Codex runs no checks; the reviewer runs the suite and the smoke
  test after the change lands.

## Report

`specs/report/<phase>/<task-id>-report.md`

## Notes

<Known traps. Prior art. Anything deliberately out of scope.>
