# <task-id> — <short title>

<!-- Lives in `specs/<phase>/`; its report goes in `specs/report/<phase>/`.
     Read this file when writing a spec. Do not copy the shape of a sibling
     spec: each copy inherits the last one's bloat and nothing pulls it back. -->

## Goal

<One or two sentences: what should be true when this is done, and why it
matters. If the change is app-only or host-only, say so here.>

## Scope

Edit only:
- `path/to/file.py`

Do not touch: <anything adjacent that could plausibly be dragged in>.

<Registry delta: state `+N` or `0`, never an absolute count — a sibling spec
landing first moves both numbers. Ask for the before and after to be reported.>

Never any file under `specs/` except this task's report: a spec is the contract
the work is graded against, so an implementer that edits it is grading itself.
Report a wrong or stale spec instead of correcting it.

## Context

<Only what Codex would otherwise get wrong. Not a tour of code it can read.>

Belongs here:
- traps — an enumeration that must be extended, a check asserting the exact
  shape you are changing, a fixture built in thirty-two places, a guard that
  fails when a path already exists;
- an interface two tasks share, written verbatim and marked fixed;
- the current behaviour, where the spec exists because it is wrong.

Every sentence here comes from a command that was actually run. Run
`./scripts/spec_claims.sh <spec>` before handover: it executes the commands the
spec quotes and flags claims that something is absent.

## Contract

<What the thing must do. Exact values only where a reasonable implementation
could differ — a distinction like empty-list versus null, a join that could go
either way. Where you have read the code, be exact: paths, symbols, line
numbers. Where you have not, state the outcome required and let Codex find the
mechanism; a step prescribed on a guessed premise is dead weight when the
premise is wrong, and it stops the run.>

## Acceptance criteria

1. <A property, not a restatement of the Contract. Name what would fail
   without the change: "conflating the two fails it", "an implementation built
   from the payload's keys returns [] here".>
2. ...

## Tests

<What to add, and to which file.>

<Do not write a mutation table, and do not revert-and-restore to prove a test
can fail. Break-testing cost roughly twenty-four round trips per run and never
found a defect. The guarantee comes from the review, which runs the property
independently, reads the fixtures for the case they never construct, and diffs
against the real old code when a change claims to preserve behaviour. Instead,
name in each criterion the assertion that would fail without the change.>

## Validation

<Scope this to what the change can reach. A command that cannot be affected by
the change is pure cost.>

- `python3 scripts/check.py --only <area>` — matches most changes; match the
  selector to the surface you touched.
- `python3 scripts/check.py` — the whole suite, once before you finish.
- `python3 scripts/checks/_selfcheck.py` — only when this change adds, renames
  or reorders a check name. Once, never inside a loop.
- `.venv/bin/python scripts/smoke_host.py` — only when the host changed.
- `.venv/bin/python scripts/smoke_tui.py` — only when the runtime or the TUI
  changed. A change under `symphonai_app/` cannot reach either.
- `git status --porcelain` and `git diff --check` — always. Paste the first
  verbatim.

## Report

Write the report to `specs/report/<phase>/<task-id>-report.md`. Sections, in
order: `Summary`, `Changed Files` (path plus what changed), `Validation` (each
command and its real result), `Acceptance Criteria` (each numbered criterion,
met or not), and `Notes` (anything left undone, and any pre-existing
working-tree change you did not make). This is the one path outside `Scope` you
may create.

## Notes

<Known traps. Prior art. Anything deliberately out of scope.>
