# <task-id> — <short title> report

<!-- Lives in `specs/report/<phase>/<task-id>-report.md`, the one path outside a
     spec's Scope that the implementer may create. Read this file when writing a
     report; a report is read against its spec, section by section. -->

## Summary

<Two or three sentences: what changed and why it was needed. If the run stopped
without implementing, say so here and put the reason first — a stopped run is a
result, not a failure, and the spec being wrong is the most useful thing a
report can say.>

## Changed Files

- `path/to/file.py`: <what changed in it, and why that was the change>

<One line per file. A file that changed for a reason the spec did not ask for
belongs in `Risks & Notes` as well.>

## Validation

- `<command>`: <its real result — counts, pass/fail, the actual output line>

<Every command the spec's `Validation` listed, each with what it really
returned. A command whose result is not written down did not happen. Say which
commands you did not run and why. When a command needs an environment — a
temporary `SYMPHONAI_SESSIONS_DIR`, local socket access — name it.>

- Registry: <before> before, <after> after, from
  `python3 scripts/check.py --list | wc -l`.
- `git status --porcelain`: <verbatim, in a fenced block>
- `git diff --check`: <result>

## Acceptance Criteria

1. <Met or not met, and the assertion that carries it. For a criterion that
   names an assertion which would fail without the change, say where that
   assertion lives.>
2. ...

<Answer every numbered criterion in the spec's order, including the ones that
were not met. "Met" without saying what makes it met cannot be checked.>

## Risks & Notes

<Anything uncertain, anything left undone, and any pre-existing working-tree
change you did not make. A decision the spec left to you belongs here with the
reason you chose it. So does anything you noticed and deliberately did not
touch.>
