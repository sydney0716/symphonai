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

- `git status --porcelain`: <verbatim, in a fenced block>
- `git diff --check`: <result>

<Run no checks or smoke test; the reviewer runs them after the change lands. If
you tried something in a scratch run to show a criterion is met, say what you
ran and what it returned.>

## Acceptance Criteria

1. <Met or not met, and what shows it: the code that does it, the command
   output, or — only when the spec added one — the check that protects it.>
2. ...

<Answer every numbered criterion in the spec's order, including the ones that
were not met. "Met" without saying what shows it cannot be checked.>

## Risks & Notes

<Anything uncertain, anything left undone, and any pre-existing working-tree
change you did not make. A decision the spec left to you belongs here with the
reason you chose it. So does anything you noticed and deliberately did not
touch.>
