# Orchestra Directory

The `.orchestra/` directory stores state and coordination artifacts for this
repository's Orchestra multi-agent workflow. It is used by the supervisor to
track tasks, worker assignments, task-specific worktrees, and run history.

## Layout

- `tasks.json` is the task graph and workflow state file. It records tasks,
  workers, runs, assignments, and the current status of each task.
- `tasks/` contains per-task artifacts such as task specs, review notes, or
  other files produced while coordinating a specific task.
- `runs/` contains run logs and other execution history for Orchestra activity.
- `worktrees/` contains isolated Git worktrees created for individual
  implementation tasks.

Task status values in `tasks.json` are:

`CREATED`, `READY`, `RUNNING`, `REVIEW`, `BLOCKED`, `RETRY`, `COMPLETE`,
`FAILED`, `CANCELLED`.

For source-modifying tasks, Orchestra creates a dedicated worktree and branch:

- Branch: `orchestra/<task-id>-<name>`
- Path: `.orchestra/worktrees/<task-id>-<name>`

## Validation Notes

Some validation commands may fail inside a Codex worker's sandbox because of
filesystem or permission restrictions. For example, Git index operations such
as `git add -N` can require access outside the worker's writable area. When
that happens, the supervisor should re-run the validation command outside the
sandbox to confirm the result.
