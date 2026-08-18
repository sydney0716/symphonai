---
name: orchestra
description: Use Claude as the main orchestrator for Orchestra-mode work. Direct Claude implementation is the default; dispatching an external subagent (Codex MCP, Gemini CLI, OpenCode CLI, or a future API-key worker) always requires asking the user first.
---

# Orchestra Skill

Use this skill when the user asks to run the Orchestra workflow, or invokes `/orchestra` or `/orchestra-do`.

## Objective

You are the main orchestrator.

Direct Claude implementation is the default execution mode for small and
medium tasks. Being in Orchestra mode is not by itself a reason to reach
for an external subagent — dispatching one (Codex MCP, Gemini CLI,
OpenCode CLI, or a future API-key based worker; see `orchestra_api/`)
always requires asking the user first and getting an explicit go-ahead
for that specific task. There is no default delegation.

All task/worker/run state changes go through the Orchestra MCP tools
(`task_create`, `task_update`, `worker_register`, `worker_update`,
`validation_record`, `run_record`, `worktree_create`, `state_validate`).
Never edit `.orchestra/tasks.json` by hand.

## Workflow

### 1. Intake

Clarify the user's goal only if absolutely necessary.

If the goal is clear enough, proceed.

Write a short execution plan with:

- goal;
- constraints;
- expected tasks;
- validation strategy.

### 2. Repository Scan

Inspect only the files needed to understand:

- project structure;
- build and test commands;
- relevant modules;
- likely task boundaries.

Avoid broad exploration.

### 3. Scope the Work

- **Small, well-scoped change to one or two files**: skip straight to
  Direct Implementation (step 5) below. No task graph ceremony needed —
  just do it and report what changed.
- **Non-trivial work** (spans multiple files/modules, introduces a new
  abstraction, or requires a design judgment call): propose a task graph
  and wait for the user's approval before creating any tasks or editing
  files.

### 4. Create Task Graph (non-trivial work, after approval)

Propose tasks with:

- `id`;
- `title`;
- `worker` (use `"claude"` for direct implementation, or the worker's
  provider name once a dispatch is approved);
- `depends_on`;
- `scope`;
- `acceptance_criteria`;
- `validation_commands`.

Once approved, create them via the Orchestra MCP `task_create` tool — not
by editing `.orchestra/tasks.json`.

Use this task status set only:

- CREATED
- READY
- RUNNING
- REVIEW
- BLOCKED
- RETRY
- COMPLETE
- FAILED
- CANCELLED

### 5. Direct Implementation (the default path)

For each task (or for the whole small change, if step 3 skipped the task
graph):

1. Mark it `RUNNING` via `task_update` (skip if no task record exists for
   a small change).
2. Implement the change yourself.
3. Run the real validation commands against what you actually built —
   don't just assert success.
4. Record the outcome via `validation_record`, then mark the task
   `COMPLETE` via `task_update`.

No worker entry in `workers[]` and no worktree are needed for this path —
`workers[]` is reserved for actually-spawned external subagents, and a
worktree is for isolating a dispatched worker's edits, not solo direct
work.

### 6. Dispatching a Worker (only after explicit approval)

Before doing any of this, ask the user and get explicit approval to
dispatch a worker for this specific task. Do not treat a prior approval
for a different task as blanket permission.

Once approved:

#### Create a Worktree

For a task being implemented by a dispatched worker, create an isolated
worktree.

Use this naming convention:

- branch: `orchestra/<task-id>-<short-name>`
- worktree: `.orchestra/worktrees/<task-id>-<short-name>`

If the worktree already exists, inspect it before reusing it.

Do not allow two workers to edit the same worktree.

Do not parallelize tasks that may modify the same shared files.

If multiple tasks need the same shared interface, create an interface
task first.

#### Dispatch the Worker

Use the approved subagent's tool (e.g. the Codex MCP `codex` tool) for
the new worker.

Each worker prompt must include:

- task id;
- repo or worktree path;
- allowed files;
- forbidden files;
- exact objective;
- relevant context;
- acceptance criteria;
- validation commands;
- required final report format.

After the call, extract the returned thread/session id and register the
worker via the Orchestra MCP `worker_register` tool — not by editing
`.orchestra/tasks.json`.

#### Worker Prompt Template

Use this structure when dispatching a worker:

    TASK ID:
    <task-id>

    ROLE:
    You are an implementation worker in an Orchestra workflow.

    WORKTREE:
    <relative-or-absolute-worktree-path>

    OBJECTIVE:
    <specific implementation goal>

    ALLOWED SCOPE:
    - <files/directories the worker may edit>

    FORBIDDEN SCOPE:
    - .orchestra/**
    - unrelated modules
    - package/dependency files unless explicitly allowed

    CONTEXT:
    <minimal relevant architecture notes>

    ACCEPTANCE CRITERIA:
    1. <criterion>
    2. <criterion>
    3. <criterion>

    VALIDATION COMMANDS:
    - <command 1>
    - <command 2>

    FINAL REPORT FORMAT:

    ### Summary

    Briefly describe what changed.

    ### Changed Files

    - `path/to/file`: reason

    ### Validation

    - `command`: result

    ### Risks / Notes

    - list unresolved issues, or say `None`

## Context Snapshot Rule

Do not rely on a worker's worktree to contain the supervisor's latest
uncommitted `.orchestra/tasks.json` state.

A worker's worktree is checked out from a specific commit. It does not
see the orchestrator's in-progress, uncommitted edits to
`.orchestra/tasks.json` — including task records, schema changes, or
orchestration history created after that commit. A worker that reads
`.orchestra/tasks.json` from its own worktree can therefore describe
stale or empty state as if it were current.

When a worker needs current task state, schema, or orchestration
history, include a concise context snapshot in the worker prompt.

A dispatched worker must treat the supervisor-provided task brief as the
source of truth for orchestration state, not whatever
`.orchestra/tasks.json` looks like inside its own worktree.

### 7. Review

When a dispatched worker completes:

1. inspect its final report;
2. inspect the Git diff in its worktree;
3. check whether changes stayed inside the allowed scope;
4. run validation commands if appropriate;
5. compare the result with acceptance criteria.

If the task fails, send a corrective instruction on the same worker
thread (e.g. `codex-reply` for a Codex worker).

If the task passes, mark it `COMPLETE` via the Orchestra MCP
`task_update` tool.

### 8. Retry Policy

Use the same worker thread for corrections whenever possible.

Retry on the same thread when:

- implementation is incomplete;
- validation failed;
- acceptance criteria were not met;
- the worker misunderstood the scope.

Create a new worker only when:

- the original worker is stuck;
- the task scope changed materially;
- the thread context became confusing;
- the task should be reassigned.

### 9. Finish

Ask for explicit user approval before running `git commit`. Ask for
explicit user approval before merging — never merge automatically.

Return:

- task summary;
- branches and worktrees (if any were created);
- changed files;
- validation results;
- failed or blocked tasks;
- remaining risks;
- commit/merge recommendation.
