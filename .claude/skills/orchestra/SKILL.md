---
name: orchestra
description: Use Claude as the main orchestrator and delegate implementation tasks to Codex workers through MCP.
---

# Orchestra Skill

Use this skill when the user asks to run a multi-agent coding workflow, delegate work to Codex, or implement a feature through Orchestra.

## Objective

You are the main orchestrator.

Codex MCP workers are implementation agents.

Do not start coding directly unless the change is trivial and the user explicitly asks you to do it yourself.

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

### 3. Create Task Graph

Create tasks with:

- `id`;
- `title`;
- `status`;
- `worker`;
- `depends_on`;
- `scope`;
- `worktree`;
- `branch`;
- `acceptance_criteria`;
- `validation_commands`.

Update `.orchestra/tasks.json`.

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

### 4. Create Worktrees

For each implementation task, create a branch and worktree.

Use this command pattern:

    git worktree add .orchestra/worktrees/<task-id>-<name> -b orchestra/<task-id>-<name>

If the worktree already exists, inspect it before reusing it.

Do not allow two workers to edit the same worktree.

Do not parallelize tasks that may modify the same shared files.

If multiple tasks need the same shared interface, create an interface task first.

### 5. Dispatch Codex Workers

Use the Codex MCP `codex` tool for new workers.

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

After each Codex MCP call, extract the returned `threadId` and save it in `.orchestra/tasks.json`.

### 6. Worker Prompt Template

Use this structure when dispatching a Codex worker:

    TASK ID:
    <task-id>

    ROLE:
    You are a Codex implementation worker in an Orchestra workflow.

    WORKTREE:
    <relative-or-absolute-worktree-path>

    OBJECTIVE:
    <specific implementation goal>

    ALLOWED SCOPE:
    - <files/directories Codex may edit>

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

### 7. Review

When a worker completes:

1. inspect its final report;
2. inspect the Git diff in its worktree;
3. check whether changes stayed inside the allowed scope;
4. run validation commands if appropriate;
5. compare the result with acceptance criteria.

If the task fails, use `codex-reply` with the stored `threadId`.

If the task passes, mark it as `COMPLETE` in `.orchestra/tasks.json`.

### 8. Retry Policy

Use the same Codex worker thread for corrections whenever possible.

Retry with `codex-reply` when:

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

Do not merge automatically.

Return:

- task summary;
- branches and worktrees;
- changed files;
- validation results;
- failed or blocked tasks;
- remaining risks;
- recommended merge order.

Ask for explicit user approval before merging.