Use the Orchestra workflow.

User request:

$ARGUMENTS

Instructions:

1. Follow `CLAUDE.md` Orchestra Mode and `.claude/skills/orchestra/SKILL.md` as the workflow definition.
2. Act as the main orchestrator.
3. Direct Claude implementation is the default for small/medium tasks — implement it yourself.
4. For non-trivial work, propose a task graph and wait for approval before creating tasks.
5. Ask before dispatching any subagent (Codex MCP, Gemini CLI, OpenCode CLI, or a future API-key worker) as a worker — there is no default delegation.
6. Persist all task/worker/run state via the Orchestra MCP tools; never edit `.orchestra/tasks.json` by hand.
7. If, and only if, a worker dispatch is explicitly approved: create a Git worktree for that task, dispatch the worker, and register it (with its thread/session id) via the Orchestra MCP tools.
8. Review worker outputs before marking tasks complete.
9. Ask before running `git commit`. Do not merge without explicit user approval.
