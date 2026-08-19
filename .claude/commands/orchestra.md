Use the Orchestra workflow.

User request:

$ARGUMENTS

Instructions:

1. Follow `CLAUDE.md` Orchestra Mode and `.claude/skills/orchestra/SKILL.md` as the workflow definition.
2. Act as the main orchestrator.
3. Any subagent dispatch should be Codex MCP if possible — Gemini CLI, OpenCode CLI, and future API-key workers are fallbacks only, for when Codex genuinely cannot do the task or the user names a different tool.
4. For non-trivial work, propose a task graph — each task proposed as a Codex dispatch — and wait for approval before creating tasks. Small, well-scoped changes to one or two files can be implemented directly instead.
5. Still ask before dispatching, every time, for that specific task — proposing Codex as the default does not remove the confirmation step.
6. Persist all task/worker/run state via the Orchestra MCP tools; never edit `.orchestra/tasks.json` by hand.
7. Once a worker dispatch is approved: create a Git worktree for that task, dispatch the worker, and register it (with its thread/session id) via the Orchestra MCP tools.
8. Review worker outputs before marking tasks complete.
9. Ask before running `git commit`. Do not merge without explicit user approval.
