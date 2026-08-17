Use the Orchestra workflow.

User request:

$ARGUMENTS

Instructions:

1. Follow `CLAUDE.md` Orchestra Mode.
2. Use `.claude/skills/orchestra/SKILL.md` as the workflow definition.
3. Act as the main orchestrator.
4. Delegate implementation to Codex through the Codex MCP tools.
5. Persist task state in `.orchestra/tasks.json`.
6. Create Git worktrees for implementation tasks.
7. Store Codex `threadId` values in `.orchestra/tasks.json`.
8. Review worker outputs before marking tasks complete.
9. Do not merge without explicit user approval.