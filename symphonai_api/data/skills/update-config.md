+++
name = "update-config"
description = "Change SymphonAI's settings from plain words: hooks, permissions, role models and efforts, MCP servers."
when_to_use = "When the person asks to change how SymphonAI behaves, or says 'from now on', 'whenever', 'always' or 'never' about what the agent should do automatically."
+++
# Update config: turn a request into SymphonAI settings

1. **Find the setting.** Settings live in TOML:
   - user: `~/.symphonai/config.toml`, for every project;
   - project: `.symphonai/config.toml`, for this repository.

   Read the existing files first. Common sections:
   - `[[hooks]]`: a command run on an event:
     - `on`: a list of event names, such as `PreToolUse`;
     - `command`: an argv list;
     - optional `timeout_seconds` and `blocking`.
   - `[roles.<name>]`: `provider` and `model` (both required), and optional `effort`, for leader, planner, implementer, reviewer, worker or explorer.
   - `[permissions]`, `[always_deny]`, `[sandbox]` and `[mcp]`: what agents may do, and which tool servers start.
   - `[compaction]`, `[budgets]`, `[memory]` and `[lsp]`: how runs use context, tokens, memory and language servers.
2. **Choose the scope.**
   - **User config only:**
     - `[[hooks]]` and `[mcp]` servers, because they run programs; a repository's hooks stop the configuration from loading unless the person trusts it;
     - `[roles]`, `[permissions]` grants, `[always_deny]` and `[sandbox] readable`, which a project config ignores with a warning.
   - **Everything else** goes where the person wants it to apply.
3. **Show the exact change** as a TOML block, with the file it goes in, and ask the person to approve with ask_user.
4. **Apply.**
   - **Project config:** hand the change to the implementer as a spec. Its Validation parses the file: `python3 -c "import tomllib; tomllib.load(open('.symphonai/config.toml','rb'))"`.
   - **User config:** you cannot write outside the project. Give the person the block to add to `~/.symphonai/config.toml` themselves.
5. **Say when it takes effect:** settings are read when SymphonAI starts, so restart it. A setting it cannot read is reported then, naming the key.
