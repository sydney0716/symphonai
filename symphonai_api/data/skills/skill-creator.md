+++
name = "skill-creator"
description = "Write a new SymphonAI skill from a description or from work just done in this conversation."
when_to_use = "When the person asks to make, write or save a skill, or to turn what was just done into a reusable procedure."
+++
# Skill creator: write a new skill

1. **Find what the skill is for.** Use the person's description, or the steps just taken in this conversation. Ask with ask_user only for what you cannot infer: the name, and when it should be used.
2. **Write it short.** A skill is a procedure another agent follows without this conversation. Keep only the steps, the commands that matter and the traps learned. No history, no explanations of what the code already shows.
3. **Use the format exactly:** a file `<name>.md` with TOML front matter between `+++` lines, then a Markdown body.
   - `name`: lowercase words joined by hyphens, equal to the file name without `.md`.
   - `description`: one sentence saying what it does.
   - `when_to_use`: one sentence saying when an agent should pick it.
   - No other keys.
4. **Place it.**
   - **Project skill** (most skills): hand `.symphonai/skills/<name>.md` to the implementer as a spec. Its Validation checks the file exists and starts with `+++`.
   - **Personal skill, for every project:** you cannot write outside the project. Show the person the full file, to save as `~/.symphonai/skills/<name>.md`.
5. **Finish.** Tell the person the skill's name, that `/<name>` runs it, and that it is picked up when SymphonAI next starts.
