+++
name = "check-before-push"
description = "Before a commit, push or pull request, run the project's checks that the change could break, and report what passed and failed."
when_to_use = "When the person asks to check before committing or pushing, runs /commit or /pr and wants checks first, or asks whether the change is ready."
+++
# Check before push

1. **See the change.** Run `git status` and `git diff` (staged and unstaged) to know which files and areas changed.
2. **Find how this project checks itself.** Look in this order, and use what exists:
   - the project's instructions (`.symphonai/INSTRUCTIONS.md`, `AGENTS.md`, `CONTRIBUTING.md`);
   - CI workflows (`.github/workflows/`);
   - task files (`package.json` scripts, `Makefile`, `pyproject.toml`, `justfile`, `Cargo.toml`, `go.mod`).

   Note the test, lint, format-check and type-check commands.
3. **Pick what the change could break.**
   - Run checks scoped to the changed area when the project supports it, such as one package's tests or the changed files' lint.
   - Run the full suite only when the change touches shared code, or nothing narrower exists.
   - Never run commands that deploy, publish, push or change remote state.
4. **Run them and read the output.**
   - A check that fails is reported with its command and the failing part of the output. Do not fix it unless the person asks.
   - A check that cannot run, such as a missing tool or service, is reported as not run, with the reason.
5. **Report** one line per check: the command, then passed, failed or not run. End with whether the change looks ready to push.
