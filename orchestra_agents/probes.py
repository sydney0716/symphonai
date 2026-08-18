"""Static, candidate capability specs for known coding-agent CLIs.

Everything in this module is a CANDIDATE derived from published CLI docs and
--help references, not a confirmed runtime fact about any machine this code
actually runs on. A flag listed here may not exist in the installed version
of a given CLI, may have been renamed, or may behave differently than
documented upstream.

Local behavior must be verified by `scripts/probe_cli_agents.py` before any
adapter relies on these specs. No adapter should treat an entry in this
module as a confirmed capability without a passing probe result (see
`ValidationResult` in `orchestra_agents.models`).

Sourced from, at research time:
- Claude Code CLI reference (code.claude.com/docs/en/cli-reference)
- OpenAI Codex CLI developer-commands docs
- Gemini CLI headless.md docs

This module holds data only. Importing it executes no subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestra_agents.models import ProviderCapability


@dataclass(frozen=True)
class CliProbeSpec:
    """A candidate description of one CLI binary's scriptable surface.

    Every field is an unverified candidate until a probe run confirms it
    against the actual installed binary; see `scripts/probe_cli_agents.py`.
    """

    binary: str
    version_flag_candidates: tuple[str, ...]
    help_flag_candidates: tuple[str, ...]
    non_interactive_flag_candidates: tuple[str, ...]
    structured_output_flag_candidates: tuple[str, ...]
    resume_flag_candidates: tuple[str, ...]
    cwd_flag_candidates: tuple[str, ...]
    sandbox_permission_flag_candidates: tuple[str, ...]
    notes: str = ""

    def as_capabilities(self) -> list[ProviderCapability]:
        """Represent this spec's candidate flag groups as ProviderCapability entries.

        Every entry comes back with `confirmed=False`: nothing in this
        module has been probed against a real binary yet.
        """
        groups = {
            "non_interactive": self.non_interactive_flag_candidates,
            "structured_output": self.structured_output_flag_candidates,
            "resume": self.resume_flag_candidates,
            "cwd": self.cwd_flag_candidates,
            "sandbox_permission": self.sandbox_permission_flag_candidates,
        }
        return [
            ProviderCapability(
                name=group_name,
                supported=bool(flags),
                detail=(
                    f"candidate flags: {', '.join(flags)}"
                    if flags
                    else "no candidate flags recorded"
                ),
                confirmed=False,
            )
            for group_name, flags in groups.items()
        ]


CLAUDE_PROBE = CliProbeSpec(
    binary="claude",
    version_flag_candidates=("--version",),
    help_flag_candidates=("--help",),
    non_interactive_flag_candidates=("-p", "--print"),
    structured_output_flag_candidates=(
        "--output-format text",
        "--output-format json",
        "--output-format stream-json",
        "--input-format stream-json",
        "--json-schema",
    ),
    resume_flag_candidates=(
        "--resume",
        "-r",
        "--continue",
        "-c",
        "--session-id",
        "--fork-session",
    ),
    cwd_flag_candidates=(),
    sandbox_permission_flag_candidates=(
        "--permission-mode",
        "--dangerously-skip-permissions",
        "--max-turns",
        "--max-budget-usd",
    ),
    notes=(
        "Candidate flags per the Claude Code CLI reference; not yet probed "
        "against a local install. No dedicated cwd flag was found in the "
        "docs -- it presumably runs in the process's own working directory, "
        "but that is itself an unverified assumption until probed."
    ),
)

CODEX_PROBE = CliProbeSpec(
    binary="codex",
    version_flag_candidates=("--version",),
    help_flag_candidates=("--help", "exec --help"),
    non_interactive_flag_candidates=("exec", "e"),
    structured_output_flag_candidates=(
        "--json",
        "--experimental-json",
        "--output-schema",
        "--output-last-message",
        "-o",
    ),
    resume_flag_candidates=(
        "exec resume",
        "exec resume --last",
        "exec resume --all",
    ),
    cwd_flag_candidates=(),
    sandbox_permission_flag_candidates=(
        "--sandbox",
        "-s",
        "--ask-for-approval",
        "-a",
        "--dangerously-bypass-approvals-and-sandbox",
        "--ephemeral",
    ),
    notes=(
        "Candidate flags per OpenAI Codex CLI developer-commands docs; not "
        "yet probed against a local install. No dedicated cwd flag was "
        "found in the docs -- same unverified process-cwd assumption as "
        "the claude entry."
    ),
)

GEMINI_PROBE = CliProbeSpec(
    binary="gemini",
    version_flag_candidates=("--version",),
    help_flag_candidates=("--help",),
    non_interactive_flag_candidates=("-p", "--prompt"),
    structured_output_flag_candidates=(
        "--output-format text",
        "--output-format json",
    ),
    resume_flag_candidates=("--resume", "-r"),
    cwd_flag_candidates=(),
    sandbox_permission_flag_candidates=(),
    notes=(
        "Candidate flags per Gemini CLI headless.md; not yet probed against "
        "a local install. Documented upstream caveat (gemini-cli issue "
        "#14180): when --resume is used, only the deprecated --prompt flag "
        "reliably feeds text in, not stdin or a positional prompt argument. "
        "Treat --resume here as especially unverified until "
        "scripts/probe_cli_agents.py confirms local behavior. No dedicated "
        "cwd or sandbox/permission flag was found in the docs."
    ),
)

ALL_PROBES: tuple[CliProbeSpec, ...] = (CLAUDE_PROBE, CODEX_PROBE, GEMINI_PROBE)
