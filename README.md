# SymphonAI

A coding-agent runtime built around a leader model that delegates to subagents,
and around keeping the person supervising them oriented.

## Why it exists

**Many models, by API — not one subscription.** Tools like Claude Code and
Codex bind you to a single vendor's model and a single vendor's idea of how an
agent should behave. Here a run is assembled from providers: Anthropic, OpenAI,
Gemini, and any OpenAI-compatible endpoint. You choose which model leads and
which models do the work, and the leader and its subagents need not be the same
model or the same vendor. Nothing about the runtime assumes one provider —
tool-schema shaping and model discovery are decided by wire format, so a new
OpenAI-compatible vendor is configuration rather than code.

**Spec and report as the contract between agents.** The leader does not hand a
subagent a vague instruction and hope. It hands over a written spec — file
paths, symbols, the contract to satisfy, the acceptance criteria — and the
subagent hands back a report saying what it changed, what it ran, and which
criteria it met or missed. That exchange is the unit of work. It is also what
makes the result reviewable: a report can be checked against a spec, whereas a
transcript can only be read.

**A dashboard, because agentic work loses the human.** The failure mode of
delegating to models is not bad code, it is cognitive debt — after enough
automated steps, nobody can say what state the project is in or why a decision
was made. So the project's own progress is data, and the views over it are
generated from that data rather than written by hand: the roadmap, what phase
is in flight, what each agent is doing, what a run cost. This is the part still
being built out; today it is a roadmap renderer, and it grows into the app.

## What exists today

The runtime is real and covered by 281 checks that make no network calls:

- **Providers** — Anthropic, OpenAI, Gemini, and OpenAI-compatible endpoints,
  with model discovery, typed retry that separates foreground from background
  work, and vendor round-trip fidelity (a provider's opaque metadata goes back
  untouched).
- **Agent loop** — tools, permissions with typed decisions and named modes,
  parallel execution of concurrency-safe calls with mutations as barriers,
  cancellation that reaches into HTTP reads and running shell commands.
- **Leader and subagents** — dispatch, per-subagent failure breakers, budgets
  for turns, wall time, tokens, and cost.
- **Context** — compaction, tool-result offload to an addressable store, an
  instruction hierarchy with provenance, and accounting for what is loaded.
- **Sessions** — an append-only JSONL transcript per run, resume, and fork from
  an earlier message.
- **Tools** — read, write, edit, multi-edit, list, glob, grep, shell, fetch,
  and search, each gated by permission.

The runtime package is standard-library only. The Textual UI is optional and is a
placeholder until the real app lands.

## Running it

```bash
python3 scripts/check.py                 # the check suite (--only selects a subset)
pip install -e ".[tui]"                  # optional Textual UI
python3 scripts/tui.py                   # run it
```

API keys are read from the environment and never from a file in the repository.

## Where it is going

Streaming, a host process that owns a run and serves an event channel, agents
defined as files, MCP as a client, and then the app that replaces the terminal
UI. The runtime comes first; the interface is deliberately last, so it is not
built twice on top of a runtime that is still moving.

---

This repository holds the runtime. The specs, design notes, and roadmap data
that drive development are kept in the private working repository.
