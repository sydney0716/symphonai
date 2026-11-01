+++
name = "cleanup"
description = "Review the changed code for reuse, quality and efficiency, then fix what is worth fixing."
when_to_use = "When the person asks to clean up, simplify or tidy recent changes, or runs /cleanup."
+++
# Cleanup: review the changed code, then fix it

Review every changed file for reuse, quality and efficiency, and fix what is worth fixing. This is not a bug hunt.

## 1. Find the change

Run `git diff` (and `git diff --staged` if anything is staged). If nothing has changed, use the files changed earlier in this conversation. If the person named a focus, keep to it.

## 2. Three reviews at once

Dispatch three explorers in the background in one turn, then wait for all three. Give each the full diff and one of these briefs. Each returns findings with path:line and the existing code to use instead, where there is one.

**Reuse.** For each change, search the codebase for existing helpers and utilities that already do what the new code does: shared modules, utility files, files next to the changed ones. Flag a new function that duplicates an existing one, and inline logic that an existing function already covers: string and path handling, environment checks, type checks.

**Quality.** Flag:
- state that duplicates existing state, or values that could be derived;
- a function that grows another parameter instead of being generalised;
- near-copies of a block that should be one shared piece;
- internals exposed past an existing boundary;
- raw strings where a constant or enum already exists;
- comments that only say what the code does or narrate the change. Keep comments that explain a non-obvious why.

**Efficiency.** Flag:
- repeated work: the same file read twice, duplicate calls, a call per item in a loop;
- independent steps run one after another;
- new blocking work on startup or on a hot path;
- updates that fire when nothing changed;
- checking that something exists right before using it, instead of handling the error;
- unbounded growth, and missing cleanup of listeners or processes;
- reading or loading everything to use one piece.

## 3. Fix

Merge the three reports. Drop findings that are wrong or not worth the change, without arguing them. Hand the rest to the implementer as one spec, with the project's existing tests as its Validation. Then tell the person what was fixed and what was dropped, or that the code was already clean.
