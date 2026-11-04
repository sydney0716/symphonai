"""Codex-format patches, preflighted before any file is changed."""

from __future__ import annotations


from symphonai_api.models import ToolCall, ToolResult
from symphonai_api.permissions import approval_diff
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.edit import _diff_result
from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
from symphonai_api.tools.metadata import ResultHint, ToolEffect, ToolMetadata, patch_paths


def parse_patch(text):
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch" or lines[-1].strip() != "*** End Patch":
        raise ValueError("patch must start with *** Begin Patch and end with *** End Patch")
    operations = []
    index = 1
    while index < len(lines) - 1:
        header = lines[index]
        kind = next((kind for kind in ("Add", "Delete", "Update") if header.startswith(f"*** {kind} File: ")), None)
        if kind is None:
            raise ValueError(f"invalid patch header: {header}")
        path = header.split(": ", 1)[1]
        if not path.strip():
            raise ValueError("patch path is blank")
        index += 1
        move = None
        if kind == "Update" and lines[index].startswith("*** Move to: "):
            move = lines[index][13:]
            if not move.strip():
                raise ValueError(f"{path}: move destination is blank")
            index += 1
        body = []
        while index < len(lines) - 1 and not any(lines[index].startswith(f"*** {value} File: ") for value in ("Add", "Delete", "Update")):
            body.append(lines[index])
            index += 1
        if kind == "Delete" and body:
            raise ValueError(f"{path}: delete takes no hunks")
        if kind == "Add" and any(not line.startswith("+") for line in body):
            raise ValueError(f"{path}: added lines must start with +")
        operations.append((kind, path, move, body))
    if not operations:
        raise ValueError("patch has no files")
    return operations


_PUNCTUATION = str.maketrans({**dict.fromkeys("‐‑‒–—―−", "-"), **dict.fromkeys("‘’‚‛", "'"),
                                **dict.fromkeys("“”„‟", '"'), **dict.fromkeys("\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000", " ")})


def _seek(lines, pattern, start, eof=False):
    if not pattern:
        return len(lines) if eof else start
    positions = range(max(start, len(lines) - len(pattern)) if eof else start, len(lines) - len(pattern) + 1)
    for normalize in (lambda value: value, str.rstrip, str.strip, lambda value: value.strip().translate(_PUNCTUATION)):
        expected = [normalize(value) for value in pattern]
        normalized = [normalize(value) for value in lines]
        for position in positions:
            if normalized[position:position + len(pattern)] == expected:
                return position
    return None


def update_text(path, text, body):
    lines = text.splitlines()
    cursor = 0
    index = 0
    while index < len(body):
        header = body[index]
        if header != "@@" and not header.startswith("@@ "):
            raise ValueError(f"{path}: expected @@ hunk, got {header}")
        index += 1
        if header.startswith("@@ "):
            context = header[3:]
            found = _seek(lines, [context], cursor)
            if found is None:
                raise ValueError(f"{path}: context not found: {context}")
            cursor = found + 1
        old, new = [], []
        eof = False
        while index < len(body) and not body[index].startswith("@@"):
            line = body[index]
            index += 1
            if line == "*** End of File":
                eof = True
                break
            if not line:
                line = " "
            if line[0] not in " +-":
                raise ValueError(f"{path}: invalid hunk line: {line}")
            if line[0] in " -":
                old.append(line[1:])
            if line[0] in " +":
                new.append(line[1:])
        if not old and not new:
            raise ValueError(f"{path}: empty hunk")
        found = _seek(lines, old, cursor, eof)
        if found is None:
            raise ValueError(f"{path}: context not found:\n" + "\n".join(old))
        lines[found:found + len(old)] = new
        cursor = found + len(new)
    return "\n".join(lines) + ("\n" if lines else "")


class ApplyPatchTool(LocalTool):
    name = "apply_patch"
    description = "Apply a Codex patch: *** Begin Patch, Add/Delete/Update File, optional Move to, @@ hunks, *** End Patch. Read existing files before editing."
    parameters = {"type": "object", "properties": {"input": {"type": "string", "minLength": 1}}, "required": ["input"]}

    def __init__(self, ledger, checkpoints=None):
        self._ledger = ledger
        self._checkpoints = checkpoints

    def metadata(self, arguments):
        return ToolMetadata(ToolEffect.DESTRUCTIVE, concurrency_safe=False,
                            paths=tuple(patch_paths(arguments.get("input"))) or None, result_hint=ResultHint.DIFF)

    def validate(self, arguments):
        try:
            parse_patch(arguments["input"])
        except ValueError as exc:
            return str(exc)
        return None

    def _execute(self, tool_call, policy, cancel=None):
        originals, changes, labels = {}, {}, []
        try:
            operations = parse_patch(tool_call.arguments["input"])
            # Deny every forbidden path before even reading another patch file.
            for _, path, move, _ in operations:
                for name in (path, move) if move else (path,):
                    decision = policy.check_read(name) if policy.mode in ("ask", "auto") else policy.check_write(name)
                    if not decision.allowed:
                        return ToolResult(tool_call.id, ok=False, error=f"{name}: {decision.reason}")
            for kind, path, move, body in operations:
                resolved = policy._resolve_within_root(path)
                destination = policy._resolve_within_root(move) if move else resolved
                for name, target in ((path, resolved), (move or path, destination)):
                    if target in originals:
                        continue
                    originals[target] = (target.read_bytes(), target.stat().st_mode) if target.exists() else None
                    if originals[target] is not None:
                        originals[target][0].decode("utf-8")
                        stale = self._ledger.check(target)
                        if stale:
                            raise ValueError(f"{name}: {stale}")
                if resolved in changes or (move and destination in changes):
                    raise ValueError(f"{path}: repeated patch path")
                original = originals[resolved]
                if kind == "Add":
                    if original is not None:
                        raise ValueError(f"{path}: file already exists")
                    updated = "\n".join(line[1:] for line in body) + ("\n" if body else "")
                else:
                    if original is None:
                        raise ValueError(f"{path}: file does not exist")
                    updated = None if kind == "Delete" else update_text(path, original[0].decode("utf-8"), body)
                if move and destination != resolved:
                    if originals[destination] is not None:
                        raise ValueError(f"{move}: move destination already exists")
                    changes[resolved] = None
                changes[destination] = updated.encode("utf-8") if updated is not None else None
                labels.append(f"{'move' if move else kind.lower()} {path}" + (f" -> {move}" if move else ""))
            if policy.mode in ("ask", "auto"):
                for path, content in changes.items():
                    old = originals[path]
                    decision = policy.check_write(str(path), diff=approval_diff(
                        old[0].decode("utf-8") if old else None,
                        content.decode("utf-8") if content is not None else "", str(path)))
                    if not decision.allowed:
                        return ToolResult(tool_call.id, ok=False, error=f"{path}: {decision.reason}")
                    if old is not None:
                        stale = self._ledger.check(path)
                        if stale:
                            raise ValueError(f"{path}: {stale}")
            if cancel is not None:
                cancel.raise_if_cancelled()
            # All hunks, permissions and read guards passed. Checkpoint every path
            # before the first mutation, so additions, deletions and moves undo too.
            if self._checkpoints is not None:
                for path in changes:
                    self._checkpoints.before_write(path)
            written = []
            try:
                for path, content in changes.items():
                    written.append(path)
                    if content is None:
                        path.unlink()
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                if self._checkpoints is not None:
                    for path in changes:
                        self._checkpoints.after_write(path)
            except OSError:
                for path in reversed(written):
                    original = originals[path]
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_bytes(original[0])
                        path.chmod(original[1])
                raise
            results = []
            for path, content in changes.items():
                original = originals[path]
                old = original[0].decode("utf-8") if original else ""
                new = content.decode("utf-8") if content is not None else ""
                relative = path.relative_to(policy.repo_root.resolve()).as_posix()
                results.append(_diff_result(tool_call.id, relative, old, new))
                if content is not None:
                    self._ledger.record(path, full=True, content=new)
            return ToolResult(tool_call.id, ok=True, content="\n".join(labels), payload={
                "kind": "file_diff", "path": ", ".join(result.payload["path"] for result in results),
                "diff": "\n".join(result.content for result in results),
                "lines_added": sum(result.payload["lines_added"] for result in results),
                "lines_removed": sum(result.payload["lines_removed"] for result in results),
                "truncated": any(result.payload["truncated"] for result in results),
                "hunks": [dict(hunk, path=result.payload["path"]) for result in results for hunk in result.payload["hunks"]],
            })
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(tool_call.id, ok=False, error=str(exc))


class _CompletedWrite(LocalTool):
    name = "completed_patch"
    description = "An already applied file change."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def __init__(self, result):
        self.result = result

    def metadata(self, arguments):
        return ToolMetadata(ToolEffect.READ_ONLY, concurrency_safe=False, paths=(arguments["path"],))

    def _execute(self, tool_call, policy, cancel=None):
        return self.result


class DiagnosticsAfterPatch(DiagnosticsAfterWrite):
    def _execute(self, tool_call, policy, cancel=None):
        result = self._tool.execute(tool_call, policy, cancel=cancel)
        if result.ok:
            for path in patch_paths(tool_call.arguments["input"]):
                resolved = policy._resolve_within_root(path)
                if resolved.is_file():
                    result = DiagnosticsAfterWrite(_CompletedWrite(result), self._manager).execute(
                        ToolCall(tool_call.id, "completed_patch", {"path": path}), policy, cancel=cancel)
        return result
