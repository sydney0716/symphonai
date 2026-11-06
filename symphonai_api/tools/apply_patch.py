"""Codex-format patches, preflighted before any file is changed."""

from __future__ import annotations


from symphonai_api.models import ToolResult
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.edit import _diff_result, approve_and_write
from symphonai_api.tools.diagnostics import DiagnosticsAfterWrite
from symphonai_api.tools.metadata import ToolMetadata, patch_paths


def parse_patch(text):
    lines = text.strip().replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "*** Begin Patch" or lines[-1].strip() != "*** End Patch":
        raise ValueError("patch must start with *** Begin Patch and end with *** End Patch")
    operations = []
    index = 1
    while index < len(lines) - 1:
        header = lines[index]
        kind = next((kind for kind in ("Add", "Delete", "Update") if header.startswith(f"*** {kind} File: ")), None)
        if kind is None:
            raise ValueError(f"invalid patch header: {header}")
        path = header.split(": ", 1)[1].strip()
        if not path.strip():
            raise ValueError("patch path is blank")
        index += 1
        move = None
        if kind == "Update" and lines[index].startswith("*** Move to: "):
            move = lines[index][13:].strip()
            if not move.strip():
                raise ValueError(f"{path}: move destination is blank")
            index += 1
        body = []
        while index < len(lines) - 1 and not any(lines[index].startswith(f"*** {value} File: ") for value in ("Add", "Delete", "Update")):
            body.append(lines[index])
            index += 1
        if kind == "Delete" and body:
            raise ValueError(f"{path}: delete takes no hunks")
        if kind == "Update" and not body and move is None:
            raise ValueError(f"{path}: update has no hunks or move destination")
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
        return len(lines)
    positions = range(max(start, len(lines) - len(pattern)) if eof else start, len(lines) - len(pattern) + 1)
    for normalize in (lambda value: value, str.rstrip, str.strip, lambda value: value.strip().translate(_PUNCTUATION)):
        expected = [normalize(value) for value in pattern]
        normalized = [normalize(value) for value in lines]
        for position in positions:
            if normalized[position:position + len(pattern)] == expected:
                return position
    return None


def update_text(path, text, body):
    parts = text.split("\n")
    lines = [part[:-1] if part.endswith("\r") else part for part in parts[:-1]]
    endings = ["\r\n" if part.endswith("\r") else "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
        endings.append("")
    ending = next((value for value in endings if value), "\n")
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
        old, new, sources = [], [], []
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
                sources.append(len(old) - 1 if line[0] == " " else None)
        if not old and not new:
            raise ValueError(f"{path}: empty hunk")
        found = _seek(lines, old, cursor, eof)
        if found is None:
            raise ValueError(f"{path}: context not found:\n" + "\n".join(old))
        retained = set(sources)
        removed_endings = iter(endings[found + offset] for offset in range(len(old)) if offset not in retained)
        new_endings = [endings[found + source] if source is not None else next(removed_endings, ending) for source in sources]
        new = [lines[found + source] if source is not None else line for line, source in zip(new, sources)]
        lines[found:found + len(old)] = new
        endings[found:found + len(old)] = new_endings
        cursor = found + len(new)
    if not lines:
        return ""
    endings[-1] = (endings[-1] or ending) if text.endswith("\n") else ""
    return "".join(line + (value or ending) for line, value in zip(lines[:-1], endings[:-1])) + lines[-1] + endings[-1]


class ApplyPatchTool(LocalTool):
    name = "apply_patch"
    description = "Apply a Codex patch: *** Begin Patch, Add/Delete/Update File, optional Move to, @@ hunks, *** End Patch. Read existing files before editing."
    parameters = {"type": "object", "properties": {"input": {"type": "string", "minLength": 1}}, "required": ["input"]}

    def __init__(self, ledger, checkpoints=None):
        self._ledger = ledger
        self._checkpoints = checkpoints

    def metadata(self, arguments):
        return ToolMetadata(concurrency_safe=False)

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
            error = approve_and_write([
                (path, str(path), originals[path][0].decode("utf-8") if originals[path] else None,
                 content.decode("utf-8") if content is not None else None,
                 content.decode("utf-8") if content is not None else None)
                for path, content in changes.items()
            ], policy, self._ledger, self._checkpoints, cancel)
            if error is not None:
                return ToolResult(tool_call.id, ok=False, error=error)
            results = []
            for path, content in changes.items():
                original = originals[path]
                old = original[0].decode("utf-8") if original else ""
                new = content.decode("utf-8") if content is not None else ""
                relative = path.relative_to(policy.repo_root.resolve()).as_posix()
                results.append(_diff_result(tool_call.id, relative, old, new))
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


class DiagnosticsAfterPatch(DiagnosticsAfterWrite):
    def _execute(self, tool_call, policy, cancel=None):
        from dataclasses import replace
        from symphonai_api.lsp import LspError
        from symphonai_api.tools.diagnostics import _utf16_to_column

        result = self._tool.execute(tool_call, policy, cancel=cancel)
        if not result.ok:
            return result
        for relative in patch_paths(tool_call.arguments["input"]):
            if cancel is not None:
                cancel.raise_if_cancelled()
            path = policy._resolve_within_root(relative)
            if not path.is_file():
                continue
            try:
                client = self._manager.client_for(path)
                if client is None:
                    continue
                diagnostics = client.wait_diagnostics(path, client.sync(path), 3.0)
                if not isinstance(diagnostics, list):
                    continue
                errors = [item for item in diagnostics if isinstance(item, dict)
                          and type(item.get("severity")) is int and item["severity"] == 1]
                if not errors:
                    continue
                errors.sort(key=lambda item: (item.get("range", {}).get("start", {}).get("line", 0),
                                              item.get("range", {}).get("start", {}).get("character", 0)))
                lines = []
                for item in errors[:20]:
                    start = item.get("range", {}).get("start", {})
                    line = int(start.get("line", 0))
                    column = _utf16_to_column(path, line, int(start.get("character", 0)))
                    lines.append(f"{path.relative_to(policy.repo_root).as_posix()}:{line + 1}:{column}: {item.get('message', '')}")
                if len(errors) > 20:
                    lines.append(f"… {len(errors) - 20} more")
                appended = f"\n\nErrors reported by {client.spec.name} after this write ({len(errors)}):\n" + "\n".join(lines)
                payload = result.payload
                if isinstance(payload, dict) and payload.get("kind") == "file_diff":
                    payload = dict(payload)
                    payload.setdefault("diff", result.content)
                result = replace(result, content=result.content + appended, payload=payload)
            except (LspError, OSError, UnicodeError, TypeError, ValueError, KeyError, AttributeError):
                continue
        return result
