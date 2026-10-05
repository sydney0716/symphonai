"""Parsing helpers for the repository's spec workflow."""

from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path


def parse_spec(path: Path, repo_root: Path) -> dict:
    path = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        raise ValueError("spec path must be inside specs/") from None
    if not relative.startswith("specs/") or path.suffix != ".md" or not path.is_file():
        raise ValueError("spec path must name an existing Markdown file under specs/")
    source = path.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in source.splitlines() if line.startswith("# ")), path.stem)
    validation = []
    in_section = False
    in_fence = False
    for line in source.splitlines():
        if line.startswith("## "):
            if in_section and in_fence:
                break
            in_section = line[3:].strip().lower() == "validation"
            continue
        if not in_section:
            continue
        if line.startswith("```"):
            if in_fence:
                break
            in_fence = line[3:].strip() in ("", "bash", "sh")
            continue
        if in_fence and line.strip():
            validation.append(line)
    report_match = re.search(r"`([^`]+)`", source[source.find("## Report"):] if "## Report" in source else "")
    report = report_match.group(1) if report_match else f"specs/report/{path.parent.name}/{path.stem}-report.md"
    report_path = Path(report)
    if report_path.is_absolute() or ".." in report_path.parts or ".env" in report_path.parts:
        raise ValueError("spec report path must remain inside the repository")
    return {"path": relative, "title": title, "text": source, "validation": validation, "report": report}


def review_verdict(answer: str) -> tuple[str, list[str]]:
    last = next((line.strip() for line in reversed(answer.splitlines()) if line.strip()), "")
    if last == "Verdict: pass":
        return "passed", []
    prefix = "Verdict: follow-ups:"
    if last.startswith(prefix):
        paths = [part.strip() for part in last[len(prefix):].split(",") if part.strip()]
        if paths:
            return "follow-ups", paths
    return "no-verdict", []


def patch_digest(patch: str) -> str:
    return hashlib.sha256(patch.encode("utf-8", errors="surrogateescape")).hexdigest()


def bind_roadmap_item(root: Path, phase_id: str, item_index: int, spec_path: str) -> None:
    path = Path(root) / "docs" / "roadmap.json"
    source = path.read_text(encoding="utf-8")
    phase_key = re.search(r'"id"\s*:\s*' + re.escape(json.dumps(phase_id)) + r'\s*,', source)
    if phase_key is None:
        raise ValueError("unknown roadmap phase")
    phase_start = source.rfind("{", 0, phase_key.start())
    phase, _ = json.JSONDecoder().raw_decode(source[phase_start:])
    if not isinstance(phase, dict) or not isinstance(phase.get("items"), list):
        raise ValueError("roadmap phase has no items")
    items_key = re.search(r'"items"\s*:\s*\[', source[phase_start:])
    if items_key is None:
        raise ValueError("roadmap phase has no items")
    cursor = phase_start + items_key.end()
    decoder = json.JSONDecoder()
    values = []
    spans = []
    while cursor < len(source):
        while cursor < len(source) and source[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(source) or source[cursor] == "]":
            break
        start = cursor
        value, end = decoder.raw_decode(source, cursor)
        values.append(value)
        spans.append((start, end))
        cursor = end
    if not 0 <= item_index < len(values):
        raise ValueError("unknown roadmap item")
    item = values[item_index]
    if isinstance(item, str):
        updated = {"title": item, "spec": [spec_path]}
    elif isinstance(item, dict):
        updated = {**item, "spec": [spec_path]}
    else:
        raise ValueError("invalid roadmap item")
    start, end = spans[item_index]
    path.write_text(source[:start] + json.dumps(updated, ensure_ascii=False) + source[end:], encoding="utf-8")


def mark_roadmap_spec_done(root: Path, spec_path: str) -> None:
    path = Path(root) / "docs" / "roadmap.json"
    source = path.read_text(encoding="utf-8")
    roadmap = json.loads(source)
    replacements = []
    decoder = json.JSONDecoder()
    for phase in roadmap.get("phases", []):
        phase_id = phase.get("id")
        match = re.search(r'"id"\s*:\s*' + re.escape(json.dumps(phase_id)) + r'\s*,', source)
        if match is None:
            continue
        phase_start = source.rfind("{", 0, match.start())
        parsed_phase, _ = decoder.raw_decode(source[phase_start:])
        items_key = re.search(r'"items"\s*:\s*\[', source[phase_start:])
        if items_key is None:
            continue
        cursor = phase_start + items_key.end()
        spans = []
        values = []
        while cursor < len(source):
            while cursor < len(source) and source[cursor] in " \t\r\n,":
                cursor += 1
            if cursor >= len(source) or source[cursor] == "]":
                break
            start = cursor
            value, end = decoder.raw_decode(source, cursor)
            spans.append((start, end))
            values.append(value)
            cursor = end
        changed = False
        for index, item in enumerate(values):
            specs = item.get("spec", []) if isinstance(item, dict) else []
            if isinstance(specs, str):
                specs = [specs]
            if spec_path not in specs:
                continue
            updated = {**item, "done": True} if isinstance(item, dict) else {"title": item, "spec": [spec_path], "done": True}
            start, end = spans[index]
            replacements.append((start, end, json.dumps(updated, ensure_ascii=False)))
            values[index] = updated
            changed = True
        if changed:
            status = "done" if values and all(isinstance(item, dict) and item.get("done") is True for item in values) else "in_progress"
            status_match = re.search(r'("status"\s*:\s*)"[^"]*"', source[phase_start:])
            if status_match:
                start = phase_start + status_match.start()
                end = phase_start + status_match.end()
                replacements.append((start, end, status_match.group(1) + json.dumps(status)))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    path.write_text(source, encoding="utf-8")
