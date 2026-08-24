#!/usr/bin/env python3
"""Render a human-readable status board from `.orchestra/tasks.json`.

The state file is the source of truth but is ~150KB of JSON, which is not
something anyone wants to read. This renders it: what is done, what is
queued, who the agents are, and what they are working on.

Deliberately a *generator* rather than a checked-in document. A hand-written
status board goes stale the moment a task changes and then quietly lies;
regenerating from state cannot. Nothing here writes -- it only reads state
and prints.

    python3 scripts/status.py              # markdown to stdout
    python3 scripts/status.py --html out.html
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / ".orchestra" / "tasks.json"

# Status values, ordered from "needs attention" to "finished", so the board
# leads with whatever is actually actionable.
STATUS_ORDER = [
    "FAILED",
    "BLOCKED",
    "RETRY",
    "RUNNING",
    "REVIEW",
    "READY",
    "CREATED",
    "CANCELLED",
    "COMPLETE",
]

DONE = {"COMPLETE", "CANCELLED"}

ROLE_DOMAINS = {
    "codex-backend": "orchestra_api/** + its tests and docs",
    "codex-frontend": "orchestra_tui/** + its tests and docs",
    "codex-researcher": "read-only; studies the reference checkouts",
    "codex-reviewer": "read-only; reviews diffs without the task brief",
}


def load_state(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"no state file at {path}")
    return json.loads(path.read_text())


def epic_of(task_id: str) -> str:
    """Group tasks by their id prefix, which is how they were named in practice."""
    for sep in ("-v", "-"):
        if sep in task_id:
            head = task_id.split(sep)[0]
            if head:
                return head
    return task_id


def summarize(state: dict) -> dict:
    tasks = state.get("tasks", [])
    workers = state.get("workers", [])
    by_status: Counter = Counter(t.get("status", "?") for t in tasks)
    open_tasks = [t for t in tasks if t.get("status") not in DONE]

    epics: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        epics[epic_of(t.get("id", "?"))][t.get("status", "?")] += 1

    validations = sum(len(t.get("validation_results", [])) for t in tasks)
    return {
        "tasks": tasks,
        "workers": workers,
        "by_status": by_status,
        "open": open_tasks,
        "epics": epics,
        "validations": validations,
    }


def render_markdown(state: dict) -> str:
    s = summarize(state)
    out: list[str] = ["# Orchestra Status Board", ""]

    total = len(s["tasks"])
    done = sum(s["by_status"][k] for k in DONE if k in s["by_status"])
    out += [
        f"**{total} tasks** — {done} done, **{len(s['open'])} open**. "
        f"{len(s['workers'])} agents. {s['validations']} recorded validations.",
        "",
    ]

    # What needs doing comes first; a board that leads with completed work
    # buries the only part anyone acts on.
    out += ["## Open", ""]
    if not s["open"]:
        out += ["Nothing open.", ""]
    else:
        for status in STATUS_ORDER:
            batch = [t for t in s["open"] if t.get("status") == status]
            for t in batch:
                out.append(f"- **[{status}]** `{t['id']}` — {t.get('title','')}")
                if t.get("worktree"):
                    out.append(f"    - worktree `{t['worktree']}` on `{t.get('branch','?')}`")
        out.append("")

    out += ["## Agents", ""]
    if not s["workers"]:
        out += ["No agents registered.", ""]
    else:
        out += ["| agent | status | task | domain |", "| --- | --- | --- | --- |"]
        for w in s["workers"]:
            wid = w.get("id", "?")
            out.append(
                f"| `{wid}` | {w.get('status','?')} | `{w.get('task_id','—')}` | "
                f"{ROLE_DOMAINS.get(wid, '—')} |"
            )
        out.append("")

    out += ["## Done, by area", ""]
    out += ["| area | complete | open |", "| --- | ---: | ---: |"]
    for epic in sorted(s["epics"]):
        counts = s["epics"][epic]
        complete = sum(counts[k] for k in DONE if k in counts)
        opened = sum(v for k, v in counts.items() if k not in DONE)
        out.append(f"| `{epic}` | {complete} | {opened} |")
    out.append("")

    return "\n".join(out)


# The presentation template lives here, next to the data that fills it, so the
# rendered page can never disagree with `tasks.json`. Editing state is what
# updates the board; nobody hand-edits HTML.
_CSS = """
:root{--ground:#f6f7f9;--surface:#fff;--surface-2:#eef1f5;--ink:#1b1f27;
--ink-soft:#48505e;--muted:#6a7385;--line:#dfe4ea;--accent:#3f6d8c;
--accent-soft:#e4edf3;--open:#b4690e;--open-soft:#fbf0e0;--done:#3f7d62;
--done-soft:#e5efea;--sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#12151a;--surface:#191d24;--surface-2:#21262f;--ink:#e7eaef;
--ink-soft:#b3bbc8;--muted:#8892a3;--line:#2b313b;--accent:#86b0cc;
--accent-soft:#1e2a34;--open:#d9963f;--open-soft:#2c2418;--done:#6fae90;
--done-soft:#1a2621}}
:root[data-theme="dark"]{--ground:#12151a;--surface:#191d24;--surface-2:#21262f;
--ink:#e7eaef;--ink-soft:#b3bbc8;--muted:#8892a3;--line:#2b313b;--accent:#86b0cc;
--accent-soft:#1e2a34;--open:#d9963f;--open-soft:#2c2418;--done:#6fae90;
--done-soft:#1a2621}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
font-size:15px;line-height:1.6;margin:0;padding:3rem 1.25rem 5rem}
.wrap{max-width:64rem;margin:0 auto;display:flex;flex-direction:column;gap:2.75rem}
header{display:flex;flex-direction:column;gap:.4rem}
h1{font-size:clamp(1.6rem,3.5vw,2.1rem);font-weight:600;letter-spacing:-.02em;
margin:0;text-wrap:balance}
.sub{color:var(--muted);margin:0;max-width:62ch}
.stamp{font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;
text-transform:uppercase;color:var(--accent)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
gap:1px;background:var(--line);border:1px solid var(--line);border-radius:6px;
overflow:hidden}
.stat{background:var(--surface);padding:1.1rem 1.25rem;display:flex;
flex-direction:column;gap:.15rem}
.stat .n{font-family:var(--mono);font-size:1.9rem;font-weight:600;line-height:1.1;
font-variant-numeric:tabular-nums;letter-spacing:-.03em}
.stat .k{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.stat.is-open .n{color:var(--open)}.stat.is-done .n{color:var(--done)}
section{display:flex;flex-direction:column;gap:1rem}
h2{font-size:.78rem;font-weight:600;letter-spacing:.11em;text-transform:uppercase;
color:var(--muted);margin:0;padding-bottom:.6rem;border-bottom:1px solid var(--line)}
.items{display:flex;flex-direction:column;gap:.7rem}
.item{background:var(--surface);border:1px solid var(--line);
border-left:3px solid var(--open);border-radius:5px;padding:.95rem 1.15rem;
display:flex;flex-direction:column;gap:.35rem}
.item-head{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem}
.pill{font-family:var(--mono);font-size:.68rem;font-weight:600;letter-spacing:.07em;
padding:.16rem .5rem;border-radius:3px;background:var(--open-soft);color:var(--open);
border:1px solid color-mix(in srgb,var(--open) 30%,transparent)}
.tid{font-family:var(--mono);font-size:.84rem;color:var(--accent)}
.item p{margin:0;color:var(--ink-soft)}
.why{font-size:.87rem;color:var(--muted)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;
color:var(--muted);font-weight:600;padding:.8rem 1rem;border-bottom:1px solid var(--line);
white-space:nowrap}
td{padding:.78rem 1rem;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:none}
td.m,th.m{font-family:var(--mono);font-size:.84rem}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.chip{display:inline-block;font-family:var(--mono);font-size:.68rem;font-weight:600;
letter-spacing:.06em;padding:.14rem .45rem;border-radius:3px;background:var(--done-soft);
color:var(--done);border:1px solid color-mix(in srgb,var(--done) 28%,transparent)}
.ro{background:var(--accent-soft);color:var(--accent);
border-color:color-mix(in srgb,var(--accent) 28%,transparent)}
.bar{height:6px;border-radius:3px;background:var(--surface-2);overflow:hidden;min-width:5rem}
.bar span{display:block;height:100%;background:var(--done);border-radius:3px}
.bar.open span{background:var(--open)}
footer{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:1.25rem}
code{font-family:var(--mono);font-size:.85em;background:var(--surface-2);
padding:.1rem .35rem;border-radius:3px}
"""


def render_html(state: dict) -> str:
    """Render the styled board directly from state.

    Takes `state`, not the markdown, so the page is generated from the same
    source of truth rather than from a scraped rendering of it.
    """
    s = summarize(state)
    total = len(s["tasks"])
    done = sum(s["by_status"][k] for k in DONE if k in s["by_status"])
    e = escape

    stats = [
        ("", total, "Tasks"),
        ("is-done", done, "Complete"),
        ("is-open", len(s["open"]), "Open"),
        ("", len(s["workers"]), "Agents"),
        ("", s["validations"], "Validations"),
    ]
    stat_html = "".join(
        f'<div class="stat {c}"><span class="n">{n}</span><span class="k">{k}</span></div>'
        for c, n, k in stats
    )

    if s["open"]:
        items = []
        for status in STATUS_ORDER:
            for t in (x for x in s["open"] if x.get("status") == status):
                # The first acceptance criterion doubles as the "why this matters"
                # line -- it is where the reason a task exists is actually written.
                why = (t.get("acceptance_criteria") or [""])[0]
                items.append(
                    '<div class="item"><div class="item-head">'
                    f'<span class="pill">{e(status)}</span>'
                    f'<span class="tid">{e(t.get("id",""))}</span></div>'
                    f'<p>{e(t.get("title",""))}</p>'
                    f'<p class="why">{e(why)}</p></div>'
                )
        open_html = f'<div class="items">{"".join(items)}</div>'
    else:
        open_html = '<p class="why">Nothing open.</p>'

    rows = []
    for w in s["workers"]:
        wid = w.get("id", "?")
        read_only = wid in ("codex-researcher", "codex-reviewer")
        chip = f'<span class="chip{" ro" if read_only else ""}">' + e(
            "READ-ONLY" if read_only else w.get("status", "?")
        ) + "</span>"
        rows.append(
            f'<tr><td class="m">{e(wid)}</td><td>{chip}</td>'
            f'<td>{e(ROLE_DOMAINS.get(wid, "—"))}</td>'
            f'<td class="m">{e(str(w.get("thread_id",""))[:8])}…</td></tr>'
        )
    agents_html = (
        '<div class="scroll"><table><thead><tr><th>Agent</th><th>State</th>'
        '<th>Domain</th><th class="m">Thread</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        if rows
        else '<p class="why">No agents registered.</p>'
    )

    area_rows = []
    for epic in sorted(s["epics"], key=lambda x: -sum(s["epics"][x].values())):
        counts = s["epics"][epic]
        complete = sum(counts[k] for k in DONE if k in counts)
        opened = sum(v for k, v in counts.items() if k not in DONE)
        total_e = complete + opened
        pct = round(100 * complete / total_e) if total_e else 0
        bar_cls = "bar" if complete else "bar open"
        area_rows.append(
            f'<tr><td class="m">{e(epic)}</td>'
            f'<td><div class="{bar_cls}"><span style="width:{pct}%"></span></div></td>'
            f'<td class="num">{complete}</td><td class="num">{opened}</td></tr>'
        )
    areas_html = (
        '<div class="scroll"><table><thead><tr><th>Area</th><th>Progress</th>'
        '<th class="num">Done</th><th class="num">Open</th></tr></thead>'
        f'<tbody>{"".join(area_rows)}</tbody></table></div>'
    )

    return f"""<title>Orchestra Status Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>{_CSS}</style>
<div class="wrap">
<header>
<p class="stamp">Generated from .orchestra/tasks.json</p>
<h1>Orchestra Status Board</h1>
<p class="sub">{done} of {total} tasks complete, {len(s['open'])} open, across {len(s['workers'])} agents.</p>
</header>
<div class="stats">{stat_html}</div>
<section><h2>Open — what needs doing</h2>{open_html}</section>
<section><h2>Agents — who is doing what</h2>{agents_html}</section>
<section><h2>Work by area</h2>{areas_html}</section>
<footer>Regenerate with <code>python3 scripts/status.py --html out.html</code>.
State in <code>.orchestra/tasks.json</code> is the source of truth; this page is
rendered from it, never hand-edited.</footer>
</div>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument("--html", default=None, help="render the styled board to this path")
    parser.add_argument("--quiet", action="store_true", help="with --html, skip the markdown")
    args = parser.parse_args(argv)

    state = load_state(Path(args.state))
    if args.html:
        Path(args.html).write_text(render_html(state))
        print(f"wrote {args.html}", file=sys.stderr)
        if args.quiet:
            return 0
    print(render_markdown(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
