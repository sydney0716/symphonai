export function splitDiff(value) {
  const lines = String(value ?? "").split("\n");
  const rows = [];
  let removed = [];
  let added = [];
  let inHunk = false;
  let remaining = null;

  function flushChanges() {
    const count = Math.max(removed.length, added.length);
    for (let index = 0; index < count; index += 1) {
      rows.push({ kind: "change", old: removed[index] ?? "", new: added[index] ?? "" });
    }
    removed = [];
    added = [];
  }

  for (const line of lines) {
    if (line.startsWith("diff --git ")) { inHunk = false; remaining = null; }
    if (line.startsWith("@@")) {
      inHunk = true;
      const header = line.match(/^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@/);
      remaining = header ? [Number(header[1] ?? 1), Number(header[2] ?? 1)] : null;
      flushChanges();
      rows.push({ kind: "hunk", text: line });
    } else if (line.startsWith("-") && (inHunk || !line.startsWith("---"))) {
      removed.push(line);
    } else if (line.startsWith("+") && (inHunk || !line.startsWith("+++"))) {
      added.push(line);
    } else if (line.startsWith(" ")) {
      flushChanges();
      rows.push({ kind: "context", old: line, new: line });
    } else {
      flushChanges();
      rows.push({ kind: "header", text: line });
    }
    if (inHunk && remaining && !line.startsWith("@@")) {
      if (line[0] === " " || line[0] === "-") remaining[0] -= 1;
      if (line[0] === " " || line[0] === "+") remaining[1] -= 1;
      if (remaining[0] <= 0 && remaining[1] <= 0) inHunk = false;
    }
  }
  flushChanges();
  return rows;
}
