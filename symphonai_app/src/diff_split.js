export function splitDiff(value) {
  const lines = String(value ?? "").split("\n");
  const rows = [];
  let removed = [];
  let added = [];

  function flushChanges() {
    const count = Math.max(removed.length, added.length);
    for (let index = 0; index < count; index += 1) {
      rows.push({ kind: "change", old: removed[index] ?? "", new: added[index] ?? "" });
    }
    removed = [];
    added = [];
  }

  for (const line of lines) {
    if (line.startsWith("@@")) {
      flushChanges();
      rows.push({ kind: "hunk", text: line });
    } else if (line.startsWith("-") && !line.startsWith("---")) {
      removed.push(line);
    } else if (line.startsWith("+") && !line.startsWith("+++")) {
      added.push(line);
    } else if (line.startsWith(" ")) {
      flushChanges();
      rows.push({ kind: "context", old: line, new: line });
    } else {
      flushChanges();
      rows.push({ kind: "header", text: line });
    }
  }
  flushChanges();
  return rows;
}
