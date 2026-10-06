function itemId(item, index) {
  const path = Array.isArray(item.spec) ? item.spec[0] : item.spec;
  if (typeof path !== "string") return `item-${index}`;
  const filename = path.split("/").at(-1) ?? path;
  return filename.replace(/\.md$/, "").split("-", 1)[0] || `item-${index}`;
}

function shortTitle(value) {
  const title = typeof value === "string" ? value : "";
  return title.length > 16 ? `${title.slice(0, 16)}…` : title;
}

function valuesFor(source, id) {
  return source instanceof Map ? source.get(id) : source?.[id];
}

export function layoutPhase(items, { followUps = {}, titles = {}, states = {} } = {}) {
  const nodes = items.map((item, index) => {
    const id = itemId(item, index);
    const title = shortTitle(valuesFor(titles, id) || item.title);
    const followUpsForItem = valuesFor(followUps, id) ?? [];
    return {
      id,
      title,
      followUps: followUpsForItem.map((followUp) => typeof followUp === "string"
        ? { id: followUp, title: shortTitle(valuesFor(titles, followUp) || followUp) }
        : { ...followUp, title: shortTitle(valuesFor(titles, followUp.id) || followUp.title || followUp.id) }),
      tags: [],
      state: valuesFor(states, id) ?? (item.done ? "done" : "todo"),
      after: Array.isArray(item.after) ? item.after : [],
      done: item.done === true || valuesFor(states, id) === "done" || valuesFor(states, id) === "committed",
    };
  });
  const byId = new Map();
  for (const node of nodes) {
    if (byId.has(node.id)) return { error: `duplicate id ${node.id}` };
    byId.set(node.id, node);
  }
  const parents = new Map(nodes.map((node) => [node.id, []]));
  for (const node of nodes) {
    for (const dependency of node.after) {
      if (byId.has(dependency)) parents.get(node.id).push(dependency);
      else if (valuesFor(titles, dependency) !== undefined) node.tags.push(`← ${dependency}`);
      else return { error: `unknown dependency ${dependency}` };
    }
  }

  const rowsById = new Map();
  const visiting = new Set();
  function rowFor(id) {
    if (rowsById.has(id)) return rowsById.get(id);
    if (visiting.has(id)) throw new Error(`cycle at ${id}`);
    visiting.add(id);
    const row = parents.get(id).length === 0
      ? 0
      : 1 + Math.max(...parents.get(id).map(rowFor));
    visiting.delete(id);
    rowsById.set(id, row);
    return row;
  }
  try {
    for (const node of nodes) rowFor(node.id);
  } catch (error) {
    return { error: error.message };
  }

  const maxRow = Math.max(-1, ...rowsById.values());
  const logicalColumns = new Map();
  const rows = [];
  const idToRowNode = new Map();
  for (let row = 0; row <= maxRow; row += 1) {
    const members = nodes.filter((node) => rowsById.get(node.id) === row);
    members.sort((left, right) => {
      const average = (node) => {
        const columns = parents.get(node.id).map((id) => logicalColumns.get(id));
        return columns.length ? columns.reduce((sum, value) => sum + value, 0) / columns.length : 0;
      };
      return average(left) - average(right) || left.id.localeCompare(right.id);
    });
    if (members.length >= 3) {
      const group = {
        id: `group-${row}`,
        title: "parallel",
        followUps: [],
        tags: members.flatMap((member) => member.tags),
        state: "group",
        row,
        column: 0,
        group: members.map((member) => ({ id: member.id, title: member.title, followUps: member.followUps, tags: member.tags, state: member.state })),
      };
      rows.push([group]);
      for (const [column, member] of members.entries()) {
        logicalColumns.set(member.id, column < 2 ? column : 0);
        idToRowNode.set(member.id, group);
      }
    } else {
      const positioned = members.map((member, column) => {
        logicalColumns.set(member.id, members.length === 1 ? 0 : column);
        const positioned = {
          id: member.id,
          title: member.title,
          followUps: member.followUps,
          tags: member.tags,
          state: member.state,
          row,
          column: members.length === 1 ? 0 : column,
        };
        idToRowNode.set(member.id, positioned);
        return positioned;
      });
      rows.push(positioned);
    }
  }

  let collapsedIds = [];
  let collapsedRows = 0;
  while (collapsedRows < rows.length && rows[collapsedRows].length > 0) {
    const rowIds = nodes.filter((node) => rowsById.get(node.id) === collapsedRows).map((node) => node.id);
    if (rowIds.length === 0 || rowIds.some((id) => !byId.get(id).done)) break;
    collapsedIds.push(...rowIds);
    collapsedRows += 1;
  }
  if (collapsedRows === rows.length) {
    collapsedRows = 0;
    collapsedIds = [];
  }
  const visibleRows = rows.slice(collapsedRows).map((row, index) => row.map((node) => ({ ...node, row: index })));
  const edges = [];
  for (const node of nodes) {
    for (const dependency of parents.get(node.id)) {
      if (collapsedIds.includes(dependency)) continue;
      const from = idToRowNode.get(dependency);
      const to = idToRowNode.get(node.id);
      const edge = { from: from.id, to: to.id };
      const fromRow = rowsById.get(dependency);
      const toRow = rowsById.get(node.id);
      const fromColumn = logicalColumns.get(dependency);
      const toColumn = logicalColumns.get(node.id);
      const blocked = toRow - fromRow > 1 && nodes.some((candidate) => {
        const candidateRow = rowsById.get(candidate.id);
        return candidateRow > fromRow && candidateRow < toRow &&
          logicalColumns.get(candidate.id) === fromColumn &&
          candidate.id !== dependency && candidate.id !== node.id;
      });
      if (blocked) {
        node.tags.push(`← ${dependency}`);
      } else if (!edges.some((existing) => existing.from === edge.from && existing.to === edge.to)) {
        edges.push(edge);
      }
    }
  }
  return { rows: visibleRows, edges, summary: collapsedIds.length ? `✓ ${collapsedIds.join(" ")}` : "" };
}
