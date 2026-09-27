export class SettingsError extends Error {
  constructor(message) {
    super(message);
    this.name = "SettingsError";
  }
}

function settings(reply) {
  if (reply === null || typeof reply !== "object" || Array.isArray(reply)) {
    throw new SettingsError("settings reply must be an object");
  }
  return reply.settings;
}

function section(reply, name) {
  const entries = settings(reply)?.[name];
  return Array.isArray(entries) ? entries : [];
}

function formatValue(value) {
  if (typeof value === "boolean") {
    return value ? "on" : "off";
  }
  if (Array.isArray(value)) {
    return value.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(", ");
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function generalRows(reply) {
  return section(reply, "config")
    .map(({ key, value, scope }) => ({ key, value: formatValue(value), scope }))
    .sort((left, right) => left.key.localeCompare(right.key));
}

export function modelRows(reply) {
  return section(reply, "providers")
    .map(({ name, env_var, key_present }) => ({
      name,
      envVar: env_var,
      present: key_present === true,
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function serverRows(reply) {
  return section(reply, "mcp_servers")
    .map(({ name, command, started }) => ({ name, command, started }));
}

export function rosterRows(reply, kind) {
  if (!["skills", "plugins", "agents"].includes(kind)) {
    return [];
  }
  return section(reply, kind)
    .map(({ name, path }) => ({ name, path }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function definitionScope(path) {
  return typeof path === "string" && path.startsWith(".symphonai/")
    ? "project"
    : "user";
}

export function agentRows(reply) {
  const visible = section(reply, "agents").map(({ name, path, scope }) => ({
    name,
    path,
    scope: scope === "project" || scope === "user" ? scope : definitionScope(path),
    reason: "",
    editable: Boolean(path),
  }));
  const withheld = inventoryRows(reply)
    .filter(({ scope, directory, names }) =>
      scope && typeof directory === "string" && /(?:^|\/)agents$/.test(directory)
      && Array.isArray(names)
    )
    .flatMap(({ scope, directory, names, reason }) => names.map((name) => ({
      name,
      path: `${directory.replace(/\/$/, "")}/${name}.toml`,
      scope,
      reason,
      editable: false,
    })));
  return [...visible, ...withheld].sort((left, right) =>
    left.name.localeCompare(right.name) || left.scope.localeCompare(right.scope)
  );
}

function emptyAgentFields() {
  return {
    prompt: "",
    provider: "",
    model: "",
    effort: "",
    tools: [],
    denyTools: [],
  };
}

function commentIndex(line) {
  let quote = "";
  let escaped = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (quote === '"' && escaped) {
      escaped = false;
    } else if (quote === '"' && character === "\\") {
      escaped = true;
    } else if (quote && character === quote) {
      quote = "";
    } else if (!quote && (character === '"' || character === "'")) {
      quote = character;
    } else if (!quote && character === "#") {
      return index;
    }
  }
  return -1;
}

function parseValue(value) {
  const trimmed = value.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    if (trimmed.startsWith("'") && trimmed.endsWith("'")) {
      return trimmed.slice(1, -1);
    }
    return trimmed;
  }
}

function scanAgentText(text) {
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const assignments = [];
  let table = "";
  for (const [index, line] of lines.entries()) {
    const body = line.replace(/\r?\n$/, "");
    const end = commentIndex(body);
    const content = (end < 0 ? body : body.slice(0, end));
    const section = content.match(/^\s*\[([^\]]+)\]\s*$/);
    if (section) {
      table = section[1].trim();
      continue;
    }
    const assignment = content.match(/^(\s*[A-Za-z0-9_-]+\s*=\s*)(.*?)(\s*)$/);
    if (!assignment) continue;
    const key = assignment[1].match(/[A-Za-z0-9_-]+(?=\s*=)/)?.[0];
    if (!key) continue;
    assignments.push({
      index,
      table,
      key,
      value: parseValue(assignment[2]),
      prefix: assignment[1],
      trailing: assignment[3],
      comment: end < 0 ? "" : body.slice(end),
      ending: line.slice(body.length),
    });
  }
  return { lines, assignments };
}

export function parseAgentText(text) {
  const fields = {
    ...emptyAgentFields(),
  };
  if (typeof text !== "string") {
    return fields;
  }
  for (const assignment of scanAgentText(text).assignments) {
    const { table, key, value } = assignment;
    if (table === "model" && ["provider", "model", "effort"].includes(key)) {
      fields[key] = typeof value === "string" ? value : "";
    } else if (table === "" && key === "prompt") {
      fields.prompt = typeof value === "string" ? value : "";
    } else if (table === "" && key === "tools") {
      fields.tools = Array.isArray(value) ? value.filter((item) => typeof item === "string") : [];
    } else if (table === "" && key === "deny_tools") {
      fields.denyTools = Array.isArray(value) ? value.filter((item) => typeof item === "string") : [];
    }
  }
  return fields;
}

function tomlString(value) {
  return JSON.stringify(value ?? "");
}

function tomlArray(values) {
  return `[${values.map((value) => tomlString(value)).join(", ")}]`;
}

export function composeAgentText(fields, original) {
  const next = {
    ...emptyAgentFields(),
    ...fields,
    tools: Array.isArray(fields?.tools) ? fields.tools.filter(Boolean) : [],
    denyTools: Array.isArray(fields?.denyTools) ? fields.denyTools.filter(Boolean) : [],
  };
  const output = rewriteAgentText(original, next, original === "");
  return output;
}

function rewriteAgentText(original, fields, creating) {
  const { lines, assignments } = scanAgentText(original);
  const ending = lines.find((line) => line.endsWith("\r\n")) ? "\r\n" : "\n";
  const desired = [
    { table: "", key: "prompt", field: "prompt", value: fields.prompt, empty: creating ? false : fields.prompt === "" },
    { table: "", key: "tools", field: "tools", value: fields.tools, empty: fields.tools.length === 0 },
    { table: "", key: "deny_tools", field: "denyTools", value: fields.denyTools, empty: fields.denyTools.length === 0 },
    { table: "model", key: "provider", field: "provider", value: fields.provider, empty: fields.provider === "" },
    { table: "model", key: "model", field: "model", value: fields.model, empty: fields.model === "" },
    { table: "model", key: "effort", field: "effort", value: fields.effort, empty: fields.effort === "" },
  ];
  const edits = new Map();
  const insertions = new Map();
  const present = new Set();
  for (const item of desired) {
    const match = assignments.find(({ table, key }) => table === item.table && key === item.key);
    const current = item.field === "tools" || item.field === "denyTools"
      ? (Array.isArray(match?.value) ? match.value.filter((value) => typeof value === "string") : [])
      : (typeof match?.value === "string" ? match.value : "");
    if (match) present.add(item.field);
    const changed = JSON.stringify(current) !== JSON.stringify(item.value);
    if (match && changed && item.empty) {
      edits.set(match.index, null);
    } else if (match && changed) {
      const value = Array.isArray(item.value) ? tomlArray(item.value) : tomlString(item.value);
      edits.set(match.index, `${match.prefix}${value}${match.trailing}${match.comment}${match.ending}`);
    } else if (!match && !item.empty) {
      const value = Array.isArray(item.value) ? tomlArray(item.value) : tomlString(item.value);
      const line = `${item.key} = ${value}${ending}`;
      const location = item.table === "" ? topLevelEnd(lines) : modelEnd(lines);
      if (!insertions.has(location)) insertions.set(location, []);
      insertions.get(location).push({ table: item.table, line });
    }
  }
  if (creating && !present.has("prompt") && fields.prompt === "") {
    insertions.set(0, [{ table: "", line: `prompt = ${tomlString(fields.prompt)}${ending}` }]);
  }

  const output = [];
  let modelHeaderAdded = false;
  for (let index = 0; index <= lines.length; index += 1) {
    if (insertions.has(index)) {
      const additions = insertions.get(index);
      if (index === lines.length && lines.length > 0 && !lines.at(-1).endsWith("\n")) {
        output.push(ending);
      }
      for (const item of additions) {
        if (item.table === "model" && !hasModelTable(lines) && !modelHeaderAdded) {
          output.push(`[model]${ending}`);
          modelHeaderAdded = true;
        }
        output.push(item.line);
      }
    }
    if (index === lines.length) break;
    if (edits.has(index)) {
      const replacement = edits.get(index);
      if (replacement !== null) output.push(replacement);
      continue;
    }
    output.push(lines[index]);
  }
  return output.join("");
}

function topLevelEnd(lines) {
  const index = lines.findIndex((line) => /^\s*\[[^\]]+\]/.test(line));
  return index < 0 ? lines.length : index;
}

function hasModelTable(lines) {
  return lines.some((line) => /^\s*\[model\]\s*(?:#.*)?(?:\r?\n)?$/.test(line));
}

function modelEnd(lines) {
  let header = -1;
  for (let index = 0; index < lines.length; index += 1) {
    if (/^\s*\[model\]\s*(?:#.*)?(?:\r?\n)?$/.test(lines[index])) {
      header = index;
      break;
    }
  }
  if (header < 0) return lines.length;
  for (let index = header + 1; index < lines.length; index += 1) {
    if (/^\s*\[[^\]]+\]/.test(lines[index])) return index;
  }
  return lines.length;
}

export function inventoryRows(reply) {
  return section(reply, "withheld")
    .map(({ scope, directory, names, reason }) => ({
      scope,
      directory,
      names,
      reason: reason || "No reason recorded.",
    }));
}

export function ceilingRows(reply) {
  const ceiling = settings(reply)?.ceiling ?? {};
  const capabilities = [
    "allowed_write_scope", "fetch_allowlist", "fetch_enabled", "modes",
    "shell_allowlist", "shell_enabled",
  ];
  return capabilities.map((capability) => {
    const entry = ceiling[capability];
    let value;
    if (entry == null) {
      value = "not set";
    } else if (Array.isArray(entry)) {
      value = entry.length === 0
        ? "none"
        : entry.map((item) => Array.isArray(item) ? item.join(" ") : item).join(", ");
    } else {
      value = formatValue(entry);
    }
    return { capability, value };
  });
}

export function trustRows(reply) {
  return section(reply, "trust")
    .map(({ root, allow }) => ({ root, allow }))
    .sort((left, right) => left.root.localeCompare(right.root));
}

export function hookRows(reply) {
  return section(reply, "hooks")
    .map(({ event, command }) => ({ event, command }))
    .sort((left, right) =>
      left.event.localeCompare(right.event) || left.command.localeCompare(right.command)
    );
}
