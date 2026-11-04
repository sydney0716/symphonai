import { button as makeButton, errorText, append, element, listen, replace } from "./render.js";
import { reloadSummary } from "./commands.js";

export const EDITOR_KEY = "symphonai.editor";

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

export function permissionRows(reply) {
  if (reply === null || typeof reply !== "object" || Array.isArray(reply)) {
    throw new SettingsError("permissions reply must be an object");
  }
  const shell = Array.isArray(reply.shell) ? reply.shell : [];
  const fetch = Array.isArray(reply.fetch) ? reply.fetch : [];
  return [
    ...shell.filter(Array.isArray).map((value) => ({
      kind: "shell", value, label: value.join(" "),
    })),
    ...fetch.filter((value) => typeof value === "string").map((value) => ({
      kind: "fetch", value, label: value,
    })),
  ];
}

export function hookRows(reply) {
  return section(reply, "hooks")
    .map(({ event, command }) => ({ event, command }))
    .sort((left, right) =>
      left.event.localeCompare(right.event) || left.command.localeCompare(right.command)
    );
}

function renderPermissions(section, context) {
  const { loadPermissions } = context;
  void loadPermissions();
  return;
}

function renderMemory(section, context) {
  const { loadMemory } = context;
  void loadMemory();
  return;
}

function renderModels(section, context) {
  const { document, boundary, settingsContent, settingsTable, providerRows } = context;
  const models = modelRows(context.settingsReply);
  const rows = models.map(({ name, envVar, present }) => [
    name, envVar, present ? "present" : "absent",
  ]);
  const table = settingsTable(["Provider", "Environment variable", "Status"], rows);
  const controls = element(document, "div", { className: "credential-controls" });
  for (const [index, { envVar }] of models.entries()) {
    const row = element(document, "div");
    const label = element(document, "label", { text: `${envVar}: ` });
    const input = element(document, "input");
    input.type = "password";
    input.autocomplete = "off";
    const save = makeButton(document, "Save");
    const remove = makeButton(document, "Remove");
    const notice = element(document, "span");
    async function write(value) {
      save.disabled = true;
      remove.disabled = true;
      try {
        await boundary.storeCredential(envVar, value);
        if (input.value === value) input.value = "";
        table.children[index + 1].children[2].textContent = value ? "present" : "absent";
        const providerRow = providerRows.find((item) => item.env_var === envVar);
        if (providerRow) providerRow.key_present = Boolean(value);
        notice.textContent = value ? "Key stored." : "Key removed.";
      } catch (error) {
        const detail = errorText(error).trim();
        notice.textContent = detail ? `Could not save the key: ${detail}` : "Could not save the key.";
      } finally {
        save.disabled = false;
        remove.disabled = false;
      }
    }
    listen(save, "click", () => input.value ? write(input.value) : undefined);
    listen(remove, "click", () => write(""));
    append(label, input);
    append(row, label, save, remove, notice);
    append(controls, row);
  }
  replace(
    settingsContent,
    element(document, "h2", { text: "Models" }),
    table,
    controls,
  );
  return;
}

function renderMcp(section, context) {
  const { document, global, boundary, settingsContent, settingsTable } = context;
  const servers = context.settingsReply?.settings?.mcp_servers ?? [];
  const table = settingsTable(["Server", "Command", "State"], servers.map(({ name, command, state, error, started }) => [
    name, command, state === "needs_sign_in" ? "needs sign-in" : state === "failed" ? `failed: ${error}` : state || (started ? "started" : "not started"),
  ]));
  servers.forEach((server, index) => {
    if (server.state !== "needs_sign_in") return;
    const signIn = makeButton(document, "Sign in", async () => {
      const tab = global.open?.("about:blank", "_blank");
      if (tab) tab.opener = null;
      signIn.disabled = true;
      try {
        const reply = await boundary.mcpSignIn(server.name);
        if (tab) tab.location.href = reply.authorization_url;
        else global.open?.(reply.authorization_url, "_blank", "noopener,noreferrer");
        global.addEventListener?.("focus", async () => {
          try {
            context.settingsReply = await boundary.settings();
            if (context.route?.page === "settings" && context.route.section === "mcp") renderMcp(section, context);
          } catch { /* Keep the last server state if refreshing fails. */ }
        }, { once: true });
      } catch (error) {
        tab?.close();
        append(settingsContent, element(document, "p", { className: "error", text: errorText(error) }));
      } finally { signIn.disabled = false; }
    });
    append(table.children[index + 1].children[2], signIn);
  });
  replace(settingsContent, element(document, "h2", { text: "MCP servers" }), table);
}

function renderRoster(section, context) {
  const { document, global, boundary, settingsContent, settingsTable, getSkillCommands } = context;

  const rows = rosterRows(context.settingsReply, section);
  const table = settingsTable(["Name", "Path"], rows.map(({ name, path }) => [name, path]));
  for (const [index, { path }] of rows.entries()) {
    if (path) {
      const copy = makeButton(document, "Copy", () => global.navigator?.clipboard?.writeText?.(path));
      append(table.children[index + 1].children[1], copy);
    }
  }
  replace(
    settingsContent,
    element(document, "h2", { text: section[0].toUpperCase() + section.slice(1) }),
    table,
  );
  return;
}

function renderHooks(section, context) {
  const { document, settingsContent, settingsTable } = context;
  const rows = hookRows(context.settingsReply).map(({ event, command }) => [event, command]);
  replace(
    settingsContent,
    element(document, "h2", { text: "Hooks" }),
    rows.length === 0
      ? element(document, "p", { text: "No hooks are configured." })
      : settingsTable(["Event", "Command"], rows),
  );
  return;
}

function renderCeiling(section, context) {
  const { document, settingsContent, settingsTable } = context;
  const rows = ceilingRows(context.settingsReply).map(({ capability, value }) => [capability, value]);
  replace(
    settingsContent,
    element(document, "h2", { text: "Ceiling" }),
    element(document, "p", { text: 'The most any agent may be granted. "not set" means the ceiling does not limit this.' }),
    settingsTable(["Capability", "Value"], rows),
  );
  return;
}

function renderTrust(section, context) {
  const { document, settingsContent, settingsTable } = context;
  const rows = trustRows(context.settingsReply).map(({ root, allow }) => [
    root, allow.length === 0 ? "nothing" : allow.join(", "),
  ]);
  replace(
    settingsContent,
    element(document, "h2", { text: "Trust" }),
    rows.length === 0
      ? element(document, "p", { text: "No directories are trusted." })
      : settingsTable(["Root", "Allows"], rows),
  );
  return;
}

function renderInventory(section, context) {
  const { document, settingsContent, settingsTable } = context;
  const rows = inventoryRows(context.settingsReply).map(({ scope, directory, names, reason }) => [
    scope, directory, names.join(", "), reason,
  ]);
  replace(
    settingsContent,
    element(document, "h2", { text: "Inventory" }),
    rows.length === 0
      ? element(document, "p", { text: "Nothing was withheld." })
      : settingsTable(["Scope", "Directory", "Names", "Reason"], rows),
  );
  return;
}

function renderGeneral(section, context) {
  const { document, global, settingsContent, settingsTable, statusFields, saveStatusFields, showConversationUsage, NOTIFICATIONS_KEY } = context;
  const rows = generalRows(context.settingsReply).map(({ key, value, scope }) => [
    key, value, scope,
  ]);
  const configTable = settingsTable(["Setting", "Value", "Scope"], rows);
  const rawValues = new Map((context.settingsReply?.settings?.config ?? []).map(row => [row.key, row]));
  rows.forEach(([key], index) => {
    const { value, scope } = rawValues.get(key);
    const cell = configTable.children[index + 1].children[1];
    if (!["boolean", "number", "string"].includes(typeof value)) {
      append(cell, element(document, "span", { text: " Open file to edit this table or list." }));
      return;
    }
    if (scope === "private" || scope === "session") {
      append(cell, element(document, "span", {
        text: ` Set in ${scope} scope. Edit that scope's source; this host supports saving only user and project settings.`,
      }));
      return;
    }
    const input = document.createElement("input");
    input.type = typeof value === "boolean" ? "checkbox" : typeof value === "number" ? "number" : "text";
    input.setAttribute("aria-label", key);
    if (typeof value === "boolean") input.checked = value;
    else input.value = String(value);
    if (typeof value === "number") input.step = "any";
    const target = document.createElement("select");
    target.setAttribute("aria-label", `Save ${key} to`);
    for (const scope of ["user", "project"]) {
      const option = element(document, "option", { text: scope });
      option.value = scope;
      append(target, option);
    }
    target.value = scope === "user" ? "user" : "project";
    const notice = element(document, "span", { className: "error" });
    notice.setAttribute("role", "status");
    const save = makeButton(document, "Save", async () => {
      save.disabled = true;
      notice.textContent = "";
      try {
        let next = input.value;
        if (typeof value === "boolean") next = input.checked;
        else if (typeof value === "number") {
          if (!input.value.trim() || !Number.isFinite(Number(input.value))) throw new Error("Enter a number.");
          next = Number(input.value);
        }
        await context.boundary.setConfig(key, next, target.value);
        await context.refreshSettings();
      } catch (error) { notice.textContent = `Could not save ${key}: ${errorText(error)}`; }
      finally { save.disabled = false; }
    });
    save.setAttribute("aria-label", `Save ${key}`);
    replace(cell, input, target, save, notice);
  });
  let theme = "system";
  try {
    const storedTheme = global.localStorage?.getItem("symphonai.theme");
    if (storedTheme === "light" || storedTheme === "dark") theme = storedTheme;
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
  const themeSelect = document.createElement("select");
  themeSelect.setAttribute("aria-label", "Theme");
  for (const [value, label] of [["system", "System"], ["light", "Light"], ["dark", "Dark"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    themeSelect.append(option);
  }
  themeSelect.value = theme;
  listen(themeSelect, "change", () => {
    const selected = ["light", "dark"].includes(themeSelect.value)
      ? themeSelect.value
      : "system";
    try {
      global.localStorage?.setItem("symphonai.theme", selected);
    } catch {
      // Storage is optional in embedded and privacy-restricted browsers.
    }
    if (selected === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", selected);
    }
  });
  const editorSelect = document.createElement("select");
  editorSelect.setAttribute("aria-label", "Editor");
  const editors = [["none", "None"], ["vscode", "VS Code"], ["cursor", "Cursor"], ["zed", "Zed"]];
  for (const [value, label] of editors) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    editorSelect.append(option);
  }
  let editor = "none";
  try {
    const stored = global.localStorage?.getItem(EDITOR_KEY);
    if (editors.some(([value]) => value === stored)) editor = stored;
  } catch { /* Storage is optional. */ }
  editorSelect.value = editor;
  listen(editorSelect, "change", () => {
    const chosen = editors.some(([value]) => value === editorSelect.value) ? editorSelect.value : "none";
    try { global.localStorage?.setItem(EDITOR_KEY, chosen); }
    catch { /* Storage is optional. */ }
    global.dispatchEvent?.(new (global.Event ?? Event)(EDITOR_KEY));
  });
  const statusOptions = [
    ["provider", "Provider"], ["model", "Model"], ["effort", "Effort"],
    ["mode", "Mode"], ["goal", "Goal"], ["context", "Context"], ["usage", "Usage"],
  ];
  const statusGroup = element(document, "section", { className: "status-line-settings" });
  append(statusGroup, element(document, "h3", { text: "Status line" }));
  for (const [key, label] of statusOptions) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = statusFields[key];
    checkbox.setAttribute("aria-label", label);
    listen(checkbox, "change", () => {
      statusFields[key] = checkbox.checked;
      saveStatusFields(global, statusFields);
      showConversationUsage();
    });
    append(statusGroup, append(element(document, "label"), checkbox, element(document, "span", { text: label })));
  }
  const notificationGroup = element(document, "section", { className: "notification-settings" });
  append(notificationGroup, element(document, "h3", { text: "Notifications" }));
  const notificationCheckbox = document.createElement("input");
  notificationCheckbox.type = "checkbox";
  notificationCheckbox.checked = context.notificationsEnabled;
  notificationCheckbox.setAttribute("aria-label", "Notifications");
  const notificationNotice = element(document, "p", {
    className: "notification-settings-message",
    text: context.notificationMessage,
  });
  listen(notificationCheckbox, "change", async () => {
    context.notificationMessage = "";
    if (!notificationCheckbox.checked) {
      context.notificationsEnabled = false;
    } else {
      try {
        if (typeof global.Notification !== "function" || typeof global.Notification.requestPermission !== "function") {
          throw new Error("notifications unavailable");
        }
        const permission = await global.Notification.requestPermission();
        if (permission !== "granted") throw new Error("notifications unavailable");
        context.notificationsEnabled = true;
      } catch {
        context.notificationsEnabled = false;
        notificationCheckbox.checked = false;
        context.notificationMessage = "Notifications are not available here.";
      }
    }
    try {
      global.localStorage?.setItem(NOTIFICATIONS_KEY, String(context.notificationsEnabled));
    } catch {
      // Storage is optional in embedded and privacy-restricted browsers.
    }
    notificationNotice.textContent = context.notificationMessage;
  });
  append(notificationGroup, append(element(document, "label"), notificationCheckbox, element(document, "span", { text: "Notifications" })), notificationNotice);
  const keymapNotice = element(document, "section");
  if (context.keymapRejected?.length) {
    append(keymapNotice, element(document, "h3", { text: "Rejected keyboard shortcuts" }));
    for (const binding of context.keymapRejected) {
      append(keymapNotice, element(document, "p", { text: `${binding.chord}: ${binding.action} — ${binding.reason}` }));
    }
  }
  replace(
    settingsContent,
    element(document, "h2", { text: "General" }),
    element(document, "h3", { text: "Appearance" }),
    themeSelect,
    element(document, "h3", { text: "Editor" }),
    editorSelect,
    statusGroup,
    notificationGroup,
    keymapNotice,
    configTable,
  );
}

const SETTINGS_RENDERERS = Object.freeze({
  permissions: renderPermissions, memory: renderMemory, models: renderModels,
  mcp: renderMcp, skills: renderRoster, plugins: renderRoster, agents: renderAgents,
  hooks: renderHooks, ceiling: renderCeiling, trust: renderTrust, inventory: renderInventory,
});

export function renderSettingsSection(section, context) {
  (SETTINGS_RENDERERS[section] ?? renderGeneral)(section, context);
  if (["skills", "plugins", "agents", "mcp"].includes(section)) {
    const notice = element(context.document, "p");
    notice.setAttribute("role", "status");
    const reload = makeButton(context.document, "Reload", async () => {
      reload.disabled = true;
      try {
        const reply = await context.boundary.reloadExtensions();
        await context.refreshSettings();
        const result = element(context.document, "p", { text: reloadSummary(reply) });
        result.setAttribute("role", "status");
        append(context.settingsContent, result);
      } catch (error) { notice.textContent = `Could not reload extensions: ${errorText(error)}`; }
      finally { reload.disabled = false; }
    });
    append(context.settingsContent, reload, notice);
  }
}

export function renderMemorySettings(context) {
  const { document, boundary, settingsContent, loadMemory, renderMemoryContent } = context;
  const children = [
    element(document, "h2", { text: "Memory" }),
    element(document, "p", {
      text: "What the agent remembered from your conversations. It reads these at the start of each new conversation.",
    }),
  ];
  if (context.memoryLoading) {
    children.push(element(document, "p", { text: "Loading memory…" }));
  } else if (context.memoryLoadError) {
    const retry = makeButton(document, "Retry", () => loadMemory());
    children.push(
      element(document, "p", { className: "memory-load-error", text: "Couldn't load memory." }),
      retry,
    );
  } else if (context.memoryReply) {
    for (const [scope, label] of [["project", "Project"], ["global", "Global"]]) {
      const group = element(document, "section", { className: "memory-group" });
      append(group, element(document, "h3", { text: label }));
      const memorySet = context.memoryReply[scope] ?? {};
      const entries = memorySet.entries ?? [];
      const last = memorySet.last_consolidation;
      if (last) {
        append(group, element(document, "p", {
          text: `Last consolidation: ${last.time}; ${last.merged.length} merges, ${last.removed.length} removed.${last.restored_at ? ` Restored ${last.restored_at}.` : ""}`,
        }));
        for (const merge of last.merged) {
          append(group, element(document, "p", { text: `Merged: ${(merge.previous ?? []).map(item => item.text).join("; ")} → ${merge.text}` }));
        }
        for (const removal of last.removed) {
          append(group, element(document, "p", { text: `Removed: ${removal.text} — ${removal.reason}` }));
        }
      }
      if (memorySet.consolidation_error) append(group, element(document, "p", { className: "error", text: memorySet.consolidation_error }));
      if (memorySet.has_previous) {
        const notice = element(document, "p", { className: "error" });
        notice.setAttribute("role", "status");
        const restore = makeButton(document, "Restore previous", async () => {
          restore.disabled = true;
          try {
            await boundary.restoreMemory(scope);
            await loadMemory();
          } catch (error) { notice.textContent = `Could not restore memory: ${errorText(error)}`; }
          finally { restore.disabled = false; }
        });
        append(group, restore, notice);
      }
      if (memorySet.error === "unreadable") {
        append(group, element(document, "p", {
          className: "memory-load-error",
          text: "This memory file couldn't be read, so nothing in it was changed.",
        }));
      } else if (entries.length === 0) {
        append(group, element(document, "p", { className: "memory-empty", text: "Nothing remembered yet." }));
      }
      for (const entry of memorySet.error === "unreadable" ? [] : entries) {
        const row = element(document, "article", { className: "memory-entry" });
        if (entry.id === context.editingMemoryId) {
          const textarea = element(document, "textarea", { className: "memory-edit-input" });
          textarea.value = entry.text;
          const save = makeButton(document, "Save");
          const cancel = makeButton(document, "Cancel");
          listen(save, "click", async () => {
            try {
              const reply = await boundary.updateMemory(entry.id, textarea.value);
              context.memoryReply[scope].entries = entries.map((item) =>
                item.id === entry.id ? reply.entry : item
              );
              context.editingMemoryId = null;
              context.memoryEditError = "";
            } catch (error) {
              context.memoryEditError = errorText(error);
            }
            if (context.route?.page === "settings" && context.route.section === "memory") {
              renderMemoryContent();
            }
          });
          listen(cancel, "click", () => {
            context.editingMemoryId = null;
            context.memoryEditError = "";
            renderMemoryContent();
          });
          append(row, textarea, save, cancel);
          if (context.memoryEditError) {
            append(row, element(document, "p", {
              className: "memory-edit-error",
              text: context.memoryEditError,
            }));
          }
        } else if (entry.id === context.confirmingMemoryDeleteId) {
          const confirm = element(document, "p", { text: "Delete this memory?" });
          const remove = makeButton(document, "Delete");
          const cancel = makeButton(document, "Cancel");
          listen(remove, "click", async () => {
            try {
              await boundary.deleteMemory(entry.id);
              context.memoryReply[scope].entries = entries.filter((item) => item.id !== entry.id);
              context.confirmingMemoryDeleteId = null;
            } catch (error) {
              context.memoryEditError = errorText(error);
            }
            renderMemoryContent();
          });
          listen(cancel, "click", () => {
            context.confirmingMemoryDeleteId = null;
            context.memoryEditError = "";
            renderMemoryContent();
          });
          append(row, element(document, "p", { className: "memory-entry-text", text: entry.text }), confirm, remove, cancel);
          if (context.memoryEditError) {
            append(row, element(document, "p", {
              className: "memory-edit-error",
              text: context.memoryEditError,
            }));
          }
        } else {
          const date = typeof entry.written_at === "string"
            ? new Date(entry.written_at).toLocaleDateString()
            : "";
          const metadata = [date, entry.agent_name].filter(Boolean).join(" · ");
          const edit = makeButton(document, "Edit");
          const remove = makeButton(document, "Delete");
          listen(edit, "click", () => {
            context.editingMemoryId = entry.id;
            context.memoryEditError = "";
            renderMemoryContent();
          });
          listen(remove, "click", () => {
            context.confirmingMemoryDeleteId = entry.id;
            context.memoryEditError = "";
            renderMemoryContent();
          });
          append(
            row,
            element(document, "p", { className: "memory-entry-text", text: entry.text }),
            element(document, "p", { className: "memory-entry-meta", text: metadata }),
            append(element(document, "div", { className: "memory-entry-actions" }), edit, remove),
          );
        }
        append(group, row);
      }
      children.push(group);
    }
  }
  replace(settingsContent, ...children);
}

export function renderPermissionSettings(context) {
  const { document, boundary, settingsContent, renderPermissionsContent } = context;
  const children = [element(document, "h2", { text: "Permissions" })];
  if (context.permissionsLoading) {
    children.push(element(document, "p", { text: "Loading permissions…" }));
  } else if (context.permissionsReply === null) {
    children.push(element(document, "p", { className: "permissions-error", text: context.permissionsError }));
  } else {
    const rows = permissionRows(context.permissionsReply);
    if (context.permissionsError) {
      children.push(element(document, "p", { className: "permissions-error", text: context.permissionsError }));
    }
    if (rows.length === 0) {
      children.push(element(document, "p", { text: "No saved permissions." }));
    } else {
      for (const permission of rows) {
        const row = element(document, "div", { className: "permission-row" });
        const remove = makeButton(document, "Remove", async () => {
          context.permissionsError = "";
          try {
            await boundary.removePermission(permission.kind, permission.value);
            context.permissionsReply = {
              ...context.permissionsReply,
              [permission.kind]: context.permissionsReply[permission.kind].filter((value) => (
                Array.isArray(permission.value)
                  ? JSON.stringify(value) !== JSON.stringify(permission.value)
                  : value !== permission.value
              )),
            };
          } catch (error) {
            const detail = errorText(error).trim();
            context.permissionsError = detail ? `Could not remove permission: ${detail}` : "Could not remove permission.";
          }
          renderPermissionsContent();
        });
        append(
          row,
          element(document, "span", { text: permission.kind === "shell" ? "Command" : "Host" }),
          element(document, "code", { text: permission.label }),
          remove,
        );
        children.push(row);
      }
    }
  }
  replace(settingsContent, ...children);
}

function renderAgents(section, context) {
  const { document, boundary, settingsContent, getSkillCommands } = context;
  const tableRoot = element(document, "div");
  const editorRoot = element(document, "div", { className: "agent-editor" });
  const newAgent = makeButton(document, "New agent");

  function renderAgentRows() {
    const rows = agentRows(context.settingsReply);
    const table = element(document, "table", { className: "settings-table" });
    const heading = element(document, "tr");
    append(heading, ...["Name", "Scope", "Path", "Reason", "Action"].map((text) =>
      element(document, "th", { text })
    ));
    append(table, heading);
    for (const rowData of rows) {
      const row = element(document, "tr", { className: "settings-row" });
      const action = element(document, "td");
      if (rowData.editable) {
        const open = makeButton(document, "Open", () => openAgent(rowData));
        append(action, open);
      } else if (rowData.reason) {
        const withheld = makeButton(document, "Withheld");
        withheld.disabled = true;
        append(action, withheld);
      }
      append(
        row,
        element(document, "td", { text: rowData.name }),
        element(document, "td", { text: rowData.scope }),
        element(document, "td", { text: rowData.path }),
        element(document, "td", { text: rowData.reason }),
        action,
      );
      append(table, row);
    }
    replace(tableRoot, table);
  }

  function field(document, tagName, className, value) {
    const control = element(document, tagName, { className });
    control.value = value;
    control.name = {
      "agent-name": "name",
      "agent-scope": "scope",
      "agent-prompt": "prompt",
      "agent-provider": "provider",
      "agent-model": "model",
      "agent-effort": "effort",
      "agent-tools": "tools",
      "agent-deny-tools": "deny_tools",
    }[className] ?? className;
    return control;
  }

  function renderEditor(values, identity = null, original = "", message = "") {
    const form = element(document, "form", { className: "agent-form" });
    const name = field(document, "input", "agent-name", identity?.name ?? values.name ?? "");
    const scope = field(document, "select", "agent-scope", identity?.scope ?? "project");
    replace(scope, ...["project", "user"].map((value) => {
      const option = element(document, "option", { text: value });
      option.value = value;
      return option;
    }));
    scope.value = identity?.scope ?? "project";
    name.disabled = Boolean(identity);
    scope.disabled = Boolean(identity);
    const prompt = field(document, "textarea", "agent-prompt", values.prompt);
    const provider = field(document, "input", "agent-provider", values.provider);
    const model = field(document, "input", "agent-model", values.model);
    const effort = field(document, "input", "agent-effort", values.effort);
    const tools = field(document, "textarea", "agent-tools", values.tools.join("\n"));
    const denyTools = field(document, "textarea", "agent-deny-tools", values.denyTools.join("\n"));
    const status = element(document, "p", { className: "agent-status", text: message });
    const save = makeButton(document, "Save", null, { className: "agent-save primary" });
    save.type = "submit";
    const labels = [
      ["Name", name], ["Scope", scope], ["Prompt", prompt],
      ["Model provider", provider], ["Model id", model], ["Model effort", effort],
      ["Allowed tools", tools], ["Denied tools", denyTools],
    ];
    for (const [labelText, control] of labels) {
      const label = element(document, "label", { text: labelText });
      append(label, control);
      append(form, label);
    }
    append(form, save, status);
    listen(form, "submit", async (event) => {
      event.preventDefault();
      const currentName = name.value.trim();
      const currentScope = scope.value;
      const text = composeAgentText({
        prompt: prompt.value,
        provider: provider.value.trim(),
        model: model.value.trim(),
        effort: effort.value.trim(),
        tools: tools.value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean),
        denyTools: denyTools.value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean),
      }, original);
      try {
        const reply = await boundary.saveAgent(currentName, currentScope, text);
        context.settingsReply = await boundary.settings();
        context.skillCommands = getSkillCommands(context.settingsReply);
        renderAgentRows();
        status.textContent = reply?.message || "definition saved; it will take effect on the next run";
      } catch (error) {
        status.textContent = errorText(error);
      }
    });
    replace(editorRoot, form);
  }

  async function openAgent(rowData) {
    const status = findAgentStatus();
    if (status) status.textContent = "Loading definition…";
    try {
      const reply = await boundary.agent(rowData.name, rowData.scope);
      const original = typeof reply?.text === "string" ? reply.text : "";
      renderEditor(parseAgentText(original), rowData, original);
    } catch (error) {
      if (status) status.textContent = errorText(error);
    }
  }

  function findAgentStatus() {
    const form = editorRoot.children[0];
    for (const child of form?.children ?? []) {
      if (child.className === "agent-status") return child;
    }
    return null;
  }

  const description = element(document, "textarea", { className: "agent-description" });
  description.maxLength = 2000;
  description.name = "description";
  const descriptionLabel = element(document, "label", { text: "Describe an agent" });
  append(descriptionLabel, description);
  const draftStatus = element(document, "p", { className: "agent-status" });
  draftStatus.setAttribute("role", "status");
  const draftButton = makeButton(document, "Draft", async () => {
    if (draftButton.disabled) return;
    draftButton.disabled = true;
    draftButton.textContent = "Drafting…";
    draftStatus.textContent = "";
    try {
      const latest = await boundary.settings();
      const selection = latest?.settings?.selection;
      draftButton.textContent = `Drafting with ${selection?.model || selection?.provider || "selected model"}…`;
      const draft = await boundary.draftAgent(description.value);
      renderEditor({
        ...emptyAgentFields(), ...draft, denyTools: draft.deny_tools ?? [],
      }, null, "", `Draft · not saved${draft.dropped?.length ? ` · Dropped tools: ${draft.dropped.join(", ")}` : ""}`);
    } catch (error) {
      draftStatus.textContent = errorText(error);
    } finally {
      draftButton.disabled = false;
      draftButton.textContent = "Draft";
    }
  });
  const draftControls = element(document, "div");
  append(draftControls, descriptionLabel, draftButton, draftStatus);
  listen(newAgent, "click", () => renderEditor(parseAgentText(""), null, ""));
  renderAgentRows();
  replace(
    settingsContent,
    element(document, "h2", { text: "Agents" }),
    newAgent,
    draftControls,
    tableRoot,
    editorRoot,
  );
  return;
}
