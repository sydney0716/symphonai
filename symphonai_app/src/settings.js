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
