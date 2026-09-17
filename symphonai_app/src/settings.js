export class SettingsError extends Error {
  constructor(message) {
    super(message);
    this.name = "SettingsError";
  }
}

function section(reply, name) {
  if (reply === null || typeof reply !== "object" || Array.isArray(reply)) {
    throw new SettingsError("settings reply must be an object");
  }
  const entries = reply.settings?.[name];
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
