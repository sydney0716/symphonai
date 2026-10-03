export const COMMANDS = Object.freeze([
  Object.freeze({
    name: "mode",
    aliases: Object.freeze([]),
    description: "Choose the permission mode",
    argumentHint: "",
  }),
  Object.freeze({
    name: "model",
    aliases: Object.freeze([]),
    description: "Choose the provider, model and effort",
    argumentHint: "[<provider> [<id> [<effort>]]]",
  }),
  Object.freeze({
    name: "effort",
    aliases: Object.freeze([]),
    description: "Choose the effort for the current model",
    argumentHint: "[<value>]",
  }),
  Object.freeze({
    name: "help",
    aliases: Object.freeze([]),
    description: "Show the commands",
    argumentHint: "",
  }),
  Object.freeze({
    name: "new",
    aliases: Object.freeze(["clear"]),
    description: "Start a new chat",
    argumentHint: "",
  }),
  Object.freeze({
    name: "plan",
    aliases: Object.freeze([]),
    description: "Switch plan mode on or off",
    argumentHint: "",
  }),
  Object.freeze({
    name: "resume",
    aliases: Object.freeze(["continue"]),
    description: "Reopen a past conversation in this project",
    argumentHint: "[<search>]",
  }),
  Object.freeze({
    name: "cost",
    aliases: Object.freeze([]),
    description: "Show what this conversation has used",
    argumentHint: "",
  }),
]);

export function matchCommands(text) {
  if (typeof text !== "string" || !text.startsWith("/")) return [];
  const query = text.slice(1).toLowerCase();
  if (/\s/.test(query)) return [];
  if (query === "") return COMMANDS;

  const ranked = COMMANDS.map((command) => {
    const names = [command.name, ...command.aliases].map((name) => name.toLowerCase());
    if (names.some((name) => name === query)) return { command, tier: 0 };
    if (names.some((name) => name.startsWith(query))) return { command, tier: 1 };
    if (names.some((name) => name.includes(query))) return { command, tier: 2 };
    if (command.description.toLowerCase().split(/\s+/).some((word) => word.startsWith(query))) {
      return { command, tier: 3 };
    }
    return null;
  }).filter(Boolean);
  const nameMatches = ranked.filter((entry) => entry.tier < 3);
  const selected = nameMatches.length > 0
    ? nameMatches
    : ranked.filter((entry) => entry.tier === 3);
  return [0, 1, 2, 3].flatMap((tier) => selected
    .filter((entry) => entry.tier === tier)
    .map((entry) => entry.command));
}
