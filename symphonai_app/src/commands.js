import { errorText, element } from "./render.js";

export const COMMANDS = Object.freeze([
  Object.freeze({
    name: "mode",
    maxArgs: 0,
    handler: handleMode,
    aliases: Object.freeze([]),
    description: "Choose the permission mode",
    argumentHint: "",
  }),
  Object.freeze({
    name: "model",
    maxArgs: 2,
    handler: handleModel,
    aliases: Object.freeze([]),
    description: "Choose the provider and model",
    argumentHint: "[<provider> [<id>]]",
  }),
  Object.freeze({
    name: "effort",
    maxArgs: 1,
    handler: handleEffort,
    aliases: Object.freeze([]),
    description: "Choose the effort for the current model",
    argumentHint: "[<value>]",
  }),
  Object.freeze({
    name: "help",
    maxArgs: 0,
    handler: handleHelp,
    aliases: Object.freeze([]),
    description: "Show the commands",
    argumentHint: "",
  }),
  Object.freeze({
    name: "new",
    maxArgs: 0,
    handler: handleNew,
    aliases: Object.freeze(["clear"]),
    description: "Start a new chat",
    argumentHint: "",
  }),
  Object.freeze({
    name: "plan",
    maxArgs: 0,
    handler: handlePlan,
    aliases: Object.freeze([]),
    description: "Switch plan mode on or off",
    argumentHint: "",
  }),
  Object.freeze({
    name: "spec",
    maxArgs: 0,
    handler: handleSpec,
    aliases: Object.freeze([]),
    description: "Switch spec mode on or off",
    argumentHint: "",
  }),
  Object.freeze({
    name: "resume",
    handler: handleResume,
    aliases: Object.freeze(["continue"]),
    description: "Reopen a past conversation in this project",
    argumentHint: "[<search>]",
  }),
  Object.freeze({
    name: "cost",
    maxArgs: 0,
    handler: handleCost,
    aliases: Object.freeze([]),
    description: "Show what this conversation has used",
    argumentHint: "",
  }),
  Object.freeze({
    name: "stats",
    maxArgs: 0,
    handler: handleStats,
    aliases: Object.freeze([]),
    description: "Show usage and cost over time",
    argumentHint: "",
  }),
  Object.freeze({
    name: "context",
    maxArgs: 0,
    handler: handleContext,
    aliases: Object.freeze([]),
    description: "Show what fills the context window",
    argumentHint: "",
  }),
  Object.freeze({
    name: "compact",
    handler: handleCompact,
    aliases: Object.freeze([]),
    description: "Summarize the conversation so far to free context",
    argumentHint: "[<instructions>]",
  }),
  Object.freeze({
    name: "goal",
    handler: handleGoal,
    aliases: Object.freeze([]),
    description: "Keep working toward a goal",
    argumentHint: "[<objective> [-- <check command>] | pause | resume | clear]",
  }),
  Object.freeze({
    name: "init",
    maxArgs: 0,
    handler: handleInit,
    aliases: Object.freeze([]),
    description: "Have the agent write .symphonai/INSTRUCTIONS.md",
    argumentHint: "",
  }),
  Object.freeze({
    name: "rename",
    minArgs: 1,
    handler: handleRename,
    aliases: Object.freeze([]),
    description: "Rename this conversation",
    argumentHint: "<title>",
  }),
  Object.freeze({
    name: "copy",
    maxArgs: 0,
    handler: handleCopy,
    aliases: Object.freeze([]),
    description: "Copy the last reply as Markdown",
    argumentHint: "",
  }),
  Object.freeze({
    name: "export",
    maxArgs: 0,
    handler: handleExport,
    aliases: Object.freeze([]),
    description: "Download this conversation as Markdown",
    argumentHint: "",
  }),
  Object.freeze({
    name: "review",
    handler: handleReview,
    aliases: Object.freeze([]),
    description: "Review the uncommitted changes",
    argumentHint: "[<focus>]",
  }),
  Object.freeze({
    name: "commit",
    handler: handleCommit,
    aliases: Object.freeze([]),
    description: "Commit selected changes",
    argumentHint: "[<message>]",
  }),
  Object.freeze({
    name: "pr",
    maxArgs: 0,
    handler: handlePr,
    aliases: Object.freeze([]),
    description: "Open a pull request",
    argumentHint: "",
  }),
]);

export function matchCommands(text, extra = []) {
  if (typeof text !== "string" || !text.startsWith("/")) return [];
  const commands = [...COMMANDS, ...extra];
  const query = text.slice(1).toLowerCase();
  if (/\s/.test(query)) return [];
  if (query === "") return commands;

  const ranked = commands.map((command) => {
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

async function handleMode(context, { args, command, typedName, usage }) {
  const { openModePicker } = context;
  openModePicker();
  return;
}

async function handleModel(context, { args, command, typedName, usage }) {
  const { openDefaultModelPicker, openModelPicker, boundary, chooseTypedModel, answerCommand } = context;
  if (args.length === 0) {
    await openDefaultModelPicker();
    return;
  }
  if (args.length === 1) {
    await openModelPicker(args[0]);
    return;
  }
  if (args.length === 2) {
    const [provider, model] = args;
    let listing;
    try {
      listing = await boundary.models(provider);
    } catch (error) {
      answerCommand(errorText(error));
      return;
    }
    if (listing?.state === "unknown") {
      await chooseTypedModel(provider, model);
      return;
    }
    if (
      listing?.state === "available"
      && Array.isArray(listing.models)
      && !listing.models.some((row) => row.id === model)
    ) {
      await chooseTypedModel(provider, model);
      return;
    }
    if (listing?.state === "available" && Array.isArray(listing.models)) {
      await chooseTypedModel(provider, model, listing.models.find((row) => row.id === model)?.efforts);
    } else {
      await openModelPicker(provider, model);
    }
    return;
  }

}

async function handleEffort(context, { args, command, typedName, usage }) {
  const { runEffort } = context;
  await runEffort(args);
  return;
}

async function handleHelp(context, { args, command, typedName, usage }) {
  const { answerCommand } = context;
  const lines = COMMANDS.map(({ name, aliases, description, argumentHint }) => (
    `/${name}${argumentHint ? ` ${argumentHint}` : ""} — ${description}`
    + (aliases.length > 0 ? ` (also ${aliases.map((alias) => `/${alias}`).join(", ")})` : "")
  ));
  if (context.skillCommands.length > 0) {
    lines.push("Skills:", ...context.skillCommands.map(({ name, argumentHint, description }) => (
      `/${name} ${argumentHint} — ${description}`
    )));
  }
  answerCommand(lines.join("\n"));
  return;
}

async function handleNew(context, { args, command, typedName, usage }) {
  const { startNewChat } = context;
  await startNewChat();
  return;
}

async function handleRename(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, refreshSessionLists, showProjects } = context;
  const title = command.slice(typedName.length).trim();
  if (!context.currentSessionId) {
    answerCommand(usage);
    return;
  }
  try {
    await boundary.renameSession(context.currentSessionId, title);
    await refreshSessionLists();
    showProjects();
    answerCommand("Conversation renamed.");
  } catch (error) {
    answerCommand(errorText(error));
  }
  return;
}

async function handleCopy(context, { args, command, typedName, usage }) {
  const { answerCommand, transcript, global } = context;
  const latest = [...transcript.model].reverse().find((item) => item.type === "text" && typeof item.text === "string");
  if (!latest) {
    answerCommand("Nothing to copy yet.");
    return;
  }
  try {
    const clipboard = global.navigator?.clipboard;
    if (typeof clipboard?.writeText !== "function") throw new Error("clipboard is unavailable");
    await clipboard.writeText(latest.text);
    answerCommand("Copied the last reply.");
  } catch (error) {
    answerCommand(`Could not copy: ${errorText(error)}`);
  }
  return;
}

async function handleExport(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, global, document } = context;
  if (!context.currentSessionId) {
    answerCommand("There is nothing to export.");
    return;
  }
  try {
    const exported = await boundary.exportSession(context.currentSessionId);
    const url = global.URL.createObjectURL(new global.Blob([exported.markdown], {
      type: "text/markdown;charset=utf-8",
    }));
    const link = element(document, "a");
    link.href = url;
    link.download = exported.filename;
    link.click();
    global.URL.revokeObjectURL(url);
    answerCommand("Conversation exported.");
  } catch (error) {
    answerCommand(`Could not export: ${errorText(error)}`);
  }
  return;
}

async function handlePlan(context, { args, command, typedName, usage }) {
  const { togglePlanMode } = context;
  await togglePlanMode();
  return;
}

async function handleSpec(context, { args, command, typedName, usage }) {
  const { answerCommand, showConversationUsage } = context;
  context.specMode = !context.specMode;
  showConversationUsage();
  answerCommand(context.specMode
    ? "Spec mode on: each message starts a spec loop."
    : "Spec mode off.");
  return;
}

async function handleResume(context, { args, command, typedName, usage }) {
  const { resumeConversation } = context;
  await resumeConversation(args);
  return;
}

async function handleCost(context, { args, command, typedName, usage }) {
  const { showCost } = context;
  await showCost(args);
  return;
}

async function handleStats(context) {
  await context.showUsage();
}

async function handleContext(context, { args, command, typedName, usage }) {
  const { showContext } = context;
  await showContext(args);
  return;
}

async function handleCompact(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, refreshConversation } = context;
  answerCommand("Compacting…");
  try {
    const instructions = command.slice(typedName.length).trim();
    const result = await boundary.compact(instructions || undefined);
    answerCommand(result.changed
      ? `Compacted: ${result.before_tokens} → ${result.after_tokens} tokens, ${result.dropped_messages} messages summarized.`
      : "Nothing to compact yet.");
  } catch (error) {
    answerCommand(errorText(error));
  }
  await refreshConversation();
  return;
}

async function handleGoal(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, showConversationUsage } = context;
  if (args.length === 0) {
    try {
      const reply = await boundary.conversationStats();
      const goal = reply?.conversation?.goal;
      if (!goal) {
        answerCommand("No goal. Set one with /goal <objective> [-- <check command>].");
        return;
      }
      const lines = [
        goal.objective,
        `Check: ${goal.check?.length ? goal.check.join(" ") : "none (the agent reports completion)"}`,
        `${goal.phase} · round ${goal.rounds} of ${goal.max_rounds}${goal.reason ? ` · ${goal.reason}` : ""}`,
      ];
      if (goal.last_check) {
        lines.push(goal.last_check.ok
          ? "Last check: passed"
          : goal.last_check.exit === null
            ? "Last check: timed out"
            : `Last check: exit ${goal.last_check.exit}`);
        if (goal.last_check.output) lines.push(goal.last_check.output);
      }
      answerCommand(lines.join("\n"));
    } catch (error) {
      answerCommand(errorText(error));
    }
    return;
  }
  const action = args[0];
  if (["pause", "resume", "clear"].includes(action)) {
    if (args.length !== 1) {
      answerCommand(usage);
      return;
    }
    try {
      const reply = await boundary.goalState(action);
      context.conversation = { ...(context.conversation ?? {}), goal: reply.goal };
      showConversationUsage();
      answerCommand({ pause: "Goal paused.", resume: "Goal resumed.", clear: "Goal cleared." }[action]);
    } catch (error) {
      answerCommand(errorText(error));
    }
    return;
  }
  const freeText = command.slice(typedName.length).trim();
  const separator = freeText.indexOf(" -- ");
  if (separator < 0 && (freeText === "--" || freeText.startsWith("-- ") || freeText.endsWith(" --"))) {
    answerCommand(usage);
    return;
  }
  const objective = (separator < 0 ? freeText : freeText.slice(0, separator)).trim();
  const checkText = separator < 0 ? "" : freeText.slice(separator + 4).trim();
  if (!objective || (separator >= 0 && !checkText)) {
    answerCommand(usage);
    return;
  }
  try {
    const reply = await boundary.setGoal(objective, checkText ? checkText.split(/\s+/) : []);
    context.conversation = { ...(context.conversation ?? {}), goal: reply.goal };
    showConversationUsage();
    answerCommand(`Goal set. Round 1 of ${reply.goal?.max_rounds ?? 10} started.`);
  } catch (error) {
    answerCommand(error?.status === 409
      ? "A run is active. Wait for it to finish, then set the goal."
      : errorText(error));
  }
  return;
}

async function handleInit(context, { args, command, typedName, usage }) {
  const { answerCommand, showPromptError, submitCommandPrompt, INIT_PROMPT } = context;
  if ((context.conversation?.mode ?? context.currentMode) === "plan") {
    answerCommand("Plan mode cannot write files. Switch with /plan, then run /init.");
    return;
  }
  context.promptFailure = "";
  showPromptError();
  await submitCommandPrompt(INIT_PROMPT);
  return;
}

async function handleReview(context, { args, command, typedName, usage }) {
  const { showPromptError, submitCommandPrompt, REVIEW_PROMPT } = context;
  const focus = command.slice(typedName.length).trim();
  context.promptFailure = "";
  showPromptError();
  await submitCommandPrompt(`${REVIEW_PROMPT}${focus ? `\n\nFocus on: ${focus}` : ""}`);
  return;
}

async function handleCommit(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, showTranscript } = context;
  let status;
  try {
    status = await boundary.gitStatus();
  } catch (error) {
    answerCommand(errorText(error));
    return;
  }
  if (status?.repo === false) {
    answerCommand("Not a git repository.");
    return;
  }
  const files = Array.isArray(status?.files) ? status.files : [];
  if (files.length === 0) {
    answerCommand("Nothing to commit.");
    return;
  }
  context.commandEntry.type = "commit-card";
  context.commandEntry.message = command.slice(typedName.length).trim();
  context.commandEntry.files = files.map((file) => ({
    ...file,
    checked: true,
    paths: [...new Set([file.old_path, file.path].filter((path) => typeof path === "string"))],
  }));
  context.commandEntry.error = "";
  showTranscript();
  return;
}

async function handlePr(context, { args, command, typedName, usage }) {
  const { boundary, answerCommand, showTranscript } = context;
  try {
    const info = await boundary.pullRequestInfo();
    context.commandEntry.type = "pull-request-card";
    context.commandEntry.problem = info.problem || "";
    context.commandEntry.branch = info.branch || "";
    context.commandEntry.base = info.base || "";
    context.commandEntry.title = info.title || "";
    context.commandEntry.body = "";
    context.commandEntry.error = "";
    showTranscript();
  } catch (error) {
    answerCommand(errorText(error));
  }
}
