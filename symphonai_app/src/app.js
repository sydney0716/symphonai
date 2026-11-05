import { createApprovals } from "./approvals.js";
import { createAgentBoard } from "./agents.js";
import { createClient } from "./client.js";
import { parseMarkdown } from "./markdown.js";
import { highlight } from "./highlight.js";
import { resolveHost } from "./host_handle.js";
import { decodeEvent, FILES_CHANGED_EVENT_TYPE } from "./protocol.js";
import { button as makeButton, errorText, append, element, listen, renderDiff, renderMarkdownBlocks, renderSplitDiff, renderTranscript, replace } from "./render.js";
import { splitDiff } from "./diff_split.js";
import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "./route.js";
import { COMMANDS, matchCommands } from "./commands.js";
import keymapDefaults from "../keys.default.json" with { type: "json" };
import { lookup, mergeKeymap, parseKeymap } from "./keys.js";
import { createPicker } from "./picker.js";
import { EDITOR_KEY, renderSettingsSection, renderMemorySettings, renderPermissionSettings } from "./settings.js";
import { createTranscript } from "./transcript.js";
import { createTurnState, DISPATCHING, RUNNING } from "./turn.js";

const userKeymapFile = await import("../keys.json", { with: { type: "json" } })
  .then(({ default: keymap }) => keymap)
  .catch(() => ({}));

const STATUS_LINE_KEY = "symphonai.status-line.v2";
const NOTIFICATIONS_KEY = "symphonai.notifications";
const PROMPT_DRAFTS_KEY = "symphonai.drafts";
const DEFAULT_STATUS_FIELDS = Object.freeze({
  provider: false, model: true, effort: true, mode: false,
  goal: false, context: false, usage: true,
});

function stateLabel(state) {
  const value = typeof state === "string" && state ? state : "unknown";
  const label = value.replaceAll("_", " ");
  return label[0].toUpperCase() + label.slice(1);
}

function readStatusFields(global) {
  try {
    const current = global.localStorage?.getItem(STATUS_LINE_KEY);
    const stored = JSON.parse(current ?? global.localStorage?.getItem("symphonai.status-line") ?? "null");
    return Object.fromEntries(Object.keys(DEFAULT_STATUS_FIELDS).map((key) => [
      key,
      key === "usage" && current == null ? true
        : typeof stored?.[key] === "boolean" ? stored[key] : DEFAULT_STATUS_FIELDS[key],
    ]));
  } catch {
    return { ...DEFAULT_STATUS_FIELDS };
  }
}

function saveStatusFields(global, fields) {
  try {
    global.localStorage?.setItem(STATUS_LINE_KEY, JSON.stringify(fields));
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
}

export const INIT_PROMPT = `Please analyze this repository and write .symphonai/INSTRUCTIONS.md, which SymphonAI loads into every conversation in this project.

Include:
1. The commands used most often: how to build, lint and run the tests, including how to run a single test.
2. The high-level architecture: the big picture that takes reading several files to understand.

Rules:
- If .symphonai/INSTRUCTIONS.md already exists, read it and improve it instead of starting over.
- Keep it short: only what an agent would get wrong without it. Do not list every file or directory, and do not add generic advice such as writing tests or handling errors.
- If README.md, CLAUDE.md, AGENTS.md, .cursorrules, .cursor/rules/ or .github/copilot-instructions.md exist, carry over the parts that matter.
- Do not invent sections or facts the repository does not support.`;

export const REVIEW_PROMPT = `Review the uncommitted changes in this repository. Do not change any file.

1. Read them with \`git status\` and \`git diff HEAD\`, and read every untracked file that is not ignored.
2. Read the code each change depends on, and run the narrowest tests that exercise it.
3. Report findings, most severe first. For each: the file and line, what is wrong, and the input or situation that shows it. Leave out style preferences.
4. If nothing is worth fixing, say so in one line.`;

function promptCommand(text) {
  if (text === INIT_PROMPT) return "/init";
  if (text === REVIEW_PROMPT) return "/review";
  const focusPrefix = `${REVIEW_PROMPT}\n\nFocus on: `;
  if (typeof text === "string" && text.startsWith(focusPrefix)) {
    return `/review ${text.slice(focusPrefix.length)}`;
  }
  if (typeof text === "string") {
    const mentioned = /^((?:Use the [^\n]+ skill\.\n)+)\n([\s\S]*)$/.exec(text);
    if (mentioned) {
      const names = [...mentioned[1].matchAll(/Use the (.+) skill\./g)].map((match) => match[1].toLowerCase());
      const tokens = [...mentioned[2].matchAll(/(?:^|\s)@([^\s]+)/g)]
        .map((match) => match[1].replace(/[.,;:!?)}\]]+$/, "").toLowerCase());
      if (names.every((name) => tokens.includes(name))) return mentioned[2];
    }
  }
  const skill = typeof text === "string" && /^Use the (\S+) skill\.(?:\n\n([\s\S]*))?$/.exec(text);
  return skill ? `/${skill[1]}${skill[2] ? ` ${skill[2]}` : ""}` : text;
}

function projectGroups(sessions, currentRoot) {
  const groups = new Map();
  for (const session of sessions) {
    const root = typeof session.repo_root === "string" ? session.repo_root : "";
    if (!groups.has(root)) {
      groups.set(root, { root, sessions: [], updatedAt: "" });
    }
    const group = groups.get(root);
    group.sessions.push(session);
    group.updatedAt = [group.updatedAt, session.updated_at ?? ""].sort().at(-1);
  }
  if (!groups.has(currentRoot)) {
    groups.set(currentRoot, { root: currentRoot, sessions: [], updatedAt: "" });
  }
  for (const group of groups.values()) {
    group.sessions.sort((left, right) =>
      (right.updated_at ?? "").localeCompare(left.updated_at ?? "")
    );
  }
  return [...groups.values()].sort((left, right) => {
    if (left.root === currentRoot) return -1;
    if (right.root === currentRoot) return 1;
    return right.updatedAt.localeCompare(left.updatedAt) || left.root.localeCompare(right.root);
  });
}

function projectName(root) {
  return root === "" ? "Unknown project" : root.split("/").filter(Boolean).at(-1);
}

function clientFor(global, supplied) {
  if (supplied) {
    return supplied;
  }
  const handshake = resolveHost(global);
  return createClient({
    port: handshake.port,
    token: handshake.token,
    fetch: global.fetch.bind(global),
  });
}

const ROUTE_KEY = "symphonai.route";
const RAIL_KEY = "symphonai.rail-tools";
const TERMINAL_PANEL_KEY = "symphonai.terminal-panel";
const DIFF_MODE_KEY = "symphonai.diff-mode";
const SIDEBAR_SESSION_LIMIT = 200;
const RAIL_TOOL_NAMES = ["Agents", "Roadmap", "Changes", "Files", "Jobs", "Loop", "Notes"];
const RAIL_TOOL_ICONS = Object.freeze({
  Agents: "icon-tool-agents",
  Roadmap: "icon-tool-roadmap",
  Changes: "icon-tool-changes",
  Files: "icon-tool-files",
  Jobs: "icon-tool-jobs",
  Loop: "icon-tool-loop",
  Notes: "icon-tool-notes",
});

function storedRail(global) {
  try {
    const value = JSON.parse(global.localStorage?.getItem(RAIL_KEY) ?? "null");
    const open = Array.isArray(value?.open)
      ? [...new Set(value.open.filter((name) => RAIL_TOOL_NAMES.includes(name)))]
      : ["Agents", "Roadmap"];
    const selected = open.includes(value?.selected) ? value.selected : open[0] ?? "";
    return { open, selected };
  } catch {
    return { open: ["Agents", "Roadmap"], selected: "Agents" };
  }
}

function rememberRail(global, open, selected) {
  try {
    global.localStorage?.setItem(RAIL_KEY, JSON.stringify({ open, selected }));
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
}

function storedRoute(global) {
  try {
    const fragment = global.localStorage?.getItem(ROUTE_KEY);
    return fragment ? parseRoute(fragment) : DEFAULT_ROUTE;
  } catch {
    return DEFAULT_ROUTE;
  }
}

function rememberRoute(global, route) {
  try {
    global.localStorage?.setItem(ROUTE_KEY, formatRoute(route));
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
}

function storedDiffMode(global) {
  try {
    return global.localStorage?.getItem(DIFF_MODE_KEY) === "side-by-side" ? "side-by-side" : "unified";
  } catch {
    return "unified";
  }
}

function rememberDiffMode(global, mode) {
  try {
    global.localStorage?.setItem(DIFF_MODE_KEY, mode);
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
}

export async function start({ global, document, client }) {
  const boundary = clientFor(global, client);
  let diffMode = storedDiffMode(global);
  const shell = document.getElementById("app-shell");
  const sidebar = document.getElementById("sidebar");
  const homeLink = document.getElementById("home-link");
  const newChat = document.getElementById("new-chat");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const railToggle = document.getElementById("rail-toggle");
  setToggleLabel(sidebarToggle, "sidebar", false);
  setToggleLabel(railToggle, "status", false);
  const pageLinks = document.getElementById("page-links");
  const pageRoot = document.getElementById("page");
  const chatPane = document.getElementById("chat-pane");
  const agentsRoot = document.getElementById("agents");
  const roadmapRoot = document.getElementById("roadmap");
  const changesRoot = document.getElementById("changes");
  const filesRoot = document.getElementById("files");
  const jobsRoot = element(document, "div", { className: "rail-pane jobs-pane" });
  jobsRoot.setAttribute("role", "tabpanel");
  jobsRoot.hidden = true;
  const notesRoot = element(document, "div", { className: "rail-pane notes-pane" });
  notesRoot.setAttribute("role", "tabpanel");
  notesRoot.hidden = true;
  const loopsRoot = element(document, "div", { className: "rail-pane loop-pane" });
  loopsRoot.setAttribute("role", "tabpanel");
  loopsRoot.hidden = true;
  const railTabsRoot = document.getElementById("rail-tabs");
  const railToolsRoot = document.getElementById("rail-tools");
  railToolsRoot.before(jobsRoot, loopsRoot, notesRoot);
  let { open: openRailTools, selected: selectedRailTool } = storedRail(global);
  let loops = [];
  let selectedLoopId = "";
  let selectedLoop = null;
  let loopError = "";
  let loopActionError = "";
  let loopActionPending = false;
  let loopCancelConfirm = false;
  let loopReviseNote = "";
  let selectedCardId = "";
  let cardDetail = null;
  let cardDetailLoading = false;
  let cardDetailError = "";
  let loopCommitError = "";
  let loopCommitPending = false;
  let loopDiscardConfirm = false;
  let loopCommitMessage = "";
  let jobsCommandDraft = "";
  let jobsPoll = null;
  const runNotice = document.getElementById("run-notice");
  const chatRoot = document.getElementById("chat");
  const jumpLatest = document.getElementById("jump-latest");
  const approvalsRoot = document.getElementById("approvals");
  const form = document.getElementById("prompt-form");
  const input = document.getElementById("prompt");
  const promptError = document.getElementById("prompt-error");
  const attachmentList = document.getElementById("prompt-attachments");
  const pendingFiles = [];
  const pastedTextChips = [];
  const previewMarks = [];
  let previewMarkSession = null;
  let previewMarksSequence = 0;
  const previewDeliveries = new Map();
  let nextPastedTextId = 1;
  const promptAttachments = [];
  let following = true;
  let savedScrollTop = 0;
  input.required = false;
  const commandMenu = element(document, "div", { className: "command-menu" });
  const fileMenu = element(document, "div", { className: "file-menu" });
  const historySearchBar = element(document, "div", { className: "history-search-bar" });
  historySearchBar.hidden = true;
  const shortcutHelpPanel = element(document, "div", { className: "shortcut-help-panel" });
  shortcutHelpPanel.hidden = true;
  const historySearchInput = element(document, "input", { className: "history-search-input" });
  historySearchInput.type = "search";
  historySearchInput.setAttribute("aria-label", "Search prompt history");
  historySearchInput.placeholder = "Search prompt history";
  const historySearchResult = element(document, "span", { className: "history-search-result" });
  const historySearchCount = element(document, "span", { className: "history-search-count" });
  append(historySearchBar, historySearchInput, historySearchResult, historySearchCount);
  const editConflictPanel = element(document, "div", { className: "edit-conflict-panel" });
  editConflictPanel.hidden = true;
  const pickerHost = element(document, "div", { className: "picker-host" });
  form.before(commandMenu, fileMenu, historySearchBar, shortcutHelpPanel, editConflictPanel, pickerHost);
  let quickOpenState = null;
  let quickOpenTimer = null;
  let quickOpenVersion = 0;
  let commandMatches = [];
  let highlightedCommand = 0;
  let fileMatches = [];
  let highlightedFile = 0;
  let fileFragment = null;
  let fileSearchSerial = 0;
  const turn = createTurnState();
  const promptAction = document.getElementById("prompt-action") ?? makeButton(document, "Send", null, { type: "submit", className: "primary composer-action" });
  if (!promptAction.parentNode) append(form, promptAction);
  const stopMirror = makeButton(document, "Stop", () => userCommandPending ? boundary.stop("") : perform(turn.stop()), { className: "primary composer-stop" });
  stopMirror.title = "Stop the running turn";
  const stopStatus = append(element(document, "div", { className: "conversation-usage composer-stop-status" }),
    element(document, "span", { text: "Running" }), stopMirror);
  stopStatus.hidden = true;
  listen(promptAction, "click", async event => {
    if (turn.state === RUNNING || turn.state === DISPATCHING || userCommandPending) {
      event.preventDefault();
      if (userCommandPending) await boundary.stop("");
      else await perform(turn.stop());
    }
  });
  let historyPrompts = null;
  const pagePrompts = [];
  let historyIndex = -1;
  let historyDraft = "";
  let recalledPrompt = null;
  let historySearchDraft = "";
  let historySearchMatches = [];
  let historySearchIndex = 0;
  let editingPreviousPrompt = null;
  let lastIdleEscapeAt = 0;
  let suppressEditUntil = 0;
  let escapeStopRequested = false;
  const approvalDiffs = new Map();
  const approvalClient = new Proxy(boundary, {
    get(target, property, receiver) {
      if (property === "approvals") {
        return async (...args) => {
          const reply = await boundary.approvals(...args);
          for (const approval of reply?.pending ?? []) {
            if (typeof approval.approval_id === "string" && typeof approval.diff === "string") {
              approvalDiffs.set(approval.approval_id, approval.diff);
            }
          }
          return reply;
        };
      }
      return Reflect.get(target, property, receiver);
    },
  });
  const approvals = createApprovals({ client: approvalClient });
  const approvalDenyNotes = new Map();
  const transcript = createTranscript();
  const agentTranscript = createTranscript();
  let agentView = null;
  let agentViewGeneration = 0;
  let preserveChatScroll = false;
  const agentViewHeader = element(document, "header", { className: "agent-conversation-header" });
  agentViewHeader.hidden = true;
  chatRoot.before(agentViewHeader);
  const missingKeyNotice = element(document, "p", { className: "missing-key-notice" });
  append(
    missingKeyNotice,
    element(document, "span", { text: "Add an API key to start: " }),
    modelsLink(),
    element(document, "span", { text: "." }),
  );
  missingKeyNotice.hidden = true;
  chatRoot.before(missingKeyNotice);
  let commandEntry = null;
  let userCommandPending = false;
  const board = createAgentBoard();
  const readRoadmap = async () => {
    try {
      const reply = await boundary.roadmap();
      return { goals: Array.isArray(reply?.goals) ? reply.goals : [], error: "" };
    } catch (error) {
      return { goals: [], error: errorText(error) };
    }
  };
  let project = await boundary.project();
  const projectless = project?.repo_root == null;
  let initialSessions = [];
  let initialArchivedSessions = [];
  let initialRoadmap = { goals: [], error: "" };
  let settingsReply;
  let healthReply;
  let conversationReply = { conversation: null };
  let initialProjects = { recent: [] };
  let mcpReply = { prompts: [] };
  if (projectless) {
    [settingsReply, healthReply, initialProjects] = await Promise.all([
      boundary.settings(),
      boundary.health().catch(() => null),
      (boundary.projects?.() ?? Promise.resolve({ recent: [] })).catch(() => ({ recent: [] })),
    ]);
  } else {
    [initialSessions, initialArchivedSessions, initialRoadmap, settingsReply, healthReply, conversationReply, mcpReply] = await Promise.all([
      boundary.sessions(SIDEBAR_SESSION_LIMIT),
      boundary.sessions(SIDEBAR_SESSION_LIMIT, undefined, true),
      readRoadmap(),
      boundary.settings(),
      boundary.health().catch(() => null),
      boundary.conversationStats().catch(() => ({ conversation: null })),
      (boundary.mcpPrompts?.() ?? Promise.resolve({ prompts: [] })).catch(() => ({ prompts: [] })),
    ]);
  }
  const providerRows = (settingsReply?.settings?.providers ?? []).map((row) => ({ ...row }));
  const getSkillCommands = (reply) => {
    const builtIns = new Set(COMMANDS.flatMap((command) => [command.name, ...command.aliases].map((name) => name.toLowerCase())));
    return (Array.isArray(reply?.settings?.skills) ? reply.settings.skills : [])
      .filter((skill) => typeof skill?.name === "string" && skill.name && !builtIns.has(skill.name.toLowerCase()))
      .map(({ name }) => ({
        name,
        aliases: [],
        description: `Use the ${name} skill`,
        argumentHint: `[<request>]`,
      }));
  };
  let skillCommands = getSkillCommands(settingsReply);

  async function refreshSettings() {
    settingsReply = await boundary.settings();
    skillCommands = getSkillCommands(settingsReply);
    if (boundary.mcpPrompts) {
      try { mcpCommands = getMcpCommands(await boundary.mcpPrompts()); }
      catch { mcpCommands = []; }
    }
    if (route?.page === "settings") renderSettingsSection(route.section, settingsContext());
    if (input.value) await renderComposerMenus();
    return settingsReply;
  }
  const getMcpCommands = reply => (Array.isArray(reply?.prompts) ? reply.prompts : []).map((prompt) => ({
    name: `mcp.${prompt.server}.${prompt.name}`, aliases: [], description: prompt.description || "MCP prompt",
    argumentHint: (prompt.arguments || []).map((argument) => `${argument.required ? "<" : "["}${argument.name}${argument.required ? ">" : "]"}`).join(" "),
    prompt,
  }));
  let mcpCommands = getMcpCommands(mcpReply);
  const defaultKeymap = parseKeymap(JSON.stringify(keymapDefaults));
  const userKeymap = global.__symphonai?.keys ?? global.keymap ?? userKeymapFile;
  let userBindings;
  try { userBindings = parseKeymap(JSON.stringify(userKeymap)); }
  catch (error) {
    userBindings = new Map();
    userBindings.rejected = [{ chord: "User keymap", action: "", reason: errorText(error) }];
  }
  const { bindings: keymap, rejected: keymapRejected } = mergeKeymap(defaultKeymap, userBindings);
  const platform = /mac/i.test(global.navigator?.platform ?? "") ? "darwin" : "other";
  const allModes = ["ask", "plan", "auto", "allow"];
  const ceilingModes = settingsReply?.settings?.ceiling?.modes;
  const permittedModes = Array.isArray(ceilingModes)
    ? allModes.filter((mode) => ceilingModes.includes(mode))
    : allModes;
  const configuredMode = settingsReply?.settings?.mode ?? "ask";
  const launchMode = permittedModes.includes(configuredMode)
    ? configuredMode
    : (permittedModes[0] ?? "ask");
  let currentMode = conversationReply?.conversation?.mode ?? launchMode;
  let specMode = false;
  let selection = settingsReply?.settings?.selection ?? { provider: "", model: null, effort: null };
  let rememberedPlanMode = null;
  let sessions = initialSessions;
  let archivedSessions = initialArchivedSessions;
  let editingSessionId = null;
  let editingSessionTitle = "";
  let focusSessionRename = false;
  let sessionRowError = "";
  let sessionArchiveError = "";
  let confirmingSessionId = null;
  let currentSessionId = conversationReply?.conversation?.session_id ?? null;
  transcript.setSessionId(currentSessionId);
  let conversation = conversationReply?.conversation ?? null;
  let pendingNewChat = null;
  function bindTranscriptLeader() {
    if ((conversation?.session_id ?? null) !== currentSessionId) return;
    const leader = conversation?.agents?.find((agent) => agent.parent_agent_id == null);
    transcript.setRootAgentId(leader?.agent_id ?? null);
  }
  bindTranscriptLeader();
  let draftSaveTimer = null;

  async function refreshSessionLists() {
    const [active, archived] = await Promise.all([
      boundary.sessions(SIDEBAR_SESSION_LIMIT),
      boundary.sessions(SIDEBAR_SESSION_LIMIT, undefined, true),
    ]);
    sessions = active;
    archivedSessions = archived;
  }

  function draftKey() {
    return JSON.stringify([project?.repo_root ?? "", currentSessionId ?? "new"]);
  }

  function readDrafts() {
    try {
      const stored = global.localStorage?.getItem(PROMPT_DRAFTS_KEY);
      const drafts = JSON.parse(stored ?? "[]");
      return Array.isArray(drafts)
        ? drafts.filter((draft) => (
          typeof draft?.key === "string"
          && typeof draft?.text === "string"
          && Number.isFinite(draft?.savedAt)
        ))
        : [];
    } catch {
      return [];
    }
  }

  function stopDraftTimer() {
    if (draftSaveTimer === null) return;
    try {
      (global.clearTimeout ?? clearTimeout)(draftSaveTimer);
    } catch {
      // A timer host may be unavailable alongside storage.
    }
    draftSaveTimer = null;
  }

  function saveDraft() {
    stopDraftTimer();
    try {
      const key = draftKey();
      const drafts = readDrafts().filter((draft) => draft.key !== key);
      if (input.value || pastedTextChips.length > 0) {
        drafts.push({
          key,
          text: input.value,
          pastes: pastedTextChips.map(({ id, text, placeholder, lines }) => ({
            id, text, placeholder, lines,
          })),
          savedAt: Date.now(),
        });
        drafts.sort((left, right) => left.savedAt - right.savedAt);
        if (drafts.length > 50) drafts.splice(0, drafts.length - 50);
      }
      global.localStorage?.setItem(PROMPT_DRAFTS_KEY, JSON.stringify(drafts));
    } catch {
      // Drafts are optional when storage is unavailable.
    }
  }

  function restoreDraft() {
    const key = draftKey();
    const draft = readDrafts().find((entry) => entry.key === key);
    input.value = draft?.text ?? "";
    pastedTextChips.splice(0);
    for (const paste of Array.isArray(draft?.pastes) ? draft.pastes : []) {
      if (
        Number.isInteger(paste?.id)
        && typeof paste.text === "string"
        && typeof paste.placeholder === "string"
        && Number.isInteger(paste.lines)
        && input.value.includes(paste.placeholder)
      ) {
        pastedTextChips.push({ ...paste, expanded: false });
        nextPastedTextId = Math.max(nextPastedTextId, paste.id + 1);
      }
    }
    showPendingAttachments();
  }

  function clearDraft() {
    stopDraftTimer();
    try {
      const key = draftKey();
      const drafts = readDrafts();
      const remaining = drafts.filter((draft) => draft.key !== key);
      if (remaining.length !== drafts.length) {
        global.localStorage?.setItem(PROMPT_DRAFTS_KEY, JSON.stringify(remaining));
      }
    } catch {
      // Drafts are optional when storage is unavailable.
    }
  }

  function queueDraftSave() {
    stopDraftTimer();
    try {
      const schedule = typeof global.setTimeout === "function"
        ? global.setTimeout.bind(global)
        : setTimeout;
      draftSaveTimer = schedule(() => {
        draftSaveTimer = null;
        saveDraft();
      }, 300);
    } catch {
      // The composer remains usable when timers are unavailable.
    }
  }

  restoreDraft();
  let roadmapGoals = Array.isArray(initialRoadmap.goals) ? initialRoadmap.goals : [];
  let roadmapError = initialRoadmap.error;
  let showRailTools = false;
  const settingsPane = element(document, "section", { className: "settings-pane" });
  const settingsSections = element(document, "nav", { className: "settings-sections" });
  const settingsContent = element(document, "div", { className: "settings-content" });
  append(settingsPane, element(document, "h1", { text: "Settings" }), settingsSections, settingsContent);
  let promptFailure = "";
  let route;
  const usagePane = element(document, "main", { className: "usage-page" });
  let usageDays = 30;
  let usageSequence = 0;
  const conversationUsage = element(document, "p", { className: "conversation-usage" });
  append(chatPane, conversationUsage);
  const statusFields = readStatusFields(global);
  let notificationsEnabled = false;
  try {
    notificationsEnabled = global.localStorage?.getItem(NOTIFICATIONS_KEY) === "true";
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
  let notificationMessage = "";
  let memoryReply = null;
  let memoryLoadError = null;
  let memoryLoading = false;
  let memoryLoadSequence = 0;
  let editingMemoryId = null;
  let memoryEditError = "";
  let confirmingMemoryDeleteId = null;
  let permissionsReply = null;
  let permissionsError = "";
  let permissionsLoading = false;

  function settingsTable(headings, rows) {
    const table = element(document, "table", { className: "settings-table" });
    const heading = element(document, "tr");
    append(heading, ...headings.map((text) => element(document, "th", { text })));
    append(table, heading);
    for (const cells of rows) {
      const row = element(document, "tr", { className: "settings-row" });
      append(row, ...cells.map((text) => element(document, "td", { text })));
      append(table, row);
    }
    return table;
  }

  function renderMemoryContent() {
    renderMemorySettings(settingsContext());
  }

  async function loadMemory() {
    const sequence = ++memoryLoadSequence;
    memoryLoading = true;
    memoryLoadError = null;
    renderMemoryContent();
    try {
      const reply = await boundary.memory();
      if (sequence !== memoryLoadSequence) return;
      memoryReply = reply;
    } catch {
      if (sequence !== memoryLoadSequence) return;
      memoryLoadError = true;
    }
    memoryLoading = false;
    if (route?.page === "settings" && route.section === "memory") {
      renderMemoryContent();
    }
  }

  function renderPermissionsContent() {
    renderPermissionSettings(settingsContext());
  }

  async function loadPermissions() {
    permissionsReply = null;
    permissionsError = "";
    permissionsLoading = true;
    renderPermissionsContent();
    try {
      permissionsReply = await boundary.permissions();
    } catch (error) {
      const detail = errorText(error).trim();
      permissionsError = detail ? `Could not load permissions: ${detail}` : "Could not load permissions.";
    }
    permissionsLoading = false;
    if (route?.page === "settings" && route.section === "permissions") {
      renderPermissionsContent();
    }
  }

  function settingsContext() {
    return {
      document,
      NOTIFICATIONS_KEY,
      renderMemoryContent,
      renderPermissionsContent,
      global,
      boundary,
      settingsContent,
      settingsTable,
      loadPermissions,
      loadMemory,
      providerRows,
      getSkillCommands,
      refreshSettings,
      statusFields,
      saveStatusFields,
      keymapRejected,
      showConversationUsage,
      get settingsReply() { return settingsReply; },
      set settingsReply(value) { settingsReply = value; },
      get skillCommands() { return skillCommands; },
      set skillCommands(value) { skillCommands = value; },
      get notificationsEnabled() { return notificationsEnabled; },
      set notificationsEnabled(value) { notificationsEnabled = value; },
      get notificationMessage() { return notificationMessage; },
      set notificationMessage(value) { notificationMessage = value; },

      get memoryLoading() { return memoryLoading; },
      set memoryLoading(value) { memoryLoading = value; },
      get memoryLoadError() { return memoryLoadError; },
      set memoryLoadError(value) { memoryLoadError = value; },
      get memoryReply() { return memoryReply; },
      set memoryReply(value) { memoryReply = value; },
      get editingMemoryId() { return editingMemoryId; },
      set editingMemoryId(value) { editingMemoryId = value; },
      get memoryEditError() { return memoryEditError; },
      set memoryEditError(value) { memoryEditError = value; },
      get confirmingMemoryDeleteId() { return confirmingMemoryDeleteId; },
      set confirmingMemoryDeleteId(value) { confirmingMemoryDeleteId = value; },
      get permissionsLoading() { return permissionsLoading; },
      set permissionsLoading(value) { permissionsLoading = value; },
      get permissionsReply() { return permissionsReply; },
      set permissionsReply(value) { permissionsReply = value; },
      get permissionsError() { return permissionsError; },
      set permissionsError(value) { permissionsError = value; },
      get route() { return route; },
      set route(value) { route = value; },
    };
  }

  function showSettings(section) {
    renderSettingsSection(section, settingsContext());
  }

  let pageNavLinks = [];
  let changesDirty = false;
  function editorFileLink(path, line = 1) {
    let scheme;
    try { scheme = global.localStorage?.getItem(EDITOR_KEY); }
    catch { return null; }
    const label = new Map([["vscode", "VS Code"], ["cursor", "Cursor"], ["zed", "Zed"]]).get(scheme);
    if (!label || typeof project?.repo_root !== "string") return null;
    const absolute = `${project.repo_root.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
    const encoded = absolute.split("/").map((part) => encodeURIComponent(part)).join("/");
    const link = element(document, "a", { text: `Open in ${label}` });
    link.href = `${scheme}://file${encoded}:${line}:1`;
    link.setAttribute("role", "button");
    listen(link, "keydown", (event) => {
      if (event.key === " ") { event.preventDefault(); link.click(); }
    });
    return link;
  }

  global.addEventListener?.(EDITOR_KEY, () => {
    renderFiles();
    if (selectedRailTool === "Changes" || route.page === "changes") void showChanges();
  });

  function firstChangedLine(diff) {
    let line = null;
    for (const text of diff.split("\n")) {
      const hunk = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(text);
      if (hunk) { line = Number(hunk[1]); continue; }
      if (line === null) continue;
      if (text.startsWith("+") || text.startsWith("-")) return Math.max(1, line);
      if (text.startsWith(" ")) line++;
    }
    return 1;
  }

  let changesSequence = 0, jobsSequence = 0, sourceSequence = 0, loopSequence = 0, cardSequence = 0;
  let jobsForm = null, jobsList = null, jobsNotice = null;
  const jobRows = new Map();
  const approvalAnswers = new Map();
  const approvalFeedback = new Map();
  async function showChanges() {
    const sequence = ++changesSequence;
    const sessionId = currentSessionId;
    replace(
      changesRoot,
      element(document, "h1", { text: "Changes" }),
      element(document, "p", { className: "changes-loading", text: "Loading changes…" }),
    );
    const [conversationResult, gitResult] = await Promise.allSettled([
      boundary.changes(),
      boundary.gitStatus(),
    ]);
    if (sequence !== changesSequence || sessionId !== currentSessionId) return;
    const reply = conversationResult.status === "fulfilled" ? conversationResult.value : null;
    const gitReply = gitResult.status === "fulfilled" ? gitResult.value : null;

    function showConflict(paths, payload) {
      const warning = element(document, "section", { className: "changes-conflict" });
      const list = element(document, "ul");
      for (const path of paths) append(list, element(document, "li", { text: path }));
      append(
        warning,
        element(document, "p", { text: "These files changed outside the agent:" }),
        list,
      );
      const force = makeButton(document, "Revert anyway", async () => {
        try {
          await boundary.revertChanges({ ...payload, force: true });
          await showChanges();
        } catch (error) {
          if (error?.status === 409 && Array.isArray(error.paths) && error.paths.length > 0) {
            showConflict(error.paths, payload);
          } else {
            replace(
              warning,
              element(document, "p", {
                className: "error",
                text: errorText(error),
              }),
            );
          }
        }
      });
      append(warning, force);
      changesRoot.append(warning);
    }

    async function revert(payload) {
      try {
        await boundary.revertChanges(payload);
        await showChanges();
      } catch (error) {
        if (error?.status === 409 && Array.isArray(error.paths) && error.paths.length > 0) {
          showConflict(error.paths, payload);
        } else {
          changesRoot.append(element(document, "p", {
            className: "error",
            text: errorText(error),
          }));
        }
      }
    }

    async function worktreeAction(name, action, row) {
      try {
        await boundary[action](name);
        await showChanges();
      } catch (error) {
        row.append(element(document, "p", {
          className: "error",
          text: errorText(error),
        }));
      }
    }

    const files = Array.isArray(reply?.files) ? reply.files : [];
    const turns = Array.isArray(reply?.turns) ? reply.turns : [];
    const worktrees = Array.isArray(reply?.worktrees) ? reply.worktrees : [];
    function diffView(diff) {
      const view = element(document, "div", { className: "changes-diff" });
      view.setAttribute("data-mode", diffMode);
      const unified = element(document, "div", { className: "diff-unified" });
      append(unified, renderDiff(document, diff));
      append(view, unified, renderSplitDiff(document, diff, { splitDiff }));
      return view;
    }
    const modeToggle = makeButton(document, diffMode === "side-by-side" ? "Unified" : "Side by side", null, { className: "changes-diff-toggle" });
    modeToggle.setAttribute("aria-pressed", String(diffMode === "side-by-side"));
    listen(modeToggle, "click", () => {
      diffMode = diffMode === "side-by-side" ? "unified" : "side-by-side";
      rememberDiffMode(global, diffMode);
      void showChanges();
    });
    const toolbar = element(document, "div", { className: "changes-toolbar" });
    append(toolbar, modeToggle);
    const children = [element(document, "h1", { text: "Changes" }), toolbar];
    const workingTree = element(document, "section", { className: "changes-working-tree" });
    append(workingTree, element(document, "h2", { text: "Working tree" }));
    if (gitResult.status === "rejected") {
      append(workingTree, element(document, "p", {
        className: "error",
        text: errorText(gitResult.reason),
      }));
    } else if (gitReply?.repo === false) {
      append(workingTree, element(document, "p", { text: "Not a git repository." }));
    } else {
      append(workingTree, element(document, "p", {
        className: "changes-branch",
        text: `Branch: ${gitReply?.branch ?? "(detached HEAD)"}`,
      }));
      const workingFiles = Array.isArray(gitReply?.files) ? gitReply.files : [];
      if (workingFiles.length === 0) {
        append(workingTree, element(document, "p", { text: "No uncommitted changes." }));
      }
      for (const file of workingFiles) {
        const row = element(document, "details", { className: "changes-working-file" });
        const summary = element(document, "summary");
        append(summary, element(document, "span", { text: `${file.path} · ${file.status}` }));
        const editorLink = editorFileLink(file.path, firstChangedLine(file.diff ?? ""));
        if (editorLink) append(summary, " ", editorLink);
        if (file.staged) append(summary, element(document, "span", { className: "changes-staged", text: "Staged" }));
        let diffLoaded = file.diff_deferred !== true;
        let loadingDiff = null;
        const diff = file.diff_deferred === true
          ? renderDiff(document, "")
          : diffView(file.diff ?? "");
        function loadDiff() {
          if (diffLoaded) return Promise.resolve(true);
          if (loadingDiff) return loadingDiff;
          diff.textContent = "Loading diff…";
          loadingDiff = Promise.resolve().then(() => boundary.gitDiff(file.path)).then((result) => {
            const updatedLink = editorFileLink(file.path, firstChangedLine(result.diff ?? ""));
            if (editorLink && updatedLink) editorLink.href = updatedLink.href;
            diffLoaded = true;
            row.replaceChildren(summary, diffView(result.diff ?? ""));
            return true;
          }).catch((error) => {
            diff.textContent = errorText(error);
            diff.className = "diff error";
            return false;
          }).finally(() => { loadingDiff = null; });
          return loadingDiff;
        }
        listen(row, "toggle", () => { if (row.open) void loadDiff(); });
        if (editorLink) listen(editorLink, "click", async (event) => {
          if (diffLoaded) return;
          event.preventDefault();
          event.stopPropagation();
          if (await loadDiff()) editorLink.click();
        });
        append(row, summary, diff);
        append(workingTree, row);
      }
    }
    children.push(workingTree);
    if (conversationResult.status === "rejected") {
      children.push(element(document, "p", {
        className: "error",
        text: errorText(conversationResult.reason),
      }));
    }
    if (conversationResult.status === "fulfilled" && files.length === 0 && worktrees.length === 0) {
      children.push(element(document, "p", { text: "No changes in this conversation." }));
    }
    if (worktrees.length > 0) {
      children.push(element(document, "h2", { text: "Worktrees" }));
      for (const worktree of worktrees) {
        const row = element(document, "section", { className: "changes-worktree" });
        append(row, element(document, "h3", { text: worktree.name }));
        append(row, element(document, "p", {
          className: "changes-paths",
          text: Array.isArray(worktree.files) ? worktree.files.join(", ") : "",
        }));
        append(row, element(document, "p", {
          className: "changes-state",
          text: "Not applied — these changes are in the worktree, not your files.",
        }));
        append(row, diffView(worktree.diff ?? ""));
        const apply = makeButton(document, "Apply worktree", () => worktreeAction(worktree.name, "applyWorktree", row));
        const discard = makeButton(document, "Discard worktree", () => worktreeAction(worktree.name, "discardWorktree", row));
        append(row, apply, discard);
        children.push(row);
      }
    }
    if (files.length > 0) {
      children.push(element(document, "h2", { text: "Files" }));
      for (const file of files) {
        const row = element(document, "section", { className: "changes-file" });
        const header = element(document, "h3", { text: `${file.path} · ${file.status}` });
        const editorLink = editorFileLink(file.path, firstChangedLine(file.diff ?? ""));
        if (editorLink) append(header, " ", editorLink);
        append(row, header);
        if (file.changed_outside) {
          append(row, element(document, "p", {
            className: "changes-outside",
            text: "Changed outside the agent.",
          }));
        }
        append(row, diffView(file.diff ?? ""));
        const button = makeButton(document, "Revert", () => revert({ path: file.path }));
        append(row, button);
        children.push(row);
      }
    }
    if (turns.length > 0) {
      children.push(element(document, "h2", { text: "Prompts" }));
      for (const turn of turns) {
        const row = element(document, "section", { className: "changes-turn" });
        append(row, element(document, "p", { text: turn.prompt || "(empty prompt)" }));
        append(row, element(document, "p", {
          className: "changes-paths",
          text: Array.isArray(turn.paths) ? turn.paths.join(", ") : "",
        }));
        const button = makeButton(document, "Revert this prompt and later", () => revert({ key: turn.key }));
        append(row, button);
        children.push(row);
      }
    }
    replace(changesRoot, ...children);
    changesDirty = false;
  }

  function jobsPaneVisible() {
    return selectedRailTool === "Jobs" && openRailTools.includes("Jobs") && !showRailTools;
  }

  function updateJobsPolling(jobs) {
    const shouldPoll = jobsPaneVisible() && jobs.some((job) => job.status === "running");
    if (shouldPoll && jobsPoll === null && typeof global.setInterval === "function") {
      jobsPoll = global.setInterval(() => void showJobs(), 1000);
    } else if (!shouldPoll && jobsPoll !== null) {
      global.clearInterval?.(jobsPoll);
      jobsPoll = null;
    }
  }

  let notesRequest = 0;

  function refreshOpenNotes() {
    if (selectedRailTool === "Notes") return showNotes();
    return undefined;
  }

  async function showNotes() {
    if (selectedRailTool !== "Notes" || showRailTools) return;
    const request = ++notesRequest;
    const sessionId = currentSessionId;
    replace(notesRoot, element(document, "h1", { text: "Notes" }));
    const empty = "No notes yet. The leader starts notes once the conversation is long.";
    if (!sessionId) {
      append(notesRoot, element(document, "p", { text: empty }));
      return;
    }
    try {
      const reply = await boundary.sessionNotes(sessionId);
      if (request !== notesRequest || sessionId !== currentSessionId || selectedRailTool !== "Notes") return;
      if (!reply.text) {
        append(notesRoot, element(document, "p", { text: empty }));
        return;
      }
      const updatedAt = Date.parse(reply.updated_at);
      if (Number.isFinite(updatedAt)) {
        const updated = new Date(updatedAt);
        const time = updated.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
        const label = updated.toDateString() === new Date().toDateString()
          ? time
          : `${updated.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${time}`;
        append(notesRoot, element(document, "p", {
          className: "activity-output-label", text: `Updated ${label}`,
        }));
      }
      const body = element(document, "div", { className: "assistant" });
      append(body, ...renderMarkdownBlocks(document, reply.text, parseMarkdown, { highlight }));
      append(notesRoot, body);
    } catch (error) {
      if (request !== notesRequest || sessionId !== currentSessionId || selectedRailTool !== "Notes") return;
      append(notesRoot, element(document, "p", {
        className: "error", text: `Could not load notes: ${errorText(error)}`,
      }));
    }
  }

  const terminalPanel = document.getElementById("terminal-panel");
  const terminalTabs = element(document, "div", { className: "terminal-tabs" });
  terminalTabs.setAttribute("role", "tablist");
  terminalTabs.setAttribute("aria-label", "Terminals");
  const terminalViews = element(document, "div", { className: "terminal-views" });
  const terminalError = element(document, "p", { className: "terminal-error error" });
  terminalError.setAttribute("aria-live", "polite");
  const terminalResizeHandle = element(document, "div", { className: "terminal-resize-handle" });
  terminalResizeHandle.setAttribute("role", "separator");
  terminalResizeHandle.setAttribute("aria-label", "Resize terminal panel");
  terminalResizeHandle.setAttribute("aria-orientation", "horizontal");
  terminalResizeHandle.tabIndex = 0;
  const terminalToggle = makeButton(document, "Terminal", () => toggleTerminal(), { className: "terminal-toggle" });
  terminalToggle.setAttribute("aria-controls", "terminal-panel");
  terminalToggle.disabled = projectless;
  const terminalSessions = new Map();
  const terminalPending = new Map();
  let terminalOpen = false, terminalHeight = 280, terminalLoading = false, terminalSelected = null;
  let terminalNumber = 0;
  try {
    const saved = JSON.parse(global.localStorage?.getItem(TERMINAL_PANEL_KEY) ?? "null");
    terminalOpen = saved?.open === true;
    if (Number.isFinite(saved?.height)) terminalHeight = saved.height;
  } catch {
    // Storage is optional in embedded and privacy-restricted browsers.
  }
  append(terminalPanel, terminalResizeHandle, terminalTabs, terminalViews, terminalError);

  function saveTerminalPanel() {
    try {
      global.localStorage?.setItem(TERMINAL_PANEL_KEY, JSON.stringify({ open: terminalOpen, height: terminalHeight }));
    } catch {
      // Storage is optional in embedded and privacy-restricted browsers.
    }
  }

  function sizeTerminalPanel(height = terminalHeight) {
    const maximum = Math.max(120, (global.innerHeight || 800) * 0.8);
    terminalHeight = Math.max(120, Math.min(height, maximum));
    terminalPanel.setAttribute("style", `height: ${terminalHeight}px`);
    terminalResizeHandle.setAttribute("aria-valuemin", "120");
    terminalResizeHandle.setAttribute("aria-valuemax", String(maximum));
    terminalResizeHandle.setAttribute("aria-valuenow", String(terminalHeight));
  }

  function terminalOptions() {
    const style = global.getComputedStyle?.(terminalPanel);
    const token = (name) => style?.getPropertyValue(name).trim();
    return {
      fontFamily: token("--font-code") || "monospace", fontSize: 13, scrollback: 2000,
      theme: {
        background: token("--ui-inset"), foreground: token("--ui-text"),
        cursor: token("--ui-text"), cursorAccent: token("--ui-inset"),
        selectionBackground: token("--ui-selected"), selectionForeground: token("--ui-text"),
        black: token("--ui-text-secondary"), brightBlack: token("--ui-text-secondary"),
        red: token("--ui-danger"), brightRed: token("--ui-danger"),
        green: token("--ui-success"), brightGreen: token("--ui-success"),
        yellow: token("--ui-warning"), brightYellow: token("--ui-warning"),
        blue: token("--ui-accent"), brightBlue: token("--ui-accent"),
        magenta: token("--ui-danger"), brightMagenta: token("--ui-danger"),
        cyan: token("--ui-accent"), brightCyan: token("--ui-accent"),
        white: token("--ui-text"), brightWhite: token("--ui-text"),
      },
    };
  }

  function fitTerminal() {
    const session = terminalSessions.get(terminalSelected);
    if (terminalPanel.hidden || !session || session.exited) return;
    try {
      session.terminal.options = terminalOptions();
      session.fit.fit();
      void boundary.terminalResize(session.id, session.terminal.cols, session.terminal.rows).catch((error) => {
        terminalError.textContent = errorText(error);
      });
    } catch (error) { terminalError.textContent = errorText(error); }
  }

  function renderTerminalTabs() {
    const tabs = [];
    for (const session of terminalSessions.values()) {
      const selected = terminalSelected === session.id;
      session.view.hidden = !selected;
      const row = element(document, "div", { className: "terminal-tab-item" });
      const tab = makeButton(document, `Shell ${session.number}${session.exited ? " (exited)" : ""}`, () => {
        terminalSelected = session.id;
        renderTerminalTabs(); fitTerminal(); session.terminal.focus();
      });
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(selected));
      tab.setAttribute("aria-controls", session.view.id);
      const close = makeButton(document, "×", () => closeTerminal(session));
      close.setAttribute("aria-label", `Close terminal ${session.number}`);
      close.setAttribute("title", `Close terminal ${session.number}`);
      append(row, tab, close);
      tabs.push(row);
    }
    const add = makeButton(document, "+", () => addTerminal());
    add.setAttribute("aria-label", "Add terminal");
    add.setAttribute("title", "Add terminal");
    add.disabled = terminalLoading || terminalSessions.size >= 4 || !global.Terminal || !global.FitAddon?.FitAddon;
    const close = makeButton(document, "Hide", () => toggleTerminal());
    close.setAttribute("aria-label", "Hide terminal panel");
    replace(terminalTabs, ...tabs, add, close);
  }

  function attachTerminal(saved) {
    const view = element(document, "div", { className: "terminal-view" });
    view.id = `terminal-${saved.id}`;
    view.setAttribute("role", "tabpanel");
    append(terminalViews, view);
    const terminal = new global.Terminal(terminalOptions());
    const fit = new global.FitAddon.FitAddon();
    terminal.loadAddon(fit);
    terminal.open(view);
    const session = { id: saved.id, number: ++terminalNumber, terminal, fit, view, exited: false, end: saved.end ?? Array.from(saved.output || "").length, pendingInput: "", inputInFlight: false };
    terminalSessions.set(saved.id, session);
    terminal.attachCustomKeyEventHandler((event) => {
      event.stopPropagation();
      if (lookup(keymap, event, { platform }) === "toggle-terminal") {
        event.preventDefault();
        if (event.type === "keydown") void toggleTerminal();
        return false;
      }
      return true;
    });
    let replaying = true;
    async function sendInput() {
      session.inputInFlight = true;
      try {
        while (session.pendingInput && !session.exited) {
          const data = session.pendingInput;
          session.pendingInput = "";
          try { await boundary.terminalInput(session.id, data); }
          catch (error) { terminalError.textContent = errorText(error); }
        }
      } finally { session.inputInFlight = false; }
    }
    terminal.onData((data) => {
      if (replaying || session.exited) return;
      session.pendingInput += data;
      if (!session.inputInFlight) void sendInput();
    });
    let replay = saved.output || "";
    const pending = terminalPending.get(saved.id) || [];
    terminalPending.delete(saved.id);
    for (const payload of pending) {
      replay += unseenTerminalOutput(session, payload);
      if (payload.exited) {
        session.exited = true;
        replay += `\r\n[exited ${payload.exit_code}]\r\n`;
      }
    }
    // Query replies and keystrokes are both stale until replay parsing ends.
    if (replay) terminal.write(replay, () => { replaying = false; });
    else replaying = false;
    terminalSelected ??= session.id;
    return session;
  }

  function unseenTerminalOutput(session, payload) {
    const offset = payload.offset ?? session.end;
    const length = Array.from(payload.data).length;
    const skipped = Math.max(0, session.end - offset);
    session.end = Math.max(session.end, offset + length);
    if (skipped >= length) return "";
    return skipped ? Array.from(payload.data).slice(skipped).join("") : payload.data;
  }

  function terminalOutput(payload) {
    const session = terminalSessions.get(payload.terminal_id);
    if (!session) {
      if (terminalLoading) {
        const pending = terminalPending.get(payload.terminal_id) || [];
        pending.push(payload);
        terminalPending.set(payload.terminal_id, pending);
      }
      return;
    }
    session.terminal.write(unseenTerminalOutput(session, payload));
    if (payload.exited) {
      session.exited = true;
      session.terminal.write(`\r\n[exited ${payload.exit_code}]\r\n`);
      renderTerminalTabs();
    }
  }

  async function addTerminal() {
    if (terminalLoading || terminalSessions.size >= 4) return;
    terminalLoading = true;
    terminalError.textContent = "";
    renderTerminalTabs();
    try {
      const reply = await boundary.terminalStart();
      attachTerminal({ id: reply.id, output: "" });
      terminalSelected = reply.id;
    } catch (error) { terminalError.textContent = errorText(error); }
    finally { terminalLoading = false; terminalPending.clear(); }
    renderTerminalTabs(); fitTerminal();
    if (terminalOpen) terminalSessions.get(terminalSelected)?.terminal.focus();
  }

  async function closeTerminal(session) {
    try { await boundary.terminalStop(session.id); }
    catch (error) { terminalError.textContent = errorText(error); return; }
    session.exited = true;
    session.terminal.dispose(); session.view.remove();
    terminalSessions.delete(session.id);
    if (terminalSelected === session.id) terminalSelected = terminalSessions.keys().next().value ?? null;
    renderTerminalTabs(); fitTerminal();
    if (terminalOpen) terminalSessions.get(terminalSelected)?.terminal.focus();
  }

  async function toggleTerminal() {
    if (projectless) return;
    terminalOpen = !terminalOpen;
    terminalPanel.hidden = !terminalOpen;
    terminalToggle.setAttribute("aria-expanded", String(terminalOpen));
    saveTerminalPanel();
    if (!terminalOpen) { input.focus(); return; }
    await openTerminalPanel();
  }

  async function openTerminalPanel() {
    terminalPanel.hidden = false;
    sizeTerminalPanel();
    if (!global.Terminal || !global.FitAddon?.FitAddon) {
      terminalError.textContent = "Terminal unavailable";
      renderTerminalTabs();
      return;
    }
    if (!terminalLoading) {
      terminalLoading = true;
      terminalError.textContent = "";
      renderTerminalTabs();
      try {
        const reply = await boundary.terminals();
        for (const saved of reply.terminals.filter((item) => !item.exited)) {
          if (!terminalSessions.has(saved.id) && terminalSessions.size < 4) attachTerminal(saved);
        }
      } catch (error) { terminalError.textContent = errorText(error); }
      finally { terminalLoading = false; terminalPending.clear(); }
      if (!terminalSessions.size && !terminalError.textContent) await addTerminal();
    }
    renderTerminalTabs(); fitTerminal();
    if (terminalOpen) terminalSessions.get(terminalSelected)?.terminal.focus();
  }

  let terminalDrag = null;
  listen(terminalResizeHandle, "pointerdown", (event) => {
    terminalDrag = { y: event.clientY, height: terminalHeight };
    terminalResizeHandle.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });
  listen(terminalResizeHandle, "pointermove", (event) => {
    if (!terminalDrag) return;
    sizeTerminalPanel(terminalDrag.height + terminalDrag.y - event.clientY);
    fitTerminal();
  });
  const finishTerminalDrag = () => { terminalDrag = null; saveTerminalPanel(); };
  listen(terminalResizeHandle, "pointerup", finishTerminalDrag);
  listen(terminalResizeHandle, "pointercancel", finishTerminalDrag);
  listen(terminalResizeHandle, "keydown", (event) => {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    sizeTerminalPanel(terminalHeight + (event.key === "ArrowUp" ? 20 : -20));
    saveTerminalPanel(); fitTerminal();
  });
  global.addEventListener?.("resize", () => { sizeTerminalPanel(); fitTerminal(); });
  if (global.ResizeObserver) new global.ResizeObserver(() => fitTerminal()).observe(terminalViews);
  const updateTerminalTheme = () => {
    for (const session of terminalSessions.values()) session.terminal.options = terminalOptions();
  };
  if (global.MutationObserver) new global.MutationObserver(updateTerminalTheme)
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  global.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", updateTerminalTheme);
  sizeTerminalPanel();
  terminalPanel.hidden = !terminalOpen;
  terminalToggle.setAttribute("aria-expanded", String(terminalOpen));

  async function showJobs() {
    const sequence = ++jobsSequence;
    if (!jobsPaneVisible()) { updateJobsPolling([]); return; }
    if (!jobsForm) {
      jobsForm = element(document, "div", { className: "jobs-command-form" });
      jobsList = element(document, "div");
      jobsNotice = element(document, "p");
      const command = element(document, "input", { className: "jobs-command" });
      command.type = "text";
      command.placeholder = "Command to run in the background";
      command.setAttribute("aria-label", "Background command");
      command.value = jobsCommandDraft;
      const run = makeButton(document, "Run in background");
      run.disabled = !command.value.trim();
      listen(command, "input", () => { jobsCommandDraft = command.value; run.disabled = !command.value.trim(); });
      listen(run, "click", async () => {
        const value = command.value;
        if (!value.trim()) return;
        run.disabled = true;
        try {
          await boundary.startJob(value);
          if (command.value === value) { command.value = ""; jobsCommandDraft = ""; }
          await showJobs();
        } catch (error) { jobsNotice.textContent = errorText(error); }
        finally { run.disabled = !command.value.trim(); }
      });
      append(jobsForm, command, run);
      replace(jobsRoot, element(document, "h1", { text: "Jobs" }), jobsForm, jobsNotice, jobsList);
    }
    let jobs;
    try { const reply = await boundary.jobs(); jobs = Array.isArray(reply?.jobs) ? reply.jobs : []; }
    catch (error) {
      if (sequence !== jobsSequence) return;
      jobsNotice.textContent = errorText(error);
      updateJobsPolling([]);
      return;
    }
    if (sequence !== jobsSequence || !jobsPaneVisible()) return;
    jobsNotice.textContent = jobs.length ? "" : "No background jobs.";
    const ids = new Set(jobs.map(job => job.id));
    for (const [id, row] of jobRows) if (!ids.has(id)) { row.root.remove(); jobRows.delete(id); }
    const now = global.Date?.now?.() ?? Date.now();
    for (const job of jobs) {
      let row = jobRows.get(job.id);
      if (!row) {
        row = { root: element(document, "details", { className: "jobs-row" }),
          label: element(document, "span"), owner: element(document, "span"),
          status: element(document, "span", { className: "jobs-status" }), elapsed: element(document, "span"),
          output: element(document, "pre", { className: "jobs-output" }), dropped: element(document, "p") };
        row.stop = makeButton(document, "Stop", async () => {
          try { await boundary.stopJob(job.id); await showJobs(); }
          catch (error) { jobsNotice.textContent = errorText(error); }
        });
        append(row.root, append(element(document, "summary"), row.label, row.owner, row.status, row.elapsed, row.stop), row.dropped, row.output);
        jobRows.set(job.id, row);
        append(jobsList, row.root);
      }
      row.label.textContent = job.label;
      row.owner.textContent = job.owner === "person" ? "you" : "agent";
      row.status.textContent = job.status === "exited" ? `exited (exit code ${job.exit_code})` : job.status;
      const start = Date.parse(job.started_at), end = job.ended_at ? Date.parse(job.ended_at) : now;
      row.elapsed.textContent = Number.isFinite(start) && Number.isFinite(end) ? `${Math.max(0, Math.floor((end - start) / 1000))}s` : "time unavailable";
      row.stop.hidden = job.status !== "running";
      if (row.output.textContent !== (job.output_tail ?? "")) row.output.textContent = job.output_tail ?? "";
      row.dropped.textContent = job.output_dropped ? "Earlier output was dropped." : "";
    }
    updateJobsPolling(jobs);
  }

  function showPage(nextRoute) {
    if (route?.page === "chat") {
      savedScrollTop = chatRoot.scrollTop;
    }
    route = nextRoute;
    if (projectless && projectStartScreen) {
      replace(pageRoot, projectStartScreen);
      return;
    }
    for (const { page, link } of pageNavLinks) {
      if (page === route.page) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    for (const { section, link } of sectionLinks) {
      const currentSection = route.section || "general";
      if (route.page === "settings" && section === currentSection) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
    if (route.page === "settings") {
      showSettings(route.section);
    }
    const pane = route.page === "usage" ? usagePane : route.page === "settings" ? settingsPane : chatPane;
    replace(pageRoot, pane, stopStatus, terminalPanel);
    showPromptError();
    terminalPanel.hidden = !terminalOpen;
    if (!terminalPanel.hidden) fitTerminal();
    if (route.page === "chat") {
      chatRoot.scrollTop = following ? chatRoot.scrollHeight : savedScrollTop;
      jumpLatest.hidden = following;
    }
    updateMissingKeyNotice();
  }

  async function showUsage(days = usageDays) {
    usageDays = days;
    const sequence = ++usageSequence;
    showPage({ page: "usage", section: "" });
    const title = element(document, "h1", { text: "Usage" });
    const ranges = element(document, "div", { className: "usage-ranges" });
    ranges.setAttribute("aria-label", "Usage period");
    for (const range of [7, 30, 90]) {
      const button = makeButton(document, `${range} days`, () => showUsage(range));
      button.setAttribute("aria-pressed", String(range === days));
      append(ranges, button);
    }
    const content = element(document, "div", { className: "usage-content" });
    append(content, element(document, "p", { text: "Loading usage…" }));
    replace(usagePane, title, ranges, content);
    try {
      const report = await boundary.usage(days);
      if (sequence !== usageSequence) return;
      replace(content);
      append(content, element(document, "p", { className: "usage-scope", text:
        "Tokens and estimates cover saved calls across projects. Vendor costs cover the organization and may lag recent calls." }));
      const counts = new Intl.NumberFormat();
      const balanceNumbers = new Intl.NumberFormat(undefined, { maximumFractionDigits: 20 });
      const amountText = (value) => {
        if (!value) return "unavailable";
        if (!value.currency) return `${balanceNumbers.format(Number(value.amount))} balance`;
        try {
          const money = new Intl.NumberFormat(undefined, {
            style: "currency", currency: value.currency, minimumFractionDigits: 2, maximumFractionDigits: 2,
          });
          const exact = Number(value.exact_amount ?? value.amount);
          if (exact !== 0 && Math.abs(exact) < 0.01) {
            return exact < 0 ? `>-${money.format(0.01)}` : `<${money.format(0.01)}`;
          }
          return money.format(Number(value.amount));
        } catch {
          return `${balanceNumbers.format(Number(value.exact_amount ?? value.amount))} ${value.currency}`;
        }
      };
      const vendorNames = {
        anthropic: "Anthropic", openai: "OpenAI", gemini: "Google Gemini", deepseek: "DeepSeek",
        kimi: "Moonshot (Kimi)", grok: "xAI (Grok)", openrouter: "OpenRouter",
        qwen: "Alibaba Qwen (DashScope)", ollama: "Ollama (local)", groq: "Groq", together: "Together AI",
      };
      const providers = report.providers ?? [];
      if (providers.length === 0) append(content, element(document, "p", { text: "No saved usage or connected balances for this period." }));
      for (const provider of providers) {
        const section = element(document, "section", { className: "usage-provider" });
        append(section, element(document, "h2", { text: vendorNames[provider.provider] ?? provider.provider }));
        const table = element(document, "table", { className: "usage-table" });
        const header = element(document, "tr");
        for (const name of ["Model", "Input", "Cached", "Output", "Cost", "Balance"]) {
          const cell = element(document, "th", { text: name });
          cell.setAttribute("scope", "col");
          append(header, cell);
        }
        const head = element(document, "thead");
        append(head, header);
        const body = element(document, "tbody");
        const addRow = (label, input, cached, output, cost = "", balance = "") => {
          const row = element(document, "tr");
          for (const value of [label, counts.format(input), counts.format(cached), counts.format(output), cost, balance]) append(row, element(document, "td", { text: String(value) }));
          append(body, row);
        };
        for (const model of provider.models ?? []) {
          addRow(model.model, model.input_tokens, (model.cache_read_tokens ?? 0) + (model.cache_write_tokens ?? 0), model.output_tokens);
        }
        addRow("Total", provider.input_tokens ?? 0, (provider.cache_read_tokens ?? 0) + (provider.cache_write_tokens ?? 0),
          provider.output_tokens ?? 0, provider.cost ? `${amountText(provider.cost)} · ${provider.cost.source}` : "unavailable",
          provider.balance?.map(amountText).join("; ") || "unavailable");
        append(table, head, body);
        append(section, table);
        if (provider.cost?.data_refreshed_at) append(section, element(document, "p", { className: "usage-refreshed", text: `Vendor data refreshed at ${provider.cost.data_refreshed_at}` }));
        if (provider.error) append(section, element(document, "p", { className: "error", text: provider.error }));
        append(content, section);
      }
      append(content, element(document, "h2", { text: "Daily tokens (UTC)" }));
      const daily = report.daily ?? [];
      const largest = Math.max(1, ...daily.map((day) => day.input_tokens + day.output_tokens));
      const bars = element(document, "div", { className: "usage-daily" });
      for (const day of daily) {
        const total = day.input_tokens + day.output_tokens;
        const row = element(document, "div", { className: "usage-day" });
        const track = element(document, "div", { className: "usage-bar-track" });
        const bar = element(document, "span", { className: "usage-bar" });
        bar.setAttribute("style", `width: ${Math.max(0, total / largest * 100)}%`);
        bar.setAttribute("aria-hidden", "true");
        append(track, bar);
        append(row, element(document, "span", { text: day.date }), track, element(document, "span", { text: counts.format(total) }));
        append(bars, row);
      }
      append(content, bars);
      if ((report.projects ?? []).length) {
        append(content, element(document, "h2", { text: "By project" }));
        for (const project of report.projects) append(content, element(document, "p", { className: "usage-project", text:
          `${project.repo_root || "Unknown project"}: ${counts.format(project.input_tokens)} input · ${counts.format(project.cache_read_tokens + project.cache_write_tokens)} cached · ${counts.format(project.output_tokens)} output` }));
      }
      for (const warning of report.warnings ?? []) append(content, element(document, "p", { className: "usage-warning", text: warning }));
    } catch (error) {
      if (sequence === usageSequence) replace(content, element(document, "p", { className: "error", text: `Could not load usage: ${errorText(error)}` }));
    }
  }

  function navigate(nextRoute, { updateFragment = true } = {}) {
    const next = parseRoute(formatRoute(nextRoute));
    showPage(next);
    rememberRoute(global, next);
    if (updateFragment && global.location) {
      global.location.hash = formatRoute(next);
    }
  }

  const sectionLinks = ["general", "models", "mcp", "skills", "plugins", "agents", "memory", "hooks", "ceiling", "permissions", "trust", "inventory"].map((section) => {
    const link = element(document, "a", {
      text: section[0].toUpperCase() + section.slice(1),
    });
    link.href = formatRoute({ page: "settings", section });
    listen(link, "click", (event) => {
      event.preventDefault();
      navigate({ page: "settings", section });
    });
    return { section, link };
  });
  replace(settingsSections, ...sectionLinks.map(({ link }) => link));

  const pageOrder = ["chat", "settings"];
  const links = pageOrder.map((page) => {
    const pageRoute = { page, section: "" };
    const link = element(document, "a", {
      ...(page === "settings" ? { className: "icon-settings" } : {}),
      text: page[0].toUpperCase() + page.slice(1),
    });
    link.href = formatRoute(pageRoute);
    listen(link, "click", (event) => {
      event.preventDefault();
      navigate(pageRoute);
    });
    return link;
  });
  pageNavLinks = pageOrder.map((page, index) => ({ page, link: links[index] }));
  const [chatLink, settingsLink] = links;
  replace(pageLinks, chatLink);
  listen(homeLink, "click", () => navigate({ page: "chat", section: "" }));

  const projectsRoot = element(document, "section", { className: "projects" });
  let projectActionError = "";
  let folderPicker = null;
  let projectStartScreen = null;

  async function openProjectAt(path, sessionId = null) {
    saveDraft();
    closeAgentView();
    projectActionError = "";
    try {
      const opened = await boundary.openProject(path);
      showConfigurationWarnings(opened?.config_warnings);
      if (sessionId) await boundary.openSession(sessionId);
      global.location?.reload?.();
      return true;
    } catch (error) {
      projectActionError = errorText(error);
      if (folderPicker) renderFolderPicker();
      else if (projectless) renderProjectStart();
      showProjects();
      return false;
    }
  }

  function closeFolderPicker() {
    folderPicker = null;
    replace(projectless ? pageRoot : pickerHost);
    if (projectless) renderProjectStart();
  }

  function renderFolderPicker() {
    if (!folderPicker) return;
    const dialog = element(document, "section", { className: "folder-picker" });
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-labelledby", "folder-picker-title");
    const heading = element(document, "h2", { text: "Open a project folder" });
    heading.id = "folder-picker-title";
    const path = element(document, "p", { className: "folder-picker-path", text: folderPicker.listing?.path ?? "Loading folders…" });
    const up = makeButton(document, "Up");
    up.disabled = !folderPicker.listing?.parent;
    listen(up, "click", () => folderPicker.listing?.parent && loadDirectories(folderPicker.listing.parent));
    const entries = element(document, "div", { className: "folder-picker-entries" });
    for (const entry of folderPicker.listing?.entries ?? []) {
      const button = makeButton(document, entry.name, () => loadDirectories(entry.path));
      append(entries, button);
    }
    const error = element(document, "p", { className: "project-action-error", text: folderPicker.error || projectActionError });
    const open = makeButton(document, "Open this folder", null, { className: "primary" });
    open.disabled = !folderPicker.listing?.path;
    listen(open, "click", () => folderPicker.listing?.path && openProjectAt(folderPicker.listing.path));
    const cancel = makeButton(document, "Cancel", closeFolderPicker);
    append(dialog, heading, path, up, entries, error, open, cancel);
    replace(projectless ? pageRoot : pickerHost, dialog);
    (folderPicker.listing?.entries?.[0] ? entries.children[0] : open).focus?.();
  }

  async function loadDirectories(path) {
    if (!folderPicker) return;
    try {
      folderPicker.listing = await boundary.directories(path);
      folderPicker.error = "";
    } catch (error) {
      folderPicker.error = errorText(error);
    }
    renderFolderPicker();
  }

  function showFolderPicker() {
    projectActionError = "";
    folderPicker = { listing: null, error: "" };
    renderFolderPicker();
    void loadDirectories();
  }

  function renderProjectStart() {
    if (!projectless) return;
    projectStartScreen = element(document, "section", { className: "project-start" });
    append(projectStartScreen, element(document, "h1", { text: "Open a project" }));
    const recent = Array.isArray(initialProjects?.recent) ? initialProjects.recent : [];
    const list = element(document, "div", { className: "project-start-list" });
    for (const item of recent) {
      const row = element(document, "div", { className: "project-start-row" });
      const label = `${item.name || projectName(item.repo_root)}${item.exists ? "" : " · missing"}`;
      if (item.exists) {
        const button = makeButton(document, label);
        button.setAttribute("title", item.repo_root);
        listen(button, "click", () => openProjectAt(item.repo_root));
        append(row, button);
      } else {
        append(row, element(document, "span", { text: label }));
      }
      append(list, row);
    }
    const openFolder = makeButton(document, "Open folder…", showFolderPicker, { className: "primary" });
    append(projectStartScreen, list, openFolder,
      element(document, "p", { className: "project-action-error", text: projectActionError }));
    replace(pageRoot, projectStartScreen);
  }

  function dismissTopLayer() {
    if (folderPicker) closeFolderPicker();
    else if (quickOpenState) closeQuickOpen();
    else if (!shortcutHelpPanel.hidden) shortcutHelpPanel.hidden = true;
    else if (!historySearchBar.hidden) closeHistorySearch(true);
    else if (fileFragment !== null) closeFileMenu();
    else if (commandMatches.length) closeCommandMenu();
    else if (!findBar.hidden) closeFind();
    else if (selectedCardId && selectedRailTool === "Loop") returnToLoopBoard();
    else if (agentView) closeAgentView();
    else return false;
    return true;
  }
  listen(document, "keydown", event => {
    if (event.defaultPrevented || event.key !== "Escape") return;
    if (dismissTopLayer()) {
      event.preventDefault();
      event.stopPropagation?.();
    } else if (event.target !== input && (turn.state === RUNNING || turn.state === DISPATCHING || userCommandPending)) {
      event.preventDefault();
      event.stopPropagation?.();
      void (userCommandPending ? boundary.stop("") : perform(turn.stop()));
    }
  });

  const findBar = element(document, "div", { className: "transcript-find" });
  findBar.hidden = true;
  const findInput = element(document, "input", { className: "transcript-find-input" });
  findInput.type = "search";
  findInput.setAttribute("aria-label", "Find in conversation");
  const findCount = element(document, "span", { className: "transcript-find-count", text: "0 of 0" });
  const findClose = makeButton(document, "Close");
  chatRoot.before(findBar);
  append(findBar, findInput, findCount, findClose);
  let findReplacements = [];
  let findMatches = [];
  let activeFindMatch = -1;

  function clearFindHighlights() {
    for (const { parent, original, wrapper } of findReplacements) {
      if (wrapper.parentNode === parent) parent.replaceChild(original, wrapper);
    }
    findReplacements = [];
    findMatches = [];
    activeFindMatch = -1;
  }

  function updateFindCount() {
    findCount.textContent = findMatches.length === 0
      ? "0 of 0"
      : `${activeFindMatch + 1} of ${findMatches.length}`;
  }

  function highlightFindMatches() {
    clearFindHighlights();
    const query = findInput.value.trim();
    if (!query) {
      updateFindCount();
      return;
    }
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(escaped, "gi");
    function visit(node) {
      if (node.nodeType === 3 || (node.nodeType === 1 && !(node.childNodes?.length) && node._textContent)) {
        const source = node.textContent;
        pattern.lastIndex = 0;
        const matches = [...source.matchAll(pattern)];
        if (matches.length === 0 || !node.parentNode) return;
        const parent = node.parentNode;
        const wrapper = element(document, "span", { className: "transcript-find-text" });
        let cursor = 0;
        for (const match of matches) {
          const start = match.index;
          if (start > cursor) wrapper.append(document.createTextNode(source.slice(cursor, start)));
          const highlight = element(document, "mark", {
            className: "transcript-find-match",
            text: match[0],
          });
          wrapper.append(highlight);
          findMatches.push(highlight);
          cursor = start + match[0].length;
        }
        if (cursor < source.length) wrapper.append(document.createTextNode(source.slice(cursor)));
        parent.replaceChild(wrapper, node);
        findReplacements.push({ parent, original: node, wrapper });
        return;
      }
      for (const child of [...(node.childNodes ?? [])]) visit(child);
    }
    visit(chatRoot);
    activeFindMatch = findMatches.length > 0 ? 0 : -1;
    updateFindCount();
    findMatches[0]?.scrollIntoView?.({ block: "nearest" });
  }

  let findOpener = null;

  function closeFind() {
    clearFindHighlights();
    findInput.value = "";
    findBar.hidden = true;
    findCount.textContent = "0 of 0";
    let visible = findOpener && document.documentElement.contains(findOpener);
    for (let node = findOpener; visible && node?.nodeType === 1; node = node.parentNode) {
      const style = global.getComputedStyle?.(node);
      if (node.hidden || node.inert || style?.display === "none"
        || style?.visibility === "hidden" || style?.visibility === "collapse") visible = false;
    }
    (visible ? findOpener : input).focus();
    findOpener = null;
  }

  function openFind() {
    if (findBar.hidden) findOpener = document.activeElement;
    if (route.page !== "chat") navigate({ page: "chat" });
    findBar.hidden = false;
    findInput.focus();
    findInput.setSelectionRange?.(0, findInput.value.length);
  }

  function stepFind(direction) {
    if (findMatches.length === 0) return;
    activeFindMatch = (activeFindMatch + direction + findMatches.length) % findMatches.length;
    updateFindCount();
    findMatches[activeFindMatch]?.scrollIntoView?.({ block: "nearest" });
  }

  listen(findInput, "input", highlightFindMatches);
  listen(findInput, "keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      stepFind(event.shiftKey ? -1 : 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeFind();
    }
  });
  listen(findClose, "click", closeFind);

  function showTranscript() {
    const onChatPage = route.page === "chat";
    if (onChatPage && !preserveChatScroll) {
      following = chatRoot.scrollHeight - chatRoot.scrollTop - chatRoot.clientHeight <= 24;
    }
    const actions = {
      promptText: promptCommand,
      highlight,
      copyText: async (text) => {
        const clipboard = global.navigator?.clipboard;
        if (typeof clipboard?.writeText !== "function") {
          throw new Error("clipboard is unavailable");
        }
        await clipboard.writeText(text);
      },
      onViewChanges: () => openRailTool("Changes"),
      openSourceFile: async (path) => {
        await openRailTool("Files");
        await openSourceFile(path);
      },
      onCommit: async (entry) => {
        const paths = entry.files.filter((file) => file.checked)
          .flatMap((file) => file.paths ?? [file.path]);
        try {
          const result = await boundary.commit(entry.message, paths);
          entry.type = "command";
          entry.output = [`Committed ${result.commit.slice(0, 7)}: ${entry.message.split(/\r?\n/, 1)[0]}`];
          showTranscript();
          await showChanges();
        } catch (error) {
          entry.error = errorText(error);
          showTranscript();
        }
      },
      onCancelCommit: (entry) => {
        const index = transcript.model.indexOf(entry);
        if (index !== -1) transcript.model.splice(index, 1);
        showTranscript();
      },
      onOpenPullRequest: async (entry) => {
        try {
          const result = await boundary.openPullRequest(entry.title, entry.body);
          entry.type = "pull-request-done";
          entry.url = result.url;
          showTranscript();
        } catch (error) {
          entry.error = errorText(error);
          showTranscript();
        }
      },
      onClosePullRequest: (entry) => {
        const index = transcript.model.indexOf(entry);
        if (index !== -1) transcript.model.splice(index, 1);
        showTranscript();
      },
      onUndo: async (entry, force = false) => {
        entry.undoing = true;
        entry.error = "";
        entry.conflicts = [];
        showTranscript();
        try {
          await boundary.revertChanges({ key: entry.checkpointKey, ...(force ? { force: true } : {}) });
          entry.undone = true;
          entry.canUndo = false;
          entry.conflicts = [];
        } catch (error) {
          if (error?.status === 409 && Array.isArray(error.paths) && error.paths.length > 0) {
            entry.conflicts = error.paths;
          } else {
            entry.error = errorText(error);
          }
        } finally {
          entry.undoing = false;
          showTranscript();
        }
      },
      rerender: showTranscript,
    };
    if (agentView?.loading) {
      replace(chatRoot, element(document, "p", {
        className: "agent-history-status",
        text: "Loading agent history…",
      }));
    } else if (agentView?.error) {
      replace(chatRoot, element(document, "p", {
        className: "agent-history-error",
        text: `Could not load ${agentView.name}'s conversation: ${agentView.error}`,
      }));
    } else {
      renderTranscript(
        document,
        chatRoot,
        agentView ? agentTranscript.model : transcript.model,
        parseMarkdown,
        agentView ? { copyText: actions.copyText } : actions,
        {
          requestAnimationFrame: global.requestAnimationFrame?.bind(global),
          isCurrent: () => !agentView?.loading && !agentView?.error,
          editingEntry: agentView ? null : editingPreviousPrompt,
          viewKey: agentView?.agentId ?? null,
          returnScrollTop: agentView?.returnScrollTop,
          afterRender: () => {
            if (!findBar.hidden && findInput.value) highlightFindMatches();
            if (route.page === "chat" && !preserveChatScroll && following) {
              chatRoot.scrollTop = chatRoot.scrollHeight;
              jumpLatest.hidden = true;
            }
          },
        },
      );
    }
    if (editingPreviousPrompt && !agentView) {
      const promptIndex = transcript.model
        .slice(0, transcript.model.indexOf(editingPreviousPrompt))
        .filter((entry) => entry.type === "prompt").length;
      const promptNode = Array.from(chatRoot.children)
        .filter((node) => node.className === "prompt")[promptIndex];
      if (promptNode) promptNode.className += " editing-previous-prompt";
    }
    if (!findBar.hidden && findInput.value) highlightFindMatches();
    updateMissingKeyNotice();
    if (!onChatPage) return;
    if (preserveChatScroll) return;
    if (following) {
      chatRoot.scrollTop = chatRoot.scrollHeight;
      jumpLatest.hidden = true;
    } else {
      jumpLatest.hidden = false;
    }
  }

  function renderAgentViewHeader() {
    showAgents();
    if (!agentView) {
      agentViewHeader.hidden = true;
      replace(agentViewHeader);
      form.hidden = false;
      showPromptError();
      return;
    }
    replace(
      agentViewHeader,
      element(document, "h2", { className: "agent-conversation-title", text: agentView.name }),
    );
    agentViewHeader.hidden = false;
    form.hidden = true;
    showPromptError();
  }

  async function openAgentView(rowData) {
    if (!currentSessionId || rowData.name === "leader" || !rowData.parentAgentId) return;
    const generation = ++agentViewGeneration;
    agentView = {
      agentId: rowData.agentId,
      name: rowData.name,
      sessionId: currentSessionId,
      loading: true,
      error: "",
      returnScrollTop: chatRoot.scrollTop ?? 0,
      returnFollowing: following,
    };
    agentTranscript.reset();
    agentTranscript.setSessionId(currentSessionId);
    agentTranscript.setRootAgentId(rowData.agentId);
    renderAgentViewHeader();
    following = true;
    preserveChatScroll = true;
    navigate({ page: "chat", section: "" });
    showTranscript();
    chatRoot.scrollTop = chatRoot.scrollHeight;
    jumpLatest.hidden = true;
    preserveChatScroll = false;
    try {
      const reply = await boundary.agentHistory(currentSessionId, rowData.agentId);
      if (generation !== agentViewGeneration || !agentView) return;
      if (!Array.isArray(reply?.messages)) throw new Error("host returned no agent messages");
      for (const message of reply.messages) {
        agentTranscript.apply({
          kind: "event",
          payload: {
            ...message,
            type: "HistoryMessage",
            agent_id: rowData.agentId,
            session_id: currentSessionId,
          },
        });
      }
      agentView.loading = false;
      showTranscript();
    } catch (error) {
      if (generation !== agentViewGeneration || !agentView) return;
      agentView.loading = false;
      agentView.error = errorText(error);
      showTranscript();
    }
  }

  function closeAgentView() {
    if (!agentView) return;
    const previous = agentView;
    agentView = null;
    agentViewGeneration += 1;
    agentTranscript.reset();
    agentTranscript.setSessionId(null);
    renderAgentViewHeader();
    following = previous.returnFollowing;
    preserveChatScroll = true;
    showTranscript();
    chatRoot.scrollTop = previous.returnScrollTop;
    jumpLatest.hidden = following;
    preserveChatScroll = false;
  }

  async function reconcileEditCard(card) {
    card.canUndo = false;
    if (!card.checkpointKey || card.undone) return;
    try {
      const reply = await boundary.changes();
      const turn = (Array.isArray(reply?.turns) ? reply.turns : [])
        .find((entry) => entry?.key === card.checkpointKey);
      if (!Array.isArray(turn?.paths)) return;
      const paths = new Set(turn.paths.filter((path) => typeof path === "string"));
      card.files = card.files.filter((file) => paths.has(file.path));
      card.added = card.files.reduce((total, file) => total + file.added, 0);
      card.removed = card.files.reduce((total, file) => total + file.removed, 0);
      card.canUndo = card.files.length > 0 && paths.size > 0;
    } catch {
      card.canUndo = false;
    }
  }

  listen(chatRoot, "scroll", () => {
    if (route.page !== "chat") return;
    following = chatRoot.scrollHeight - chatRoot.scrollTop - chatRoot.clientHeight <= 24;
    jumpLatest.hidden = following;
  });
  listen(jumpLatest, "mousedown", (event) => event.preventDefault());
  listen(jumpLatest, "click", () => {
    following = true;
    chatRoot.scrollTop = chatRoot.scrollHeight;
    jumpLatest.hidden = true;
  });

  async function markSessionViewed(sessionId) {
    try {
      await boundary.sessionViewed(sessionId);
      await refreshSessionLists();
      showProjects();
    } catch (error) {
      sessionRowError = errorText(error);
      showProjects();
    }
  }

  let sessionReplay = null;
  let sessionOpenSerial = 0;

  async function openSession(runId) {
    pendingNewChat = null;
    if (runNotice.textContent.startsWith("Could not open conversation history")) runNotice.textContent = "";
    sessionReplay?.cancel();
    sessionReplay = null;
    const serial = ++sessionOpenSerial;
    closeAgentView();
    const targetSession = [...sessions, ...archivedSessions].find((session) => session.run_id === runId);
    if (targetSession?.repo_root && targetSession.repo_root !== project.repo_root) {
      return openProjectAt(targetSession.repo_root, runId);
    }
    saveDraft();
    const replay = { sessionId: runId, live: [], cancelled: false, cancel() { this.cancelled = true; } };
    sessionReplay = replay;
    const root = project.repo_root;
    const current = () => serial === sessionOpenSerial && currentSessionId === runId && project.repo_root === root;
    transcript.reset();
    currentSessionId = runId;
    void refreshOpenNotes();
    transcript.setSessionId(currentSessionId);
    restoreDraft();
    showProjects();
    showTranscript();
    clearConversationAgents();
    try {
      let page = await boundary.openSession(runId);
      if (!current()) return;
      const replayId = page?.replay_id, expected = page?.history_frames;
      if (typeof replayId !== "string" || !Number.isInteger(expected) || expected < 0) {
        throw new Error("The host did not report a complete history replay.");
      }
      const history = [];
      while (true) {
        if (!current() || replay.cancelled) return;
        if (page.replay_id !== replayId || page.history_frames !== expected || !Array.isArray(page.history) || page.offset !== history.length) {
          throw new Error("The host sent an invalid history page.");
        }
        history.push(...page.history);
        if (page.next_offset === null) break;
        if (page.next_offset !== history.length || page.history.length === 0 || history.length >= expected) {
          throw new Error("The host sent an invalid history offset.");
        }
        page = await boundary.historyPage(replayId, page.next_offset);
      }
      if (history.length !== expected) throw new Error("The host sent an incomplete history replay.");
      const stats = await boundary.conversationStats();
      if (!current()) return;
      conversation = stats?.conversation ?? null;
      bindTranscriptLeader();
      for (const frame of history) {
        if (!current()) return;
        await onFrame(frame, true, true);
      }
      const live = [];
      for (const frame of replay.live) {
        if (!current()) return;
        if (["SessionStarted", "SessionEnded"].includes(frame.payload?.type)) await onFrame(frame, true, true);
        else live.push(frame);
      }
      if (!current()) return;
      sessionReplay = null;
      following = true;
      navigate({ page: "chat", section: "" });
      preserveChatScroll = true;
      showAgents();
      showTranscript();
      chatRoot.scrollTop = chatRoot.scrollHeight;
      jumpLatest.hidden = true;
      preserveChatScroll = false;

      const cards = transcript.model.filter(entry => entry.type === "edit-card" && entry.checkpointKey && !entry.undone);
      if (cards.length) {
        changesDirty = true;
        const afterPaint = global.requestAnimationFrame ? callback => global.requestAnimationFrame(callback) : queueMicrotask;
        afterPaint(() => afterPaint(async () => {
          if (!current()) return;
          try {
            const changes = await boundary.changes();
            if (!current()) return;
            let changed = false;
            for (const card of cards) {
              if (!transcript.model.includes(card) || card.undone) continue;
              const turn = (Array.isArray(changes?.turns) ? changes.turns : [])
                .find(turn => turn?.key === card.checkpointKey);
              if (!Array.isArray(turn?.paths)) continue;
              const paths = new Set(turn.paths.filter(path => typeof path === "string"));
              const files = card.files.filter(file => paths.has(file.path));
              const canUndo = files.length > 0 && paths.size > 0;
              changed ||= canUndo !== card.canUndo;
              card.canUndo = canUndo;
            }
            if (changed) showTranscript();
          } catch { /* Replayed cards stay non-undoable if Changes is unavailable. */ }
        }));
      }
      for (const frame of live) {
        if (!current()) return;
        await onFrame(frame, true);
      }
      await refreshPreviewMarks();
      if (!current()) return;
      showInstructionWarnings(conversation);
      if (permittedModes.includes(conversation?.mode)) currentMode = conversation.mode;
      showConversationUsage();
      showAgents();
      await markSessionViewed(runId);
      if (current()) await refreshSelectedChanges();
    } catch (error) {
      if (!current()) return;
      sessionReplay = null;
      try {
        const stats = await boundary.conversationStats();
        if (serial !== sessionOpenSerial || project.repo_root !== root) return;
        conversation = stats?.conversation ?? null;
        currentSessionId = conversation?.session_id ?? null;
      } catch { /* Keep the attempted session selected while the host is unreachable. */ }
      transcript.reset();
      transcript.setSessionId(currentSessionId);
      restoreDraft();
      bindTranscriptLeader();
      if (permittedModes.includes(conversation?.mode)) currentMode = conversation.mode;
      navigate({ page: "chat", section: "" });
      showProjects();
      showTranscript();
      showConversationUsage();
      showAgents();
      runNotice.textContent = `Could not open conversation history: ${errorText(error)} `;
      append(runNotice, makeButton(document, "Retry", () => openSession(runId)));
      try { await approvals.resync(); showApprovals(); } catch { /* Reconnect will retry. */ }
    } finally {
      replay.cancel();
      if (sessionReplay === replay) sessionReplay = null;
    }
  }

  function showProjects() {
    const groups = [];
    let nextRename = null;
    const openFolder = makeButton(document, "Open folder…", showFolderPicker, { className: "project-open-folder" });
    groups.push(openFolder);
    projectActionError && groups.push(element(document, "p", { className: "project-action-error", text: projectActionError }));
    for (const group of projectGroups(sessions, project.repo_root)) {
      const current = group.root === project.repo_root;
      if (projectless && !group.root) continue;
      const section = element(document, "details", {
        className: `project ${current ? "openable" : "unavailable"}`,
      });
      section.open = current;
      const heading = element(document, "summary", {
        text: current ? project.name : projectName(group.root),
      });
      append(section, heading);
      if (group.root && !current) {
        const openProject = makeButton(document, "Open project", (event) => {
          event.preventDefault();
          event.stopPropagation();
          openProjectAt(group.root);
        }, { className: "project-open-action" });
        append(heading, openProject);
      }
      if (!current && !group.root) {
        append(section, element(document, "p", {
          className: "project-status",
          text: "Not openable from this host.",
        }));
      }
      if (current && group.sessions.length === 0) {
        append(section, element(document, "p", {
          className: "project-empty",
          text: "No chats yet.",
        }));
      }
      for (const session of group.sessions) {
        const parent = sessions.find((item) => item.run_id === session.parent_session_id);
        const label = (session.title || session.run_id) + (
          session.parent_session_id ? ` · fork of ${parent?.title || session.parent_session_id}` : ""
        ) + (session.activity === "working" ? " · working" : "");
        if (!current && !group.root) {
          const unavailable = element(document, "p", {
            className: "session-link unavailable",
            text: label,
          });
          unavailable.setAttribute("title", label);
          append(section, unavailable);
          continue;
        }
        const row = element(document, "div", { className: "session-row" });
        const button = makeButton(document, label, null, { className: "session-link" });
        if (session.activity === "waiting" || session.unread) {
          const waiting = session.activity === "waiting";
          const dot = element(document, "span", { className: `session-dot ${waiting ? "waiting" : "unread"}` });
          dot.setAttribute("aria-label", waiting ? "Waiting" : "Unread");
          replace(button, dot, document.createTextNode(label));
          if (waiting) append(button, element(document, "span", { className: "session-waiting", text: " waiting" }));
        }
        if (session.run_id === currentSessionId) {
          button.setAttribute("aria-current", "true");
        }
        button.setAttribute("title", label);

        listen(button, "click", async () => openSession(session.run_id));
        append(row, button);
        if (editingSessionId === session.run_id) {
          const field = element(document, "input", { className: "session-rename-input" });
          if (focusSessionRename) nextRename = field;
          field.setAttribute("aria-label", "Conversation name");
          field.value = editingSessionTitle;
          listen(field, "input", () => { editingSessionTitle = field.value; });
          listen(field, "keydown", async (event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              editingSessionId = null;
              sessionRowError = "";
              showProjects();
            } else if (event.key === "Enter") {
              event.preventDefault();
              await saveSessionTitle(session.run_id, field.value);
            }
          });
          listen(field, "blur", () => {
            if (editingSessionId === session.run_id) void saveSessionTitle(session.run_id, field.value);
          });
          append(row, field);
        } else {
          const rename = makeButton(document, "", null, { className: "session-row-action icon-pencil" });
          rename.setAttribute("aria-label", "Rename");
          rename.setAttribute("title", "Rename");
          const remove = makeButton(document, "", null, { className: "session-row-action icon-trash" });
          remove.setAttribute("aria-label", "Delete");
          remove.setAttribute("title", "Delete");
          const archive = makeButton(document, "Archive", null, { className: "session-row-action icon-archive" });
          archive.setAttribute("aria-label", "Archive");
          archive.setAttribute("title", "Archive");
          listen(archive, "click", () => updateSessionArchive(session.run_id, true));
          listen(rename, "click", () => {
            editingSessionId = session.run_id;
            editingSessionTitle = session.title || "";
            focusSessionRename = true;
            confirmingSessionId = null;
            sessionRowError = "";
            showProjects();
          });
          listen(remove, "click", () => {
            confirmingSessionId = session.run_id;
            editingSessionId = null;
            sessionRowError = "";
            showProjects();
          });
          append(row, rename, archive, remove);
          if (confirmingSessionId === session.run_id) {
            const confirmation = element(document, "div", { className: "session-delete-confirmation" });
            append(confirmation, element(document, "span", { text: "Delete this conversation? This cannot be undone." }));
            const confirm = makeButton(document, "Delete");
            const cancel = makeButton(document, "Cancel");
            listen(confirm, "click", () => deleteSession(session.run_id));
            listen(cancel, "click", () => {
              confirmingSessionId = null;
              sessionRowError = "";
              showProjects();
            });
            append(confirmation, confirm, cancel);
            append(row, confirmation);
          }
        }
        if (sessionRowError && (editingSessionId === session.run_id || confirmingSessionId === session.run_id)) {
          append(row, element(document, "p", { className: "session-row-error", text: sessionRowError }));
        }
        append(section, row);
      }
      if (current) {
        const projectArchived = archivedSessions.filter((session) => session.repo_root === group.root);
        if (projectArchived.length > 0) {
          const archived = element(document, "details", { className: "archived-sessions" });
          append(archived, element(document, "summary", { text: `Archived (${projectArchived.length})` }));
          for (const session of projectArchived) {
            const row = element(document, "div", { className: "session-row archived-session-row" });
            const link = makeButton(document, session.title || session.run_id, null, { className: "session-link" });
            link.setAttribute("title", session.title || session.run_id);
            if (session.run_id === currentSessionId) link.setAttribute("aria-current", "true");
            listen(link, "click", () => openSession(session.run_id));
            const unarchive = makeButton(document, "Unarchive", null, { className: "session-row-action icon-archive" });
            unarchive.setAttribute("aria-label", "Unarchive");
            unarchive.setAttribute("title", "Unarchive");
            listen(unarchive, "click", () => updateSessionArchive(session.run_id, false));
            const remove = makeButton(document, "", null, { className: "session-row-action icon-trash" });
            remove.setAttribute("aria-label", "Delete");
            remove.setAttribute("title", "Delete");
            listen(remove, "click", () => {
              confirmingSessionId = session.run_id;
              sessionRowError = "";
              showProjects();
            });
            append(row, link, unarchive, remove);
            if (confirmingSessionId === session.run_id) {
              const confirmation = element(document, "div", { className: "session-delete-confirmation" });
              append(confirmation, element(document, "span", { text: "Delete this conversation? This cannot be undone." }));
              const confirm = makeButton(document, "Delete");
              const cancel = makeButton(document, "Cancel");
              listen(confirm, "click", () => deleteSession(session.run_id));
              listen(cancel, "click", () => {
                confirmingSessionId = null;
                sessionRowError = "";
                showProjects();
              });
              append(confirmation, confirm, cancel);
              append(row, confirmation);
            }
            if (sessionRowError && confirmingSessionId === session.run_id) {
              append(row, element(document, "p", { className: "session-row-error", text: sessionRowError }));
            }
            append(archived, row);
          }
          append(section, archived);
        }
        if (sessionArchiveError) {
          append(section, element(document, "p", { className: "session-row-error", text: sessionArchiveError }));
        }
      }
      groups.push(section);
    }
    replace(projectsRoot, ...groups);
    if (nextRename) {
      focusSessionRename = false;
      nextRename.focus();
      nextRename.setSelectionRange?.(0, nextRename.value.length);
    }
  }
  async function saveSessionTitle(runId, title) {
    const session = [...sessions, ...archivedSessions].find((item) => item.run_id === runId);
    const nextTitle = title.trim();
    if (!nextTitle || nextTitle === (session?.title || "")) {
      editingSessionId = null;
      sessionRowError = "";
      showProjects();
      return;
    }
    try {
      await boundary.renameSession(runId, nextTitle);
      editingSessionId = null;
      sessionRowError = "";
      await refreshSessionLists();
    } catch (error) {
      sessionRowError = errorText(error);
    }
    showProjects();
  }
  async function deleteSession(runId) {
    try {
      await boundary.deleteSession(runId);
      sessions = sessions.filter((session) => session.run_id !== runId);
      archivedSessions = archivedSessions.filter((session) => session.run_id !== runId);
      confirmingSessionId = null;
      sessionRowError = "";
      if (currentSessionId === runId) {
        closeAgentView();
        transcript.reset();
        currentSessionId = null;
        void refreshOpenNotes();
        transcript.setSessionId(null);
        clearConversationAgents();
        currentMode = launchMode;
        rememberedPlanMode = null;
        await refreshConversation();
      }
      showProjects();
      showTranscript();
    } catch (error) {
      sessionRowError = errorText(error);
      showProjects();
    }
  }
  async function updateSessionArchive(runId, archived) {
    try {
      await boundary.archiveSession(runId, archived);
      await refreshSessionLists();
      sessionRowError = "";
      sessionArchiveError = "";
      confirmingSessionId = null;
    } catch (error) {
      sessionArchiveError = errorText(error);
    }
    showProjects();
  }
  showProjects();
  async function startNewChat() {
    saveDraft();
    closeAgentView();
    clearConversationAgents();
    try {
      await boundary.newSession();
      transcript.reset();
      currentSessionId = null;
      await refreshPreviewMarks();
      void refreshOpenNotes();
      transcript.setSessionId(null);
      restoreDraft();
      showProjects();
      showTranscript();
      conversation = null;
      currentMode = launchMode;
      rememberedPlanMode = null;
      showConversationUsage();
      showAgents();
      await refreshSelectedChanges();
      promptFailure = "";
      showPromptError();
      navigate({ page: "chat", section: "" });
    } catch (error) {
      await refreshConversation();
      promptFailure = errorText(error);
      showPromptError();
    }
  }
  listen(newChat, "click", startNewChat);
  if (projectless) {
    newChat.hidden = true;
    pageLinks.hidden = true;
    settingsLink.hidden = true;
    replace(sidebar, homeLink, projectsRoot);
  } else {
    replace(sidebar, homeLink, newChat, pageLinks, projectsRoot, settingsLink);
  }

  function setToggleLabel(button, label, folded) {
    const text = `${folded ? "Show" : "Hide"} ${label}`;
    button.textContent = "";
    button.setAttribute("aria-label", text);
    button.setAttribute("title", text);
  }

  function toggleFold(className, button, label) {
    const classes = shell.className.split(" ");
    const folded = classes.includes(className);
    shell.className = folded
      ? classes.filter((name) => name !== className).join(" ")
      : [...classes, className].join(" ");
    setToggleLabel(button, label, !folded);
  }

  listen(sidebarToggle, "click", () => {
    toggleFold("sidebar-folded", sidebarToggle, "sidebar");
  });
  listen(railToggle, "click", () => toggleFold("rail-folded", railToggle, "status"));

  const fragment = typeof global.location?.hash === "string" ? global.location.hash : "";
  if (projectless) renderProjectStart();
  else showPage(fragment ? parseRoute(fragment) : storedRoute(global));
  runNotice.textContent = healthReply?.state === "active"
    ? "A run started before this page was opened is still in progress."
    : "";
  if (typeof global.addEventListener === "function") {
    global.addEventListener("hashchange", () => {
      navigate(parseRoute(global.location?.hash ?? ""), { updateFragment: false });
    });
  }

  function showPromptError() {
    const busy = turn.state === RUNNING || turn.state === DISPATCHING || userCommandPending;
    promptAction.textContent = busy ? "Stop" : "Send";
    promptAction.title = busy ? "Stop the running turn" : "Send message";
    promptAction.setAttribute("aria-label", promptAction.textContent);
    stopStatus.hidden = !busy || (route?.page === "chat" && !form.hidden);
    promptError.textContent = promptFailure;
    if (promptFailure === "Add an API key in Settings before sending a message.") {
      append(promptError, element(document, "span", { text: " " }), modelsLink());
    }
  }

  function modelsLink() {
    const link = element(document, "a", { text: "Settings › Models" });
    link.href = formatRoute({ page: "settings", section: "models" });
    listen(link, "click", (event) => {
      event.preventDefault();
      navigate({ page: "settings", section: "models" });
    });
    return link;
  }

  function updateMissingKeyNotice() {
    if (!missingKeyNotice) return;
    missingKeyNotice.hidden = route.page !== "chat"
      || Boolean(agentView)
      || transcript.model.length > 0
      || providerRows.some((row) => row.key_present);
  }

  function rememberPrompt(text) {
    if (!text) return;
    if (pagePrompts[0] !== text) pagePrompts.unshift(text);
    if (historyPrompts !== null && historyPrompts[0] !== text) historyPrompts.unshift(text);
    historyIndex = -1;
    recalledPrompt = null;
    historyDraft = "";
  }

  async function loadPromptHistory() {
    if (historyPrompts !== null) return true;
    try {
      const reply = await boundary.history(100);
      const combined = [];
      for (const prompt of [...pagePrompts, ...(Array.isArray(reply?.prompts) ? reply.prompts : [])]) {
        if (typeof prompt === "string" && prompt !== "" && combined.at(-1) !== prompt) {
          combined.push(prompt);
        }
      }
      historyPrompts = combined;
      return true;
    } catch {
      return false;
    }
  }

  function showHistoryPrompt(prompt) {
    input.value = prompt;
    recalledPrompt = prompt;
    if (typeof input.setSelectionRange === "function") {
      input.setSelectionRange(prompt.length, prompt.length);
    }
  }

  async function handlePromptHistory(event) {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return false;
    if (historyIndex >= 0 && input.value !== recalledPrompt) {
      historyIndex = -1;
      recalledPrompt = null;
      return false;
    }
    if (event.key === "ArrowUp") {
      if (historyIndex === -1 && input.value !== "") return false;
      if (!(await loadPromptHistory()) || !historyPrompts?.length) return false;
      if (historyIndex + 1 >= historyPrompts.length) return false;
      if (historyIndex === -1) historyDraft = input.value;
      historyIndex += 1;
      event.preventDefault();
      showHistoryPrompt(historyPrompts[historyIndex]);
      return true;
    }
    if (historyIndex === -1) return false;
    event.preventDefault();
    if (historyIndex === 0) {
      historyIndex = -1;
      recalledPrompt = null;
      input.value = historyDraft;
    } else {
      historyIndex -= 1;
      showHistoryPrompt(historyPrompts[historyIndex]);
    }
    return true;
  }

  function updateHistorySearch() {
    const query = historySearchInput.value.toLocaleLowerCase();
    historySearchMatches = (historyPrompts ?? []).filter((prompt) => (
      prompt.toLocaleLowerCase().includes(query)
    ));
    historySearchIndex = 0;
    showHistorySearchMatch();
  }

  function showHistorySearchMatch() {
    const match = historySearchMatches[historySearchIndex];
    historySearchResult.textContent = match ?? (historySearchInput.value ? "no match" : "");
    historySearchCount.textContent = match
      ? `${historySearchIndex + 1} of ${historySearchMatches.length}`
      : "";
  }

  async function openHistorySearch() {
    historySearchDraft = input.value;
    if (!(await loadPromptHistory())) historyPrompts = [];
    historySearchBar.hidden = false;
    historySearchInput.value = "";
    updateHistorySearch();
    historySearchInput.focus();
  }

  function closeHistorySearch(restoreDraft) {
    historySearchBar.hidden = true;
    historySearchInput.value = "";
    historySearchMatches = [];
    if (restoreDraft) input.value = historySearchDraft;
    historySearchCount.textContent = "";
    input.focus();
  }

  function stepHistorySearch() {
    if (historySearchMatches.length === 0) return;
    historySearchIndex = (historySearchIndex + 1) % historySearchMatches.length;
    showHistorySearchMatch();
  }

  listen(historySearchInput, "input", updateHistorySearch);
  listen(historySearchInput, "keydown", (event) => {
    if (lookup(keymap, event, { platform }) === "history-search") {
      event.preventDefault();
      stepHistorySearch();
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (historySearchMatches.length > 0) {
        input.value = historySearchMatches[historySearchIndex];
        queueDraftSave();
        closeHistorySearch(false);
      }
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeHistorySearch(true);
    }
  });

  function selectPreviousMessageForEdit() {
    editingPreviousPrompt = [...transcript.model].reverse()
      .find((entry) => entry.type === "prompt") ?? null;
    if (!editingPreviousPrompt) return false;
    input.value = editingPreviousPrompt.text;
    queueDraftSave();
    showTranscript();
    input.focus();
    return true;
  }

  function cancelPreviousMessageEdit() {
    editingPreviousPrompt = null;
    editConflictPanel.hidden = true;
    replace(editConflictPanel);
    input.value = "";
    queueDraftSave();
    showTranscript();
    input.focus();
  }

  function showEditConflict(paths, text, recordId) {
    const force = makeButton(document, "Fork anyway", () => forkEditedPrompt(text, recordId, true));
    const cancel = makeButton(document, "Cancel edit", cancelPreviousMessageEdit);
    const list = element(document, "ul");
    for (const path of paths) append(list, element(document, "li", { text: path }));
    replace(
      editConflictPanel,
      element(document, "p", { text: "These files changed outside the agent:" }),
      list,
      force,
      cancel,
    );
    editConflictPanel.hidden = false;
  }

  async function forkEditedPrompt(text, recordId, force = false) {
    try {
      const reply = await boundary.forkSession(currentSessionId, recordId, force);
      if (typeof reply?.run_id !== "string") throw new Error("The host did not return the forked conversation.");
      editingPreviousPrompt = null;
      editConflictPanel.hidden = true;
      replace(editConflictPanel);
      await openSession(reply.run_id);
      input.value = text;
      queueDraftSave();
      return form.requestSubmit();
    } catch (error) {
      if (error?.status === 409 && Array.isArray(error.paths) && error.paths.length > 0) {
        showEditConflict(error.paths, text, recordId);
      } else {
        promptFailure = errorText(error);
        showPromptError();
      }
    }
  }

  async function submitEditedPreviousPrompt(text) {
    const promptIndex = transcript.model.indexOf(editingPreviousPrompt);
    const earlierPrompts = transcript.model
      .slice(0, promptIndex)
      .filter((entry) => entry.type === "prompt");
    if (earlierPrompts.length === 0) {
      editingPreviousPrompt = null;
      editConflictPanel.hidden = true;
      showTranscript();
      await startNewChat();
      input.value = text;
      return form.requestSubmit();
    }
    const reply = transcript.model.slice(0, promptIndex).reverse()
      .find((entry) => entry.type === "text" && typeof entry.recordId === "string");
    if (!reply || !currentSessionId) {
      promptFailure = "Could not find the earlier reply needed to fork this conversation.";
      showPromptError();
      return;
    }
    return forkEditedPrompt(text, reply.recordId);
  }

  async function refreshPreviewMarks() {
    if (projectless || !boundary.previewMarks) return;
    if (previewMarkSession !== currentSessionId) {
      previewMarks.splice(0);
      showPendingAttachments();
    }
    const sequence = ++previewMarksSequence;
    try {
      const reply = await boundary.previewMarks();
      if (sequence !== previewMarksSequence || reply.session_id !== currentSessionId) return;
      previewMarkSession = reply.session_id;
      previewMarks.splice(0, previewMarks.length, ...(reply.marks ?? []));
      showPendingAttachments();
    } catch (error) {
      promptFailure = `Could not load preview marks: ${errorText(error)}`;
      showPromptError();
    }
  }

  async function openPreview(args) {
    if (args[0] === "stop") {
      await boundary.previewStop();
      answerCommand("Preview stopped.");
      return;
    }
    const window = global.open?.("", "_blank", "popup,width=1100,height=800");
    if (!window) {
      answerCommand("Allow pop-up windows to open the preview.");
      return;
    }
    window.opener = null;
    try {
      const reply = await boundary.previewStart(args[0] ?? "");
      window.location.href = reply.preview_url;
    } catch (error) {
      window.close();
      answerCommand(errorText(error));
    }
  }

  function withPreviewMarks(text, marks) {
    return [text, ...marks.map((mark, index) => {
      const fence = "`".repeat(Math.max(3, ...[...mark.html.matchAll(/`+/g)].map((match) => match[0].length + 1)));
      return `## Preview mark ${index + 1}\nComment: ${mark.comment}\nURL: ${mark.url}\nSelector: ${mark.selector}\nSource${mark.source_guess ? " (guess)" : ""}: ${mark.source_hint || "unknown"}\n${fence}html\n${mark.html}\n${fence}`;
    })].filter(Boolean).join("\n\n");
  }

  function showPendingAttachments() {
    const files = pendingFiles.map((file, index) => {
      const chip = element(document, "span", {
        className: "attachment-chip",
        text: `${file.name || "image"} · ${file.size}`,
      });
      const remove = makeButton(document, "Remove", () => {
        pendingFiles.splice(index, 1);
        showPendingAttachments();
      });
      return append(chip, remove);
    });
    const pastes = pastedTextChips.map((paste) => {
      const chip = element(document, "span", {
        className: "attachment-chip pasted-text-chip",
      });
      const label = element(document, "span", {
        text: `Pasted text #${paste.id} · ${paste.lines} lines`,
      });
      const expand = makeButton(document, paste.expanded ? "Hide" : "Expand", () => {
        paste.expanded = !paste.expanded;
        showPendingAttachments();
      });
      const remove = makeButton(document, "Remove", () => {
        input.value = input.value.replace(paste.placeholder, "");
        pastedTextChips.splice(pastedTextChips.indexOf(paste), 1);
        showPendingAttachments();
        queueDraftSave();
      });
      append(chip, label, expand, remove);
      if (paste.expanded) {
        append(chip, element(document, "pre", {
          className: "pasted-text-preview",
          text: paste.text,
        }));
      }
      return chip;
    });
    const marks = previewMarks.map((mark) => {
      const chip = element(document, "span", { className: "attachment-chip preview-mark-chip", text:
        `${mark.text?.slice(0, 40) || mark.selector} · ${mark.comment}` });
      append(chip, makeButton(document, "Remove", async () => {
        try {
          await boundary.removePreviewMarks([mark.id], previewMarkSession);
          await refreshPreviewMarks();
        } catch (error) {
          promptFailure = errorText(error);
          showPromptError();
        }
      }));
      return chip;
    });
    replace(attachmentList, ...files, ...pastes, ...marks);
  }

  function expandPastedText(value) {
    let result = "";
    let cursor = 0;
    while (cursor < value.length) {
      let next = null;
      for (const paste of pastedTextChips) {
        const index = value.indexOf(paste.placeholder, cursor);
        if (index !== -1 && (next === null || index < next.index)) next = { paste, index };
      }
      if (next === null) {
        result += value.slice(cursor);
        break;
      }
      result += value.slice(cursor, next.index) + next.paste.text;
      cursor = next.index + next.paste.placeholder.length;
    }
    return result;
  }

  function addFiles(files) {
    const allowed = new Set(["image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"]);
    for (const file of files) {
      if (pendingFiles.length >= 10) {
        promptFailure = "You can attach up to 10 files.";
        break;
      }
      if (file.size > 5_000_000 && !["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
        promptFailure = `${file.name || "File"} is over the 5,000,000 byte limit.`;
        continue;
      }
      if (!allowed.has(file.type)) {
        promptFailure = `${file.name || "File"} must be a PNG, JPEG, GIF, WEBP or PDF.`;
        continue;
      }
      pendingFiles.push(file);
      promptFailure = "";
    }
    showPendingAttachments();
    showPromptError();
  }

  async function encodePendingFiles() {
    return Promise.all(pendingFiles.map(async (file) => {
      let source = file;
      if (["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
        let bitmap;
        try {
          bitmap = await (global.createImageBitmap ?? globalThis.createImageBitmap)(file);
          const longest = Math.max(bitmap.width, bitmap.height);
          if (longest > 1568) {
            const canvas = document.createElement("canvas");
            const scale = 1568 / longest;
            canvas.width = Math.max(1, Math.round(bitmap.width * scale));
            canvas.height = Math.max(1, Math.round(bitmap.height * scale));
            canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            const resized = await new Promise((resolve) => canvas.toBlob(resolve, file.type, 0.9));
            if (resized && (resized.type === file.type || (file.type === "image/webp" && resized.type === "image/png"))) {
              source = resized;
            }
          }
        } catch {
          // Undecodable images retain their original bytes.
        } finally {
          bitmap?.close?.();
        }
      }
      if (source.size > 5_000_000) {
        throw new Error(`${file.name || "File"} is over the 5,000,000 byte limit.`);
      }
      const bytes = new Uint8Array(await source.arrayBuffer());
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }
      return {
        data: (global.btoa ?? globalThis.btoa)(binary),
        filename: file.name || undefined,
        kind: file.type === "application/pdf" ? "document" : "image",
      };
    }));
  }

  async function refreshRoadmap() {
    const reply = await readRoadmap();
    if (reply.error) {
      roadmapError = reply.error;
    } else {
      roadmapGoals = Array.isArray(reply.goals) ? reply.goals : [];
      roadmapError = "";
    }
    renderRoadmapUI();
  }

  function renderRoadmapUI() {
    const list = element(document, "ol", { className: "roadmap-goals" });
    for (const goal of roadmapGoals) {
      const row = element(document, "li", { className: "roadmap-goal" });
      const detail = element(document, "div", { className: "roadmap-goal-detail" });
      if (typeof goal.loop_id === "string" && goal.loop_id) {
        const link = makeButton(document, goal.goal, null, { className: "roadmap-goal-link" });
        link.setAttribute("title", "Open loop");
        listen(link, "click", () => {
          selectedLoopId = goal.loop_id;
          void openRailTool("Loop");
        });
        append(detail, link);
      } else {
        append(detail, element(document, "span", { className: "roadmap-goal-text", text: goal.goal }));
      }
      append(detail, element(document, "span", { className: "roadmap-goal-state", text: stateLabel(goal.state) }));
      append(row, detail);
      append(list, row);
    }
    const children = [element(document, "h2", { text: "Roadmap" })];
    if (roadmapError) children.push(element(document, "p", { className: "error", text: roadmapError }));
    if (roadmapGoals.length === 0 && !roadmapError) {
      children.push(element(document, "p", { className: "roadmap-empty", text: "No goals yet. Ask the leader to add one." }));
    }
    children.push(list);
    replace(roadmapRoot, ...children);
  }

  renderRoadmapUI();
  const fileListings = new Map();
  const expandedFileDirectories = new Set();
  const fileTreeErrors = new Map();
  let filesDirty = false;
  let selectedFilePath = null;
  let fileSourceText = null;
  let fileSourceError = null;
  let fileSourceLoading = false;

  function renderFiles() {
    if (selectedFilePath !== null) {
      const back = makeButton(document, "Back", () => {
        selectedFilePath = null;
        fileSourceText = null;
        fileSourceError = null;
        fileSourceLoading = false;
        renderFiles();
      }, { className: "files-back" });
      const header = element(document, "div", { className: "files-viewer-header" });
      append(header, back, element(document, "strong", {
        className: "files-viewer-path",
        text: selectedFilePath,
      }));
      const editorLink = editorFileLink(selectedFilePath, 1);
      if (editorLink) append(header, " ", editorLink);
      const children = [header];
      if (fileSourceLoading) {
        children.push(element(document, "p", { text: "Loading…" }));
      } else if (fileSourceError) {
        const message = {
          404: "Not found",
          413: "Too large to show (over 1 MiB)",
          415: "Not a text file",
        }[fileSourceError.status] ?? "Could not read file";
        children.push(element(document, "p", { className: "files-source-error", text: message }));
      } else if (typeof fileSourceText === "string") {
        const lines = fileSourceText.split("\n");
        const width = String(lines.length).length;
        const source = element(document, "div", { className: "files-source" });
        append(
          source,
          element(document, "pre", {
            className: "files-line-numbers",
            attributes: { "aria-hidden": "true" },
            text: lines.map((_, index) => String(index + 1).padStart(width, " ")).join("\n"),
          }),
          element(document, "pre", { className: "files-code", text: fileSourceText }),
        );
        children.push(source);
      }
      replace(filesRoot, ...children);
      return;
    }

    const rows = [];
    function appendDirectory(path, depth) {
      const listing = fileListings.get(path);
      if (!listing) return;
      for (const entry of listing.entries ?? []) {
        const childPath = path ? `${path}/${entry.name}` : entry.name;
        const isDirectory = entry.kind === "dir";
        const row = makeButton(document, "", null, { className: `files-tree-row${isDirectory ? " files-directory" : " files-file"}` });
        row.setAttribute("style", `padding-left: calc(var(--space-4) * ${depth})`);
        if (isDirectory) {
          const expanded = expandedFileDirectories.has(childPath);
          append(row,
            element(document, "span", { className: "files-chevron", text: expanded ? "▾" : "▸" }),
            element(document, "span", { className: "files-name", text: entry.name }),
          );
          row.setAttribute("aria-expanded", String(expanded));
          listen(row, "click", async () => {
            if (expandedFileDirectories.has(childPath)) {
              expandedFileDirectories.delete(childPath);
              renderFiles();
              return;
            }
            expandedFileDirectories.add(childPath);
            renderFiles();
            await loadFileDirectory(childPath);
          });
        } else {
          append(row,
            element(document, "span", { className: "files-chevron" }),
            element(document, "span", { className: "files-name", text: entry.name }),
          );
          listen(row, "click", () => openSourceFile(childPath));
        }
        rows.push(row);
        if (isDirectory && expandedFileDirectories.has(childPath)) {
          appendDirectory(childPath, depth + 1);
          if (fileTreeErrors.has(childPath)) {
            rows.push(element(document, "p", {
              className: "files-source-error",
              text: "Could not load directory",
            }));
          }
        }
      }
      if (listing.truncated) {
        const more = element(document, "p", { className: "files-tree-more", text: "More files not shown" });
        more.setAttribute("style", `padding-left: calc(var(--space-4) * ${depth})`);
        rows.push(more);
      }
    }
    appendDirectory("", 0);
    if (fileTreeErrors.has("")) rows.push(element(document, "p", {
      className: "files-source-error", text: "Could not load directory",
    }));
    if (!fileListings.has("") && !fileTreeErrors.has("")) {
      rows.push(element(document, "p", { text: "Loading…" }));
    }
    replace(filesRoot, ...rows);
  }

  async function loadFileDirectory(path) {
    try {
      fileListings.set(path, await boundary.tree(path));
      fileTreeErrors.delete(path);
    } catch {
      fileTreeErrors.set(path, true);
    }
    renderFiles();
  }

  async function openSourceFile(path) {
    const sequence = ++sourceSequence;
    selectedFilePath = path;
    fileSourceText = null;
    fileSourceError = null;
    fileSourceLoading = true;
    renderFiles();
    try {
      const reply = await boundary.source(path);
      if (sequence !== sourceSequence || selectedFilePath !== path) return;
      fileSourceText = reply.text;
    } catch (error) {
      if (sequence !== sourceSequence) return;
      fileSourceError = error;
    } finally {
      if (sequence !== sourceSequence) return;
      fileSourceLoading = false;
      renderFiles();
    }
  }

  async function refreshFiles() {
    filesDirty = false;
    const paths = ["", ...expandedFileDirectories];
    await Promise.all(paths.map(async (path) => {
      try {
        fileListings.set(path, await boundary.tree(path));
        fileTreeErrors.delete(path);
      } catch {
        fileTreeErrors.set(path, true);
      }
    }));
    if (selectedFilePath !== null) await openSourceFile(selectedFilePath);
    else renderFiles();
  }

  function renderLoopCardDetails() {
    const details = element(document, "section", { className: "loop-card-details" });
    const back = makeButton(document, "Back to the board", returnToLoopBoard);
    append(details, back);
    if (cardDetailLoading) {
      append(details, element(document, "p", { text: "Loading card…" }));
      return details;
    }
    if (cardDetailError) {
      append(details, element(document, "p", { className: "error", text: cardDetailError }));
      return details;
    }
    if (!cardDetail) return details;

    const card = cardDetail.card ?? selectedLoop.cards?.find((item) => item.id === selectedCardId) ?? {};
    append(details, element(document, "h3", {
      text: `${card.id ?? selectedCardId} · ${card.title ?? ""} · ${stateLabel(card.state)}`,
    }));
    const markdownSection = (title, source) => {
      const section = element(document, "section", { className: "loop-card-section" });
      append(section, element(document, "h4", { text: title }));
      append(section, ...renderMarkdownBlocks(document, typeof source === "string" ? source : "", parseMarkdown, { highlight }));
      return section;
    };
    append(details, markdownSection("Spec", cardDetail.spec));
    append(details, markdownSection("Report", cardDetail.report));

    const change = element(document, "section", { className: "loop-card-section" });
    append(change, element(document, "h4", { text: "Change" }));
    if (cardDetail.diff_truncated) {
      append(change, element(document, "p", { className: "loop-diff-truncated", text: "Diff truncated." }));
    }
    append(change, renderDiff(document, cardDetail.diff ?? ""));
    append(details, change);

    const review = cardDetail.review ?? {};
    const reviewSection = element(document, "section", { className: "loop-card-section" });
    append(reviewSection, element(document, "h4", { text: "Review" }));
    append(reviewSection, element(document, "p", {
      className: "loop-review-verdict",
      text: review.verdict ?? "No review verdict",
    }));
    append(reviewSection, ...renderMarkdownBlocks(document, typeof review.text === "string" ? review.text : "", parseMarkdown, { highlight }));
    const followUps = Array.isArray(review.follow_ups) ? review.follow_ups : [];
    if (followUps.length > 0) {
      const list = element(document, "ul", { className: "loop-follow-ups" });
      for (const followUp of followUps) {
        const id = typeof followUp === "string" ? followUp : followUp?.id;
        if (typeof id !== "string") continue;
        const button = makeButton(document, id, () => loadLoopCard(id), { className: "loop-follow-up" });
        const item = element(document, "li");
        append(item, button);
        append(list, item);
      }
      append(reviewSection, list);
    }
    append(details, reviewSection);

    if (typeof cardDetail.session_id === "string" && cardDetail.session_id) {
      const open = makeButton(document, "Open conversation", () => openSession(cardDetail.session_id));
      append(details, open);
    }
    return details;
  }

  function renderLoopCheckpoint() {
    const section = element(document, "section", { className: "loop-checkpoint" });
    const summary = selectedLoop.summary ?? {};
    if (["awaiting_commit", "done"].includes(selectedLoop.state)) {
      append(section, element(document, "h3", { text: "Commit checkpoint" }));
      const commits = Array.isArray(summary.commits) ? summary.commits : [];
      if (commits.length > 0) {
        append(section, element(document, "h4", { text: "Commits" }));
        const list = element(document, "ul", { className: "loop-commits" });
        for (const item of commits) {
          const shortId = typeof item.commit === "string" ? item.commit.slice(0, 8) : "";
          append(list, element(document, "li", { text: `${item.card} · ${shortId} · ${item.title}` }));
        }
        append(section, list);
      }
      if (Array.isArray(summary.files)) {
        append(section, element(document, "h4", { text: "Changed files" }));
        const list = element(document, "ul", { className: "loop-files" });
        for (const file of summary.files) {
          append(list, element(document, "li", { text: typeof file === "string" ? file : file.path ?? "" }));
        }
        append(section, list);
      }
      if (Array.isArray(summary.validation)) {
        append(section, element(document, "h4", { text: "Validation" }));
        for (const result of summary.validation) {
          const output = element(document, "details", { className: "loop-validation" });
          append(
            output,
            element(document, "summary", { text: `${result.command} · exit ${result.exit}` }),
            element(document, "pre", { text: result.output ?? "" }),
          );
          append(section, output);
        }
      }
    }
    if (loopCommitMessage) {
      append(section, element(document, "p", { className: "loop-commit-message", text: loopCommitMessage }));
    }
    if (selectedLoop.state === "done") {
      append(section, element(document, "p", { text: `Branch: ${selectedLoop.branch ?? ""}` }));
      append(section, element(document, "p", { text: "Merge this branch with git." }));
    }
    if (selectedLoop.state === "awaiting_commit") {
      const keep = makeButton(document, "Keep the branch", null, { className: "primary" });
      keep.disabled = loopCommitPending;
      listen(keep, "click", () => runLoopCommit("approve"));
      append(section, keep);
      if (loopDiscardConfirm) {
        append(section, element(document, "p", {
          className: "loop-discard-prompt",
          text: `Discard this work and delete ${selectedLoop.branch ?? ""}?`,
        }));
        const discard = makeButton(document, "Discard");
        discard.disabled = loopCommitPending;
        listen(discard, "click", () => runLoopCommit("decline"));
        const keep = makeButton(document, "Keep");
        keep.disabled = loopCommitPending;
        listen(keep, "click", () => {
          loopDiscardConfirm = false;
          renderLoops();
        });
        append(section, discard, keep);
      } else {
        const discard = makeButton(document, "Discard");
        discard.disabled = loopCommitPending;
        listen(discard, "click", () => {
          loopDiscardConfirm = true;
          renderLoops();
        });
        append(section, discard);
      }
    }
    if (loopCommitError) {
      append(section, element(document, "p", { className: "error loop-commit-error", text: loopCommitError }));
    }
    return section;
  }

  function returnToLoopBoard() {
    selectedCardId = "";
    cardDetail = null;
    cardDetailLoading = false;
    cardDetailError = "";
    renderLoops();
  }

  async function loadLoopCard(cardId) {
    const sequence = ++cardSequence, loopId = selectedLoopId;
    const current = () => sequence === cardSequence && loopId === selectedLoopId && cardId === selectedCardId;
    selectedCardId = cardId;
    cardDetail = null;
    cardDetailLoading = true;
    cardDetailError = "";
    renderLoops();
    try {
      const reply = await boundary.loopCard(loopId, cardId);
      if (!current()) return;
      cardDetail = reply;
    } catch (error) {
      if (!current()) return;
      cardDetailError = errorText(error);
    } finally {
      if (!current()) return;
      cardDetailLoading = false;
      renderLoops();
    }
  }

  async function runLoopCommit(action) {
    if (loopCommitPending) return;
    loopCommitPending = true;
    loopCommitError = "";
    renderLoops();
    try {
      const reply = await boundary.loopCommit(selectedLoopId, action);
      selectedLoop = reply?.loop ?? reply;
      loopDiscardConfirm = false;
      loopCommitMessage = action === "approve"
        ? `Kept on branch ${reply?.branch ?? selectedLoop.branch ?? ""}`
        : "";
    } catch (error) {
      loopCommitError = errorText(error);
    } finally {
      loopCommitPending = false;
      renderLoops();
    }
  }

  function renderLoops() {
    const list = element(document, "nav", { className: "loop-list" });
    list.setAttribute("aria-label", "Spec loops");
    const children = [element(document, "h2", { text: "Loop" }), list];
    if (loopError) {
      children.push(element(document, "p", { className: "error", text: loopError }));
    }
    const ordered = [...loops].sort((left, right) => (
      String(right.updated_at ?? "").localeCompare(String(left.updated_at ?? ""))
    ));
    if (ordered.length === 0 && !loopError) {
      children.push(element(document, "p", { text: "No spec loops yet." }));
    }
    for (const loop of ordered) {
      const button = makeButton(document, `${loop.goal ?? loop.id} · ${stateLabel(loop.state)}`, null, { className: "loop-list-item" });
      button.setAttribute("aria-selected", String(loop.id === selectedLoopId));
      listen(button, "click", () => loadLoop(loop.id));
      append(list, button);
    }
    if (selectedLoop) {
      if (selectedCardId) {
        children.push(renderLoopCardDetails());
      } else {
      const detail = element(document, "section", { className: "loop-detail" });
      append(detail, element(document, "h3", { text: selectedLoop.goal ?? "" }));
      append(detail, element(document, "p", {
        className: "loop-state",
        text: `State: ${stateLabel(selectedLoop.state)}${selectedLoop.reason ? ` · ${selectedLoop.reason}` : ""}`,
      }));
      const lanes = element(document, "div", { className: "loop-lanes" });
      const cardsById = new Map((selectedLoop.cards ?? []).map((card) => [card.id, card]));
      for (const lane of selectedLoop.lanes ?? []) {
        const column = element(document, "section", { className: "loop-lane" });
        append(column, element(document, "h4", { text: `Lane ${lane.id}` }));
        const cards = (lane.cards ?? []).map((id) => cardsById.get(id)).filter(Boolean);
        for (const card of cards) {
          const cardElement = element(document, "article", {
            className: `loop-card${card.state === "dropped" ? " loop-card-dimmed" : ""}`,
          });
          const open = makeButton(document, `${card.id} · ${card.title ?? ""} · ${stateLabel(card.state)}`, () => loadLoopCard(card.id), { className: "loop-card-open" });
          append(cardElement, open);
          if (selectedLoop.state === "awaiting_plan" && card.state !== "dropped") {
            const drop = makeButton(document, "Drop");
            drop.disabled = loopActionPending;
            listen(drop, "click", () => runLoopAction(() => boundary.loopPlan(
              selectedLoopId,
              "drop",
              { cards: [card.id] },
            )));
            append(cardElement, drop);
          }
          append(column, cardElement);
        }
        append(lanes, column);
      }
      append(detail, lanes);
      const actions = element(document, "div", { className: "loop-actions" });
      if (selectedLoop.state === "awaiting_plan") {
        const approve = makeButton(document, "Approve plan");
        approve.disabled = loopActionPending;
        listen(approve, "click", () => runLoopAction(() => boundary.loopPlan(selectedLoopId, "approve")));
        append(actions, approve);

        const revise = element(document, "div", { className: "loop-revise" });
        const note = element(document, "input", { className: "loop-revise-note" });
        note.type = "text";
        note.value = loopReviseNote;
        note.setAttribute("aria-label", "Revision note");
        note.setAttribute("placeholder", "Note for the planner");
        note.disabled = loopActionPending;
        listen(note, "input", () => { loopReviseNote = note.value; });
        const reviseButton = makeButton(document, "Revise…");
        reviseButton.disabled = loopActionPending;
        listen(reviseButton, "click", () => runLoopAction(() => boundary.loopPlan(
          selectedLoopId,
          "revise",
          { note: loopReviseNote },
        )));
        append(revise, note, reviseButton);
        append(actions, revise);
      }
      if (["planning", "awaiting_plan", "running", "validating", "needs_attention"].includes(selectedLoop.state)) {
        if (loopCancelConfirm) {
          append(actions, element(document, "p", {
            className: "loop-cancel-prompt",
            text: "Cancel this loop? Its branch and work will be removed.",
          }));
          const confirm = makeButton(document, "Cancel loop");
          confirm.disabled = loopActionPending;
          listen(confirm, "click", () => runLoopAction(() => boundary.loopCancel(selectedLoopId)));
          const keep = makeButton(document, "Keep running");
          keep.disabled = loopActionPending;
          listen(keep, "click", () => {
            loopCancelConfirm = false;
            renderLoops();
          });
          append(actions, confirm, keep);
        } else {
          const cancel = makeButton(document, "Cancel loop");
          cancel.disabled = loopActionPending;
          listen(cancel, "click", () => {
            loopCancelConfirm = true;
            renderLoops();
          });
          append(actions, cancel);
        }
      }
      if (selectedLoop.state === "needs_attention") {
        const resume = makeButton(document, "Resume");
        resume.disabled = loopActionPending;
        listen(resume, "click", () => runLoopAction(() => boundary.loopResume(selectedLoopId)));
        append(actions, resume);
      }
      if (loopActionError) {
        append(actions, element(document, "p", { className: "error loop-action-error", text: loopActionError }));
      }
      append(detail, actions);
      children.push(detail);
      children.push(renderLoopCheckpoint());
      }
    } else if (selectedLoopId && !loopError) {
      children.push(element(document, "p", { text: "Loading loop…" }));
    }
    replace(loopsRoot, ...children);
  }

  async function loadLoop(id) {
    const sequence = ++loopSequence;
    if (typeof id !== "string" || !id) return;
    if (selectedLoopId !== id) {
      loopActionError = "";
      loopReviseNote = "";
      loopCancelConfirm = false;
      selectedCardId = "";
      cardDetail = null;
      cardDetailError = "";
      cardDetailLoading = false;
      loopCommitError = "";
      loopCommitMessage = "";
      loopDiscardConfirm = false;
    }
    selectedLoopId = id;
    selectedLoop = null;
    loopError = "";
    loopCancelConfirm = false;
    renderLoops();
    try {
      const reply = await boundary.loop(id);
      if (sequence !== loopSequence) return;
      selectedLoop = reply.loop;
    } catch (error) {
      if (sequence !== loopSequence) return;
      loopError = errorText(error);
    }
    renderLoops();
  }

  async function runLoopAction(action) {
    if (loopActionPending) return;
    loopActionPending = true;
    loopActionError = "";
    renderLoops();
    try {
      const reply = await action();
      selectedLoop = reply?.loop ?? reply;
      loopCancelConfirm = false;
      loopReviseNote = "";
    } catch (error) {
      loopActionError = errorText(error);
    } finally {
      loopActionPending = false;
      renderLoops();
    }
  }

  async function loadLoops() {
    loopError = "";
    try {
      const reply = await boundary.loops();
      loops = Array.isArray(reply?.loops) ? reply.loops : [];
      if (!loops.some((loop) => loop.id === selectedLoopId)) {
        selectedLoopId = loops[0]?.id ?? "";
      }
      renderLoops();
      if (selectedLoopId) await loadLoop(selectedLoopId);
      else {
        selectedLoop = null;
        renderLoops();
      }
    } catch (error) {
      loopError = errorText(error);
      renderLoops();
    }
  }

  function renderRail() {
    const tabs = openRailTools.map((name) => {
      const selected = name === selectedRailTool;
      const wrapper = element(document, "div", { className: "rail-tab-item" });
      wrapper.setAttribute("aria-selected", String(selected));
      const tab = makeButton(document, selected ? name : "", null, { className: `rail-tab ${RAIL_TOOL_ICONS[name]}` });
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(selected));
      tab.setAttribute("aria-label", name);
      tab.setAttribute("title", name);
      listen(tab, "click", () => selectRailTool(name));
      append(wrapper, tab);
      if (selected) {
        const close = makeButton(document, "×", null, { className: "rail-tab-close" });
        close.setAttribute("aria-label", `Close ${name}`);
        close.setAttribute("title", `Close ${name}`);
        listen(close, "click", () => closeRailTool(name));
        append(wrapper, close);
      }
      return wrapper;
    });
    const add = makeButton(document, "+", null, { className: "rail-tab-add" });
    add.setAttribute("aria-label", "Open a tool");
    listen(add, "click", () => {
      showRailTools = !showRailTools;
      renderRail();
    });
    replace(railTabsRoot, ...tabs, add);

    const tools = RAIL_TOOL_NAMES.map((name) => {
      const row = makeButton(document, name, () => openRailTool(name), { className: "rail-tool" });
      return row;
    });
    replace(railToolsRoot, element(document, "h2", { text: "Tools" }), ...tools);
    const panes = { Agents: agentsRoot, Roadmap: roadmapRoot, Changes: changesRoot, Files: filesRoot, Jobs: jobsRoot, Loop: loopsRoot, Notes: notesRoot };
    for (const [name, pane] of Object.entries(panes)) {
      pane.hidden = showRailTools || name !== selectedRailTool;
    }
    renderLoops();
    renderFiles();
    railToolsRoot.hidden = !showRailTools && openRailTools.length > 0;
  }

  renderRail();

  function selectRailTool(name) {
    if (!openRailTools.includes(name)) return openRailTool(name);
    const changed = selectedRailTool !== name;
    selectedRailTool = name;
    showRailTools = false;
    if (name !== "Jobs") updateJobsPolling([]);
    rememberRail(global, openRailTools, selectedRailTool);
    renderRail();
    if (name === "Loop") return loadLoops();
    if (name === "Changes") return changed || changesDirty ? showChanges() : undefined;
    if (name === "Files") return changed || filesDirty ? refreshFiles() : undefined;
    if (name === "Notes") return showNotes();
    if (!changed) return undefined;
    if (name === "Jobs") return showJobs();
    return undefined;
  }

  function openRailTool(name) {
    if (!RAIL_TOOL_NAMES.includes(name)) return;
    if (!openRailTools.includes(name)) openRailTools = [...openRailTools, name];
    return selectRailTool(name);
  }

  function closeRailTool(name) {
    openRailTools = openRailTools.filter((tool) => tool !== name);
    if (selectedRailTool === name) selectedRailTool = openRailTools.at(-1) ?? "";
    showRailTools = openRailTools.length === 0;
    if (selectedRailTool !== "Jobs") updateJobsPolling([]);
    rememberRail(global, openRailTools, selectedRailTool);
    renderRail();
    if (selectedRailTool === "Changes") return showChanges();
    if (selectedRailTool === "Jobs") return showJobs();
    if (selectedRailTool === "Notes") return showNotes();
    return undefined;
  }

  async function refreshSelectedChanges() {
    if (selectedRailTool === "Changes") await showChanges();
  }

  if (selectedRailTool === "Changes") void showChanges();
  if (selectedRailTool === "Files") void refreshFiles();
  if (selectedRailTool === "Jobs") void showJobs();
  if (selectedRailTool === "Notes") void showNotes();

  function costText(cost) {
    if (!cost || typeof cost.amount !== "string" || typeof cost.currency !== "string") return "";
    let money = `${cost.currency} ${cost.amount}`;
    const amount = Number(cost.amount);
    const belowCent = amount > 0 && amount < 0.01;
    if (Number.isFinite(amount)) {
      try {
        money = new Intl.NumberFormat(undefined, {
          style: "currency", currency: cost.currency,
        }).format(belowCent ? 0.01 : amount);
        if (belowCent) money = `<${money}`;
      } catch { /* Keep the currency code when Intl cannot format it. */ }
    }
    return `${cost.estimated ? "~" : ""}${money}`;
  }

  function usageText(usage) {
    if (!usage || !Number.isInteger(usage.total_tokens)) {
      return "";
    }
    const tokens = `${usage.total_tokens.toLocaleString()} tokens`;
    const money = costText(usage.cost);
    return money ? `${money} · ${tokens}`
      : `${tokens} · ${usage.unpriced_model ? `no price for ${usage.unpriced_model}` : "no prices set"}`;
  }

  function showConversationUsage() {
    const context = conversation?.context;
    const provider = conversation?.provider ?? selection.provider;
    const model = conversation?.model ?? selection.model;
    const modelText = typeof model === "string" && model ? model : "";
    const activeMode = conversation?.mode ?? currentMode;
    const mode = permittedModes.includes(activeMode) ? activeMode : "";
    const providerText = typeof provider === "string" && provider ? provider : "";
    const activeEffort = conversation?.effort ?? selection.effort;
    const effort = typeof activeEffort === "string" && activeEffort ? activeEffort : "";
    const identityStatus = [
      statusFields.provider && providerText,
      statusFields.model && modelText,
    ].filter(Boolean).join(" / ");
    const primaryStatus = [identityStatus, statusFields.effort && effort].filter(Boolean).join(" ");
    const extraStatus = [
      mode && mode !== "ask" && (statusFields.mode ? `Mode ${mode}` : mode),
      mode === "ask" && statusFields.mode && `Mode ${mode}`,
      specMode && "Spec mode",
      statusFields.goal && conversation?.goal && `Goal ${conversation.goal.phase} ${conversation.goal.rounds}/${conversation.goal.max_rounds}`,
      statusFields.context && Number.isInteger(context?.used_tokens) && Number.isInteger(context?.budget_tokens)
        ? `Context ${context.used_tokens} / ${context.budget_tokens} tokens`
        : "",
      statusFields.usage && usageText(conversation?.usage),
    ].filter(Boolean);
    replace(conversationUsage, element(document, "span", {
      className: "conversation-status", text: [primaryStatus, ...extraStatus].filter(Boolean).join(" · "),
    }), terminalToggle);
    conversationUsage.className = `conversation-usage${mode === "plan" ? " plan" : ""}${specMode ? " spec-mode" : ""}`;
  }

  function waitingForApproval() {
    return [...approvals.state.values()].some((question) =>
      question.state === "open" || question.state === "answering"
    );
  }

  function updateWindowTitle() {
    if (waitingForApproval()) {
      document.title = "● Waiting for you · SymphonAI";
    } else if (turn.state === RUNNING || turn.state === DISPATCHING) {
      document.title = `Working… · ${project?.name || projectName(project?.repo_root ?? "")} · SymphonAI`;
    } else {
      document.title = project?.name
        ? `${project.name} · SymphonAI`
        : project?.repo_root ? `${projectName(project.repo_root)} · SymphonAI` : "SymphonAI";
    }
  }

  function conversationTitle(sessionId) {
    return sessions.find((session) => session.run_id === sessionId)?.title
      || (sessionId === currentSessionId && conversation?.title)
      || project?.name
      || projectName(project?.repo_root ?? "");
  }

  function notify(title, body, sessionId) {
    if (!notificationsEnabled || (document.hidden !== true && document.hasFocus?.() !== false)) return;
    const Notification = global.Notification;
    if (typeof Notification !== "function" || Notification.permission !== "granted") return;
    const notification = new Notification(title, { body, tag: sessionId });
    notification.onclick = () => {
      global.focus?.();
      void openSession(sessionId);
    };
  }

  function showAgents() {
    const rows = [...board.rows];
    for (const agent of conversation?.agents ?? []) {
      const replayed = rows.find((row) => row.noticeOnly && row.name === agent.name);
      if (replayed) Object.assign(replayed, {
        agentId: agent.agent_id, parentAgentId: agent.parent_agent_id ?? null,
        state: agent.state === "cancelled" && agent.parent_agent_id ? "closed" : agent.state,
        noticeOnly: false,
      });
      if (!rows.some((row) => row.agentId === agent.agent_id)) {
        rows.push({
          agentId: agent.agent_id,
          parentAgentId: agent.parent_agent_id ?? null,
          name: agent.name,
          state: agent.state === "cancelled" && agent.parent_agent_id ? "closed" : agent.state,
          tool: "",
        });
      }
    }
    const verbs = {
      read_file: ["reading", "read"],
      edit_file: ["editing", "edited"],
      multi_edit_file: ["editing", "edited"],
      write_file: ["editing", "edited"],
      run_shell: ["running", "ran"],
      grep: ["searching", "searched"],
      glob: ["searching", "searched"],
      list_files: ["searching", "searched"],
      web_fetch: ["fetching", "fetched"],
      web_search: ["searching the web for", "searched"],
      dispatch_subagent: ["dispatching", "dispatched"],
    };
    const description = (tool, target) =>
      [verbs[tool]?.[0] ?? tool, target].filter(Boolean).join(" ");
    const usageByAgent = new Map(
      (conversation?.agents ?? []).map((agent) => [agent.agent_id, agent]),
    );
    const stateWords = {
      running: "running",
      waiting: "waiting for approval",
      stopping: "stopping",
      finished: "finished",
      failed: "failed",
      stopped: "stopped",
      closed: "closed",
    };
    const displayState = (state) => state === "pending" ? "running"
      : state === "cancelled" ? "stopped"
        : stateWords[state] ?? "";
    if (rows.length === 0) {
      replace(agentsRoot, element(document, "p", { text: "Nothing is running." }));
      return;
    }
    const byId = new Map(rows.map((row) => [row.agentId, row]));
    const nodes = new Map(rows.map(({ agentId, parentAgentId, name, state, tool, target, progress, question }) => {
      const status = displayState(state);
      const activity = status === "running"
        ? (tool ? description(tool, target) : "thinking")
        : status === "waiting for approval"
          ? description(tool, target)
          : "";
      const row = element(document, "div", { className: "agent-row" });
      if (agentView ? agentView.agentId === agentId : name === "leader") {
        row.setAttribute("aria-current", "true");
      }
      if (name === "leader" || parentAgentId) {
        row.tabIndex = 0;
        row.setAttribute("role", "button");
        row.setAttribute("aria-label", name === "leader" ? "Back to the conversation" : `Open ${name}'s conversation`);
        listen(row, "click", (event) => {
          event.stopPropagation?.();
          return name === "leader" ? closeAgentView() : openAgentView({ agentId, parentAgentId, name });
        });
        listen(row, "keydown", (event) => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          event.stopPropagation?.();
          return name === "leader" ? closeAgentView() : openAgentView({ agentId, parentAgentId, name });
        });
      }
      append(row, element(document, "span", {
        className: "agent-name",
        text: name,
      }));
      const detail = [status, activity, usageText(usageByAgent.get(agentId))]
        .filter(Boolean).join(" · ");
      if (detail) {
        append(row, element(document, "span", {
          className: "agent-detail",
          text: detail,
        }));
      }
      if (progress) append(row, element(document, "span", {
        className: "agent-detail", text: progress,
      }));
      if (question) append(row, element(document, "span", {
        className: "agent-detail", text: `Asks: ${question}`,
      }));
      return [agentId, row];
    }));
    const children = new Map();
    const roots = [];
    for (const row of rows) {
      const seen = new Set([row.agentId]);
      let ancestor = byId.get(row.parentAgentId);
      while (ancestor && !seen.has(ancestor.agentId)) {
        seen.add(ancestor.agentId);
        ancestor = byId.get(ancestor.parentAgentId);
      }
      const parent = ancestor ? null : nodes.get(row.parentAgentId);
      if (!parent) {
        roots.push(nodes.get(row.agentId));
        continue;
      }
      if (!children.has(row.parentAgentId)) {
        children.set(row.parentAgentId, element(document, "div", { className: "agent-children" }));
        append(parent, children.get(row.parentAgentId));
      }
      append(children.get(row.parentAgentId), nodes.get(row.agentId));
    }
    replace(agentsRoot, ...roots);
  }

  function clearConversationAgents() {
    board.clear();
    conversation = null;
    showConversationUsage();
    showAgents();
  }

  const instructionWarningsShown = new Set();
  const configWarningsShown = new Set();

  function showConfigurationWarnings(warnings) {
    if (!Array.isArray(warnings)) return;
    for (const warning of warnings) {
      if (typeof warning !== "string" || configWarningsShown.has(warning)) continue;
      configWarningsShown.add(warning);
      answerCommand(`Configuration warning: ${warning}`);
    }
  }

  function showInstructionWarnings(current) {
    showConfigurationWarnings(current?.config_warnings);
    const sessionId = current?.session_id;
    const warnings = current?.instruction_warnings;
    if (!sessionId || instructionWarningsShown.has(sessionId) || !Array.isArray(warnings) || warnings.length === 0) return;
    instructionWarningsShown.add(sessionId);
    for (const warning of warnings) answerCommand(`Instruction warning: ${warning}`);
  }

  showConversationUsage();
  showAgents();
  showConfigurationWarnings(settingsReply?.settings?.config_warnings);
  showInstructionWarnings(conversation);

  async function refreshConversation() {
    try {
      const reply = await boundary.conversationStats();
      conversation = reply?.conversation ?? null;
      bindTranscriptLeader();
      showTranscript();
      showInstructionWarnings(conversation);
      if (permittedModes.includes(conversation?.mode)) {
        currentMode = conversation.mode;
      }
      showConversationUsage();
      showAgents();
    } catch {
      // A failed status refresh does not change the conversation itself.
    }
  }

  function showApprovals() {
    const children = [];
    for (const drafts of [approvalAnswers, approvalFeedback]) {
      for (const id of drafts.keys()) if (!["open", "answering"].includes(approvals.state.get(id)?.state)) drafts.delete(id);
    }
    for (const id of approvalDenyNotes.keys()) {
      if (approvals.state.get(id)?.state !== "open") approvalDenyNotes.delete(id);
    }
    for (const id of approvalDiffs.keys()) {
      if (approvals.state.get(id)?.state !== "open") approvalDiffs.delete(id);
    }
    const recordDecisionAtCurrentPosition = (entry, question) => {
      const toolCallId = !question.session_id || question.session_id === currentSessionId
        ? question.tool_call_id : undefined;
      const removeRecord = transcript.recordDecision(entry, toolCallId);
      showTranscript();
      return () => {
        removeRecord();
        showTranscript();
      };
    };
    for (const [id, question] of approvals.state) {
      if (question.state !== "open" && question.state !== "answering") continue;
      const row = element(document, "section", { className: `approval ${question.state} ${question.kind ?? "approval"}` });
      if (question.session_id && question.session_id !== currentSessionId) {
        const session = sessions.find((item) => item.run_id === question.session_id);
        append(row, element(document, "p", {
          className: "approval-session",
          text: session?.title || question.session_id,
        }));
        const open = makeButton(document, "Open", () => openSession(question.session_id));
        append(row, open);
      }
      if (question.kind === "question") {
        append(row, element(document, "p", { text: question.target }));
        if (question.state === "open") {
          const recordAnswer = async (allowed, answerText = "") => {
            const removeRecord = recordDecisionAtCurrentPosition({
              type: "user-question",
              question: question.target,
              answer: allowed ? answerText : "Skipped",
            }, question);
            try {
              await approvals.answer(id, allowed, "", false, answerText);
            } catch (error) {
              removeRecord();
              throw error;
            }
            const resolved = approvals.state.get(id);
            if (resolved?.state !== "resolved") removeRecord();
            showApprovals();
          };
          for (const option of question.options ?? []) {
            const button = makeButton(document, option, () => recordAnswer(true, option));
            append(row, button);
          }
          const answerInput = element(document, "textarea");
          answerInput.setAttribute("aria-label", "Your answer");
          answerInput.value = approvalAnswers.get(id) ?? "";
          listen(answerInput, "input", () => approvalAnswers.set(id, answerInput.value));
          const answerButton = makeButton(document, "Answer", () => {
            const answer = answerInput.value.trim();
            if (answer) void recordAnswer(true, answer);
          });
          const answerLabel = element(document, "label", { text: "Your answer" });
          append(answerLabel, answerInput);
          const skip = makeButton(document, "Skip", () => recordAnswer(false));
          append(row, answerLabel, answerButton, skip);
        }
        children.push(row);
        continue;
      }
      if (question.kind === "plan") {
        append(row, ...renderMarkdownBlocks(document, question.target, parseMarkdown, { highlight }));
        if (question.state === "open") {
          for (const [mode, label] of [["ask", "Approve, ask before changes"], ["allow", "Approve, allow changes"]]) {
            if (!(question.options ?? []).includes(mode)) continue;
            const approve = makeButton(document, label, async () => {
              const removeRecord = recordDecisionAtCurrentPosition({
                type: "user-plan", plan: question.target,
                outcome: mode === "ask" ? "Approved — ask before changes" : "Approved — allow changes",
              }, question);
              try {
                await approvals.answer(id, true, "", false, mode);
              } catch (error) {
                removeRecord();
                throw error;
              }
              const resolved = approvals.state.get(id);
              if (resolved?.state === "resolved") {
                currentMode = mode;
                if (conversation) conversation = { ...conversation, mode };
                rememberedPlanMode = null;
                showConversationUsage();
                showTranscript();
              } else {
                removeRecord();
              }
              showApprovals();
            });
            append(row, approve);
          }
          const feedback = element(document, "textarea");
          feedback.setAttribute("aria-label", "Feedback");
          feedback.value = approvalFeedback.get(id) ?? "";
          listen(feedback, "input", () => approvalFeedback.set(id, feedback.value));
          const feedbackLabel = element(document, "label", { text: "Feedback" });
          append(feedbackLabel, feedback);
          const keep = makeButton(document, "Keep planning", async () => {
            const reason = feedback.value.trim();
            const removeRecord = recordDecisionAtCurrentPosition({
              type: "user-plan", plan: question.target,
              outcome: reason ? `Kept planning: ${reason}` : "Kept planning",
            }, question);
            try {
              await approvals.answer(id, false, reason);
            } catch (error) {
              removeRecord();
              throw error;
            }
            const resolved = approvals.state.get(id);
            if (resolved?.state !== "resolved") removeRecord();
            showApprovals();
          });
          append(row, feedbackLabel, keep);
        }
        children.push(row);
        continue;
      }
      append(
        row,
        element(document, "p", {
          text: `${question.operation}: ${question.target} — ${question.details}`,
        }),
      );
      const approvalDiff = question.diff ?? approvalDiffs.get(id) ?? "";
      if (approvalDiff) {
        const lines = approvalDiff.split("\n");
        const diffRoot = element(document, "div", { className: "approval-diff" });
        const renderDiff = (visibleLines) => {
          const pre = element(document, "pre", { className: "diff" });
          for (const line of visibleLines) {
            const span = element(document, "span", { text: `${line}\n` });
            if (line.startsWith("+") && !line.startsWith("+++")) {
              span.className = "diff-add";
            } else if (line.startsWith("-") && !line.startsWith("---")) {
              span.className = "diff-remove";
            } else if (line.startsWith("@@")) {
              span.className = "diff-hunk";
            }
            append(pre, span);
          }
          return pre;
        };
        append(diffRoot, renderDiff(lines.slice(0, 20)));
        if (lines.length > 20) {
          const expand = makeButton(document, `Show all ${lines.length} lines`, () => replace(diffRoot, renderDiff(lines)));
          append(diffRoot, expand);
        }
        append(row, diffRoot);
      }
      if (question.state === "open") {
        const allow = listen(
          makeButton(document, "Allow"),
          "click",
          async () => {
            await approvals.answer(id, true, "");
            showApprovals();
          },
        );

        if (question.remember) {
          const remember = listen(
            makeButton(document, `Allow, and don't ask again for ${question.remember} in this chat`),
            "click",
            async () => {
              await approvals.answer(id, true, "", true);
              showApprovals();
            },
          );

          const always = listen(
            makeButton(document, `Always allow ${question.remember} in this project`),
            "click",
            async () => {
              await approvals.answer(id, true, "", "project");
              showApprovals();
            },
          );

          append(row, allow, remember, always);
        } else {
          append(row, allow);
        }
        const denyNote = element(document, "input");
        denyNote.type = "text";
        denyNote.placeholder = "Tell the agent why (optional)";
        denyNote.setAttribute("aria-label", "Tell the agent why (optional)");
        denyNote.value = approvalDenyNotes.get(id) ?? "";
        listen(denyNote, "input", () => {
          approvalDenyNotes.set(id, denyNote.value);
        });
        const denyWithNote = async () => {
          const note = denyNote.value.trim();
          approvalDenyNotes.delete(id);
          await approvals.answer(
            id,
            false,
            note ? `Denied by the person: ${note}` : "",
          );
          showApprovals();
        };
        listen(denyNote, "keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void denyWithNote();
          }
        });
        const deny = listen(
          makeButton(document, "Deny"),
          "click",
          denyWithNote,
        );

        append(row, denyNote, deny);
      }
      children.push(row);
    }
    replace(approvalsRoot, ...children);
    updateWindowTitle();
  }

  async function activeRuntimeRun() {
    try {
      const reply = await boundary.health();
      return reply?.state === "active" ? { runId: reply.runtime_run_id } : null;
    } catch {
      return null;
    }
  }

  const promptDrafts = new Map();
  let submittingPrompt = false;

  function finishPromptDraft(action, accepted) {
    const draft = promptDrafts.get(action.id);
    promptDrafts.delete(action.id);
    if (!draft) return;
    if (draft.sessionId !== currentSessionId || draft.root !== project.repo_root) return;
    if (accepted && input.value === draft.text
        && pendingFiles.length === draft.files.length
        && pendingFiles.every((file, index) => file === draft.files[index])) {
      input.value = "";
      clearDraft();
      pendingFiles.length = 0;
      pastedTextChips.length = 0;
    } else if (!accepted && input.value === "" && pendingFiles.length === 0) {
      input.value = draft.text;
      pendingFiles.push(...draft.files);
      pastedTextChips.splice(0, pastedTextChips.length, ...draft.pastes);
      saveDraft();
    }
    showPendingAttachments();
  }

  async function perform(actions) {
    updateWindowTitle();
    for (const action of actions) {
      if (action.kind === "withdrawn") promptDrafts.delete(action.id);
      if (action.kind === "prompt") {
        const presentation = {
          text: action.text,
          attachments: (action.attachments ?? []).map(({ kind, filename }) => ({ kind, filename })),
        };
        if (presentation.attachments.length > 0) promptAttachments.push(presentation);
        const startingChat = currentSessionId === null ? { frames: [] } : null;
        if (startingChat) pendingNewChat = startingChat;
        try {
          const reply = await boundary.prompt(
            action.text,
            (action.attachments ?? []).map(({ data, filename }) => ({ data, ...(filename ? { filename } : {}) })),
          );
          if (reply?.conflict === true) {
            finishPromptDraft(action, false);
            if (pendingNewChat === startingChat) pendingNewChat = null;
            const pendingIndex = promptAttachments.indexOf(presentation);
            if (pendingIndex !== -1) promptAttachments.splice(pendingIndex, 1);
            const active = await activeRuntimeRun();
            if (active) {
              promptFailure = "A run is already in progress.";
              await perform(turn.adopted(active.runId));
            } else {
              turn.rejected("the host refused a prompt it is no longer running");
              promptFailure = "The message was not sent. Send it again.";
            }
            turn.withdraw(action.id);
          } else {
            finishPromptDraft(action, true);
            let previewClearError = "";
            const delivery = previewDeliveries.get(action.text);
            if (delivery) {
              try {
                await boundary.removePreviewMarks(delivery.ids, delivery.sessionId);
                previewDeliveries.delete(action.text);
                previewMarks.splice(0, previewMarks.length, ...previewMarks.filter((mark) => !delivery.ids.includes(mark.id)));
                showPendingAttachments();
              } catch (error) {
                previewClearError = `Message sent; could not clear preview marks: ${errorText(error)}`;
              }
            }
            if (!currentSessionId && typeof reply?.run_id === "string") {
              currentSessionId = reply.run_id;
              void refreshOpenNotes();
              transcript.setSessionId(currentSessionId);
              showProjects();
            }
            promptFailure = previewClearError;
            await perform(turn.accepted());
            await refreshConversation();
            if (startingChat && pendingNewChat === startingChat) {
              while (startingChat.frames.length) await onFrame(startingChat.frames.shift(), true);
              if (pendingNewChat === startingChat) pendingNewChat = null;
            }
            try {
              await refreshSessionLists();
              showProjects();
            } catch {
              // The accepted prompt remains valid if refreshing history fails.
            }
          }
        } catch (error) {
          if (pendingNewChat === startingChat) pendingNewChat = null;
          const pendingIndex = promptAttachments.indexOf(presentation);
          if (pendingIndex !== -1) promptAttachments.splice(pendingIndex, 1);
          turn.rejected(String(error));
          turn.withdraw(action.id);
          finishPromptDraft(action, false);
          promptFailure = `Prompt failed: ${errorText(error)}`;
        }
      }
      if (action.kind === "stop") {
        await boundary.stop("");
      }
    }
    showPromptError();
    updateWindowTitle();
  }
  if (typeof global.addEventListener === "function") {
    global.addEventListener("keydown", async (event) => {
      if (event.defaultPrevented) return;
      if (lookup(keymap, event, { platform }) === "toggle-terminal") {
        event.preventDefault();
        await toggleTerminal();
        return;
      }
      if (terminalPanel.contains(event.target)) return;
      if (!quickOpenState && lookup(keymap, event, { platform }) === "quick-open") {
        event.preventDefault();
        openQuickOpen();
        return;
      }
      if (quickOpenState) return;
      if (findBar.hidden && lookup(keymap, event, { platform }) === "find") {
        event.preventDefault();
        openFind();
        return;
      }
      if (!findBar.hidden && event.key === "Escape") {
        event.preventDefault();
        closeFind();
        return;
      }
      if (!findBar.hidden) return;
      if (agentView && event.key === "Escape") {
        event.preventDefault();
        closeAgentView();
        return;
      }
      if (lookup(keymap, event, { platform }) !== "stop-run") return;
      event.preventDefault();
      if (userCommandPending) {
        await boundary.stop("");
        return;
      }
      await perform(turn.stop());
    });
  }

  listen(input, "paste", (event) => {
    const files = Array.from(event.clipboardData?.items ?? [])
      .map((item) => item.getAsFile?.())
      .filter(Boolean);
    if (files.length > 0) {
      event.preventDefault();
      addFiles(files);
      return;
    }
    const text = event.clipboardData?.getData?.("text/plain");
    if (typeof text !== "string") return;
    const lines = text.split(/\r\n|\r|\n/).length;
    if (lines < 10 && text.length < 1000) return;
    event.preventDefault();
    const id = nextPastedTextId++;
    const placeholder = `⟦Pasted text #${id}⟧`;
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? start;
    input.value = `${input.value.slice(0, start)}${placeholder}${input.value.slice(end)}`;
    input.setSelectionRange?.(start + placeholder.length, start + placeholder.length);
    pastedTextChips.push({ id, text, placeholder, lines, expanded: false });
    showPendingAttachments();
    queueDraftSave();
    renderComposerMenus();
  });
  listen(form, "dragover", (event) => event.preventDefault());
  listen(form, "drop", (event) => {
    event.preventDefault();
    addFiles(Array.from(event.dataTransfer?.files ?? []));
  });

  listen(form, "submit", async (event) => {
    event.preventDefault();
    const text = expandPastedText(input.value);
    if (editingPreviousPrompt) return submitEditedPreviousPrompt(skillMentionPrompt(text));
    const command = text.trim();
    if (command.startsWith("/")) {
      clearDraft();
      return runCommand(command);
    }
    if (command.startsWith("!") && command.length > 1) {
      if (turn.state === RUNNING || turn.state === DISPATCHING) {
        promptFailure = "Wait for the turn to finish to run a command.";
        showPromptError();
        return;
      }
      const shellCommand = command.slice(1).trim();
      if (shellCommand) {
        clearDraft();
        return runUserCommand(shellCommand, text);
      }
    }
    if (userCommandPending) {
      promptFailure = "Wait for the command to finish.";
      showPromptError();
      return;
    }
    if (specMode && text.trim()) {
      promptFailure = "";
      showPromptError();
      return startSpecLoop(text);
    }
    const selectedMarks = previewMarks.slice();
    const wireText = withPreviewMarks(skillMentionPrompt(text), selectedMarks);
    if (turn.state === RUNNING && selectedMarks.length === 0 && pendingFiles.length === 0 && text.trim() && command !== "!") {
      try {
        await boundary.steer(wireText);
        input.value = "";
        clearDraft();
        promptFailure = "";
        rememberPrompt(text);
        transcript.model.push({ type: "steer", text });
        showPromptError();
        showTranscript();
        return;
      } catch {
        // A rejected steer follows the existing queued-prompt path.
      }
    }
    if (providerRows.length > 0 && !providerRows.some((row) => row.key_present)) {
      promptFailure = "Add an API key in Settings before sending a message.";
      showPromptError();
      return;
    }
    if (text === "" && pendingFiles.length === 0 && selectedMarks.length === 0) return;
    let attachments = [];
    if (submittingPrompt) return;
    const draft = { text: input.value, files: [...pendingFiles], pastes: [...pastedTextChips],
      sessionId: currentSessionId, root: project.repo_root };
    submittingPrompt = true;
    if (pendingFiles.length > 0) {
      try {
        attachments = await encodePendingFiles();
      } catch {
        submittingPrompt = false;
        promptFailure = "Could not read the attached file.";
        showPromptError();
        return;
      }
    }
    promptFailure = "";
    rememberPrompt(text);
    if (selectedMarks.length) previewDeliveries.set(wireText, { ids: selectedMarks.map((mark) => mark.id), sessionId: previewMarkSession });
    const actions = turn.submit(wireText, attachments);
    const message = actions[0] ?? turn.queue.at(-1);
    if (message) promptDrafts.set(message.id, draft);
    if (actions.length === 0) {
      input.value = "";
      clearDraft();
      pendingFiles.length = 0;
      pastedTextChips.length = 0;
      showPendingAttachments();
    }
    showPromptError();
    try { return await perform(actions); }
    finally { submittingPrompt = false; }
  });

  listen(input, "input", () => {
    lastIdleEscapeAt = 0;
    queueDraftSave();
    const remainingPastes = pastedTextChips.filter((paste) => input.value.includes(paste.placeholder));
    if (remainingPastes.length !== pastedTextChips.length) {
      pastedTextChips.splice(0, pastedTextChips.length, ...remainingPastes);
      showPendingAttachments();
    }
    if (historyIndex >= 0) {
      historyIndex = -1;
      recalledPrompt = null;
    }
    return renderComposerMenus();
  });
  listen(input, "keydown", handleComposerKeydown);

  function answerCommand(text) {
    if (!commandEntry || !transcript.model.includes(commandEntry)) {
      commandEntry = { type: "command", command: "", output: [] };
      transcript.model.push(commandEntry);
    }
    commandEntry.output.push(text);
    showTranscript();
  }

  async function startSpecLoop(goal) {
    try {
      const reply = await boundary.startLoop(goal);
      input.value = "";
      clearDraft();
      commandEntry = null;
      transcript.model.push({
        type: "command",
        command: "",
        output: [`Started a spec loop: ${goal}`],
      });
      showTranscript();
      selectedLoopId = reply.loop?.id ?? "";
      await openRailTool("Loop");
    } catch (error) {
      commandEntry = null;
      transcript.model.push({
        type: "command",
        command: "",
        output: [errorText(error)],
      });
      showTranscript();
    }
  }

  function showPicker(options) {
    const picker = createPicker({
      document,
      keymap,
      platform,
      ...options,
    });
    replace(pickerHost, picker);
    picker.focus?.();
  }

  function closePicker() {
    replace(pickerHost);
    input.focus?.();
  }

  function closeQuickOpen() {
    clearTimeout(quickOpenTimer);
    quickOpenTimer = null;
    quickOpenVersion += 1;
    const opener = quickOpenState?.opener;
    quickOpenState = null;
    replace(pickerHost);
    opener?.focus?.();
  }

  function openQuickOpen() {
    if (quickOpenState) return;
    const opener = document.activeElement;
    const overlay = element(document, "section", { className: "quick-open-overlay" });
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Quick open");
    const dialog = element(document, "div", { className: "quick-open-dialog" });
    append(dialog, element(document, "h2", { text: "Quick open" }));
    const search = element(document, "input", { className: "quick-open-input" });
    search.type = "search";
    search.setAttribute("aria-label", "Search project files");
    search.placeholder = "Search files by name or path";
    const results = element(document, "div", { className: "quick-open-results" });
    results.setAttribute("role", "listbox");
    const message = element(document, "p", { className: "quick-open-message", text: "Type to search project files." });
    let rows = [];
    let selected = -1;

    function renderResults() {
      replace(results, ...rows.map((file, index) => {
        const path = typeof file === "string" ? file : file.path;
        const name = path.split("/").at(-1);
        const folder = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "Project root";
        const row = makeButton(document, "", null, { className: `quick-open-result${index === selected ? " focused" : ""}` });
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", String(index === selected));
        append(row,
          element(document, "span", { className: "quick-open-name", text: name }),
          element(document, "span", { className: "quick-open-folder", text: folder }),
        );
        listen(row, "click", () => void choose(file));
        return row;
      }));
    }

    async function choose(file) {
      const path = typeof file === "string" ? file : file.path;
      closeQuickOpen();
      await openRailTool("Files");
      await openSourceFile(path);
    }

    function searchFiles() {
      clearTimeout(quickOpenTimer);
      const query = search.value.trim();
      const version = ++quickOpenVersion;
      rows = [];
      selected = -1;
      renderResults();
      message.textContent = query ? "Searching…" : "Type to search project files.";
      if (!query) return;
      quickOpenTimer = setTimeout(async () => {
        try {
          const reply = await boundary.files(query, 50);
          if (!quickOpenState || version !== quickOpenVersion) return;
          rows = Array.isArray(reply?.files) ? reply.files : [];
          selected = rows.length ? 0 : -1;
          renderResults();
          message.textContent = rows.length ? "" : "No matching files.";
        } catch (error) {
          if (!quickOpenState || version !== quickOpenVersion) return;
          rows = [];
          renderResults();
          message.textContent = errorText(error);
        }
      }, 120);
    }

    listen(search, "input", searchFiles);
    listen(overlay, "keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeQuickOpen();
      } else if (event.key === "ArrowDown" && rows.length) {
        event.preventDefault();
        selected = (selected + 1) % rows.length;
        renderResults();
      } else if (event.key === "ArrowUp" && rows.length) {
        event.preventDefault();
        selected = (selected - 1 + rows.length) % rows.length;
        renderResults();
      } else if (event.key === "Enter" && selected >= 0) {
        event.preventDefault();
        void choose(rows[selected]);
      }
    });
    quickOpenState = { opener, overlay };
    append(dialog, search, results, message);
    append(overlay, dialog);
    replace(pickerHost, overlay);
    search.focus?.();
  }

  function currentModelLabel() {
    const provider = conversation?.provider ?? selection.provider;
    const model = conversation?.model ?? selection.model;
    return model
      ? `${provider ? `${provider} / ` : ""}${model}`
      : "the current model";
  }

  async function chooseMode(mode) {
    try {
      const reply = await boundary.selectMode(mode);
      if (reply?.mode !== mode || !permittedModes.includes(reply.mode)) {
        throw new Error("Host refused the mode change.");
      }
      currentMode = reply.mode;
      if (conversation) conversation = { ...conversation, mode: reply.mode };
      closePicker();
      showConversationUsage();
      answerCommand(`Mode set to ${reply.mode}.`);
      await refreshConversation();
    } catch (error) {
      closePicker();
      answerCommand(errorText(error));
    }
  }

  async function togglePlanMode() {
    if (!permittedModes.includes("plan")) {
      answerCommand("Plan mode is not permitted here.");
      return;
    }
    const activeMode = conversation?.mode ?? currentMode;
    if (activeMode !== "plan") {
      await chooseMode("plan");
      if ((conversation?.mode ?? currentMode) === "plan") {
        rememberedPlanMode = activeMode;
      }
      return;
    }
    const fallback = permittedModes.includes("ask")
      ? "ask"
      : permittedModes.find((mode) => mode !== "plan");
    const nextMode = rememberedPlanMode && permittedModes.includes(rememberedPlanMode)
      ? rememberedPlanMode
      : fallback;
    if (!nextMode) {
      answerCommand("No non-plan mode is permitted here.");
      return;
    }
    await chooseMode(nextMode);
    if ((conversation?.mode ?? currentMode) !== "plan") {
      rememberedPlanMode = null;
    }
  }

  async function resumeConversation(args) {
    let listed;
    try {
      listed = await boundary.sessions();
    } catch (error) {
      answerCommand(errorText(error));
      return;
    }
    let matches = (Array.isArray(listed) ? listed : [])
      .filter((session) => session.repo_root === project.repo_root);
    if (args.length > 0) {
      const search = args.join(" ").toLowerCase();
      matches = matches.filter((session) => (
        (session.title ?? "").toLowerCase().includes(search)
        || (session.run_id ?? "").toLowerCase().includes(search)
      ));
      if (matches.length === 0) {
        answerCommand(`No conversation in this project matches "${args.join(" ")}".`);
        return;
      }
      if (matches.length === 1) {
        try {
          await openSession(matches[0].run_id);
        } catch (error) {
          answerCommand(errorText(error));
        }
        return;
      }
    } else {
      matches = matches.slice(0, 20);
      if (matches.length === 0) {
        answerCommand("No past conversations in this project.");
        return;
      }
    }

    const rows = matches.map((session) => ({
      id: session.run_id,
      label: session.title || session.run_id,
      current: session.run_id === currentSessionId,
    }));
    const firstNotCurrent = rows.findIndex((row) => !row.current);
    showPicker({
      title: "Resume a conversation",
      rows,
      initialIndex: firstNotCurrent < 0 ? 0 : firstNotCurrent,
      onChoose: async (row) => {
        closePicker();
        try {
          await openSession(row.id);
        } catch (error) {
          answerCommand(errorText(error));
        }
      },
      onCancel: () => {
        closePicker();
        answerCommand("Kept the current conversation.");
      },
    });
  }

  function usageLine(name, usage) {
    const details = [`input ${usage.input_tokens}`, `output ${usage.output_tokens}`];
    if (usage.cache_read_tokens > 0) details.push(`cache read ${usage.cache_read_tokens}`);
    if (usage.cache_write_tokens > 0) details.push(`cache write ${usage.cache_write_tokens}`);
    const cost = usage.cost && typeof usage.cost.amount === "string" && typeof usage.cost.currency === "string"
      ? ` · ${costText(usage.cost)}`
      : "";
    return `${name}: ${usage.total_tokens} tokens (${details.join(", ")})${cost}`;
  }

  async function showCost(args) {
    try {
      const reply = await boundary.conversationStats();
      const current = reply?.conversation;
      conversation = current ?? null;
      showConversationUsage();
      showAgents();
      if (!current?.usage || !Number.isInteger(current.usage.total_tokens)) {
        answerCommand("No usage recorded in this conversation yet.");
        return;
      }
      const lines = (current.agents ?? [])
        .filter((agent) => Number.isInteger(agent.total_tokens))
        .map((agent) => usageLine(agent.name, agent));
      lines.push(usageLine("Total", current.usage));
      answerCommand(lines.join("\n"));
    } catch (error) {
      answerCommand(errorText(error));
    }
  }

  async function showContext(args) {
    try {
      const reply = await boundary.conversationStats();
      const current = reply?.conversation;
      const context = current?.context;
      if (!context) {
        answerCommand("No context measured yet. Send a message first.");
        return;
      }

      const formatTokens = (value) => value.toLocaleString("en-US");
      const model = typeof current.model === "string" && current.model
        ? `${typeof current.provider === "string" && current.provider ? `${current.provider} / ` : ""}${current.model}`
        : "";
      const window = Number.isInteger(context.window_tokens)
        ? ` · window ${formatTokens(context.window_tokens)}`
        : "";
      const used = context.used_tokens;
      const budget = context.budget_tokens;
      const lines = [
        model ? `Context · ${model}` : "Context",
        `Used ${formatTokens(used)} of ${formatTokens(budget)} tokens before compaction (${(used / budget * 100).toFixed(1)}%)${window}`,
      ];
      const labels = {
        system_prompt: "System prompt",
        instructions: "Instructions",
        user: "Your messages",
        assistant: "Replies",
        tool_result: "Tool results",
      };
      const sourceOrder = ["system_prompt", "instructions", "user", "assistant", "tool_result"];
      const sources = context.by_source ?? {};
      const entries = sourceOrder
        .filter((source) => Object.hasOwn(sources, source))
        .map((source) => [labels[source], sources[source]]);
      entries.push(...Object.entries(sources)
        .filter(([source]) => !sourceOrder.includes(source)));
      lines.push(...entries.map(([name, tokens]) => `    ${name.padEnd(16)}${formatTokens(tokens)}`));
      lines.push(
        `Remaining before compaction: ${formatTokens(context.remaining_tokens)}`,
        "Counts are estimates, about 4 characters per token.",
      );
      answerCommand(lines.join("\n"));
    } catch (error) {
      answerCommand(errorText(error));
    }
  }

  function openModePicker() {
    const rows = permittedModes.map((mode) => ({
      id: mode,
      label: mode,
      current: mode === (conversation?.mode ?? currentMode),
      efforts: [],
    }));
    showPicker({
      title: "Permission mode",
      rows,
      initialIndex: Math.max(0, rows.findIndex((row) => row.current)),
      onChoose: (row) => chooseMode(row.id),
      onCancel: () => {
        closePicker();
        answerCommand(`Kept mode as ${conversation?.mode ?? currentMode}.`);
      },
    });
  }

  async function chooseModel(provider, row, efforts) {
    const existingEffort = conversation?.effort ?? selection.effort;
    const choice = { name: provider, model: row.id };
    const keepEffort = Boolean(existingEffort && Array.isArray(efforts) && efforts.includes(existingEffort));
    if (keepEffort) choice.effort = existingEffort;
    try {
      await boundary.selectProvider(choice);
      selection = { provider, model: row.id, effort: keepEffort ? existingEffort : null };
      if (conversation) {
        conversation = { ...conversation, provider, model: row.id };
        if (keepEffort) conversation.effort = existingEffort;
        else delete conversation.effort;
      }
      closePicker();
      showConversationUsage();
      answerCommand(keepEffort
        ? `Model set to ${provider} / ${row.id} · effort ${existingEffort}.`
        : existingEffort
          ? `Model set to ${provider} / ${row.id} · effort reset to the model's default.`
          : `Model set to ${provider} / ${row.id}.`);
      await refreshConversation();
    } catch (error) {
      closePicker();
      answerCommand(errorText(error));
    }
  }

  async function chooseTypedModel(provider, model, efforts = null) {
    const existingEffort = conversation?.effort ?? selection.effort;
    const choice = { name: provider, model };
    const keepEffort = Boolean(existingEffort && Array.isArray(efforts) && efforts.includes(existingEffort));
    if (keepEffort) choice.effort = existingEffort;
    try {
      await boundary.selectProvider(choice);
      selection = { provider, model, effort: keepEffort ? existingEffort : null };
      if (conversation) {
        conversation = { ...conversation, provider, model };
        if (keepEffort) conversation.effort = existingEffort;
        else delete conversation.effort;
      }
      closePicker();
      showConversationUsage();
      answerCommand(keepEffort
        ? `Model set to ${provider} / ${model} · effort ${existingEffort}.`
        : existingEffort
          ? `Model set to ${provider} / ${model} · effort reset to the model's default.`
          : `Model set to ${provider} / ${model}.`);
      await refreshConversation();
    } catch (error) {
      answerCommand(errorText(error));
    }
  }

  async function chooseEffort(provider, model, effort) {
    try {
      await boundary.selectProvider({ name: provider, model, effort });
      selection = { provider, model, effort };
      if (conversation) conversation = { ...conversation, provider, model, effort };
      closePicker();
      showConversationUsage();
      answerCommand(`Effort set to ${effort}.`);
      await refreshConversation();
    } catch (error) {
      closePicker();
      answerCommand(errorText(error));
    }
  }

  async function runEffort(args) {
    const provider = conversation?.provider ?? selection.provider;
    const model = conversation?.model ?? selection.model;
    if (!provider || !model) {
      answerCommand("Choose a model first with /model.");
      return;
    }

    let reply;
    try {
      reply = await boundary.models(provider);
    } catch (error) {
      answerCommand(errorText(error));
      return;
    }
    if (reply?.state !== "available" || !Array.isArray(reply.models)) {
      answerCommand(reply?.detail || `Models for ${provider} are unavailable.`);
      return;
    }

    const listedModel = reply.models.find((row) => row.id === model);
    const efforts = Array.isArray(listedModel?.efforts) ? listedModel.efforts : null;
    if (efforts?.length === 0) {
      answerCommand(`${model} does not take an effort.`);
      return;
    }

    if (args.length === 0) {
      if (!efforts) {
        answerCommand(`Effort levels for ${model} are unknown; type /effort <value>.`);
        return;
      }
      const rows = efforts.map((value) => ({
        id: value,
        label: value,
        current: value === (conversation?.effort ?? selection.effort),
      }));
      showPicker({
        title: `Effort · ${provider} / ${model}`,
        rows,
        initialIndex: Math.max(0, rows.findIndex((row) => row.current)),
        onChoose: (row) => chooseEffort(provider, model, row.id),
        onCancel: () => {
          closePicker();
          answerCommand(`Kept effort as ${conversation?.effort ?? selection.effort ?? "default"}.`);
        },
      });
      return;
    }

    const effort = args[0];
    if (efforts && !efforts.includes(effort)) {
      answerCommand(`${model} accepts: ${efforts.join(", ")}.`);
      return;
    }
    await chooseEffort(provider, model, effort);
  }

  async function openModelPicker(provider, requestedModel = "", returnToProvider = false) {
    const cancel = () => {
      if (returnToProvider) {
        openProviderPicker(provider);
        return;
      }
      closePicker();
      answerCommand(`Kept model as ${currentModelLabel()}.`);
    };
    try {
      const reply = await boundary.models(provider);
      if (reply?.state !== "available" || !Array.isArray(reply.models)) {
        showPicker({
          title: `Models · ${provider}`,
          message: reply?.detail || `Models for ${provider} are unavailable.`,
          onCancel: cancel,
        });
        return;
      }
      const effortsByModel = new Map(reply.models.map((model) => [model.id, model.efforts]));
      const rows = reply.models.map((model) => ({
        id: model.id,
        label: model.id,
        current: (conversation?.provider ?? selection.provider) === provider && (conversation?.model ?? selection.model) === model.id,
      }));
      const selected = rows.findIndex((row) => row.id === requestedModel);
      const current = rows.findIndex((row) => row.current);
      showPicker({
        title: `Models · ${provider}`,
        rows,
        initialIndex: selected >= 0 ? selected : Math.max(0, current),
        onChoose: (row) => chooseModel(provider, row, effortsByModel.get(row.id)),
        onCancel: cancel,
      });
    } catch (error) {
      showPicker({
        title: `Models · ${provider}`,
        message: errorText(error),
        onCancel: cancel,
      });
    }
  }

  function openProviderPicker(focusedProvider = "") {
    const providers = providerRows.filter((row) => row.key_present);
    const rows = providers.map((provider) => ({
      id: provider.name,
      label: provider.name,
      current: provider.name === (conversation?.provider ?? selection.provider),
    }));
    const focused = rows.findIndex((row) => row.id === focusedProvider);
    const current = rows.findIndex((row) => row.current);
    showPicker({
      title: "Choose a provider",
      rows,
      message: rows.length === 0 ? "Add a provider key in Settings to choose a model." : "",
      initialIndex: focused >= 0 ? focused : Math.max(0, current),
      onChoose: (row) => openModelPicker(row.id, "", true),
      onCancel: () => {
        closePicker();
        answerCommand((conversation?.provider ?? selection.provider)
          ? `Kept model as ${currentModelLabel()}.`
          : `Kept the default provider as ${providers[0]?.name ?? "unset"}.`);
      },
    });
  }

  async function openDefaultModelPicker() {
    const currentProvider = conversation?.provider ?? selection.provider;
    if (typeof currentProvider === "string" && currentProvider) {
      const otherKeyedProvider = providerRows.some((row) => row.key_present && row.name !== currentProvider);
      if (otherKeyedProvider) {
        openProviderPicker();
        return;
      }
      await openModelPicker(currentProvider);
      return;
    }
    const providers = providerRows.filter((row) => row.key_present);
    if (providers.length === 1) {
      await openModelPicker(providers[0].name);
    } else {
      openProviderPicker();
    }
  }

  function closeCommandMenu() {
    commandMatches = [];
    highlightedCommand = 0;
    replace(commandMenu);
  }

  function closeFileMenu() {
    fileMatches = [];
    highlightedFile = 0;
    fileFragment = null;
    fileSearchSerial += 1;
    replace(fileMenu);
  }

  function mentionSkills() {
    return (Array.isArray(settingsReply?.settings?.skills) ? settingsReply.settings.skills : [])
      .filter((skill) => typeof skill?.name === "string" && skill.name);
  }

  function skillMentionPrompt(text) {
    const known = new Map(mentionSkills().map((skill) => [skill.name.toLowerCase(), skill.name]));
    const mentioned = new Set();
    for (const match of text.matchAll(/(?:^|\s)@([^\s]+)/g)) {
      const token = match[1].replace(/[.,;:!?)}\]]+$/, "").toLowerCase();
      if (known.has(token)) mentioned.add(known.get(token));
    }
    return mentioned.size ? `${[...mentioned].map((name) => `Use the ${name} skill.`).join("\n")}\n\n${text}` : text;
  }

  function fileFragmentAtCursor() {
    const cursor = input.selectionStart ?? input.value.length;
    const before = input.value.slice(0, cursor);
    const match = /(^|\s)@([^\s]*)$/.exec(before);
    if (!match) return null;
    const start = before.length - match[0].length + match[1].length;
    return { query: match[2], start, end: start + match[2].length + 1 };
  }

  function renderFileRows() {
    const rows = fileMatches.map((match, index) => {
      const label = match.kind === "skill" ? `${match.value} · skill` : match.value;
      const button = makeButton(document, label, null, { className: `file-suggestion${index === highlightedFile ? " focused" : ""}` });
      button.setAttribute?.("role", "option");
      button.setAttribute?.("aria-selected", String(index === highlightedFile));
      listen(button, "mousedown", (event) => event.preventDefault());
      listen(button, "click", () => insertFileMention(match));
      return button;
    });
    replace(fileMenu, ...rows);
  }

  function insertFileMention(match) {
    if (!fileFragment) return;
    const value = match.value;
    const cursor = fileFragment.start + value.length + 2;
    input.value = `${input.value.slice(0, fileFragment.start)}@${value} ${input.value.slice(fileFragment.end)}`;
    input.focus();
    input.setSelectionRange?.(cursor, cursor);
    closeFileMenu();
  }

  async function renderComposerMenus() {
    const fragment = fileFragmentAtCursor();
    if (!fragment) {
      closeFileMenu();
      renderCommandMenu();
      return;
    }
    closeCommandMenu();
    fileFragment = fragment;
    highlightedFile = 0;
    const serial = ++fileSearchSerial;
    const searchedText = input.value;
    const searchedCursor = input.selectionStart ?? input.value.length;
    const skills = mentionSkills().filter((skill) => skill.name.toLowerCase().includes(fragment.query.toLowerCase()))
      .sort((left, right) => left.name.localeCompare(right.name))
      .map((skill) => ({ value: skill.name, kind: "skill" }));
    try {
      const reply = await boundary.files(fragment.query);
      if (
        serial !== fileSearchSerial
        || input.value !== searchedText
        || (input.selectionStart ?? input.value.length) !== searchedCursor
      ) return;
      fileMatches = [...(Array.isArray(reply?.files) ? reply.files : []).map((path) => ({ value: path, kind: "path" })), ...skills];
      if (fileMatches.length === 0) {
        replace(fileMenu, element(document, "div", { text: "No matching paths or skills." }));
      } else {
        renderFileRows();
      }
    } catch (error) {
      if (serial !== fileSearchSerial) return;
      fileMatches = skills;
      if (skills.length) renderFileRows();
      else replace(fileMenu, element(document, "div", { text: errorText(error) }));
    }
  }

  function renderCommandMenuRows() {
    const rows = commandMatches.map((command, index) => {
      const button = makeButton(document, `/${command.name}  ${command.argumentHint}  ${command.description}`, null, { className: `command-suggestion${index === highlightedCommand ? " focused" : ""}` });
      button.setAttribute?.("role", "option");
      button.setAttribute?.("aria-selected", String(index === highlightedCommand));
      listen(button, "mousedown", (event) => event.preventDefault());
      listen(button, "click", () => runCommand(`/${command.name}`));
      return button;
    });
    replace(commandMenu, ...rows);
  }

  function renderCommandMenu() {
    commandMatches = matchCommands(input.value, [...skillCommands, ...mcpCommands]);
    highlightedCommand = 0;
    if (commandMatches.length === 0) {
      closeCommandMenu();
      return;
    }
    renderCommandMenuRows();
  }

  function moveCommandHighlight(amount) {
    highlightedCommand = (highlightedCommand + amount + commandMatches.length) % commandMatches.length;
    renderCommandMenuRows();
  }

  function renderShortcutHelp() {
    const descriptions = {
      submit: "Send message",
      newline: "Insert a new line",
      cancel: "Cancel or close",
      "stop-run": "Stop the current run",
      "toggle-terminal": "Show or hide the terminal panel",
      "focus-input": "Focus the composer",
      "command-palette": "Open command palette",
      "quick-open": "Open quick file search",
      find: "Find in conversation",
      "history-search": "Search prompt history",
      "clear-chat": "Clear conversation",
    };
    const labels = new Map();
    for (const [chord, action] of keymap) {
      if (!action || !descriptions[action]) continue;
      const parts = chord.split("+").map((part) => {
        if (part === "mod") return platform === "darwin" ? "⌘" : "Ctrl";
        if (part === "ctrl") return "Ctrl";
        if (part === "meta") return "⌘";
        if (part === "alt") return platform === "darwin" ? "⌥" : "Alt";
        if (part === "shift") return "Shift";
        if (part === "enter") return "Enter";
        if (part === "escape") return "Esc";
        return part.length === 1 ? part.toUpperCase() : part;
      });
      if (!labels.has(action)) labels.set(action, []);
      labels.get(action).push(parts.join("+").replace("⌘+", "⌘"));
    }
    replace(shortcutHelpPanel, element(document, "strong", { text: "Keyboard shortcuts" }));
    for (const [action, keys] of labels) {
      append(shortcutHelpPanel, element(document, "div", {
        className: "shortcut-help-row",
        text: `${descriptions[action]} · ${keys.join(", ")}`,
      }));
    }
  }

  async function handleComposerKeydown(event) {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Escape" && dismissTopLayer()) {
      event.preventDefault();
      event.stopPropagation?.();
      return;
    }
    if (!shortcutHelpPanel.hidden) {
      shortcutHelpPanel.hidden = true;
      if (event.key === "Escape") {
        event.preventDefault();
        input.focus();
        return;
      }
    }
    if (event.key === "?" && input.value === "" && !editingPreviousPrompt) {
      event.preventDefault();
      renderShortcutHelp();
      shortcutHelpPanel.hidden = false;
      return;
    }
    if (lookup(keymap, event, { platform }) === "history-search") {
      event.preventDefault();
      await openHistorySearch();
      return;
    }
    if (fileFragment !== null) {
      if (fileMatches.length > 0 && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
        event.preventDefault();
        highlightedFile = (highlightedFile + (event.key === "ArrowDown" ? 1 : -1) + fileMatches.length) % fileMatches.length;
        renderFileRows();
        return;
      }
      if (fileMatches.length > 0 && (event.key === "Tab" || event.key === "Enter")) {
        event.preventDefault();
        insertFileMention(fileMatches[highlightedFile]);
        return;
      }
      if (lookup(keymap, event, { platform }) === "cancel") {
        event.preventDefault();
        closeFileMenu();
        return;
      }
    }
    if (commandMatches.length > 0) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        moveCommandHighlight(event.key === "ArrowDown" ? 1 : -1);
        return;
      }
      if (event.key === "Tab") {
        event.preventDefault();
        input.value = `/${commandMatches[highlightedCommand].name} `;
        closeCommandMenu();
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        await runCommand(`/${commandMatches[highlightedCommand].name}`);
        return;
      }
      if (lookup(keymap, event, { platform }) === "cancel") {
        event.preventDefault();
        closeCommandMenu();
        return;
      }
      return;
    }
    if (fileFragment !== null) return;
    if (await handlePromptHistory(event)) return;
    if (lookup(keymap, event, { platform }) === "cancel") {
      const now = global.Date?.now() ?? Date.now();
      if (event.key === "Escape" && editingPreviousPrompt) {
        event.preventDefault();
        cancelPreviousMessageEdit();
        return;
      }
      if (turn.state === DISPATCHING || turn.state === RUNNING) {
        if (event.key === "Escape") {
          suppressEditUntil = now + 500;
          if (escapeStopRequested) {
            event.preventDefault();
            return;
          }
          escapeStopRequested = true;
        }
        lastIdleEscapeAt = 0;
        event.preventDefault();
        event.stopPropagation?.();
        await perform(turn.stop());
        return;
      }
      if (event.key === "Escape" && !userCommandPending && input.value === "") {
        event.preventDefault();
        if (now < suppressEditUntil) return;
        if (lastIdleEscapeAt > 0 && now - lastIdleEscapeAt <= 500) {
          lastIdleEscapeAt = 0;
          selectPreviousMessageForEdit();
        } else {
          lastIdleEscapeAt = now;
        }
        return;
      }
      lastIdleEscapeAt = 0;
      return;
    }
    lastIdleEscapeAt = 0;
    if (lookup(keymap, event, { platform }) === "submit") {
      event.preventDefault();
      await form.requestSubmit();
      return;
    }
    if (lookup(keymap, event, { platform }) === "newline") {
      const start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? start;
      input.value = `${input.value.slice(0, start)}\n${input.value.slice(end)}`;
      input.selectionStart = start + 1;
      input.selectionEnd = start + 1;
      event.preventDefault();
    }
  }

  async function submitCommandPrompt(prompt) {
    const index = transcript.model.indexOf(commandEntry);
    if (index !== -1) transcript.model.splice(index, 1);
    commandEntry = null;
    showTranscript();
    await perform(turn.submit(prompt));
  }

  async function runCommand(command) {
    closeCommandMenu();
    closeFileMenu();
    input.value = "";
    commandEntry = { type: "command", command: command.trim(), output: [] };
    transcript.model.push(commandEntry);
    showTranscript();
    const parts = command.split(/\s+/);
    const typedName = parts[0];
    const entry = COMMANDS.find((candidate) => (
      [candidate.name, ...candidate.aliases].some((name) => name.toLowerCase() === typedName.slice(1).toLowerCase())
    ));
    const args = parts.slice(1);
    if (!entry) {
      const mcp = mcpCommands.find((candidate) => candidate.name.toLowerCase() === typedName.slice(1).toLowerCase());
      if (mcp) {
        const declared = mcp.prompt.arguments || [];
        const text = command.slice(typedName.length).trim();
        const values = Object.create(null);
        let valid = true;
        if (declared.length === 1 && text) values[declared[0].name] = text;
        else if (text) {
          const pairs = /(?:^|\s)([^\s=]+)=("(?:\\.|[^"\\])*"|'[^']*'|[^\s]+)/g;
          let end = 0;
          for (const match of text.matchAll(pairs)) {
            if (text.slice(end, match.index).trim()) valid = false;
            let value = match[2];
            try {
              if (value.startsWith('"')) value = JSON.parse(value);
              else if (value.startsWith("'")) value = value.slice(1, -1);
            } catch { valid = false; }
            values[match[1]] = value;
            end = match.index + match[0].length;
          }
          if (text.slice(end).trim()) valid = false;
        }
        if (!valid) {
          answerCommand(`Usage: /${mcp.name} ${mcp.argumentHint}`);
          input.value = `/${mcp.name} `;
          input.focus();
          return;
        }
        try {
          const reply = await boundary.mcpPrompt(mcp.prompt.server, mcp.prompt.name, { ...values });
          await submitCommandPrompt(reply.text);
        } catch (error) { answerCommand(errorText(error)); }
        return;
      }
      const skill = skillCommands.find((candidate) => candidate.name.toLowerCase() === typedName.slice(1).toLowerCase());
      if (skill) {
        const request = command.slice(typedName.length).trim();
        const prompt = `Use the ${skill.name} skill.${request ? `\n\n${request}` : ""}`;
        await submitCommandPrompt(prompt);
        return;
      }
      answerCommand(`Unknown command: ${typedName}. Type / to see commands.`);
      return;
    }
    const usage = `Usage: /${entry.name}${entry.argumentHint ? ` ${entry.argumentHint}` : ""}`;
    if (entry.name === "mode" || entry.name === "model") closePicker();
    if (args.length < (entry.minArgs ?? 0) || args.length > (entry.maxArgs ?? Infinity)) {
      answerCommand(usage);
      return;
    }
    await entry.handler({
      refreshSettings,
      closePicker,
      openModePicker,
      openDefaultModelPicker,
      openModelPicker,
      boundary,
      chooseTypedModel,
      runEffort,
      answerCommand,
      startNewChat,
      refreshSessionLists,
      showProjects,
      transcript,
      global,
      togglePlanMode,
      showConversationUsage,
      resumeConversation,
      showCost,
      openPreview,
      showUsage,
      showContext,
      showPromptError,
      submitCommandPrompt,
      INIT_PROMPT,
      REVIEW_PROMPT,
      showTranscript,
      document,
      refreshConversation,
      get currentSessionId() { return currentSessionId; },
      set currentSessionId(value) { currentSessionId = value; void refreshOpenNotes(); },
      get specMode() { return specMode; },
      set specMode(value) { specMode = value; },
      get conversation() { return conversation; },
      set conversation(value) { conversation = value; },
      get currentMode() { return currentMode; },
      set currentMode(value) { currentMode = value; },
      get promptFailure() { return promptFailure; },
      set promptFailure(value) { promptFailure = value; },
      get commandEntry() { return commandEntry; },
      set commandEntry(value) { commandEntry = value; },
      get skillCommands() { return skillCommands; },
      set skillCommands(value) { skillCommands = value; },
    }, { args, command, typedName, usage });
  }

  async function runUserCommand(command, submittedText) {
    closeCommandMenu();
    closeFileMenu();
    input.value = "";
    rememberPrompt(submittedText);
    commandEntry = { type: "command", command: `! ${command}`, output: [] };
    transcript.model.push(commandEntry);
    showTranscript();
    userCommandPending = true;
    try {
      const reply = await boundary.command(command);
      if (reply.output) commandEntry.output.push(reply.output);
      if (reply.stopped) commandEntry.output.push("stopped");
      else if (reply.timed_out) commandEntry.output.push("timed out");
      else if (reply.exit_code !== 0) commandEntry.output.push(`exit ${reply.exit_code}`);
    } catch (error) {
      commandEntry.output.push(errorText(error));
    } finally {
      userCommandPending = false;
    }
    showTranscript();
  }

  function leadingTrailingRefresh(refresh) {
    let lastRefresh = null;
    let trailingTimer = null;
    return async function requestRefresh() {
      const now = global.Date?.now() ?? Date.now();
      if (lastRefresh === null || now - lastRefresh >= 1000) {
        lastRefresh = now;
        await refresh();
        return;
      }
      if (trailingTimer === null) {
        const delay = Math.max(0, 1000 - (now - lastRefresh));
        const schedule = global.setTimeout ?? setTimeout;
        trailingTimer = schedule(async () => {
          trailingTimer = null;
          lastRefresh = global.Date?.now() ?? Date.now();
          await refresh();
        }, delay);
      }
    };
  }

  const requestUsageRefresh = leadingTrailingRefresh(async () => {
    await refreshConversation();
  });

  const requestActivityRefresh = leadingTrailingRefresh(async () => {
    try {
      await refreshSessionLists();
      showProjects();
    } catch {
      // A transient listing failure does not affect the current transcript.
    }
  });

  async function refreshActivityForOtherSession(sessionId) {
    if (typeof sessionId === "string" && sessionId !== currentSessionId) {
      await requestActivityRefresh();
      return true;
    }
    return false;
  }

  async function onFrame(frame, held = false, replaying = false) {
    if (!replaying && frame.kind === "error" && (frame.dropped ?? frame.payload?.dropped) > 0) {
      runNotice.textContent = "State unknown — reconciling after an event gap…";
      await approvals.onFrame(frame);
      showApprovals();
      await reconcileConnection(false);
      return;
    }
    if (!held && pendingNewChat && typeof frame.payload?.session_id === "string") {
      pendingNewChat.frames.push(frame);
      return;
    }
    if (!replaying) {
      if (frame.kind === "event" && frame.payload?.type === "HistoryMessage" && typeof frame.payload.replay_id === "string") return;
      if (sessionReplay && frame.payload?.session_id === sessionReplay.sessionId) {
        sessionReplay.live.push(frame);
        return;
      }
    }
    if (frame.kind === "approval_requested") {
      const payload = frame.payload ?? {};
      if (typeof payload.approval_id === "string" && typeof payload.diff === "string") {
        approvalDiffs.set(payload.approval_id, payload.diff);
      }
      const sessionId = payload.session_id;
      const operation = payload.kind === "question" ? "Question"
        : payload.kind === "plan" ? "Plan" : payload.operation;
      if (typeof sessionId === "string" && typeof operation === "string") {
        notify("Waiting for you", `${conversationTitle(sessionId)} · ${operation}`, sessionId);
      }
      await approvals.onFrame(frame);
      showApprovals();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "PreviewMarkAdded") {
      if (frame.payload.repo_root === project.repo_root && frame.payload.session_id === currentSessionId) await refreshPreviewMarks();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "SessionNotesUpdated") {
      if (frame.payload.session_id === currentSessionId) await refreshOpenNotes();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "JobChanged") {
      if (jobsPaneVisible()) await showJobs();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "TerminalOutput") {
      terminalOutput(frame.payload);
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "LoopChanged") {
      if (frame.payload.change === "roadmap") { await refreshRoadmap(); return; }
      const loopId = frame.payload.loop_id;
      if (typeof loopId === "string" && typeof frame.payload.state === "string") {
        roadmapGoals = roadmapGoals.map((goal) => goal.loop_id === loopId
          ? { ...goal, state: frame.payload.state }
          : goal);
        renderRoadmapUI();
      }
      if (typeof loopId === "string" && !loops.some((loop) => loop.id === loopId)) {
        await loadLoops();
      } else if (typeof loopId === "string") {
        if (typeof frame.payload.state === "string") {
          loops = loops.map((loop) => loop.id === loopId
            ? { ...loop, state: frame.payload.state }
            : loop);
          renderLoops();
        }
        if (loopId === selectedLoopId && selectedRailTool === "Loop" && !loopsRoot.hidden) {
          await loadLoop(loopId);
          if (selectedCardId && frame.payload.card_id === selectedCardId) {
            await loadLoopCard(selectedCardId);
          }
        }
      }
      await refreshRoadmap();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === FILES_CHANGED_EVENT_TYPE) {
      if (selectedRailTool === "Changes" && !changesRoot.hidden) void showChanges();
      else changesDirty = true;
      if (selectedRailTool === "Files" && !filesRoot.hidden) void refreshFiles();
      else filesDirty = true;
      return;
    }
    const frameSessionId = frame.kind === "event" ? frame.payload?.session_id : null;
    if (frame.kind === "event" && frame.payload?.type === "PersonNotice") {
      notify("Leader", frame.payload.text, frameSessionId);
    }
    if (frame.kind === "event" && ["RunFinished", "RunFailed"].includes(frame.payload?.type)) {
      const payload = decodeEvent(frame.payload);
      const rootAgentId = frameSessionId === currentSessionId
        ? conversation?.agents?.find((agent) => agent.parent_agent_id == null)?.agent_id
        : null;
      const isRoot = (typeof rootAgentId === "string" && payload.fields.agent_id === rootAgentId)
        || (!rootAgentId && payload.fields.agent_name === "leader");
      if (isRoot && typeof frameSessionId === "string") {
        notify(payload.type === "RunFailed" ? "Failed" : "Finished", conversationTitle(frameSessionId), frameSessionId);
      }
    }
    if (await refreshActivityForOtherSession(frameSessionId)) return;
    const previousLength = transcript.model.length;
    transcript.apply(frame);
    if (
      agentView
      && frame.kind === "event"
      && frame.payload?.session_id === agentView.sessionId
      && frame.payload?.agent_id === agentView.agentId
    ) {
      agentTranscript.apply(frame);
    }
    const newEntries = transcript.model.slice(previousLength);
    for (const entry of replaying ? [] : newEntries) {
      if (entry.type === "edit-card") {
        await reconcileEditCard(entry);
      }
    }
    if (
      frame.kind === "event" && frame.payload?.type === "HistoryMessage"
      && typeof frame.payload.record_id === "string"
    ) {
      const historyEntry = transcript.model.slice(previousLength)
        .find((entry) => entry.type === "prompt" || entry.type === "text");
      if (historyEntry) historyEntry.recordId = frame.payload.record_id;
    }
    if (frame.kind === "event" && frame.payload?.type === "PromptSubmitted") {
      const match = promptAttachments.findIndex(({ text }) => text === frame.payload.text);
      if (match !== -1 && transcript.model.at(-1)?.type === "prompt") {
        transcript.model.at(-1).attachments = promptAttachments.splice(match, 1)[0].attachments;
      }
    }
    board.apply(frame);
    if (replaying) return;
    showAgents();
    showTranscript();
    const dropped = frame.dropped ?? frame.payload?.dropped;
    if (frame.kind === "error" && Number.isInteger(dropped) && dropped > 0) {
      await approvals.onFrame(frame);
      showApprovals();
      return;
    }
    if (frame.kind !== "event") {
      return;
    }
    const event = decodeEvent(frame.payload);
    if (event.type === "GoalChanged") {
      conversation = {
        ...(conversation ?? {}),
        goal: event.fields.change === "clear"
          ? null
          : {
            ...(conversation?.goal ?? {}),
            phase: event.fields.phase,
            rounds: event.fields.rounds,
            max_rounds: event.fields.max_rounds,
            reason: event.fields.reason,
            last_check: event.fields.last_check,
          },
      };
      showTranscript();
      showConversationUsage();
    }
    if (
      event.type === "BackgroundUsageRecorded" &&
      event.fields.session_id === currentSessionId
    ) {
      await requestUsageRefresh();
    }
    if (
      (event.type === "RunFinished" || event.type === "RunFailed") &&
      (healthReply?.runtime_run_id === null ||
        (typeof healthReply?.runtime_run_id === "string" &&
          healthReply.runtime_run_id.length > 0 &&
          healthReply.runtime_run_id === event.fields.run_id))
    ) {
      runNotice.textContent = "";
    }
    await perform(turn.event({ ...event, ...event.fields }));
    if (event.type === "RunFinished" || event.type === "RunFailed") escapeStopRequested = false;
    updateWindowTitle();
    if (event.type === "RunFinished" || event.type === "RunFailed") {
      await refreshConversation();
      await refreshSelectedChanges();
      const rootAgentId = conversation?.agents?.find((agent) => agent.parent_agent_id == null)?.agent_id;
      if (typeof rootAgentId === "string" && event.fields.agent_id === rootAgentId) {
        if (route.page === "chat" && !agentView && !document.hidden) await markSessionViewed(currentSessionId);
        await requestActivityRefresh();
      }
    }
  }

  let subscription = null, reconnectTimer = null, reconnectAttempt = 0;
  let connectionClosed = false;
  async function reconcileConnection(reopen = true) {
    healthReply = await boundary.health();
    const stats = await boundary.conversationStats();
    conversation = stats?.conversation ?? null;
    const sessionId = conversation?.session_id ?? null;
    if (reopen && sessionId) await openSession(sessionId);
    else if (sessionId !== currentSessionId) {
      currentSessionId = sessionId;
      transcript.reset();
      transcript.setSessionId(sessionId);
      showProjects();
      showTranscript();
    }
    await approvals.resync();
    showApprovals();
    if (healthReply?.state === "active") {
      if (turn.state !== RUNNING && turn.state !== DISPATCHING) { turn.submit(""); turn.accepted(); }
      if (healthReply.runtime_run_id) turn.event({ type: "RunStarted", run_id: healthReply.runtime_run_id });
    } else if (turn.state === RUNNING || turn.state === DISPATCHING) {
      if (!turn.activeRunId) turn.event({ type: "RunStarted", run_id: "reconciled" });
      await perform(turn.event({ type: "RunFinished", run_id: turn.activeRunId }));
    }
    if (runNotice.textContent.startsWith("Disconnected") || runNotice.textContent.startsWith("State unknown")) runNotice.textContent = "";
    showConversationUsage();
    showPromptError();
    showAgents();
  }
  function reconnect() {
    if (connectionClosed || reconnectTimer !== null) return;
    runNotice.textContent = "Disconnected — reconnecting…";
    const delay = Math.min(500 * 2 ** reconnectAttempt++, 10000);
    reconnectTimer = (global.setTimeout ?? setTimeout)(async () => {
      reconnectTimer = null;
      if (connectionClosed) return;
      try {
        connect();
        await subscription.ready;
        await reconcileConnection();
        reconnectAttempt = 0;
      } catch { reconnect(); }
    }, delay);
  }
  function connect() {
    subscription?.close();
    const stream = boundary.events(frame => onFrame(frame));
    subscription = stream;
    const ended = () => { if (!connectionClosed && subscription === stream) reconnect(); };
    stream.done?.then(ended, ended);
  }
  await refreshPreviewMarks();
  connect();
  try {
    await subscription.ready;
    if (currentSessionId) await openSession(currentSessionId);
    await reconcileConnection(false);
  } catch { reconnect(); }
  const connection = {
    close() {
      connectionClosed = true;
      if (reconnectTimer !== null) (global.clearTimeout ?? clearTimeout)(reconnectTimer);
      subscription?.close();
    },
    get ready() { return subscription?.ready; },
    get done() { return subscription?.done; },
  };
  if (terminalOpen && !projectless) await openTerminalPanel();
  updateWindowTitle();
  showPromptError();
  return {
    approvals,
    client: boundary,
    onFrame,
    openTool: openRailTool,
    route: () => route,
    sidebar,
    subscription: connection,
    transcript,
    turn,
  };
}

const browserGlobal = globalThis;
if (browserGlobal.document && browserGlobal.__symphonai) {
  start({ global: browserGlobal, document: browserGlobal.document }).catch((error) => {
    browserGlobal.document.body.textContent = `SymphonAI failed to start: ${error}`;
  });
}
