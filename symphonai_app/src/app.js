import { createApprovals } from "./approvals.js";
import { createAgentBoard } from "./agents.js";
import { createClient } from "./client.js";
import { parseMarkdown } from "./markdown.js";
import { resolveHost } from "./host_handle.js";
import { decodeEvent } from "./protocol.js";
import { append, element, listen, renderDiff, renderMarkdownBlocks, renderTranscript, replace } from "./render.js";
import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "./route.js";
import { COMMANDS, matchCommands } from "./commands.js";
import keymapDefaults from "../keys.default.json" with { type: "json" };
import { lookup, parseKeymap } from "./keys.js";
import { createPicker } from "./picker.js";
import { agentRows, ceilingRows, composeAgentText, generalRows, hookRows, inventoryRows, modelRows, parseAgentText, rosterRows, serverRows, trustRows } from "./settings.js";
import { createTranscript } from "./transcript.js";
import { createTurnState, DISPATCHING, RUNNING } from "./turn.js";

const STATUS_LINE_KEY = "symphonai.status-line";
const NOTIFICATIONS_KEY = "symphonai.notifications";
const PROMPT_DRAFTS_KEY = "symphonai.drafts";
const DEFAULT_STATUS_FIELDS = Object.freeze({
  provider: false, model: true, effort: true, mode: false,
  goal: false, context: false, usage: false,
});

function stateLabel(state) {
  const value = typeof state === "string" && state ? state : "unknown";
  const label = value.replaceAll("_", " ");
  return label[0].toUpperCase() + label.slice(1);
}

function readStatusFields(global) {
  try {
    const stored = JSON.parse(global.localStorage?.getItem(STATUS_LINE_KEY) ?? "null");
    return Object.fromEntries(Object.keys(DEFAULT_STATUS_FIELDS).map((key) => [
      key,
      typeof stored?.[key] === "boolean" ? stored[key] : DEFAULT_STATUS_FIELDS[key],
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
  const client = createClient({
    port: handshake.port,
    token: handshake.token,
    fetch: global.fetch.bind(global),
  });
  async function request(path, options = {}) {
    const { headers: extraHeaders = {}, ...requestOptions } = options;
    const response = await global.fetch(`http://127.0.0.1:${handshake.port}${path}`, {
      ...requestOptions,
      headers: {
        Authorization: `Bearer ${handshake.token}`,
        ...extraHeaders,
      },
    });
    if (!response || response.status < 200 || response.status >= 300) {
      throw new Error(`host request failed with status ${response?.status}`);
    }
    return response.json();
  }
  client.newSession = () => request("/session/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  client.conversationStats = () => request("/conversation");
  return client;
}

const ROUTE_KEY = "symphonai.route";
const RAIL_KEY = "symphonai.rail-tools";
const SIDEBAR_SESSION_LIMIT = 200;
const RAIL_TOOL_NAMES = ["Agents", "Roadmap", "Changes", "Files", "Jobs", "Loop"];
const RAIL_TOOL_ICONS = Object.freeze({
  Agents: "icon-tool-agents",
  Roadmap: "icon-tool-roadmap",
  Changes: "icon-tool-changes",
  Files: "icon-tool-files",
  Jobs: "icon-tool-jobs",
  Loop: "icon-tool-loop",
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

export async function start({ global, document, client }) {
  const boundary = clientFor(global, client);
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
  const loopsRoot = element(document, "div", { className: "rail-pane loop-pane" });
  loopsRoot.setAttribute("role", "tabpanel");
  loopsRoot.hidden = true;
  const railTabsRoot = document.getElementById("rail-tabs");
  const railToolsRoot = document.getElementById("rail-tools");
  railToolsRoot.before(jobsRoot, loopsRoot);
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
  const promptAttachments = [];
  let following = true;
  let savedScrollTop = 0;
  input.required = false;
  const commandMenu = element(document, "div", { className: "command-menu" });
  const fileMenu = element(document, "div", { className: "file-menu" });
  const pickerHost = element(document, "div", { className: "picker-host" });
  form.before(commandMenu, fileMenu, pickerHost);
  let commandMatches = [];
  let highlightedCommand = 0;
  let fileMatches = [];
  let highlightedFile = 0;
  let fileFragment = null;
  let fileSearchSerial = 0;
  const turn = createTurnState();
  let historyPrompts = null;
  const pagePrompts = [];
  let historyIndex = -1;
  let historyDraft = "";
  let recalledPrompt = null;
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
      return { goals: [], error: error?.message || String(error) };
    }
  };
  let project = await boundary.project();
  const projectless = project?.repo_root == null;
  let initialSessions = [];
  let initialRoadmap = { goals: [], error: "" };
  let settingsReply;
  let healthReply;
  let conversationReply = { conversation: null };
  let initialProjects = { recent: [] };
  if (projectless) {
    [settingsReply, healthReply, initialProjects] = await Promise.all([
      boundary.settings(),
      boundary.health().catch(() => null),
      (boundary.projects?.() ?? Promise.resolve({ recent: [] })).catch(() => ({ recent: [] })),
    ]);
  } else {
    [initialSessions, initialRoadmap, settingsReply, healthReply, conversationReply] = await Promise.all([
      boundary.sessions(SIDEBAR_SESSION_LIMIT),
      readRoadmap(),
      boundary.settings(),
      boundary.health().catch(() => null),
      boundary.conversationStats().catch(() => ({ conversation: null })),
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
  const keymap = parseKeymap(JSON.stringify(keymapDefaults));
  const platform = /mac/i.test(global.navigator?.platform ?? "") ? "darwin" : "other";
  const allModes = ["ask", "plan", "allow"];
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
  let sessionQuery = "";
  let editingSessionId = null;
  let editingSessionTitle = "";
  let focusSessionRename = false;
  let sessionRowError = "";
  let confirmingSessionId = null;
  let sessionSearchSequence = 0;
  let currentSessionId = conversationReply?.conversation?.session_id ?? null;
  transcript.setSessionId(currentSessionId);
  let conversation = conversationReply?.conversation ?? null;
  let draftSaveTimer = null;

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
      if (input.value) {
        drafts.push({ key, text: input.value, savedAt: Date.now() });
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
    input.value = readDrafts().find((draft) => draft.key === key)?.text ?? "";
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
  let roadmapDraft = "";
  let roadmapPending = false;
  let showRailTools = false;
  const settingsPane = element(document, "section", { className: "settings-pane" });
  const settingsSections = element(document, "nav", { className: "settings-sections" });
  const settingsContent = element(document, "div", { className: "settings-content" });
  append(settingsPane, element(document, "h1", { text: "Settings" }), settingsSections, settingsContent);
  let promptFailure = "";
  let route;
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
    const children = [
      element(document, "h2", { text: "Memory" }),
      element(document, "p", {
        text: "What the agent remembered from your conversations. It reads these at the start of each new conversation.",
      }),
    ];
    if (memoryLoading) {
      children.push(element(document, "p", { text: "Loading memory…" }));
    } else if (memoryLoadError) {
      const retry = element(document, "button", { text: "Retry" });
      retry.type = "button";
      listen(retry, "click", () => loadMemory());
      children.push(
        element(document, "p", { className: "memory-load-error", text: "Couldn't load memory." }),
        retry,
      );
    } else if (memoryReply) {
      for (const [scope, label] of [["project", "Project"], ["global", "Global"]]) {
        const group = element(document, "section", { className: "memory-group" });
        append(group, element(document, "h3", { text: label }));
        const memorySet = memoryReply[scope] ?? {};
        const entries = memorySet.entries ?? [];
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
          if (entry.id === editingMemoryId) {
            const textarea = element(document, "textarea", { className: "memory-edit-input" });
            textarea.value = entry.text;
            const save = element(document, "button", { text: "Save" });
            save.type = "button";
            const cancel = element(document, "button", { text: "Cancel" });
            cancel.type = "button";
            listen(save, "click", async () => {
              try {
                const reply = await boundary.updateMemory(entry.id, textarea.value);
                memoryReply[scope].entries = entries.map((item) =>
                  item.id === entry.id ? reply.entry : item
                );
                editingMemoryId = null;
                memoryEditError = "";
              } catch (error) {
                memoryEditError = error instanceof Error ? error.message : String(error);
              }
              if (route?.page === "settings" && route.section === "memory") {
                renderMemoryContent();
              }
            });
            listen(cancel, "click", () => {
              editingMemoryId = null;
              memoryEditError = "";
              renderMemoryContent();
            });
            append(row, textarea, save, cancel);
            if (memoryEditError) {
              append(row, element(document, "p", {
                className: "memory-edit-error",
                text: memoryEditError,
              }));
            }
          } else if (entry.id === confirmingMemoryDeleteId) {
            const confirm = element(document, "p", { text: "Delete this memory?" });
            const remove = element(document, "button", { text: "Delete" });
            remove.type = "button";
            const cancel = element(document, "button", { text: "Cancel" });
            cancel.type = "button";
            listen(remove, "click", async () => {
              try {
                await boundary.deleteMemory(entry.id);
                memoryReply[scope].entries = entries.filter((item) => item.id !== entry.id);
                confirmingMemoryDeleteId = null;
              } catch (error) {
                memoryEditError = error instanceof Error ? error.message : String(error);
              }
              renderMemoryContent();
            });
            listen(cancel, "click", () => {
              confirmingMemoryDeleteId = null;
              memoryEditError = "";
              renderMemoryContent();
            });
            append(row, element(document, "p", { className: "memory-entry-text", text: entry.text }), confirm, remove, cancel);
            if (memoryEditError) {
              append(row, element(document, "p", {
                className: "memory-edit-error",
                text: memoryEditError,
              }));
            }
          } else {
            const date = typeof entry.written_at === "string"
              ? new Date(entry.written_at).toLocaleDateString()
              : "";
            const metadata = [date, entry.agent_name].filter(Boolean).join(" · ");
            const edit = element(document, "button", { text: "Edit" });
            edit.type = "button";
            const remove = element(document, "button", { text: "Delete" });
            remove.type = "button";
            listen(edit, "click", () => {
              editingMemoryId = entry.id;
              memoryEditError = "";
              renderMemoryContent();
            });
            listen(remove, "click", () => {
              confirmingMemoryDeleteId = entry.id;
              memoryEditError = "";
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

  function showSettings(section) {
    if (section === "memory") {
      void loadMemory();
      return;
    }
    if (section === "models") {
      const models = modelRows(settingsReply);
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
        const save = element(document, "button", { text: "Save" });
        save.type = "button";
        const remove = element(document, "button", { text: "Remove" });
        remove.type = "button";
        const notice = element(document, "span");
        async function write(value) {
          input.value = "";
          try {
            await boundary.storeCredential(envVar, value);
            table.children[index + 1].children[2].textContent = value ? "present" : "absent";
            const providerRow = providerRows.find((item) => item.env_var === envVar);
            if (providerRow) providerRow.key_present = Boolean(value);
            notice.textContent = value ? "Key stored." : "Key removed.";
          } catch (error) {
            const detail = typeof error?.message === "string" ? error.message.trim() : "";
            notice.textContent = detail ? `Could not save the key: ${detail}` : "Could not save the key.";
          }
        }
        listen(save, "click", () => write(input.value));
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
    if (section === "mcp") {
      const rows = serverRows(settingsReply).map(({ name, command, started }) => [
        name, command, started ? "started" : "not started",
      ]);
      replace(
        settingsContent,
        element(document, "h2", { text: "MCP servers" }),
        settingsTable(["Server", "Command", "Started"], rows),
      );
      return;
    }
    if (section === "skills" || section === "plugins" || section === "agents") {
      if (section === "agents") {
        const tableRoot = element(document, "div");
        const editorRoot = element(document, "div", { className: "agent-editor" });
        const newAgent = element(document, "button", { text: "New agent" });
        newAgent.type = "button";

        function renderAgentRows() {
          const rows = agentRows(settingsReply);
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
              const open = element(document, "button", { text: "Open" });
              open.type = "button";
              listen(open, "click", () => openAgent(rowData));
              append(action, open);
            } else if (rowData.reason) {
              const withheld = element(document, "button", { text: "Withheld" });
              withheld.type = "button";
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
          const name = field(document, "input", "agent-name", identity?.name ?? "");
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
          const save = element(document, "button", { className: "agent-save primary", text: "Save" });
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
              settingsReply = await boundary.settings();
              skillCommands = getSkillCommands(settingsReply);
              renderAgentRows();
              status.textContent = reply?.message || "definition saved; it will take effect on the next run";
            } catch (error) {
              status.textContent = error instanceof Error ? error.message : String(error);
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
            if (status) status.textContent = error instanceof Error ? error.message : String(error);
          }
        }

        function findAgentStatus() {
          const form = editorRoot.children[0];
          for (const child of form?.children ?? []) {
            if (child.className === "agent-status") return child;
          }
          return null;
        }

        listen(newAgent, "click", () => renderEditor(parseAgentText(""), null, ""));
        renderAgentRows();
        replace(
          settingsContent,
          element(document, "h2", { text: "Agents" }),
          newAgent,
          tableRoot,
          editorRoot,
        );
        return;
      }
      const rows = rosterRows(settingsReply, section);
      const table = settingsTable(["Name", "Path"], rows.map(({ name, path }) => [name, path]));
      for (const [index, { path }] of rows.entries()) {
        if (path) {
          const copy = element(document, "button", { text: "Copy" });
          copy.type = "button";
          listen(copy, "click", () => global.navigator?.clipboard?.writeText?.(path));
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
    if (section === "hooks") {
      const rows = hookRows(settingsReply).map(({ event, command }) => [event, command]);
      replace(
        settingsContent,
        element(document, "h2", { text: "Hooks" }),
        rows.length === 0
          ? element(document, "p", { text: "No hooks are configured." })
          : settingsTable(["Event", "Command"], rows),
      );
      return;
    }
    if (section === "ceiling") {
      const rows = ceilingRows(settingsReply).map(({ capability, value }) => [capability, value]);
      replace(
        settingsContent,
        element(document, "h2", { text: "Ceiling" }),
        element(document, "p", { text: 'The most any agent may be granted. "not set" means the ceiling does not limit this.' }),
        settingsTable(["Capability", "Value"], rows),
      );
      return;
    }
    if (section === "trust") {
      const rows = trustRows(settingsReply).map(({ root, allow }) => [
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
    if (section === "inventory") {
      const rows = inventoryRows(settingsReply).map(({ scope, directory, names, reason }) => [
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
    const rows = generalRows(settingsReply).map(({ key, value, scope }) => [
      key, value, scope,
    ]);
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
    notificationCheckbox.checked = notificationsEnabled;
    notificationCheckbox.setAttribute("aria-label", "Notifications");
    const notificationNotice = element(document, "p", {
      className: "notification-settings-message",
      text: notificationMessage,
    });
    listen(notificationCheckbox, "change", async () => {
      notificationMessage = "";
      if (!notificationCheckbox.checked) {
        notificationsEnabled = false;
      } else {
        try {
          if (typeof global.Notification !== "function" || typeof global.Notification.requestPermission !== "function") {
            throw new Error("notifications unavailable");
          }
          const permission = await global.Notification.requestPermission();
          if (permission !== "granted") throw new Error("notifications unavailable");
          notificationsEnabled = true;
        } catch {
          notificationsEnabled = false;
          notificationCheckbox.checked = false;
          notificationMessage = "Notifications are not available here.";
        }
      }
      try {
        global.localStorage?.setItem(NOTIFICATIONS_KEY, String(notificationsEnabled));
      } catch {
        // Storage is optional in embedded and privacy-restricted browsers.
      }
      notificationNotice.textContent = notificationMessage;
    });
    append(notificationGroup, append(element(document, "label"), notificationCheckbox, element(document, "span", { text: "Notifications" })), notificationNotice);
    replace(
      settingsContent,
      element(document, "h2", { text: "General" }),
      element(document, "h3", { text: "Appearance" }),
      themeSelect,
      statusGroup,
      notificationGroup,
      settingsTable(["Setting", "Value", "Scope"], rows),
    );
  }

  let pageNavLinks = [];
  async function showChanges() {
    replace(
      changesRoot,
      element(document, "h1", { text: "Changes" }),
      element(document, "p", { className: "changes-loading", text: "Loading changes…" }),
    );
    const [conversationResult, gitResult] = await Promise.allSettled([
      boundary.changes(),
      boundary.gitStatus(),
    ]);
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
      const force = element(document, "button", { text: "Revert anyway" });
      force.type = "button";
      listen(force, "click", async () => {
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
                text: error instanceof Error ? error.message : String(error),
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
            text: error instanceof Error ? error.message : String(error),
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
          text: error instanceof Error ? error.message : String(error),
        }));
      }
    }

    const files = Array.isArray(reply?.files) ? reply.files : [];
    const turns = Array.isArray(reply?.turns) ? reply.turns : [];
    const worktrees = Array.isArray(reply?.worktrees) ? reply.worktrees : [];
    function diffView(diff) {
      return renderDiff(document, diff);
    }
    const children = [element(document, "h1", { text: "Changes" })];
    const workingTree = element(document, "section", { className: "changes-working-tree" });
    append(workingTree, element(document, "h2", { text: "Working tree" }));
    if (gitResult.status === "rejected") {
      append(workingTree, element(document, "p", {
        className: "error",
        text: gitResult.reason instanceof Error ? gitResult.reason.message : String(gitResult.reason),
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
        if (file.staged) append(summary, element(document, "span", { className: "changes-staged", text: "Staged" }));
        let diffLoaded = file.diff_deferred !== true;
        let diffLoading = false;
        const diff = diffView(file.diff ?? "");
        listen(row, "toggle", async () => {
          if (!row.open || diffLoaded || diffLoading) return;
          diffLoading = true;
          diff.textContent = "Loading diff…";
          try {
            const result = await boundary.gitDiff(file.path);
            diffLoaded = true;
            row.replaceChildren(summary, diffView(result.diff ?? ""));
          } catch (error) {
            diff.textContent = error instanceof Error ? error.message : String(error);
            diff.className = "diff error";
          } finally {
            diffLoading = false;
          }
        });
        append(row, summary, diff);
        append(workingTree, row);
      }
    }
    children.push(workingTree);
    if (conversationResult.status === "rejected") {
      children.push(element(document, "p", {
        className: "error",
        text: conversationResult.reason instanceof Error
          ? conversationResult.reason.message
          : String(conversationResult.reason),
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
        const apply = element(document, "button", { text: "Apply worktree" });
        apply.type = "button";
        listen(apply, "click", () => worktreeAction(worktree.name, "applyWorktree", row));
        const discard = element(document, "button", { text: "Discard worktree" });
        discard.type = "button";
        listen(discard, "click", () => worktreeAction(worktree.name, "discardWorktree", row));
        append(row, apply, discard);
        children.push(row);
      }
    }
    if (files.length > 0) {
      children.push(element(document, "h2", { text: "Files" }));
      for (const file of files) {
        const row = element(document, "section", { className: "changes-file" });
        append(row, element(document, "h3", { text: `${file.path} · ${file.status}` }));
        if (file.changed_outside) {
          append(row, element(document, "p", {
            className: "changes-outside",
            text: "Changed outside the agent.",
          }));
        }
        append(row, diffView(file.diff ?? ""));
        const button = element(document, "button", { text: "Revert" });
        button.type = "button";
        listen(button, "click", () => revert({ path: file.path }));
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
        const button = element(document, "button", { text: "Revert this prompt and later" });
        button.type = "button";
        listen(button, "click", () => revert({ key: turn.key }));
        append(row, button);
        children.push(row);
      }
    }
    replace(changesRoot, ...children);
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

  async function showJobs() {
    if (!jobsPaneVisible()) {
      updateJobsPolling([]);
      return;
    }
    replace(jobsRoot, element(document, "h1", { text: "Jobs" }));
    let jobs;
    try {
      const reply = await boundary.jobs();
      jobs = Array.isArray(reply?.jobs) ? reply.jobs : [];
    } catch (error) {
      updateJobsPolling([]);
      append(jobsRoot, element(document, "p", {
        className: "error",
        text: error instanceof Error ? error.message : String(error),
      }));
      return;
    }

    const command = document.createElement("input");
    command.type = "text";
    command.className = "jobs-command";
    command.setAttribute("placeholder", "Command to run in the background");
    command.value = jobsCommandDraft;
    command.setAttribute("aria-label", "Background command");
    const run = element(document, "button", { text: "Run in background" });
    run.type = "button";
    run.disabled = !command.value.trim();
    listen(command, "input", () => {
      jobsCommandDraft = command.value;
      run.disabled = !command.value.trim();
    });
    listen(run, "click", async () => {
      const value = command.value;
      if (!value.trim()) return;
      run.disabled = true;
      try {
        await boundary.startJob(value);
        jobsCommandDraft = "";
        await showJobs();
      } catch (error) {
        run.disabled = false;
        append(jobsRoot, element(document, "p", {
          className: "error",
          text: error instanceof Error ? error.message : String(error),
        }));
      }
    });
    append(jobsRoot, append(element(document, "div", { className: "jobs-command-form" }), command, run));

    if (jobs.length === 0) {
      append(jobsRoot, element(document, "p", { text: "No background jobs." }));
    }
    const now = global.Date?.now?.() ?? Date.now();
    for (const job of jobs) {
      const row = element(document, "details", { className: "jobs-row" });
      const summary = element(document, "summary");
      const startedAt = Date.parse(job.started_at);
      const endedAt = job.ended_at ? Date.parse(job.ended_at) : now;
      const elapsed = Number.isFinite(startedAt) && Number.isFinite(endedAt)
        ? `${Math.max(0, Math.floor((endedAt - startedAt) / 1000))}s`
        : "time unavailable";
      const status = job.status === "exited" ? `exited (exit code ${job.exit_code})` : job.status;
      append(
        summary,
        element(document, "span", { text: job.label }),
        element(document, "span", { text: job.owner === "person" ? "you" : "agent" }),
        element(document, "span", { className: "jobs-status", text: status }),
        element(document, "span", { text: elapsed }),
      );
      if (job.status === "running") {
        const stop = element(document, "button", { text: "Stop" });
        stop.type = "button";
        listen(stop, "click", async () => {
          try {
            await boundary.stopJob(job.id);
            await showJobs();
          } catch (error) {
            append(row, element(document, "p", {
              className: "error",
              text: error instanceof Error ? error.message : String(error),
            }));
          }
        });
        append(summary, stop);
      }
      const output = element(document, "pre", { className: "jobs-output", text: job.output_tail ?? "" });
      if (job.output_dropped) {
        append(row, element(document, "p", { text: "Earlier output was dropped." }));
      }
      append(row, summary, output);
      append(jobsRoot, row);
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
    const pane = route.page === "settings" ? settingsPane : chatPane;
    replace(pageRoot, pane);
    if (route.page === "chat") {
      chatRoot.scrollTop = following ? chatRoot.scrollHeight : savedScrollTop;
      jumpLatest.hidden = following;
    }
    updateMissingKeyNotice();
  }

  function navigate(nextRoute, { updateFragment = true } = {}) {
    const next = parseRoute(formatRoute(nextRoute));
    showPage(next);
    rememberRoute(global, next);
    if (updateFragment && global.location) {
      global.location.hash = formatRoute(next);
    }
  }

  const sectionLinks = ["general", "models", "mcp", "skills", "plugins", "agents", "memory", "hooks", "ceiling", "trust", "inventory"].map((section) => {
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
      await boundary.openProject(path);
      if (sessionId) await boundary.openSession(sessionId);
      global.location?.reload?.();
      return true;
    } catch (error) {
      projectActionError = error instanceof Error ? error.message : String(error);
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
    const up = element(document, "button", { text: "Up" });
    up.type = "button";
    up.disabled = !folderPicker.listing?.parent;
    listen(up, "click", () => folderPicker.listing?.parent && loadDirectories(folderPicker.listing.parent));
    const entries = element(document, "div", { className: "folder-picker-entries" });
    for (const entry of folderPicker.listing?.entries ?? []) {
      const button = element(document, "button", { text: entry.name });
      button.type = "button";
      listen(button, "click", () => loadDirectories(entry.path));
      append(entries, button);
    }
    const error = element(document, "p", { className: "project-action-error", text: folderPicker.error || projectActionError });
    const open = element(document, "button", { className: "primary", text: "Open this folder" });
    open.type = "button";
    open.disabled = !folderPicker.listing?.path;
    listen(open, "click", () => folderPicker.listing?.path && openProjectAt(folderPicker.listing.path));
    const cancel = element(document, "button", { text: "Cancel" });
    cancel.type = "button";
    listen(cancel, "click", closeFolderPicker);
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
      folderPicker.error = error instanceof Error ? error.message : String(error);
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
        const button = element(document, "button", { text: label });
        button.type = "button";
        button.setAttribute("title", item.repo_root);
        listen(button, "click", () => openProjectAt(item.repo_root));
        append(row, button);
      } else {
        append(row, element(document, "span", { text: label }));
      }
      append(list, row);
    }
    const openFolder = element(document, "button", { className: "primary", text: "Open folder…" });
    openFolder.type = "button";
    listen(openFolder, "click", showFolderPicker);
    append(projectStartScreen, list, openFolder,
      element(document, "p", { className: "project-action-error", text: projectActionError }));
    replace(pageRoot, projectStartScreen);
  }

  listen(document, "keydown", (event) => {
    if (folderPicker && event.key === "Escape") {
      event.preventDefault();
      closeFolderPicker();
    } else if (selectedCardId && selectedRailTool === "Loop" && event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation?.();
      returnToLoopBoard();
    }
  });

  function showTranscript() {
    const onChatPage = route.page === "chat";
    if (onChatPage && !preserveChatScroll) {
      following = chatRoot.scrollHeight - chatRoot.scrollTop - chatRoot.clientHeight <= 24;
    }
    const actions = {
      promptText: promptCommand,
      copyText: async (text) => {
        const clipboard = global.navigator?.clipboard;
        if (typeof clipboard?.writeText !== "function") {
          throw new Error("clipboard is unavailable");
        }
        await clipboard.writeText(text);
      },
      onViewChanges: () => openRailTool("Changes"),
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
          entry.error = error instanceof Error ? error.message : String(error);
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
          entry.error = error instanceof Error ? error.message : String(error);
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
            entry.error = error instanceof Error ? error.message : String(error);
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
      );
    }
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
    if (!agentView) {
      agentViewHeader.hidden = true;
      replace(agentViewHeader);
      form.hidden = false;
      return;
    }
    const back = element(document, "button", {
      className: "agent-conversation-back",
      text: "Back to conversation",
    });
    back.type = "button";
    listen(back, "click", closeAgentView);
    replace(
      agentViewHeader,
      element(document, "h2", { className: "agent-conversation-title", text: agentView.name }),
      back,
    );
    agentViewHeader.hidden = false;
    form.hidden = true;
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
      agentView.error = error instanceof Error ? error.message : String(error);
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

  async function openSession(runId) {
    closeAgentView();
    const targetSession = sessions.find((session) => session.run_id === runId);
    if (targetSession?.repo_root && targetSession.repo_root !== project.repo_root) {
      return openProjectAt(targetSession.repo_root, runId);
    }
    saveDraft();
    const previous = transcript.snapshot();
    const previousId = currentSessionId;
    transcript.reset();
    currentSessionId = runId;
    transcript.setSessionId(currentSessionId);
    restoreDraft();
    showProjects();
    showTranscript();
    clearConversationAgents();
    try {
      await boundary.openSession(runId);
      await refreshConversation();
      await refreshSelectedChanges();
      navigate({ page: "chat", section: "" });
    } catch (error) {
      transcript.restore(previous);
      currentSessionId = previousId;
      transcript.setSessionId(currentSessionId);
      restoreDraft();
      showProjects();
      showTranscript();
      await refreshConversation();
      throw error;
    }
  }

  function showProjects() {
    const groups = [];
    const activeSearch = document.activeElement?.className === "session-search"
      ? document.activeElement
      : null;
    const searchSelection = activeSearch === null
      ? null
      : [activeSearch.selectionStart, activeSearch.selectionEnd];
    let nextSearch = null;
    let nextRename = null;
    const openFolder = element(document, "button", { className: "project-open-folder", text: "Open folder…" });
    openFolder.type = "button";
    listen(openFolder, "click", showFolderPicker);
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
        const openProject = element(document, "button", { className: "project-open-action", text: "Open project" });
        openProject.type = "button";
        listen(openProject, "click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          openProjectAt(group.root);
        });
        append(heading, openProject);
      }
      if (current) {
        const search = element(document, "input", { className: "session-search" });
        nextSearch = search;
        search.type = "search";
        search.setAttribute("placeholder", "Search conversations");
        search.setAttribute("aria-label", "Search conversations");
        search.value = sessionQuery;
        listen(search, "input", async () => {
          sessionQuery = search.value;
          const sequence = ++sessionSearchSequence;
          try {
            const results = await boundary.sessions(SIDEBAR_SESSION_LIMIT, sessionQuery);
            if (sequence !== sessionSearchSequence) return;
            sessions = results;
            showProjects();
          } catch (error) {
            sessionRowError = error instanceof Error ? error.message : String(error);
            showProjects();
          }
        });
        append(section, search);
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
        ) + (["working", "waiting"].includes(session.activity) ? ` · ${session.activity}` : "");
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
        const button = element(document, "button", {
          className: "session-link",
          text: label,
        });
        if (session.run_id === currentSessionId) {
          button.setAttribute("aria-current", "true");
        }
        button.setAttribute("title", label);
        button.type = "button";
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
          const rename = element(document, "button", { className: "session-row-action icon-pencil" });
          rename.type = "button";
          rename.setAttribute("aria-label", "Rename");
          rename.setAttribute("title", "Rename");
          const remove = element(document, "button", { className: "session-row-action icon-trash" });
          remove.type = "button";
          remove.setAttribute("aria-label", "Delete");
          remove.setAttribute("title", "Delete");
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
          append(row, rename, remove);
          if (confirmingSessionId === session.run_id) {
            const confirmation = element(document, "div", { className: "session-delete-confirmation" });
            append(confirmation, element(document, "span", { text: "Delete this conversation? This cannot be undone." }));
            const confirm = element(document, "button", { text: "Delete" });
            confirm.type = "button";
            const cancel = element(document, "button", { text: "Cancel" });
            cancel.type = "button";
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
      groups.push(section);
    }
    replace(projectsRoot, ...groups);
    if (nextRename) {
      focusSessionRename = false;
      nextRename.focus();
      nextRename.setSelectionRange?.(0, nextRename.value.length);
    } else if (activeSearch && nextSearch) {
      nextSearch.focus();
      if (searchSelection && searchSelection.every(Number.isInteger)) {
        nextSearch.setSelectionRange?.(...searchSelection);
      }
    }
  }
  async function saveSessionTitle(runId, title) {
    const session = sessions.find((item) => item.run_id === runId);
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
      sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT, sessionQuery);
    } catch (error) {
      sessionRowError = error instanceof Error ? error.message : String(error);
    }
    showProjects();
  }
  async function deleteSession(runId) {
    try {
      await boundary.deleteSession(runId);
      sessions = sessions.filter((session) => session.run_id !== runId);
      confirmingSessionId = null;
      sessionRowError = "";
      if (currentSessionId === runId) {
        closeAgentView();
        transcript.reset();
        currentSessionId = null;
        transcript.setSessionId(null);
        clearConversationAgents();
        currentMode = launchMode;
        rememberedPlanMode = null;
        await refreshConversation();
      }
      showProjects();
      showTranscript();
    } catch (error) {
      sessionRowError = error instanceof Error ? error.message : String(error);
      showProjects();
    }
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
    } catch {
      await refreshConversation();
      promptFailure = "Could not start a new chat.";
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

  function showPendingAttachments() {
    replace(attachmentList, ...pendingFiles.map((file, index) => {
      const chip = element(document, "span", {
        className: "attachment-chip",
        text: `${file.name || "image"} · ${file.size}`,
      });
      const remove = element(document, "button", { text: "Remove" });
      remove.type = "button";
      listen(remove, "click", () => {
        pendingFiles.splice(index, 1);
        showPendingAttachments();
      });
      return append(chip, remove);
    }));
  }

  function addFiles(files) {
    const allowed = new Set(["image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"]);
    for (const file of files) {
      if (pendingFiles.length >= 10) {
        promptFailure = "You can attach up to 10 files.";
        break;
      }
      if (file.size > 5_000_000) {
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
      const bytes = new Uint8Array(await file.arrayBuffer());
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

  async function saveRoadmapAction(action, { clearDraft = false } = {}) {
    if (roadmapPending) return;
    roadmapPending = true;
    roadmapError = "";
    renderRoadmapUI();
    try {
      const reply = await action();
      if (Array.isArray(reply?.goals)) roadmapGoals = reply.goals;
      if (clearDraft) roadmapDraft = "";
      if (reply?.loop?.id) {
        selectedLoopId = reply.loop.id;
        selectedLoop = reply.loop;
        loops = [reply.loop, ...loops.filter((loop) => loop.id !== reply.loop.id)];
        await openRailTool("Loop");
      }
    } catch (error) {
      roadmapError = error instanceof Error ? error.message : String(error);
    } finally {
      roadmapPending = false;
      renderRoadmapUI();
    }
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
    const form = element(document, "form", { className: "roadmap-add-form" });
    const field = element(document, "input", { className: "roadmap-add-input" });
    field.type = "text";
    field.value = roadmapDraft;
    field.setAttribute("aria-label", "Goal");
    field.setAttribute("placeholder", "Add a goal");
    field.disabled = roadmapPending;
    listen(field, "input", () => {
      roadmapDraft = field.value;
      add.disabled = roadmapPending || !roadmapDraft.trim();
    });
    const add = element(document, "button", { text: "Add goal" });
    add.type = "submit";
    add.disabled = roadmapPending || !roadmapDraft.trim();
    listen(form, "submit", (event) => {
      event.preventDefault();
      if (roadmapDraft.trim()) {
        return saveRoadmapAction(() => boundary.addRoadmapGoal(roadmapDraft.trim()), { clearDraft: true });
      }
    });
    append(form, field, add);

    const list = element(document, "ol", { className: "roadmap-goals" });
    for (const [index, goal] of roadmapGoals.entries()) {
      const row = element(document, "li", { className: "roadmap-goal" });
      const detail = element(document, "div", { className: "roadmap-goal-detail" });
      if (typeof goal.loop_id === "string" && goal.loop_id) {
        const link = element(document, "button", { className: "roadmap-goal-link", text: goal.goal });
        link.type = "button";
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
      const actions = element(document, "div", { className: "roadmap-goal-actions" });
      for (const [label, delta] of [["Move up", -1], ["Move down", 1]]) {
        const button = element(document, "button", { text: label });
        button.type = "button";
        button.disabled = roadmapPending || index + delta < 0 || index + delta >= roadmapGoals.length;
        listen(button, "click", () => saveRoadmapAction(() => boundary.moveRoadmapGoal(goal.id, index + delta)));
        append(actions, button);
      }
      const remove = element(document, "button", { text: "Remove" });
      remove.type = "button";
      remove.disabled = roadmapPending;
      listen(remove, "click", () => saveRoadmapAction(() => boundary.removeRoadmapGoal(goal.id)));
      append(actions, remove);
      if (["not_started", "cancelled"].includes(goal.state)) {
        const start = element(document, "button", { text: "Start" });
        start.type = "button";
        start.disabled = roadmapPending;
        listen(start, "click", () => saveRoadmapAction(() => boundary.startRoadmapGoal(goal.id)));
        append(actions, start);
      }
      append(row, detail, actions);
      append(list, row);
    }
    const children = [element(document, "h2", { text: "Roadmap" }), form];
    if (roadmapError) children.push(element(document, "p", { className: "error", text: roadmapError }));
    if (roadmapGoals.length === 0 && !roadmapError) {
      children.push(element(document, "p", { className: "roadmap-empty", text: "No goals yet." }));
    }
    children.push(list);
    replace(roadmapRoot, ...children);
  }

  renderRoadmapUI();
  const fileListings = new Map();
  const expandedFileDirectories = new Set();
  const fileTreeErrors = new Map();
  let selectedFilePath = null;
  let fileSourceText = null;
  let fileSourceError = null;
  let fileSourceLoading = false;

  function renderFiles() {
    if (selectedFilePath !== null) {
      const back = element(document, "button", { className: "files-back", text: "Back" });
      back.type = "button";
      listen(back, "click", () => {
        selectedFilePath = null;
        fileSourceText = null;
        fileSourceError = null;
        fileSourceLoading = false;
        renderFiles();
      });
      const header = element(document, "div", { className: "files-viewer-header" });
      append(header, back, element(document, "strong", {
        className: "files-viewer-path",
        text: selectedFilePath,
      }));
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
        const row = element(document, "button", {
          className: `files-tree-row${isDirectory ? " files-directory" : " files-file"}`,
        });
        row.type = "button";
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
    selectedFilePath = path;
    fileSourceText = null;
    fileSourceError = null;
    fileSourceLoading = true;
    renderFiles();
    try {
      const reply = await boundary.source(path);
      fileSourceText = reply.text;
    } catch (error) {
      fileSourceError = error;
    } finally {
      fileSourceLoading = false;
      renderFiles();
    }
  }

  function renderLoopCardDetails() {
    const details = element(document, "section", { className: "loop-card-details" });
    const back = element(document, "button", { text: "Back to the board" });
    back.type = "button";
    listen(back, "click", returnToLoopBoard);
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
      append(section, ...renderMarkdownBlocks(document, typeof source === "string" ? source : "", parseMarkdown));
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
    append(reviewSection, ...renderMarkdownBlocks(document, typeof review.text === "string" ? review.text : "", parseMarkdown));
    const followUps = Array.isArray(review.follow_ups) ? review.follow_ups : [];
    if (followUps.length > 0) {
      const list = element(document, "ul", { className: "loop-follow-ups" });
      for (const followUp of followUps) {
        const id = typeof followUp === "string" ? followUp : followUp?.id;
        if (typeof id !== "string") continue;
        const button = element(document, "button", { className: "loop-follow-up", text: id });
        button.type = "button";
        listen(button, "click", () => loadLoopCard(id));
        const item = element(document, "li");
        append(item, button);
        append(list, item);
      }
      append(reviewSection, list);
    }
    append(details, reviewSection);

    if (typeof cardDetail.session_id === "string" && cardDetail.session_id) {
      const open = element(document, "button", { text: "Open conversation" });
      open.type = "button";
      listen(open, "click", () => openSession(cardDetail.session_id));
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
      const keep = element(document, "button", { className: "primary", text: "Keep the branch" });
      keep.type = "button";
      keep.disabled = loopCommitPending;
      listen(keep, "click", () => runLoopCommit("approve"));
      append(section, keep);
      if (loopDiscardConfirm) {
        append(section, element(document, "p", {
          className: "loop-discard-prompt",
          text: `Discard this work and delete ${selectedLoop.branch ?? ""}?`,
        }));
        const discard = element(document, "button", { text: "Discard" });
        discard.type = "button";
        discard.disabled = loopCommitPending;
        listen(discard, "click", () => runLoopCommit("decline"));
        const keep = element(document, "button", { text: "Keep" });
        keep.type = "button";
        keep.disabled = loopCommitPending;
        listen(keep, "click", () => {
          loopDiscardConfirm = false;
          renderLoops();
        });
        append(section, discard, keep);
      } else {
        const discard = element(document, "button", { text: "Discard" });
        discard.type = "button";
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
    selectedCardId = cardId;
    cardDetail = null;
    cardDetailLoading = true;
    cardDetailError = "";
    renderLoops();
    try {
      cardDetail = await boundary.loopCard(selectedLoopId, cardId);
    } catch (error) {
      cardDetailError = error instanceof Error ? error.message : String(error);
    } finally {
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
      loopCommitError = error instanceof Error ? error.message : String(error);
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
      const button = element(document, "button", {
        className: "loop-list-item",
        text: `${loop.goal ?? loop.id} · ${stateLabel(loop.state)}`,
      });
      button.type = "button";
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
          const open = element(document, "button", {
            className: "loop-card-open",
            text: `${card.id} · ${card.title ?? ""} · ${stateLabel(card.state)}`,
          });
          open.type = "button";
          listen(open, "click", () => loadLoopCard(card.id));
          append(cardElement, open);
          if (selectedLoop.state === "awaiting_plan" && card.state !== "dropped") {
            const drop = element(document, "button", { text: "Drop" });
            drop.type = "button";
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
        const approve = element(document, "button", { text: "Approve plan" });
        approve.type = "button";
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
        const reviseButton = element(document, "button", { text: "Revise…" });
        reviseButton.type = "button";
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
          const confirm = element(document, "button", { text: "Cancel loop" });
          confirm.type = "button";
          confirm.disabled = loopActionPending;
          listen(confirm, "click", () => runLoopAction(() => boundary.loopCancel(selectedLoopId)));
          const keep = element(document, "button", { text: "Keep running" });
          keep.type = "button";
          keep.disabled = loopActionPending;
          listen(keep, "click", () => {
            loopCancelConfirm = false;
            renderLoops();
          });
          append(actions, confirm, keep);
        } else {
          const cancel = element(document, "button", { text: "Cancel loop" });
          cancel.type = "button";
          cancel.disabled = loopActionPending;
          listen(cancel, "click", () => {
            loopCancelConfirm = true;
            renderLoops();
          });
          append(actions, cancel);
        }
      }
      if (selectedLoop.state === "needs_attention") {
        const resume = element(document, "button", { text: "Resume" });
        resume.type = "button";
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
      selectedLoop = reply.loop;
    } catch (error) {
      loopError = error instanceof Error ? error.message : String(error);
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
      loopActionError = error instanceof Error ? error.message : String(error);
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
      loopError = error instanceof Error ? error.message : String(error);
      renderLoops();
    }
  }

  function renderRail() {
    const tabs = openRailTools.map((name) => {
      const selected = name === selectedRailTool;
      const wrapper = element(document, "div", { className: "rail-tab-item" });
      wrapper.setAttribute("aria-selected", String(selected));
      const tab = element(document, "button", {
        className: `rail-tab ${RAIL_TOOL_ICONS[name]}`,
        text: selected ? name : "",
      });
      tab.type = "button";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(selected));
      tab.setAttribute("aria-label", name);
      tab.setAttribute("title", name);
      listen(tab, "click", () => selectRailTool(name));
      append(wrapper, tab);
      if (selected) {
        const close = element(document, "button", { className: "rail-tab-close", text: "×" });
        close.type = "button";
        close.setAttribute("aria-label", `Close ${name}`);
        close.setAttribute("title", `Close ${name}`);
        listen(close, "click", () => closeRailTool(name));
        append(wrapper, close);
      }
      return wrapper;
    });
    const add = element(document, "button", { className: "rail-tab-add", text: "+" });
    add.type = "button";
    add.setAttribute("aria-label", "Open a tool");
    listen(add, "click", () => {
      showRailTools = !showRailTools;
      renderRail();
    });
    replace(railTabsRoot, ...tabs, add);

    const tools = RAIL_TOOL_NAMES.map((name) => {
      const row = element(document, "button", { className: "rail-tool", text: name });
      row.type = "button";
      listen(row, "click", () => openRailTool(name));
      return row;
    });
    replace(railToolsRoot, element(document, "h2", { text: "Tools" }), ...tools);
    const panes = { Agents: agentsRoot, Roadmap: roadmapRoot, Changes: changesRoot, Files: filesRoot, Jobs: jobsRoot, Loop: loopsRoot };
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
    if (!changed) return undefined;
    if (name === "Changes") return showChanges();
    if (name === "Files") return loadFileDirectory("");
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
    return undefined;
  }

  async function refreshSelectedChanges() {
    if (selectedRailTool === "Changes") await showChanges();
  }

  if (selectedRailTool === "Changes") void showChanges();
  if (selectedRailTool === "Files") void loadFileDirectory("");
  if (selectedRailTool === "Jobs") void showJobs();

  function costText(cost) {
    return cost && typeof cost.amount === "string" && typeof cost.currency === "string"
      ? `${cost.currency} ${cost.amount}`
      : "";
  }

  function usageText(usage) {
    if (!usage || !Number.isInteger(usage.total_tokens)) {
      return "";
    }
    return [`${usage.total_tokens} tokens`, costText(usage.cost)].filter(Boolean).join(" · ");
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
    conversationUsage.textContent = [primaryStatus, ...extraStatus].filter(Boolean).join(" · ");
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
      if (!rows.some((row) => row.agentId === agent.agent_id)) {
        rows.push({
          agentId: agent.agent_id,
          parentAgentId: agent.parent_agent_id ?? null,
          name: agent.name,
          state: agent.state,
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
      paused: "paused",
      stopping: "stopping",
      finished: "finished",
      failed: "failed",
      stopped: "stopped",
    };
    const displayState = (state) => state === "pending" ? "running"
      : state === "cancelled" ? "stopped"
        : stateWords[state] ?? "";
    if (rows.length === 0) {
      replace(agentsRoot, element(document, "p", { text: "Nothing is running." }));
      return;
    }
    const byId = new Map(rows.map((row) => [row.agentId, row]));
    const nodes = new Map(rows.map(({ agentId, parentAgentId, name, state, tool, target }) => {
      const status = displayState(state);
      const activity = status === "running"
        ? (tool ? description(tool, target) : "thinking")
        : status === "waiting for approval"
          ? description(tool, target)
          : "";
      const row = element(document, "div", { className: "agent-row" });
      if (name !== "leader" && parentAgentId) {
        row.tabIndex = 0;
        row.setAttribute("role", "button");
        row.setAttribute("aria-label", `Open ${name}'s conversation`);
        listen(row, "click", (event) => {
          event.stopPropagation?.();
          return openAgentView({ agentId, parentAgentId, name });
        });
        listen(row, "keydown", (event) => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          event.stopPropagation?.();
          return openAgentView({ agentId, parentAgentId, name });
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

  function showInstructionWarnings(current) {
    const sessionId = current?.session_id;
    const warnings = current?.instruction_warnings;
    if (!sessionId || instructionWarningsShown.has(sessionId) || !Array.isArray(warnings) || warnings.length === 0) return;
    instructionWarningsShown.add(sessionId);
    for (const warning of warnings) answerCommand(`Instruction warning: ${warning}`);
  }

  showConversationUsage();
  showAgents();
  showInstructionWarnings(conversation);

  async function refreshConversation() {
    try {
      const reply = await boundary.conversationStats();
      conversation = reply?.conversation ?? null;
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
        const open = element(document, "button", { text: "Open" });
        open.type = "button";
        listen(open, "click", () => openSession(question.session_id));
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
            const button = element(document, "button", { text: option });
            button.type = "button";
            listen(button, "click", () => recordAnswer(true, option));
            append(row, button);
          }
          const answerInput = element(document, "textarea");
          answerInput.setAttribute("aria-label", "Your answer");
          const answerButton = element(document, "button", { text: "Answer" });
          answerButton.type = "button";
          listen(answerButton, "click", () => {
            const answer = answerInput.value.trim();
            if (answer) void recordAnswer(true, answer);
          });
          const answerLabel = element(document, "label", { text: "Your answer" });
          append(answerLabel, answerInput);
          const skip = element(document, "button", { text: "Skip" });
          skip.type = "button";
          listen(skip, "click", () => recordAnswer(false));
          append(row, answerLabel, answerButton, skip);
        }
        children.push(row);
        continue;
      }
      if (question.kind === "plan") {
        append(row, ...renderMarkdownBlocks(document, question.target, parseMarkdown));
        if (question.state === "open") {
          for (const [mode, label] of [["ask", "Approve, ask before changes"], ["allow", "Approve, allow changes"]]) {
            if (!(question.options ?? []).includes(mode)) continue;
            const approve = element(document, "button", { text: label });
            approve.type = "button";
            listen(approve, "click", async () => {
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
          const feedbackLabel = element(document, "label", { text: "Feedback" });
          append(feedbackLabel, feedback);
          const keep = element(document, "button", { text: "Keep planning" });
          keep.type = "button";
          listen(keep, "click", async () => {
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
          const expand = element(document, "button", { text: `Show all ${lines.length} lines` });
          expand.type = "button";
          listen(expand, "click", () => replace(diffRoot, renderDiff(lines)));
          append(diffRoot, expand);
        }
        append(row, diffRoot);
      }
      if (question.state === "open") {
        const allow = listen(
          element(document, "button", { text: "Allow" }),
          "click",
          async () => {
            await approvals.answer(id, true, "");
            showApprovals();
          },
        );
        allow.type = "button";
        if (question.remember) {
          const remember = listen(
            element(document, "button", {
              text: `Allow, and don't ask again for ${question.remember} in this chat`,
            }),
            "click",
            async () => {
              await approvals.answer(id, true, "", true);
              showApprovals();
            },
          );
          remember.type = "button";
          append(row, allow, remember);
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
          element(document, "button", { text: "Deny" }),
          "click",
          denyWithNote,
        );
        deny.type = "button";
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

  async function perform(actions) {
    updateWindowTitle();
    for (const action of actions) {
      if (action.kind === "prompt") {
        const presentation = {
          text: action.text,
          attachments: (action.attachments ?? []).map(({ kind, filename }) => ({ kind, filename })),
        };
        if (presentation.attachments.length > 0) promptAttachments.push(presentation);
        try {
          const reply = await boundary.prompt(
            action.text,
            (action.attachments ?? []).map(({ data, filename }) => ({ data, ...(filename ? { filename } : {}) })),
          );
          if (reply?.conflict === true) {
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
          } else {
            if (!currentSessionId && typeof reply?.run_id === "string") {
              currentSessionId = reply.run_id;
              transcript.setSessionId(currentSessionId);
              showProjects();
            }
            promptFailure = "";
            await perform(turn.accepted());
            try {
              sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT, sessionQuery);
              showProjects();
            } catch {
              // The accepted prompt remains valid if refreshing history fails.
            }
          }
        } catch (error) {
          const pendingIndex = promptAttachments.indexOf(presentation);
          if (pendingIndex !== -1) promptAttachments.splice(pendingIndex, 1);
          turn.rejected(String(error));
          promptFailure = "Prompt failed.";
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
    }
  });
  listen(form, "dragover", (event) => event.preventDefault());
  listen(form, "drop", (event) => {
    event.preventDefault();
    addFiles(Array.from(event.dataTransfer?.files ?? []));
  });

  listen(form, "submit", async (event) => {
    event.preventDefault();
    const text = input.value;
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
    if (turn.state === RUNNING && pendingFiles.length === 0 && text.trim() && command !== "!") {
      try {
        await boundary.steer(text);
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
    if (text === "" && pendingFiles.length === 0) return;
    let attachments = [];
    if (pendingFiles.length > 0) {
      try {
        attachments = await encodePendingFiles();
      } catch {
        promptFailure = "Could not read the attached file.";
        showPromptError();
        return;
      }
    }
    input.value = "";
    clearDraft();
    pendingFiles.length = 0;
    showPendingAttachments();
    promptFailure = "";
    rememberPrompt(text);
    const actions = turn.submit(text, attachments);
    showPromptError();
    return perform(actions);
  });

  listen(input, "input", () => {
    queueDraftSave();
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
        output: [error instanceof Error ? error.message : String(error)],
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
          answerCommand(error instanceof Error ? error.message : String(error));
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
          answerCommand(error instanceof Error ? error.message : String(error));
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
      ? ` · ${usage.cost.amount} ${usage.cost.currency}`
      : "";
    return `${name}: ${usage.total_tokens} tokens (${details.join(", ")})${cost}`;
  }

  async function showCost(args) {
    if (args.length > 0) {
      answerCommand("Usage: /cost");
      return;
    }
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
      answerCommand(error instanceof Error ? error.message : String(error));
    }
  }

  async function showContext(args) {
    if (args.length > 0) {
      answerCommand("Usage: /context");
      return;
    }
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
      answerCommand(error instanceof Error ? error.message : String(error));
    }
  }

  async function runEffort(args) {
    if (args.length > 1) {
      answerCommand("Usage: /effort [<value>]");
      return;
    }
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
      answerCommand(error instanceof Error ? error.message : String(error));
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
        message: error instanceof Error ? error.message : String(error),
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

  function fileFragmentAtCursor() {
    const cursor = input.selectionStart ?? input.value.length;
    const before = input.value.slice(0, cursor);
    const match = /(^|\s)@([^\s]*)$/.exec(before);
    if (!match) return null;
    const start = before.length - match[0].length + match[1].length;
    return { query: match[2], start, end: start + match[2].length + 1 };
  }

  function renderFileRows() {
    const rows = fileMatches.map((path, index) => {
      const button = element(document, "button", {
        className: `file-suggestion${index === highlightedFile ? " focused" : ""}`,
        text: path,
      });
      button.type = "button";
      button.setAttribute?.("role", "option");
      button.setAttribute?.("aria-selected", String(index === highlightedFile));
      listen(button, "mousedown", (event) => event.preventDefault());
      listen(button, "click", () => insertFileMention(path));
      return button;
    });
    replace(fileMenu, ...rows);
  }

  function insertFileMention(path) {
    if (!fileFragment) return;
    input.value = `${input.value.slice(0, fileFragment.start)}@${path} ${input.value.slice(fileFragment.end)}`;
    input.focus();
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
    try {
      const reply = await boundary.files(fragment.query);
      if (
        serial !== fileSearchSerial
        || input.value !== searchedText
        || (input.selectionStart ?? input.value.length) !== searchedCursor
      ) return;
      fileMatches = Array.isArray(reply?.files) ? reply.files : [];
      if (fileMatches.length === 0) {
        replace(fileMenu, element(document, "div", { text: "No files found." }));
      } else {
        renderFileRows();
      }
    } catch (error) {
      if (serial !== fileSearchSerial) return;
      fileMatches = [];
      replace(fileMenu, element(document, "div", {
        text: error instanceof Error ? error.message : String(error),
      }));
    }
  }

  function renderCommandMenuRows() {
    const rows = commandMatches.map((command, index) => {
      const button = element(document, "button", {
        className: `command-suggestion${index === highlightedCommand ? " focused" : ""}`,
        text: `/${command.name}  ${command.argumentHint}  ${command.description}`,
      });
      button.type = "button";
      button.setAttribute?.("role", "option");
      button.setAttribute?.("aria-selected", String(index === highlightedCommand));
      listen(button, "mousedown", (event) => event.preventDefault());
      listen(button, "click", () => runCommand(`/${command.name}`));
      return button;
    });
    replace(commandMenu, ...rows);
  }

  function renderCommandMenu() {
    commandMatches = matchCommands(input.value, skillCommands);
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

  async function handleComposerKeydown(event) {
    if (event.isComposing || event.keyCode === 229) return;
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
      if (turn.state === DISPATCHING || turn.state === RUNNING) {
        event.preventDefault();
        await perform(turn.stop());
      }
      return;
    }
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
    if (entry.name === "mode") {
      closePicker();
      if (args.length > 0) {
        answerCommand("Usage: /mode");
        return;
      }
      openModePicker();
      return;
    }
    if (entry.name === "model") {
      closePicker();
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
          answerCommand(error instanceof Error ? error.message : String(error));
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
      answerCommand("Usage: /model [<provider> [<id>]]");
      return;
    }
    if (entry.name === "effort") {
      await runEffort(args);
      return;
    }
    if (entry.name === "help") {
      if (args.length > 0) {
        answerCommand("Usage: /help");
        return;
      }
      const lines = COMMANDS.map(({ name, aliases, description, argumentHint }) => (
        `/${name}${argumentHint ? ` ${argumentHint}` : ""} — ${description}`
        + (aliases.length > 0 ? ` (also ${aliases.map((alias) => `/${alias}`).join(", ")})` : "")
      ));
      if (skillCommands.length > 0) {
        lines.push("Skills:", ...skillCommands.map(({ name, argumentHint, description }) => (
          `/${name} ${argumentHint} — ${description}`
        )));
      }
      answerCommand(lines.join("\n"));
      return;
    }
    if (entry.name === "new") {
      if (args.length > 0) {
        answerCommand("Usage: /new");
        return;
      }
      await startNewChat();
      return;
    }
    if (entry.name === "rename") {
      const title = command.slice(typedName.length).trim();
      if (!currentSessionId || !title) {
        answerCommand("Usage: /rename <title>");
        return;
      }
      try {
        await boundary.renameSession(currentSessionId, title);
        sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT, sessionQuery);
        showProjects();
        answerCommand("Conversation renamed.");
      } catch (error) {
        answerCommand(error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (entry.name === "copy") {
      if (args.length > 0) {
        answerCommand("Usage: /copy");
        return;
      }
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
        answerCommand(`Could not copy: ${error instanceof Error ? error.message : String(error)}`);
      }
      return;
    }
    if (entry.name === "plan") {
      if (args.length > 0) {
        answerCommand("Usage: /plan");
        return;
      }
      await togglePlanMode();
      return;
    }
    if (entry.name === "spec") {
      if (args.length > 0) {
        answerCommand("Usage: /spec");
        return;
      }
      specMode = !specMode;
      showConversationUsage();
      answerCommand(specMode
        ? "Spec mode on: each message starts a spec loop."
        : "Spec mode off.");
      return;
    }
    if (entry.name === "resume") {
      await resumeConversation(args);
      return;
    }
    if (entry.name === "cost") {
      await showCost(args);
      return;
    }
    if (entry.name === "context") {
      await showContext(args);
      return;
    }
    if (entry.name === "compact") {
      answerCommand("Compacting…");
      try {
        const instructions = command.slice(typedName.length).trim();
        const result = await boundary.compact(instructions || undefined);
        answerCommand(result.changed
          ? `Compacted: ${result.before_tokens} → ${result.after_tokens} tokens, ${result.dropped_messages} messages summarized.`
          : "Nothing to compact yet.");
      } catch (error) {
        answerCommand(error instanceof Error ? error.message : String(error));
      }
      await refreshConversation();
      return;
    }
    if (entry.name === "goal") {
      const usage = "Usage: /goal [<objective> [-- <check command>] | pause | resume | clear]";
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
          answerCommand(error instanceof Error ? error.message : String(error));
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
          conversation = { ...(conversation ?? {}), goal: reply.goal };
          showConversationUsage();
          answerCommand({ pause: "Goal paused.", resume: "Goal resumed.", clear: "Goal cleared." }[action]);
        } catch (error) {
          answerCommand(error instanceof Error ? error.message : String(error));
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
        conversation = { ...(conversation ?? {}), goal: reply.goal };
        showConversationUsage();
        answerCommand(`Goal set. Round 1 of ${reply.goal?.max_rounds ?? 10} started.`);
      } catch (error) {
        answerCommand(error?.status === 409
          ? "A run is active. Wait for it to finish, then set the goal."
          : error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (entry.name === "init") {
      if (args.length > 0) {
        answerCommand("Usage: /init");
        return;
      }
      if ((conversation?.mode ?? currentMode) === "plan") {
        answerCommand("Plan mode cannot write files. Switch with /plan, then run /init.");
        return;
      }
      promptFailure = "";
      showPromptError();
      await submitCommandPrompt(INIT_PROMPT);
      return;
    }
    if (entry.name === "review") {
      const focus = command.slice(typedName.length).trim();
      promptFailure = "";
      showPromptError();
      await submitCommandPrompt(`${REVIEW_PROMPT}${focus ? `\n\nFocus on: ${focus}` : ""}`);
      return;
    }
    if (entry.name === "commit") {
      let status;
      try {
        status = await boundary.gitStatus();
      } catch (error) {
        answerCommand(error instanceof Error ? error.message : String(error));
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
      commandEntry.type = "commit-card";
      commandEntry.message = command.slice(typedName.length).trim();
      commandEntry.files = files.map((file) => ({
        ...file,
        checked: true,
        paths: [...new Set([file.old_path, file.path].filter((path) => typeof path === "string"))],
      }));
      commandEntry.error = "";
      showTranscript();
      return;
    }
    if (entry.name === "pr") {
      if (args.length > 0) {
        answerCommand("Usage: /pr");
        return;
      }
      try {
        const info = await boundary.pullRequestInfo();
        commandEntry.type = "pull-request-card";
        commandEntry.problem = info.problem || "";
        commandEntry.branch = info.branch || "";
        commandEntry.base = info.base || "";
        commandEntry.title = info.title || "";
        commandEntry.body = "";
        commandEntry.error = "";
        showTranscript();
      } catch (error) {
        answerCommand(error instanceof Error ? error.message : String(error));
      }
    }
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
      commandEntry.output.push(error instanceof Error ? error.message : String(error));
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
      sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT, sessionQuery);
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

  async function onFrame(frame) {
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
    if (frame.kind === "event" && frame.payload?.type === "JobChanged") {
      if (jobsPaneVisible()) await showJobs();
      return;
    }
    if (frame.kind === "event" && frame.payload?.type === "LoopChanged") {
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
    const frameSessionId = frame.kind === "event" ? frame.payload?.session_id : null;
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
    for (const entry of newEntries) {
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
    updateWindowTitle();
    if (event.type === "RunFinished" || event.type === "RunFailed") {
      await refreshConversation();
      await refreshSelectedChanges();
      const rootAgentId = conversation?.agents?.find((agent) => agent.parent_agent_id == null)?.agent_id;
      if (typeof rootAgentId === "string" && event.fields.agent_id === rootAgentId) {
        await requestActivityRefresh();
      }
    }
  }

  const subscription = boundary.events((frame) => onFrame(frame));
  if (subscription.ready) {
    await subscription.ready;
    if (currentSessionId) {
      const leader = conversation?.agents?.find((agent) => agent.parent_agent_id == null);
      if (leader?.state === "running") {
        turn.submit("");
        turn.accepted();
        if (typeof healthReply?.runtime_run_id === "string") {
          turn.event({ type: "RunStarted", run_id: healthReply.runtime_run_id });
        }
      }
      await openSession(currentSessionId);
    }
    try {
      await approvals.resync();
      showApprovals();
    } catch {
      // A failed refresh must not prevent the conversation from opening.
    }
  }
  updateWindowTitle();
  showPromptError();
  return {
    approvals,
    client: boundary,
    onFrame,
    openTool: openRailTool,
    route: () => route,
    sidebar,
    subscription,
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
