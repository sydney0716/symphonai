import { createApprovals } from "./approvals.js";
import { createAgentBoard } from "./agents.js";
import { createClient } from "./client.js";
import { parseMarkdown } from "./markdown.js";
import { resolveHost } from "./host_handle.js";
import { decodeEvent } from "./protocol.js";
import { renderRoadmap, parseRoadmap, specPaths, followUpsFor } from "./roadmap.js";
import { layoutPhase, layoutRoadmap } from "./roadmap_graph.js";
import { append, element, listen, renderTranscript, replace } from "./render.js";
import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "./route.js";
import { COMMANDS, matchCommands } from "./commands.js";
import keymapDefaults from "../keys.default.json" with { type: "json" };
import { lookup, parseKeymap } from "./keys.js";
import { createPicker } from "./picker.js";
import { agentRows, ceilingRows, composeAgentText, generalRows, hookRows, inventoryRows, modelRows, parseAgentText, rosterRows, serverRows, trustRows } from "./settings.js";
import { createSpecView } from "./spec_view.js";
import { createTranscript } from "./transcript.js";
import { createTurnState } from "./turn.js";

export const INIT_PROMPT = `Please analyze this repository and write .symphonai/INSTRUCTIONS.md, which SymphonAI loads into every conversation in this project.

Include:
1. The commands used most often: how to build, lint and run the tests, including how to run a single test.
2. The high-level architecture: the big picture that takes reading several files to understand.

Rules:
- If .symphonai/INSTRUCTIONS.md already exists, read it and improve it instead of starting over.
- Keep it short: only what an agent would get wrong without it. Do not list every file or directory, and do not add generic advice such as writing tests or handling errors.
- If README.md, CLAUDE.md, AGENTS.md, .cursorrules, .cursor/rules/ or .github/copilot-instructions.md exist, carry over the parts that matter.
- Do not invent sections or facts the repository does not support.`;

function filename(path) {
  return path.split("/").at(-1).replace(/\.md$/, "");
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
const SIDEBAR_SESSION_LIMIT = 200;

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
  const pageLinks = document.getElementById("page-links");
  const pageRoot = document.getElementById("page");
  const chatPane = document.getElementById("chat-pane");
  const agentsRoot = document.getElementById("agents");
  const roadmapRoot = document.getElementById("roadmap");
  const specRoot = document.getElementById("spec");
  const runNotice = document.getElementById("run-notice");
  const chatRoot = document.getElementById("chat");
  const approvalsRoot = document.getElementById("approvals");
  const form = document.getElementById("prompt-form");
  const input = document.getElementById("prompt");
  const promptError = document.getElementById("prompt-error");
  const attachmentList = document.getElementById("prompt-attachments");
  const attachmentPicker = document.getElementById("attachment-picker");
  const attachButton = document.getElementById("attach-button");
  const pendingFiles = [];
  const promptAttachments = [];
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
  const approvals = createApprovals({ client: boundary });
  const specView = createSpecView({ client: boundary });
  const transcript = createTranscript();
  let commandEntry = null;
  const board = createAgentBoard();
  const readSpecFiles = () => typeof boundary.specFiles === "function"
    ? boundary.specFiles().catch(() => ({ paths: [] }))
    : Promise.resolve({ paths: [] });
  let [project, initialSessions, roadmapReply, settingsReply, healthReply, conversationReply, initialSpecRuns, initialSpecFiles] = await Promise.all([
    boundary.project(),
    boundary.sessions(SIDEBAR_SESSION_LIMIT),
    boundary.file("docs/roadmap.json"),
    boundary.settings(),
    boundary.health().catch(() => null),
    boundary.conversationStats().catch(() => ({ conversation: null })),
    (boundary.specRuns?.() ?? Promise.resolve([])).catch(() => []),
    readSpecFiles(),
  ]);
  const providerRows = (settingsReply?.settings?.providers ?? []).map((row) => ({ ...row }));
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
  let rememberedPlanMode = null;
  let sessions = initialSessions;
  let currentSessionId = conversationReply?.conversation?.session_id ?? null;
  transcript.setSessionId(currentSessionId);
  let conversation = conversationReply?.conversation ?? null;
  let roadmap = renderRoadmap(parseRoadmap(roadmapReply.text));
  let specRuns = initialSpecRuns;
  let allSpecPaths = [...new Set([
    ...roadmap.phases.flatMap((phase) => phase.items.flatMap((item) => specPaths(item))),
    ...(Array.isArray(initialSpecFiles?.paths) ? initialSpecFiles.paths.filter((path) => typeof path === "string") : []),
  ])];
  const graphTitles = new Map();
  const graphPhaseLoaded = new Set();
  const graphTitlesReady = new Map();
  const expandedDonePhases = new Set();
  const expandedDoneGroups = new Set();
  const graphId = (path) => path.split("/").at(-1).replace(/\.md$/, "").split("-", 1)[0];
  const knownGraphPaths = () => [...new Set(allSpecPaths.flatMap((path) => [path, ...followUpsFor(path, allSpecPaths)]))];
  for (const path of knownGraphPaths()) graphTitles.set(graphId(path), "");
  const settingsPane = element(document, "section", { className: "settings-pane" });
  const settingsSections = element(document, "nav", { className: "settings-sections" });
  const settingsContent = element(document, "div", { className: "settings-content" });
  append(settingsPane, element(document, "h1", { text: "Settings" }), settingsSections, settingsContent);
  const changesPane = element(document, "section", { className: "changes-pane" });
  let promptFailure = "";
  let route;
  const conversationUsage = element(document, "p", { className: "conversation-usage" });
  append(form, conversationUsage);

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

  function showSettings(section) {
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
          } catch {
            notice.textContent = "Could not update key.";
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
          const save = element(document, "button", { className: "agent-save", text: "Save" });
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
    replace(
      settingsContent,
      element(document, "h2", { text: "General" }),
      settingsTable(["Setting", "Value", "Scope"], rows),
    );
  }

  async function showChanges() {
    replace(
      changesPane,
      element(document, "h1", { text: "Changes" }),
      element(document, "p", { className: "changes-loading", text: "Loading changes…" }),
    );
    let reply;
    try {
      reply = await boundary.changes();
    } catch (error) {
      replace(
        changesPane,
        element(document, "h1", { text: "Changes" }),
        element(document, "p", {
          className: "error",
          text: error instanceof Error ? error.message : String(error),
        }),
      );
      return;
    }

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
      changesPane.append(warning);
    }

    async function revert(payload) {
      try {
        await boundary.revertChanges(payload);
        await showChanges();
      } catch (error) {
        if (error?.status === 409 && Array.isArray(error.paths) && error.paths.length > 0) {
          showConflict(error.paths, payload);
        } else {
          changesPane.append(element(document, "p", {
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
    const children = [element(document, "h1", { text: "Changes" })];
    if (files.length === 0 && worktrees.length === 0) {
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
        append(row, element(document, "pre", { text: worktree.diff ?? "" }));
        const apply = element(document, "button", { text: "Apply" });
        apply.type = "button";
        listen(apply, "click", () => worktreeAction(worktree.name, "applyWorktree", row));
        const discard = element(document, "button", { text: "Discard" });
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
        append(row, element(document, "pre", { text: file.diff ?? "" }));
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
    replace(changesPane, ...children);
  }

  function showPage(nextRoute) {
    route = nextRoute;
    if (route.page === "settings") {
      showSettings(route.section);
    } else if (route.page === "changes") {
      void showChanges();
    }
    const pane = route.page === "settings" ? settingsPane
      : route.page === "changes" ? changesPane : chatPane;
    replace(pageRoot, pane);
  }

  function navigate(nextRoute, { updateFragment = true } = {}) {
    const next = parseRoute(formatRoute(nextRoute));
    showPage(next);
    rememberRoute(global, next);
    if (updateFragment && global.location) {
      global.location.hash = formatRoute(next);
    }
  }

  const sectionLinks = ["general", "models", "mcp", "skills", "plugins", "agents", "hooks", "ceiling", "trust", "inventory"].map((section) => {
    const link = element(document, "a", {
      text: section[0].toUpperCase() + section.slice(1),
    });
    link.href = formatRoute({ page: "settings", section });
    listen(link, "click", (event) => {
      event.preventDefault();
      navigate({ page: "settings", section });
    });
    return link;
  });
  replace(settingsSections, ...sectionLinks);

  const links = PAGES.filter((page) => page === "settings" || page === "changes").map((page) => {
    const pageRoute = { page, section: "" };
    const link = element(document, "a", {
      text: page[0].toUpperCase() + page.slice(1),
    });
    link.href = formatRoute(pageRoute);
    listen(link, "click", (event) => {
      event.preventDefault();
      navigate(pageRoute);
    });
    return link;
  });
  replace(pageLinks, ...links);
  listen(homeLink, "click", () => navigate({ page: "chat", section: "" }));

  const projectsRoot = element(document, "section", { className: "projects" });
  function showTranscript() {
    renderTranscript(document, chatRoot, transcript.model, parseMarkdown);
  }

  async function openSession(runId) {
    const previous = [...transcript.model];
    const previousId = currentSessionId;
    transcript.model.length = 0;
    currentSessionId = runId;
    transcript.setSessionId(currentSessionId);
    showTranscript();
    clearConversationAgents();
    try {
      await boundary.openSession(runId);
      await refreshConversation();
      navigate({ page: "chat", section: "" });
    } catch (error) {
      transcript.model.splice(0, transcript.model.length, ...previous);
      currentSessionId = previousId;
      transcript.setSessionId(currentSessionId);
      showTranscript();
      await refreshConversation();
      throw error;
    }
  }

  function showProjects() {
    const groups = [];
    for (const group of projectGroups(sessions, project.repo_root)) {
      const current = group.root === project.repo_root;
      const section = element(document, "details", {
        className: `project ${current ? "openable" : "unavailable"}`,
      });
      section.open = current;
      append(
        section,
        element(document, "summary", {
          text: current ? project.name : projectName(group.root),
        }),
      );
      if (!current) {
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
        if (!current) {
          const unavailable = element(document, "p", {
            className: "session-link unavailable",
            text: label,
          });
          unavailable.setAttribute("title", label);
          append(section, unavailable);
          continue;
        }
        const button = element(document, "button", {
          className: "session-link",
          text: label,
        });
        button.setAttribute("title", label);
        button.type = "button";
        listen(button, "click", async () => {
          await openSession(session.run_id);
        });
        append(section, button);
      }
      groups.push(section);
    }
    replace(projectsRoot, ...groups);
  }
  showProjects();
  async function startNewChat() {
    clearConversationAgents();
    try {
      await boundary.newSession();
      transcript.model.length = 0;
      currentSessionId = null;
      transcript.setSessionId(null);
      showTranscript();
      conversation = null;
      currentMode = launchMode;
      rememberedPlanMode = null;
      showConversationUsage();
      showAgents();
      promptFailure = "";
      showPromptError();
      navigate({ page: "chat", section: "" });
      input.value = "";
    } catch {
      await refreshConversation();
      promptFailure = "Could not start a new chat.";
      showPromptError();
    }
  }
  listen(newChat, "click", startNewChat);
  replace(sidebar, homeLink, newChat, pageLinks, projectsRoot);

  function toggleFold(className, button, label) {
    const classes = shell.className.split(" ");
    const folded = classes.includes(className);
    shell.className = folded
      ? classes.filter((name) => name !== className).join(" ")
      : [...classes, className].join(" ");
    button.textContent = folded ? `Hide ${label}` : `Show ${label}`;
  }

  listen(sidebarToggle, "click", () => {
    toggleFold("sidebar-folded", sidebarToggle, "sidebar");
  });
  listen(railToggle, "click", () => toggleFold("rail-folded", railToggle, "status"));

  const fragment = typeof global.location?.hash === "string" ? global.location.hash : "";
  showPage(fragment ? parseRoute(fragment) : storedRoute(global));
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

  let selectedSpec = null;
  let specActionError = "";

  function workflow(item, phase, itemIndex) {
    const paths = specPaths(item);
    const path = paths[0] ?? null;
    const plans = specRuns.filter((run) => run.kind === "plan" && run.phase === phase.id && run.item === itemIndex);
    const plan = plans[plans.length - 1] ?? null;
    if (path === null) {
      if (!plan) return { step: "unplanned", plan };
      if (plan.state === "running") return { step: "planning", plan };
      if (plan.state === "stopped" || plan.state === "failed") {
        const reason = plan.stopped_reason === "budget_turns" ? "turn limit" : plan.stopped_reason || plan.error || "unknown reason";
        return { step: `planner stopped: ${reason}`, plan };
      }
      if (plan.state === "finished" && plan.created?.length === 0) return { step: "planner wrote no spec", plan };
      if (plan.state === "finished" && plan.created?.length > 1) return { step: "planner wrote several specs", plan };
      return { step: "unplanned", plan };
    }
    const run = specRuns.find((candidate) => candidate.kind === "implement" && candidate.spec === path);
    if (!run) return { step: "planned", run: null };
    if (run.committed) return { step: "committed", run };
    if (run.review?.verdict === "running") return { step: "in review", run };
    if (run.review) return { step: "reviewed", run };
    return { step: run.state === "running" ? "running" : "ran", run };
  }

  function actionButton(label, callback) {
    const button = element(document, "button", { text: label });
    listen(button, "click", callback);
    return button;
  }

  async function refreshSpecState() {
    const [reply, runs, specFilesReply] = await Promise.all([
      boundary.file("docs/roadmap.json"),
      boundary.specRuns(),
      readSpecFiles(),
    ]);
    if (typeof reply?.text === "string") roadmap = renderRoadmap(parseRoadmap(reply.text));
    specRuns = runs;
    allSpecPaths = [...new Set([
      ...roadmap.phases.flatMap((phase) => phase.items.flatMap((item) => specPaths(item))),
      ...(Array.isArray(specFilesReply?.paths) ? specFilesReply.paths.filter((path) => typeof path === "string") : []),
    ])];
    graphTitles.clear();
    for (const path of knownGraphPaths()) graphTitles.set(graphId(path), "");
    graphPhaseLoaded.clear();
    renderRoadmapUI();
    if (selectedSpec) {
      const phase = roadmap.phases.find((value) => value.id === selectedSpec.phase.id);
      const item = phase?.items[selectedSpec.index] ?? selectedSpec.item;
      selectedSpec = { ...selectedSpec, phase: phase ?? selectedSpec.phase, item };
      selectedSpec.result = await specView.open(item, { specPaths: allSpecPaths });
      showSpec(selectedSpec.result, item, selectedSpec.phase, selectedSpec.index);
    }
  }

  async function runSpecAction(action, { newConversation = false } = {}) {
    specActionError = "";
    if (newConversation) clearConversationAgents();
    try {
      await action();
      if (newConversation) await refreshConversation();
      await refreshSpecState();
    } catch (error) {
      if (newConversation) await refreshConversation();
      specActionError = error?.message || "spec action failed";
      if (selectedSpec) showSpec(selectedSpec.result, selectedSpec.item, selectedSpec.phase, selectedSpec.index);
    }
  }

  function showSpec(result, item = null, phase = null, itemIndex = -1) {
    const children = [];
    if (item && phase) {
      const current = workflow(item, phase, itemIndex);
      const panel = element(document, "section", { className: "spec-workflow" });
      append(panel, element(document, "p", { className: "spec-step", text: current.step }));
      const actions = element(document, "div", { className: "spec-actions" });
      const run = current.run;
      if (current.step === "unplanned" || current.step.startsWith("planner stopped:") || current.step.startsWith("planner wrote ")) append(actions, actionButton("Plan", () => runSpecAction(() => boundary.planSpec(phase.id, itemIndex), { newConversation: true })));
      if (current.step === "planning" && current.plan) append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(current.plan.session_id))));
      if (current.plan && current.step !== "planning" && current.step !== "unplanned") append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(current.plan.session_id))));
      if (current.step === "planned" && result.specs[0]) append(actions, actionButton("Run", () => runSpecAction(() => boundary.runSpec(result.specs[0].path), { newConversation: true })));
      if (current.step === "running" && run) append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(run.session_id))));
      if (current.step === "ran" && run) {
        append(actions, actionButton("Review", () => runSpecAction(() => boundary.reviewSpec(run.session_id), { newConversation: true })));
        append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(run.session_id))));
      }
      if (current.step === "in review" && run?.review?.session_id) append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(run.review.session_id))));
      if (current.step === "reviewed" && run) {
        append(actions, element(document, "p", { text: run.review.verdict }));
        if (run.review.follow_ups?.length) append(actions, element(document, "p", { text: run.review.follow_ups.join(", ") }));
        if (["passed", "follow-ups"].includes(run.review.verdict)) {
          const message = element(document, "input");
          message.value = run.commit_message || "";
          message.setAttribute("aria-label", "Commit message");
          append(actions, message, actionButton("Commit…", () => runSpecAction(() => boundary.commitSpec(run.session_id, message.value))));
        }
        else append(actions, actionButton("Review", () => runSpecAction(() => boundary.reviewSpec(run.session_id), { newConversation: true })));
        append(actions, actionButton("Open", () => runSpecAction(() => boundary.openSession(run.review.session_id))));
      }
      if (current.step === "committed" && run) append(actions, element(document, "p", { text: run.committed.sha }));
      if (specActionError) append(panel, element(document, "p", { className: "error", text: specActionError }));
      append(panel, actions);
      children.push(panel);
    }
    for (const spec of result.specs) {
      children.push(element(document, "h3", { text: filename(spec.path) }));
      children.push(element(document, "pre", { text: spec.text }));
    }
    if (result.report) {
      children.push(element(document, "h3", { text: "Report" }));
      children.push(element(document, "pre", { text: result.report.text }));
    }
    if (result.followUps.length > 0) {
      const list = element(document, "ul", { className: "follow-ups" });
      for (const followUp of result.followUps) {
        append(list, element(document, "li", { text: filename(followUp.path) }));
      }
      children.push(element(document, "h3", { text: "Follow-ups" }), list);
    }
    if (result.error) {
      children.push(element(document, "p", { className: "error", text: result.error }));
    }
    replace(specRoot, ...children);
  }

  function roadmapItem(item, phase, itemIndex) {
    const state = workflow(item, phase, itemIndex).step;
    const button = element(document, "button", {
      className: "roadmap-item",
      text: `${item.title} · ${state}`,
    });
    listen(button, "click", async () => {
      selectedSpec = { item, phase, index: itemIndex, result: await specView.open(item, { specPaths: allSpecPaths }) };
      specActionError = "";
      showSpec(selectedSpec.result, item, phase, itemIndex);
    });
    return button;
  }

  async function loadGraphTitles(phase) {
    if (graphPhaseLoaded.has(phase.id)) return graphTitlesReady.get(phase.id);
    graphPhaseLoaded.add(phase.id);
    const paths = phase.items.flatMap((item) => {
      const spec = specPaths(item)[0];
      return spec ? [spec, ...followUpsFor(spec, allSpecPaths)] : [];
    });
    const promise = Promise.all(paths.map(async (path) => {
      try {
        const reply = await boundary.file(path);
        const firstLine = reply.text.split(/\r?\n/, 1)[0];
        const match = /^#\s+[^—]+—\s+(.+)$/.exec(firstLine);
        if (match) graphTitles.set(graphId(path), match[1].trim());
      } catch {}
    })).then(() => {
      renderRoadmapUI();
    });
    graphTitlesReady.set(phase.id, promise);
    return promise;
  }

  function openGraphSpec(item, phase, itemIndex) {
    selectedSpec = { item, phase, index: itemIndex, result: null };
    return specView.open(item, { specPaths: allSpecPaths }).then((result) => {
      selectedSpec = { ...selectedSpec, result };
      specActionError = "";
      showSpec(result, item, phase, itemIndex);
    });
  }

  function graphView(phase) {
    const titles = new Map(graphTitles);
    const followUps = new Map();
    const states = new Map();
    const itemById = new Map();
    phase.items.forEach((item, index) => {
      const paths = specPaths(item);
      const id = paths.length ? graphId(paths[0]) : `item-${index}`;
      itemById.set(id, { item, index });
      followUps.set(id, (paths.length ? followUpsFor(paths[0], allSpecPaths) : []).map((path) => ({
        id: graphId(path), path,
        title: graphTitles.get(graphId(path)) || filename(path).replace(/^[^-]+-/, ""),
      })));
      states.set(id, workflow(item, phase, index).step);
    });
    const graph = layoutPhase(phase.items, { followUps, titles, states });
    if (graph.error) {
      return [element(document, "p", { className: "error roadmap-graph-error", text: graph.error }),
        ...phase.items.map((item, index) => roadmapItem(item, phase, index))];
    }
    const container = element(document, "div", { className: "roadmap-graph" });
    for (const [rowIndex, row] of graph.rows.entries()) {
      const boxes = element(document, "div", { className: "roadmap-graph-row" });
      for (const node of row) {
        if (node.summary) {
          append(boxes, element(document, "div", { className: "roadmap-done-summary", text: node.title }));
          continue;
        }
        const widthClass = row.length < 2 || node.group ? "roadmap-box-wide" : "";
        const box = element(document, "div", { className: ["roadmap-box", widthClass, node.group ? "roadmap-group" : `roadmap-state-${node.state.replaceAll(" ", "-")}`].filter(Boolean).join(" ") });
        if (node.group) append(box, element(document, "div", { className: "roadmap-group-heading", text: "parallel" }));
        const entries = node.group ?? [node];
        for (const entry of entries) {
          const bound = itemById.get(entry.id);
          const button = element(document, "button", { className: "roadmap-box-button", text: `${entry.id}  ${entry.title}` });
          if (bound) {
            button.setAttribute("title", bound.item.title);
            listen(button, "click", () => openGraphSpec(bound.item, phase, bound.index));
          }
          append(box, button);
          for (const followUp of entry.followUps ?? []) {
            const followButton = element(document, "button", { className: "roadmap-follow-up", text: `${followUp.id}  ${followUp.title}` });
            listen(followButton, "click", () => openGraphSpec({ title: followUp.title, spec: followUp.path }, phase, bound?.index ?? -1));
            append(box, followButton);
          }
          for (const tag of entry.tags ?? []) append(box, element(document, "span", { className: "roadmap-tag", text: tag }));
        }
        boxes.append(box);
      }
      append(container, boxes);
      if (rowIndex < graph.rows.length - 1) {
        const connectorRow = element(document, "div", { className: "roadmap-connector-row" });
        for (const edge of graph.edges) {
          const from = graph.rows.flat().find((node) => node.id === edge.from);
          const to = graph.rows.flat().find((node) => node.id === edge.to);
          if (!from || !to || from.row > rowIndex || to.row <= rowIndex) continue;
          const isTargetRow = to.row === rowIndex + 1;
          const edgeClass = isTargetRow
            ? `roadmap-edge roadmap-edge-from-${from.column}-to-${to.column}`
            : `roadmap-edge roadmap-edge-through roadmap-edge-from-${from.column}`;
          append(connectorRow, element(document, "div", { className: edgeClass }));
        }
        append(container, connectorRow);
      }
    }
    return graph.summary
      ? [element(document, "div", { className: "roadmap-done-summary", text: graph.summary }), container]
      : [container];
  }

  function phaseDependencies(phase, index) {
    return phase.after ?? (index > 0 ? [roadmap.phases[index - 1].id] : []);
  }

  function donePhaseChains() {
    const dependents = new Map(roadmap.phases.map((phase) => [phase.id, []]));
    roadmap.phases.forEach((phase, index) => {
      for (const dependency of phaseDependencies(phase, index)) {
        dependents.get(dependency)?.push(phase.id);
      }
    });
    const chains = [];
    for (let index = 0; index < roadmap.phases.length;) {
      const first = roadmap.phases[index];
      const members = [first];
      let nextIndex = index + 1;
      while (first.status === "done" && nextIndex < roadmap.phases.length) {
        const previous = members.at(-1);
        const next = roadmap.phases[nextIndex];
        const dependencies = phaseDependencies(next, nextIndex);
        if (
          next.status !== "done"
          || dependencies.length !== 1
          || dependencies[0] !== previous.id
          || dependents.get(previous.id)?.length !== 1
          || dependents.get(previous.id)?.[0] !== next.id
        ) break;
        members.push(next);
        nextIndex += 1;
      }
      if (members.length > 1) chains.push({ index, members });
      index += members.length;
    }
    return chains;
  }

  function visibleRoadmapPhases() {
    const chains = donePhaseChains();
    const sourceToVisible = new Map();
    const groups = new Map();
    for (const chain of chains) {
      const first = chain.members[0];
      const last = chain.members.at(-1);
      const key = `${first.id}..${last.id}`;
      if (expandedDoneGroups.has(key)) continue;
      groups.set(chain.index, { ...chain, key, id: `done-${key}` });
      for (const phase of chain.members) sourceToVisible.set(phase.id, `done-${key}`);
    }
    const visible = [];
    for (let index = 0; index < roadmap.phases.length;) {
      const group = groups.get(index);
      if (group) {
        const first = group.members[0];
        const total = group.members.reduce((sum, phase) => sum + phase.progress.total, 0);
        visible.push({
          id: group.id,
          name: `✓ ${first.id}–${group.members.at(-1).id} · ${group.members.length} phases`,
          status: "done",
          progress: { done: total, total },
          items: [],
          after: phaseDependencies(first, index)
            .filter((id) => !group.members.some((phase) => phase.id === id))
            .map((id) => sourceToVisible.get(id) ?? id),
          collapsedMembers: group.members,
          groupKey: group.key,
        });
        index += group.members.length;
        continue;
      }
      const phase = roadmap.phases[index];
      visible.push({
        ...phase,
        after: phaseDependencies(phase, index).map((id) => sourceToVisible.get(id) ?? id),
      });
      index += 1;
    }
    return visible;
  }

  function phaseMark(status) {
    return status === "done" ? "✓" : status === "in_progress" ? "◐" : "○";
  }

  function phaseBox(node) {
    const phase = node.phase;
    if (phase.collapsedMembers) {
      const summary = element(document, "button", {
        className: "roadmap-phase-chain-summary",
        text: phase.name,
      });
      listen(summary, "click", () => {
        expandedDoneGroups.add(phase.groupKey);
        renderRoadmapUI();
      });
      return summary;
    }
    const box = element(document, "section", {
      className: `roadmap-phase-box roadmap-phase-${phase.status}`,
    });
    box.setAttribute("data-phase-id", phase.id);
    const header = element(document, "button", {
      className: "roadmap-phase-header",
      text: `${phaseMark(phase.status)} ${phase.id} ${phase.name} · ${phase.progress.done}/${phase.progress.total}`,
    });
    listen(header, "click", () => {
      if (phase.status !== "done") return;
      if (expandedDonePhases.has(phase.id)) expandedDonePhases.delete(phase.id);
      else expandedDonePhases.add(phase.id);
      renderRoadmapUI();
    });
    append(box, header);
    if (node.tags.length) {
      append(box, ...node.tags.map((tag) => element(document, "span", { className: "roadmap-tag", text: tag })));
    }
    if (phase.status !== "done" || expandedDonePhases.has(phase.id)) {
      const body = element(document, "div", { className: "roadmap-phase-tasks" });
      append(body, ...graphView(phase));
      append(box, body);
    }
    return box;
  }

  function renderRoadmapUI() {
    const graph = layoutRoadmap(visibleRoadmapPhases(), { maxColumns: 1 });
    if (graph.error) {
      replace(roadmapRoot, element(document, "p", { className: "error roadmap-graph-error", text: graph.error }));
      return;
    }
    const container = element(document, "div", { className: "roadmap-overview" });
    for (const [rowIndex, row] of graph.rows.entries()) {
      const boxes = element(document, "div", { className: "roadmap-phase-row" });
      for (const node of row) {
        if (node.group) {
          const group = element(document, "div", { className: "roadmap-phase-group" });
          append(group, ...node.group.map((member) => phaseBox(member)));
          append(boxes, group);
        } else {
          append(boxes, phaseBox(node));
        }
      }
      append(container, boxes);
      if (rowIndex < graph.rows.length - 1) {
        const connectorRow = element(document, "div", { className: "roadmap-connector-row" });
        for (const edge of graph.edges) {
          const from = graph.rows.flat().find((node) => node.id === edge.from);
          const to = graph.rows.flat().find((node) => node.id === edge.to);
          if (!from || !to || from.row > rowIndex || to.row <= rowIndex) continue;
          const isTargetRow = to.row === rowIndex + 1;
          const edgeClass = isTargetRow
            ? `roadmap-edge roadmap-edge-from-${from.column}-to-${to.column}`
            : `roadmap-edge roadmap-edge-through roadmap-edge-from-${from.column}`;
          append(connectorRow, element(document, "div", { className: edgeClass }));
        }
        append(container, connectorRow);
      }
    }
    replace(roadmapRoot, container);
    for (const phase of roadmap.phases) {
      if (phase.status !== "done" || expandedDonePhases.has(phase.id)) void loadGraphTitles(phase);
    }
  }
  renderRoadmapUI();

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
    const provider = conversation?.provider;
    const model = conversation?.model;
    const modelText = typeof model === "string" && model
      ? `${typeof provider === "string" && provider ? `${provider} / ` : ""}${model}`
      : "";
    const mode = permittedModes.includes(conversation?.mode) ? conversation.mode : "";
    const effort = typeof conversation?.effort === "string" && conversation.effort
      ? `Effort ${conversation.effort}`
      : "";
    conversationUsage.textContent = [
      modelText,
      effort,
      mode && `Mode ${mode}`,
      conversation?.goal && `Goal ${conversation.goal.phase} ${conversation.goal.rounds}/${conversation.goal.max_rounds}`,
      Number.isInteger(context?.used_tokens) && Number.isInteger(context?.budget_tokens)
        ? `Context ${context.used_tokens} / ${context.budget_tokens} tokens`
        : "",
      usageText(conversation?.usage),
    ].filter(Boolean).join(" · ");
    conversationUsage.className = `conversation-usage${mode === "plan" ? " plan" : ""}`;
  }

  function showAgents() {
    const rows = [...board.rows];
    for (const agent of conversation?.agents ?? []) {
      if (!rows.some((row) => row.agentId === agent.agent_id)) {
        rows.push({
          agentId: agent.agent_id,
          parentAgentId: agent.parent_agent_id ?? null,
          name: agent.name,
          state: agent.state ?? "done",
          tool: "",
        });
      }
    }
    const visibleRows = rows.filter((row) => ["running", "waiting"].includes(row.state));
    const usageByAgent = new Map(
      (conversation?.agents ?? []).map((agent) => [agent.agent_id, agent]),
    );
    if (visibleRows.length === 0) {
      replace(agentsRoot, element(document, "p", { text: "Nothing is running." }));
      return;
    }
    const byId = new Map(visibleRows.map((row) => [row.agentId, row]));
    const nodes = new Map(visibleRows.map(({ agentId, name, state, tool }) => [agentId, element(document, "div", {
      className: "agent-row",
      text: [name, state, tool, usageText(usageByAgent.get(agentId))]
        .filter(Boolean).join(" · "),
    })]));
    const children = new Map();
    const roots = [];
    for (const row of visibleRows) {
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
    for (const row of visibleRows) {
      if (!["running", "waiting", "paused"].includes(row.state)) continue;
      const node = nodes.get(row.agentId);
      const controls = element(document, "div", { className: "agent-controls" });
      const controlError = element(document, "p", {
        className: "agent-control-error",
        text: row.controlError ?? "",
      });
      const sendControl = async (action, text) => {
        try {
          const reply = await boundary.controlAgent(row.agentId, action, text);
          row.state = reply.state;
          row.controlError = "";
          showAgents();
        } catch (error) {
          row.controlError = error.message || "Agent control failed.";
          showAgents();
        }
      };
      const pauseOrResume = element(document, "button", {
        text: row.state === "paused" ? "Resume" : "Pause",
      });
      pauseOrResume.type = "button";
      listen(pauseOrResume, "click", () => sendControl(
        row.state === "paused" ? "resume" : "pause",
      ));
      const redirectButton = element(document, "button", { text: "Redirect" });
      redirectButton.type = "button";
      listen(redirectButton, "click", () => {
        if (node.children.some((child) => child.className === "agent-redirect")) return;
        const redirect = element(document, "div", { className: "agent-redirect" });
        const redirectInput = element(document, "input");
        redirectInput.type = "text";
        redirectInput.setAttribute?.("aria-label", `Redirect ${row.name}`);
        const send = element(document, "button", { text: "Send" });
        send.type = "button";
        const close = () => {
          const parent = redirect.parentNode;
          if (parent?.removeChild) {
            parent.removeChild(redirect);
          } else if (parent) {
            parent.children = parent.children.filter(
              (child) => child !== redirect,
            );
          }
        };
        const submit = () => {
          if (!redirectInput.value.trim()) {
            row.controlError = "Redirect text must not be blank.";
            showAgents();
            return;
          }
          void sendControl("redirect", redirectInput.value);
        };
        listen(send, "click", submit);
        listen(redirectInput, "keydown", (event) => {
          if (lookup(keymap, event, { platform }) === "cancel") {
            event.preventDefault();
            close();
          } else if (event.key === "Enter") {
            event.preventDefault();
            submit();
          }
        });
        append(redirect, redirectInput, send);
        append(node, redirect);
        redirectInput.focus?.();
      });
      const stop = element(document, "button", { text: "Stop" });
      stop.type = "button";
      listen(stop, "click", () => sendControl("stop"));
      append(controls, pauseOrResume, redirectButton, stop);
      append(node, controls, controlError);
    }
    replace(agentsRoot, ...roots);
  }

  function clearConversationAgents() {
    board.clear();
    conversation = null;
    showConversationUsage();
    showAgents();
  }

  showConversationUsage();
  showAgents();

  async function refreshConversation() {
    try {
      const reply = await boundary.conversationStats();
      conversation = reply?.conversation ?? null;
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
    for (const [id, question] of approvals.state) {
      const row = element(document, "section", { className: `approval ${question.state}` });
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
      append(
        row,
        element(document, "p", {
          text: `${question.operation}: ${question.target} — ${question.details}`,
        }),
      );
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
        const deny = listen(
          element(document, "button", { text: "Deny" }),
          "click",
          async () => {
            await approvals.answer(id, false, "");
            showApprovals();
          },
        );
        deny.type = "button";
        append(row, deny);
      }
      children.push(row);
    }
    replace(approvalsRoot, ...children);
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
            }
            promptFailure = "";
            await perform(turn.accepted());
            try {
              sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT);
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
  }

  listen(attachButton, "click", () => attachmentPicker.click());
  listen(attachmentPicker, "change", () => {
    addFiles(Array.from(attachmentPicker.files ?? []));
    attachmentPicker.value = "";
  });
  listen(input, "paste", (event) => {
    const files = Array.from(event.clipboardData?.items ?? [])
      .filter((item) => item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
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
      return runCommand(command);
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
    pendingFiles.length = 0;
    showPendingAttachments();
    promptFailure = "";
    rememberPrompt(text);
    const actions = turn.submit(text, attachments);
    showPromptError();
    return perform(actions);
  });

  listen(input, "input", () => {
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
    return conversation?.model
      ? `${conversation.provider ? `${conversation.provider} / ` : ""}${conversation.model}`
      : "the current model";
  }

  async function chooseMode(mode) {
    try {
      const reply = await boundary.selectMode(mode);
      if (reply?.mode !== mode || !permittedModes.includes(reply.mode)) {
        throw new Error("Host refused the mode change.");
      }
      currentMode = reply.mode;
      conversation = { ...(conversation ?? {}), mode: reply.mode };
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
    const existingEffort = conversation?.effort;
    const choice = { name: provider, model: row.id };
    const keepEffort = Boolean(existingEffort && Array.isArray(efforts) && efforts.includes(existingEffort));
    if (keepEffort) choice.effort = existingEffort;
    try {
      await boundary.selectProvider(choice);
      conversation = { ...(conversation ?? {}), provider, model: row.id };
      if (keepEffort) conversation.effort = existingEffort;
      else delete conversation.effort;
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
    const existingEffort = conversation?.effort;
    const choice = { name: provider, model };
    const keepEffort = Boolean(existingEffort && Array.isArray(efforts) && efforts.includes(existingEffort));
    if (keepEffort) choice.effort = existingEffort;
    try {
      await boundary.selectProvider(choice);
      conversation = { ...(conversation ?? {}), provider, model };
      if (keepEffort) conversation.effort = existingEffort;
      else delete conversation.effort;
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
      conversation = { ...(conversation ?? {}), provider, model, effort };
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
    const provider = conversation?.provider;
    const model = conversation?.model;
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
        current: value === conversation.effort,
      }));
      showPicker({
        title: `Effort · ${provider} / ${model}`,
        rows,
        initialIndex: Math.max(0, rows.findIndex((row) => row.current)),
        onChoose: (row) => chooseEffort(provider, model, row.id),
        onCancel: () => {
          closePicker();
          answerCommand(`Kept effort as ${conversation?.effort || "default"}.`);
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
        current: conversation?.provider === provider && conversation?.model === model.id,
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
      current: provider.name === conversation?.provider,
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
        answerCommand(conversation?.provider
          ? `Kept model as ${currentModelLabel()}.`
          : `Kept the default provider as ${providers[0]?.name ?? "unset"}.`);
      },
    });
  }

  async function openDefaultModelPicker() {
    const currentProvider = conversation?.provider;
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
    commandMatches = matchCommands(input.value);
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
      }
      return;
    }
    if (fileFragment !== null) return;
    if (await handlePromptHistory(event)) return;
    if (lookup(keymap, event, { platform }) === "submit") {
      event.preventDefault();
      await form.requestSubmit();
    }
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
        await openModelPicker(provider, model);
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
      answerCommand(COMMANDS.map(({ name, aliases, description, argumentHint }) => (
        `/${name}${argumentHint ? ` ${argumentHint}` : ""} — ${description}`
        + (aliases.length > 0 ? ` (also ${aliases.map((alias) => `/${alias}`).join(", ")})` : "")
      )).join("\n"));
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
    if (entry.name === "plan") {
      if (args.length > 0) {
        answerCommand("Usage: /plan");
        return;
      }
      await togglePlanMode();
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
      await perform(turn.submit(INIT_PROMPT));
    }
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

  const requestSpecRefresh = leadingTrailingRefresh(async () => {
    await refreshSpecState().catch(() => {});
  });

  async function refreshSpecsForEvent(frame) {
    const type = frame.kind === "event" ? frame.payload?.type : "";
    if (!["RunFinished", "RunFailed", "GoalChanged"].includes(type)) return;
    const sessionId = frame.payload?.session_id;
    if (typeof sessionId !== "string" || !specRuns.some((run) =>
      run.session_id === sessionId || run.review?.session_id === sessionId
    )) return;
    await requestSpecRefresh();
  }

  const requestActivityRefresh = leadingTrailingRefresh(async () => {
    try {
      sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT);
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
      await approvals.onFrame(frame);
      showApprovals();
      return;
    }
    const frameSessionId = frame.kind === "event" ? frame.payload?.session_id : null;
    await refreshSpecsForEvent(frame);
    if (await refreshActivityForOtherSession(frameSessionId)) return;
    const previousLength = transcript.model.length;
    transcript.apply(frame);
    if (
      frame.kind === "event" && frame.payload?.type === "HistoryMessage"
      && typeof frame.payload.record_id === "string"
      && transcript.model.length === previousLength + 1
    ) {
      transcript.model.at(-1).recordId = frame.payload.record_id;
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
      (event.type === "RunFinished" || event.type === "RunFailed") &&
      (healthReply?.runtime_run_id === null ||
        (typeof healthReply?.runtime_run_id === "string" &&
          healthReply.runtime_run_id.length > 0 &&
          healthReply.runtime_run_id === event.fields.run_id))
    ) {
      runNotice.textContent = "";
    }
    await perform(turn.event({ ...event, ...event.fields }));
    if (event.type === "RunFinished" || event.type === "RunFailed") {
      await refreshConversation();
    }
  }

  const subscription = boundary.events((frame) => onFrame(frame));
  showPromptError();
  return {
    approvals,
    client: boundary,
    onFrame,
    route: () => route,
    sidebar,
    specPaths: allSpecPaths,
    specView,
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
