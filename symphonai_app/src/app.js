import { createApprovals } from "./approvals.js";
import { createAgentBoard } from "./agents.js";
import { createClient } from "./client.js";
import { resolveHost } from "./host_handle.js";
import { decodeEvent } from "./protocol.js";
import { renderRoadmap, parseRoadmap, specPaths } from "./roadmap.js";
import { append, element, listen, renderTranscript, replace } from "./render.js";
import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "./route.js";
import { ceilingRows, generalRows, hookRows, inventoryRows, modelRows, rosterRows, serverRows, trustRows } from "./settings.js";
import { createSpecView } from "./spec_view.js";
import { createTranscript } from "./transcript.js";
import { createTurnState } from "./turn.js";

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
  const turn = createTurnState();
  const approvals = createApprovals({ client: boundary });
  const specView = createSpecView({ client: boundary });
  const transcript = createTranscript();
  const board = createAgentBoard();
  const [project, initialSessions, roadmapReply, settingsReply, healthReply, conversationReply] = await Promise.all([
    boundary.project(),
    boundary.sessions(SIDEBAR_SESSION_LIMIT),
    boundary.file("docs/roadmap.json"),
    boundary.settings(),
    boundary.health().catch(() => null),
    boundary.conversationStats().catch(() => ({ conversation: null })),
  ]);
  const providerRows = (settingsReply?.settings?.providers ?? []).map((row) => ({ ...row }));
  const providerControls = element(document, "div", { className: "provider-controls" });
  const providerLabel = element(document, "label", { text: "Provider " });
  const providerSelect = element(document, "select");
  const modelInput = element(document, "input");
  modelInput.placeholder = "Model (optional)";
  modelInput.ariaLabel = "Model (optional)";
  const modelList = element(document, "datalist");
  modelList.id = "provider-models";
  modelInput.setAttribute?.("list", modelList.id);
  const effortControls = element(document, "span", { className: "effort-controls" });
  let effortSelect = null;
  const modelStatus = element(document, "span", { className: "model-status" });
  const baseUrlInput = element(document, "input");
  baseUrlInput.placeholder = "Base URL (optional)";
  baseUrlInput.ariaLabel = "Base URL (optional)";
  const providerStatus = element(document, "span", { className: "provider-status" });
  function showProviderChoices() {
    const current = providerSelect.value;
    const keyed = providerRows.filter((row) => row.key_present);
    replace(providerSelect, ...keyed.map((row) => {
      const option = element(document, "option", { text: row.name });
      option.value = row.name;
      return option;
    }));
    providerSelect.value = keyed.some((row) => row.name === current) ? current : (keyed[0]?.name ?? "");
    providerSelect.disabled = keyed.length === 0;
    providerStatus.textContent = keyed.length === 0
      ? "Add an API key in Settings to choose a provider."
      : "Changes to an active chat apply to the next chat.";
  }
  showProviderChoices();
  let modelRequest = 0;
  let modelState = { state: "unknown", models: [], detail: "Choose a provider." };
  function showEfforts() {
    const previous = effortSelect?.value;
    const selected = modelState.models.find((model) => model.id === modelInput.value.trim());
    if (!selected || selected.efforts.length === 0) {
      effortSelect = null;
      replace(effortControls);
      return;
    }
    const label = element(document, "label", { text: "Effort " });
    effortSelect = element(document, "select");
    effortSelect.ariaLabel = "Effort";
    replace(effortSelect, ...selected.efforts.map((effort) => {
      const option = element(document, "option", { text: effort });
      option.value = effort;
      return option;
    }));
    effortSelect.value = selected.efforts.includes(previous) ? previous : selected.efforts[0];
    append(label, effortSelect);
    replace(effortControls, label);
  }
  function showModelState() {
    replace(modelList, ...modelState.models.map((model) => {
      const option = element(document, "option");
      option.value = model.id;
      return option;
    }));
    showEfforts();
    const typed = modelInput.value.trim();
    if (modelState.state === "available") {
      modelStatus.textContent = typed && !modelState.models.some((model) => model.id === typed)
        ? "Unavailable in the provider listing; this model will still be submitted."
        : `${modelState.models.length} models available.`;
      return;
    }
    modelStatus.textContent = modelState.detail || "Model listing unavailable.";
  }
  async function refreshModels() {
    const request = ++modelRequest;
    const provider = providerSelect.value;
    if (!provider) {
      modelState = { state: "unknown", models: [], detail: "Choose a provider." };
      showModelState();
      return;
    }
    modelStatus.textContent = "Loading models…";
    try {
      const baseUrl = baseUrlInput.value.trim();
      const reply = await boundary.models(provider, baseUrl || undefined);
      if (request !== modelRequest) return;
      const validModels = Array.isArray(reply?.models) && reply.models.every((model) =>
        model && typeof model.id === "string" && Array.isArray(model.efforts)
        && model.efforts.every((effort) => typeof effort === "string")
      );
      if (reply?.state === "available" && validModels) {
        modelState = { state: "available", models: reply.models, detail: "" };
      } else if (reply?.state === "unknown" && validModels) {
        modelState = { state: "unknown", models: [], detail: reply.detail ?? "" };
      } else {
        throw new Error("invalid model listing reply");
      }
    } catch {
      if (request !== modelRequest) return;
      modelState = { state: "unknown", models: [], detail: "Model listing unavailable." };
    }
    showModelState();
  }
  listen(providerSelect, "change", async () => {
    modelInput.value = "";
    baseUrlInput.value = "";
    await refreshModels();
  });
  listen(baseUrlInput, "change", refreshModels);
  listen(modelInput, "input", showModelState);
  append(providerLabel, providerSelect);
  append(
    providerControls,
    providerLabel,
    modelInput,
    baseUrlInput,
    modelList,
    effortControls,
    modelStatus,
    providerStatus,
  );
  append(form, providerControls);
  await refreshModels();
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
  const modeControls = element(document, "div", { className: "mode-controls" });
  const modeLabel = element(document, "label", { text: "Mode " });
  const modeSelect = element(document, "select");
  const modeStatus = element(document, "span", { className: "mode-status" });
  replace(modeSelect, ...permittedModes.map((mode) => {
    const option = element(document, "option", {
      text: mode === "plan" ? "plan · read only" : mode,
    });
    option.value = mode;
    return option;
  }));
  function showMode(message = "") {
    modeSelect.value = currentMode;
    modeControls.className = currentMode === "plan" ? "mode-controls plan" : "mode-controls";
    modeStatus.textContent = message || (
      currentMode === "plan" ? "Plan mode is read only."
        : currentMode === "ask" ? "Ask before changes."
          : "Allow changes within configured limits."
    );
  }
  listen(modeSelect, "change", async () => {
    const requested = modeSelect.value;
    modeSelect.disabled = true;
    try {
      const reply = await boundary.selectMode(requested);
      if (!permittedModes.includes(reply?.mode)) {
        throw new Error("host returned an invalid mode");
      }
      currentMode = reply.mode;
      showMode();
    } catch {
      showMode(`Mode change refused; still ${currentMode}.`);
    } finally {
      modeSelect.disabled = false;
    }
  });
  append(modeLabel, modeSelect);
  append(modeControls, modeLabel, modeStatus);
  append(form, modeControls);
  showMode();
  let sessions = initialSessions;
  let currentSessionId = null;
  let conversation = conversationReply?.conversation ?? null;
  const roadmap = renderRoadmap(parseRoadmap(roadmapReply.text));
  const allSpecPaths = roadmap.phases.flatMap((phase) =>
    phase.items.flatMap((item) => specPaths(item))
  );
  const settingsPane = element(document, "section", { className: "settings-pane" });
  const settingsSections = element(document, "nav", { className: "settings-sections" });
  const settingsContent = element(document, "div", { className: "settings-content" });
  append(settingsPane, element(document, "h1", { text: "Settings" }), settingsSections, settingsContent);
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
            if (providerRow) {
              providerRow.key_present = Boolean(value);
              showProviderChoices();
              await refreshModels();
            }
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

  function showPage(nextRoute) {
    route = nextRoute;
    if (route.page === "settings") {
      showSettings(route.section);
    }
    const pane = route.page === "settings" ? settingsPane : chatPane;
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

  const links = PAGES.filter((page) => page === "settings").map((page) => {
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
    renderTranscript(document, chatRoot, transcript.model);
    const rows = [...chatRoot.children];
    const children = [];
    for (const [index, entry] of transcript.model.entries()) {
      children.push(rows[index]);
      if (!currentSessionId || !entry.recordId || !["prompt", "text"].includes(entry.type)) {
        continue;
      }
      const button = element(document, "button", { className: "fork-message", text: "Fork here" });
      button.type = "button";
      listen(button, "click", async () => {
        const sourceId = currentSessionId;
        const previous = [...transcript.model];
        transcript.model.length = 0;
        showTranscript();
        try {
          const reply = await boundary.forkSession(sourceId, entry.recordId);
          currentSessionId = reply.run_id;
          try {
            sessions = await boundary.sessions(SIDEBAR_SESSION_LIMIT);
            showProjects();
          } catch {
            // The fork remains current if refreshing the sidebar fails.
          }
          navigate({ page: "chat", section: "" });
        } catch {
          transcript.model.splice(0, transcript.model.length, ...previous);
          showTranscript();
          promptFailure = "Fork failed.";
          showPromptError();
        }
      });
      children.push(button);
    }
    replace(chatRoot, ...children);
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
        );
        if (!current) {
          append(section, element(document, "p", {
            className: "session-link unavailable",
            text: label,
          }));
          continue;
        }
        const button = element(document, "button", {
          className: "session-link",
          text: label,
        });
        button.type = "button";
        listen(button, "click", async () => {
          const previous = [...transcript.model];
          const previousId = currentSessionId;
          transcript.model.length = 0;
          currentSessionId = session.run_id;
          showTranscript();
          try {
            await boundary.openSession(session.run_id);
            board.clear();
            conversation = null;
            showConversationUsage();
            showAgents();
            await refreshConversation();
            navigate({ page: "chat", section: "" });
          } catch (error) {
            transcript.model.splice(0, transcript.model.length, ...previous);
            currentSessionId = previousId;
            showTranscript();
            throw error;
          }
        });
        append(section, button);
      }
      groups.push(section);
    }
    replace(projectsRoot, ...groups);
  }
  showProjects();
  listen(newChat, "click", async () => {
    try {
      await boundary.newSession();
      transcript.model.length = 0;
      currentSessionId = null;
      showTranscript();
      conversation = null;
      currentMode = launchMode;
      showMode();
      board.clear();
      showConversationUsage();
      showAgents();
      promptFailure = "";
      showPromptError();
      navigate({ page: "chat", section: "" });
      input.value = "";
    } catch {
      promptFailure = "Could not start a new chat.";
      showPromptError();
    }
  });
  replace(sidebar, homeLink, newChat, projectsRoot, pageLinks);

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

  function showSpec(result) {
    const children = [];
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

  function roadmapItem(item) {
    const button = element(document, "button", {
      className: "roadmap-item",
      text: item.title,
    });
    listen(button, "click", async () => {
      showSpec(await specView.open(item, { specPaths: allSpecPaths }));
    });
    return button;
  }

  const roadmapChildren = [];
  let openedCurrentPhase = false;
  for (const phase of roadmap.phases) {
    const section = element(document, "details", { className: "roadmap-phase" });
    section.open = !openedCurrentPhase && phase.status !== "done";
    openedCurrentPhase ||= section.open;
    append(
      section,
      element(document, "summary", {
        text: `${phase.id} · ${phase.name} — ${phase.progress.done}/${phase.progress.total} · ${phase.status}`,
      }),
      ...phase.items.map(roadmapItem),
    );
    roadmapChildren.push(section);
  }
  replace(roadmapRoot, ...roadmapChildren);

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
    conversationUsage.textContent = [
      context && `Context ${context.used_tokens} / ${context.budget_tokens} tokens`,
      usageText(conversation?.usage),
    ].filter(Boolean).join(" · ");
  }

  function showAgents() {
    const rows = [...board.rows];
    for (const agent of conversation?.agents ?? []) {
      if (!rows.some((row) => row.agentId === agent.agent_id)) {
        rows.push({ agentId: agent.agent_id, parentAgentId: agent.parent_agent_id ?? null, name: agent.name, state: "done", tool: "" });
      }
    }
    const usageByAgent = new Map(
      (conversation?.agents ?? []).map((agent) => [agent.agent_id, agent]),
    );
    if (rows.length === 0) {
      replace(agentsRoot, element(document, "p", { text: "Nothing is running." }));
      return;
    }
    const byId = new Map(rows.map((row) => [row.agentId, row]));
    const nodes = new Map(rows.map(({ agentId, name, state, tool }) => [agentId, element(document, "div", {
      className: "agent-row",
      text: [name, state, tool, usageText(usageByAgent.get(agentId))]
        .filter(Boolean).join(" · "),
    })]));
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
  showConversationUsage();
  showAgents();

  async function refreshConversation() {
    try {
      const reply = await boundary.conversationStats();
      conversation = reply?.conversation ?? null;
      if (permittedModes.includes(conversation?.mode)) {
        currentMode = conversation.mode;
        showMode();
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
        const deny = listen(
          element(document, "button", { text: "Deny" }),
          "click",
          async () => {
            await approvals.answer(id, false, "");
            showApprovals();
          },
        );
        deny.type = "button";
        append(row, allow, deny);
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
        try {
          if (providerSelect.value) {
            const choice = { name: providerSelect.value };
            if (modelInput.value.trim()) choice.model = modelInput.value.trim();
            if (baseUrlInput.value.trim()) choice.base_url = baseUrlInput.value.trim();
            if (effortSelect?.value) choice.effort = effortSelect.value;
            await boundary.selectProvider(choice);
          }
          const reply = await boundary.prompt(action.text);
          if (reply?.conflict === true) {
            const active = await activeRuntimeRun();
            if (active) {
              promptFailure = "A run is already in progress.";
              await perform(turn.adopted(active.runId));
            } else {
              turn.rejected("the host refused a prompt it is no longer running");
              promptFailure = "The message was not sent. Send it again.";
            }
          } else {
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

  listen(form, "submit", (event) => {
    event.preventDefault();
    if (providerRows.length > 0 && !providerSelect.value) {
      promptFailure = "Add an API key in Settings before sending a message.";
      showPromptError();
      return;
    }
    const text = input.value;
    input.value = "";
    promptFailure = "";
    const actions = turn.submit(text);
    showPromptError();
    return perform(actions);
  });

  async function onFrame(frame) {
    const previousLength = transcript.model.length;
    transcript.apply(frame);
    if (
      frame.kind === "event" && frame.payload?.type === "HistoryMessage"
      && typeof frame.payload.record_id === "string"
      && transcript.model.length === previousLength + 1
    ) {
      transcript.model.at(-1).recordId = frame.payload.record_id;
    }
    board.apply(frame);
    showAgents();
    showTranscript();
    if (frame.kind === "approval_requested") {
      await approvals.onFrame(frame);
      showApprovals();
      return;
    }
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
