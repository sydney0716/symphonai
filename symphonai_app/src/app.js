import { createApprovals } from "./approvals.js";
import { createClient } from "./client.js";
import { resolveHost } from "./host_handle.js";
import { decodeEvent } from "./protocol.js";
import { renderRoadmap, parseRoadmap, specPaths } from "./roadmap.js";
import { append, element, listen, renderTranscript, replace } from "./render.js";
import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "./route.js";
import { generalRows, modelRows } from "./settings.js";
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
  return createClient({
    port: handshake.port,
    token: handshake.token,
    fetch: global.fetch.bind(global),
  });
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
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const pageLinks = document.getElementById("page-links");
  const pageRoot = document.getElementById("page");
  const roadmapPane = document.getElementById("roadmap-pane");
  const chatPane = document.getElementById("chat-pane");
  const roadmapRoot = document.getElementById("roadmap");
  const specRoot = document.getElementById("spec");
  const chatRoot = document.getElementById("chat");
  const approvalsRoot = document.getElementById("approvals");
  const form = document.getElementById("prompt-form");
  const input = document.getElementById("prompt");
  const promptError = document.getElementById("prompt-error");
  const turn = createTurnState();
  const approvals = createApprovals({ client: boundary });
  const specView = createSpecView({ client: boundary });
  const transcript = createTranscript();
  const [project, sessions, roadmapReply, settingsReply] = await Promise.all([
    boundary.project(),
    boundary.sessions(SIDEBAR_SESSION_LIMIT),
    boundary.file("docs/roadmap.json"),
    boundary.settings(),
  ]);
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
      const rows = modelRows(settingsReply).map(({ name, envVar, present }) => [
        name, envVar, present ? "present" : "absent",
      ]);
      replace(
        settingsContent,
        element(document, "h2", { text: "Models" }),
        settingsTable(["Provider", "Environment variable", "Status"], rows),
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
    const pane = route.page === "roadmap"
      ? roadmapPane
      : route.page === "settings" ? settingsPane : chatPane;
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

  const sectionLinks = ["general", "models"].map((section) => {
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

  const links = PAGES.map((page) => {
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

  const projectsRoot = element(document, "section", { className: "projects" });
  if (sessions.length >= SIDEBAR_SESSION_LIMIT) {
    append(projectsRoot, element(document, "p", {
      className: "session-limit",
      text: `Showing the ${SIDEBAR_SESSION_LIMIT} most recent sessions.`,
    }));
  }
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
      const label = session.title || session.run_id;
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
        await boundary.openSession(session.run_id);
        navigate({ page: "chat", section: "" });
      });
      append(section, button);
    }
    append(projectsRoot, section);
  }
  replace(sidebar, projectsRoot, pageLinks);

  listen(sidebarToggle, "click", () => {
    const folded = shell.className.includes("sidebar-folded");
    shell.className = folded ? "app-shell" : "app-shell sidebar-folded";
    sidebarToggle.textContent = folded ? "Hide sidebar" : "Show sidebar";
  });

  const fragment = typeof global.location?.hash === "string" ? global.location.hash : "";
  showPage(fragment ? parseRoute(fragment) : storedRoute(global));
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

  async function perform(actions) {
    for (const action of actions) {
      if (action.kind === "prompt") {
        try {
          const reply = await boundary.prompt(action.text);
          promptFailure = "";
          await perform(turn.accepted(reply.run_id));
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
    const text = input.value;
    input.value = "";
    promptFailure = "";
    const actions = turn.submit(text);
    showPromptError();
    return perform(actions);
  });

  async function onFrame(frame) {
    transcript.apply(frame);
    renderTranscript(document, chatRoot, transcript.model);
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
    await perform(turn.event({ ...event, ...event.fields }));
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
