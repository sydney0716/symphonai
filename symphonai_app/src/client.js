import {
  PROTOCOL_VERSION,
  ProtocolError,
  decodeFrame,
  encodeRequest,
} from "./protocol.js";
import { isObject } from "./json.js";

export async function readEventStream(body, onData) {
  if (!body || typeof body.getReader !== "function") {
    throw new ProtocolError("host event stream has no readable body");
  }
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines = [];

  function consumeLine(rawLine) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      if (dataLines.length > 0) {
        onData(dataLines.join("\n"));
        dataLines = [];
      }
      return;
    }
    if (line.startsWith(":")) {
      return;
    }
    if (line.startsWith("data:")) {
      let value = line.slice(5);
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }
      dataLines.push(value);
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      consumeLine(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer !== "") {
    consumeLine(buffer);
  }
}

export function createClient({
  port,
  token,
  fetch: fetchImplementation = globalThis.fetch,
}) {
  if (
    !Number.isInteger(port) ||
    port < 1 ||
    port > 65535 ||
    typeof token !== "string" ||
    token.length === 0
  ) {
    throw new ProtocolError("client requires a valid port and token");
  }

  const origin = `http://127.0.0.1:${port}`;
  let droppedFrames = 0;

  function headers(extra = {}) {
    return { ...extra, Authorization: `Bearer ${token}` };
  }

  async function readReply(response) {
    let value;
    try {
      value = await response.json();
    } catch {
      throw new ProtocolError("host reply is not valid JSON");
    }
    if (!isObject(value)) {
      throw new ProtocolError("host reply must be an object");
    }
    return value;
  }

  async function readListReply(response) {
    let value;
    try {
      value = await response.json();
    } catch {
      throw new ProtocolError("host reply is not valid JSON");
    }
    if (!Array.isArray(value)) {
      throw new ProtocolError("host reply must be an array");
    }
    return value;
  }

  async function requestFailure(response, detailOnError = true) {
    let detail = "";
    let failureReply = null;
    if (detailOnError) {
      try {
        const value = await response.json();
        if (isObject(value)) {
          failureReply = value;
          if (typeof value.error === "string") {
            detail = value.error;
          }
        }
      } catch {
        // Keep the status error when the host did not return JSON.
      }
    }
    const error = new Error(detail || `host request failed with status ${response.status}`);
    error.status = response.status;
    if (Array.isArray(failureReply?.paths)) {
      error.paths = failureReply.paths;
    }
    return error;
  }

  async function request(method, path, body, read = readReply, { detailOnError = true } = {}) {
    const options = { method, headers: headers() };
    if (body !== undefined) {
      options.headers = headers({ "Content-Type": "application/json" });
      options.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await fetchImplementation(`${origin}${path}`, options);
    } catch {
      throw new Error("network request failed");
    }
    if (!response || typeof response.status !== "number") {
      throw new ProtocolError("host returned a malformed response");
    }
    if (response.status < 200 || response.status >= 300) {
      throw await requestFailure(response, detailOnError);
    }
    return read(response);
  }

  async function post(kind, payload) {
    const encoded = encodeRequest(kind, payload);
    return request("POST", encoded.path, encoded.body);
  }

  const client = {
    events(onFrame) {
      if (typeof onFrame !== "function") {
        throw new ProtocolError("events requires a frame callback");
      }
      const controller = new AbortController();
      let closed = false;
      const receive = (data) => {
        const frame = decodeFrame(data);
        if (
          frame.kind === "error" &&
          Number.isInteger(frame.payload.dropped) &&
          frame.payload.dropped > 0
        ) {
          droppedFrames += frame.payload.dropped;
          onFrame({ kind: "error", dropped: frame.payload.dropped });
          return;
        }
        onFrame(frame);
      };
      const responseReady = (async () => {
        let response;
        try {
          response = await fetchImplementation(`${origin}/events`, {
            method: "GET",
            headers: headers(),
            signal: controller.signal,
          });
        } catch {
          if (closed) {
            return;
          }
          throw new Error("network request failed");
        }
        if (!response || typeof response.status !== "number") {
          throw new ProtocolError("host returned a malformed response");
        }
        if (response.status < 200 || response.status >= 300) {
          throw await requestFailure(response);
        }
        return response;
      })();
      const ready = responseReady.then(() => undefined);
      ready.catch(() => {});
      const done = (async () => {
        const response = await responseReady;
        try {
          await readEventStream(response.body, receive);
        } catch (error) {
          if (closed) {
            return;
          }
          throw error;
        }
      })();
      return {
        close() {
          closed = true;
          controller.abort();
        },
        done,
        ready,
        get droppedFrames() {
          return droppedFrames;
        },
        get hasDroppedFrames() {
          return droppedFrames > 0;
        },
      };
    },

    async prompt(text, attachments = []) {
      const encoded = encodeRequest("prompt", {
        prompt: text,
        ...(attachments.length > 0 ? { attachments } : {}),
      });
      return request("POST", encoded.path, encoded.body);
    },

    stop(reason = "") {
      return post("stop", { reason });
    },

    roadmap() {
      return request("GET", "/roadmap");
    },

    approve(id, allowed, reason = "", remember = false, answerText = "") {
      return post("approval", {
        approval_id: id,
        allowed,
        reason,
        ...(remember ? { remember } : {}),
        ...(typeof answerText === "string" && answerText.length > 0 ? { answer: answerText } : {}),
      });
    },

    steer(text) {
      return request("POST", "/steer", { text }, readReply, { detailOnError: true });
    },

    command(command) {
      return request("POST", "/command", { command }, readReply, { detailOnError: true });
    },

    sessionViewed(sessionId) {
      return request("POST", "/session/viewed", { session_id: sessionId }, readReply, { detailOnError: true });
    },

    openSession(runId) {
      return post("session/open", { run_id: runId });
    },

    mcpPrompts() { return request("GET", "/mcp/prompts"); },
    mcpPrompt(server, name, arguments_) { return request("POST", "/mcp/prompt", { server, name, arguments: arguments_ }); },
    mcpSignIn(server) { return request("POST", "/mcp/sign-in", { server }); },

    terminals() { return request("GET", "/terminals"); },
    terminalStart() { return request("POST", "/terminal/start", {}); },
    terminalInput(id, data) { return request("POST", "/terminal/input", { id, data }); },
    terminalResize(id, cols, rows) { return request("POST", "/terminal/resize", { id, cols, rows }); },
    terminalStop(id) { return request("POST", "/terminal/stop", { id }); },

    sessionNotes(sessionId) {
      const query = new URLSearchParams({ session_id: sessionId });
      return request("GET", `/session/notes?${query}`);
    },

    agentHistory(sessionId, agentId) {
      const query = new URLSearchParams({ session_id: sessionId, agent_id: agentId });
      return request("GET", `/session/agent?${query}`);
    },

    changes() {
      return request("GET", "/changes");
    },

    gitStatus() {
      return request("GET", "/git/status");
    },

    gitDiff(path) {
      return request("GET", `/git/diff?path=${encodeURIComponent(path)}`);
    },

    commit(message, paths) {
      return request("POST", "/git/commit", { message, paths }, readReply, {
        detailOnError: true,
      });
    },

    pullRequestInfo() {
      return request("GET", "/git/pull-request");
    },

    openPullRequest(title, body) {
      return request("POST", "/git/pull-request", { title, body }, readReply, {
        detailOnError: true,
      });
    },

    jobs() {
      return request("GET", "/jobs");
    },

    startJob(command) {
      return request("POST", "/jobs/start", { command }, readReply, { detailOnError: true });
    },

    stopJob(id) {
      return request("POST", "/jobs/stop", { id }, readReply, { detailOnError: true });
    },

    startLoop(goal) {
      return request("POST", "/loop/start", { goal }, readReply, { detailOnError: true });
    },

    loops() {
      return request("GET", "/loops");
    },

    loop(id) {
      return request("GET", `/loop?${new URLSearchParams({ id })}`);
    },

    loopPlan(loopId, action, options = {}) {
      return request("POST", "/loop/plan", { loop_id: loopId, action, ...options }, readReply, {
        detailOnError: true,
      });
    },

    loopCancel(loopId) {
      return request("POST", "/loop/cancel", { loop_id: loopId }, readReply, {
        detailOnError: true,
      });
    },

    loopResume(loopId) {
      return request("POST", "/loop/resume", { loop_id: loopId }, readReply, {
        detailOnError: true,
      });
    },

    loopCard(loopId, cardId) {
      const query = new URLSearchParams({ loop: loopId, card: cardId });
      return request("GET", `/loop/card?${query}`);
    },

    loopCommit(loopId, action) {
      return request("POST", "/loop/commit", { loop_id: loopId, action }, readReply, {
        detailOnError: true,
      });
    },

    memory() {
      return request("GET", "/memory");
    },

    updateMemory(id, text) {
      return request("POST", "/memory/update", { id, text }, readReply, {
        detailOnError: true,
      });
    },

    deleteMemory(id) {
      return request("POST", "/memory/delete", { id }, readReply, {
        detailOnError: true,
      });
    },

    restoreMemory(scope) {
      return request("POST", "/memory/restore", { scope }, readReply, { detailOnError: true });
    },

    revertChanges(payload) {
      return request("POST", "/changes/revert", payload, readReply, {
        detailOnError: true,
      });
    },

    applyWorktree(name) {
      return request("POST", "/worktree/apply", { name }, readReply, {
        detailOnError: true,
      });
    },

    discardWorktree(name) {
      return request("POST", "/worktree/discard", { name }, readReply, {
        detailOnError: true,
      });
    },

    forkSession(runId, recordId, force = false) {
      return request("POST", "/session/fork", {
        run_id: runId,
        record_id: recordId,
        ...(force ? { force: true } : {}),
      });
    },

    approvals() {
      return request("GET", "/approvals");
    },

    sessions(limit, query, archived = false) {
      const params = new URLSearchParams();
      if (limit !== undefined) params.set("limit", String(limit));
      if (query !== undefined && query !== "") params.set("query", query);
      if (archived) params.set("archived", "1");
      return request(
        "GET",
        params.size ? `/sessions?${params}` : "/sessions",
        undefined,
        readListReply,
      );
    },

    exportSession(runId) {
      const params = new URLSearchParams({ id: runId });
      return request("GET", `/session/export?${params}`, undefined, readReply, {
        detailOnError: true,
      });
    },

    renameSession(runId, title) {
      return request("POST", "/session/rename", { run_id: runId, title }, readReply, {
        detailOnError: true,
      });
    },

    archiveSession(runId, archived) {
      return request("POST", "/session/archive", { run_id: runId, archived }, readReply, {
        detailOnError: true,
      });
    },

    deleteSession(runId) {
      return request("POST", "/session/delete", { run_id: runId }, readReply, {
        detailOnError: true,
      });
    },

    project() {
      return request("GET", "/project");
    },

    openProject(path) {
      return request("POST", "/project/open", { path }, readReply, { detailOnError: true });
    },

    projects() {
      return request("GET", "/projects");
    },

    directories(path) {
      const query = path === undefined ? "" : `?${new URLSearchParams({ path })}`;
      return request("GET", `/directories${query}`, undefined, readReply, { detailOnError: true });
    },

    settings() {
      return request("GET", "/settings");
    },

    reloadExtensions() {
      return request("POST", "/extensions/reload", {}, readReply, { detailOnError: true });
    },

    setConfig(key, value, scope) {
      return request("POST", "/config/set", { key, value, scope }, readReply, { detailOnError: true });
    },

    permissions() {
      return request("GET", "/permissions");
    },

    removePermission(kind, value) {
      return request("POST", "/permissions/remove", { kind, value }, readReply, {
        detailOnError: true,
      });
    },

    agent(name, scope) {
      const query = new URLSearchParams({ name, scope });
      return request("GET", `/agent?${query}`);
    },

    draftAgent(description) {
      return request("POST", "/agent/draft", { description }, readReply, { detailOnError: true });
    },

    saveAgent(name, scope, text) {
      return request(
        "POST",
        "/agent",
        { name, scope, text },
        readReply,
        { detailOnError: true },
      );
    },

    models(provider, baseUrl) {
      const query = new URLSearchParams({ provider });
      if (baseUrl) query.set("base_url", baseUrl);
      return request("GET", `/models?${query}`);
    },

    selectProvider(choice) {
      return request("POST", "/provider", choice, readReply, { detailOnError: true });
    },

    selectMode(mode) {
      return request("POST", "/mode", { mode }, readReply, { detailOnError: true });
    },

    compact(instructions) {
      return request(
        "POST",
        "/compact",
        instructions === undefined ? {} : { instructions },
        readReply,
        { detailOnError: true },
      );
    },

    setGoal(objective, check, maxRounds) {
      return request(
        "POST",
        "/goal",
        {
          objective,
          check,
          ...(maxRounds === undefined ? {} : { max_rounds: maxRounds }),
        },
        readReply,
        { detailOnError: true },
      );
    },

    goalState(action) {
      return request(
        "POST",
        "/goal/state",
        { action },
        readReply,
        { detailOnError: true },
      );
    },

    newSession() {
      return request("POST", "/session/new", {}, readReply, { detailOnError: true });
    },

    previewStart(url) { return request("POST", "/preview/start", { url }, readReply, { detailOnError: true }); },
    previewStop() { return request("POST", "/preview/stop", {}, readReply, { detailOnError: true }); },
    previewMarks() { return request("GET", "/preview/marks"); },
    removePreviewMarks(ids, sessionId) {
      return request("POST", "/preview/mark", { remove: ids, session_id: sessionId }, readReply, { detailOnError: true });
    },

    usage(days = 30) {
      return request("GET", `/usage?days=${encodeURIComponent(days)}`, undefined, readReply, { detailOnError: true });
    },

    conversationStats() {
      return request("GET", "/conversation", undefined, readReply, { detailOnError: true });
    },

    storeCredential(name, value) {
      return request("POST", "/credentials", { name, value });
    },

    survey() {
      return request("GET", "/survey");
    },

    health() {
      return request("GET", "/health");
    },

    files(query = "", limit = 20) {
      const parameters = new URLSearchParams({ query, limit: String(limit) });
      return request("GET", `/files?${parameters}`);
    },

    tree(path = "") {
      const parameters = new URLSearchParams({ path });
      return request("GET", `/tree?${parameters}`);
    },

    source(path) {
      const parameters = new URLSearchParams({ path });
      return request("GET", `/source?${parameters}`);
    },

    history(limit = 100) {
      return request("GET", `/history?limit=${encodeURIComponent(String(limit))}`);
    },

    async file(path) {
      const query = new URLSearchParams({ path });
      return request("GET", `/file?${query}`);
    },

    get droppedFrames() {
      return droppedFrames;
    },

    get hasDroppedFrames() {
      return droppedFrames > 0;
    },

    protocolVersion: PROTOCOL_VERSION,
  };
  return client;
}
