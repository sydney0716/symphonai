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

  async function request(method, path, body, read = readReply, { detailOnError = false } = {}) {
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
      throw error;
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
      const done = (async () => {
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
          throw new Error(
            `host event stream failed with status ${response.status}`,
          );
        }
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
      const options = {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(encoded.body),
      };
      let response;
      try {
        response = await fetchImplementation(`${origin}${encoded.path}`, options);
      } catch {
        throw new Error("network request failed");
      }
      if (!response || typeof response.status !== "number") {
        throw new ProtocolError("host returned a malformed response");
      }
      if (response.status === 409) {
        const conflict = await readReply(response);
        if (typeof conflict.run_id !== "string") {
          throw new ProtocolError("prompt conflict omitted its run id");
        }
        return { accepted: false, conflict: true, run_id: conflict.run_id };
      }
      if (response.status < 200 || response.status >= 300) {
        throw new Error(`host request failed with status ${response.status}`);
      }
      return readReply(response);
    },

    stop(reason = "") {
      return post("stop", { reason });
    },

    controlAgent(agentId, action, text) {
      return request(
        "POST",
        "/agent/control",
        { agent_id: agentId, action, ...(text === undefined ? {} : { text }) },
        readReply,
        { detailOnError: true },
      );
    },

    changes() {
      return request("GET", "/changes");
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

    planSpec(phase, item) {
      return request("POST", "/spec/plan", { phase, item }, readReply, { detailOnError: true });
    },

    runSpec(path) {
      return request("POST", "/spec/run", { path }, readReply, { detailOnError: true });
    },

    reviewSpec(session_id) {
      return request("POST", "/spec/review", { session_id }, readReply, { detailOnError: true });
    },

    commitSpec(session_id, message) {
      return request("POST", "/spec/commit", { session_id, message }, readReply, { detailOnError: true });
    },

    specRuns() {
      return request("GET", "/spec/runs", undefined, readListReply);
    },

    approve(id, allowed, reason = "", remember = false) {
      return post("approval", {
        approval_id: id,
        allowed,
        reason,
        ...(remember ? { remember: true } : {}),
      });
    },

    openSession(runId) {
      return post("session/open", { run_id: runId });
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

    sessions(limit) {
      return request(
        "GET",
        limit === undefined ? "/sessions" : `/sessions?limit=${limit}`,
        undefined,
        readListReply,
      );
    },

    project() {
      return request("GET", "/project");
    },

    settings() {
      return request("GET", "/settings");
    },

    agent(name, scope) {
      const query = new URLSearchParams({ name, scope });
      return request("GET", `/agent?${query}`);
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

    conversationStats() {
      return request("GET", "/conversation", undefined, readReply);
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

    specFiles() {
      return request("GET", "/spec/files");
    },

    history(limit = 100) {
      return request("GET", `/history?limit=${encodeURIComponent(String(limit))}`);
    },

    async file(path) {
      const query = new URLSearchParams({ path });
      let response;
      try {
        response = await fetchImplementation(`${origin}/file?${query}`, {
          method: "GET",
          headers: headers(),
        });
      } catch {
        throw new Error("network request failed");
      }
      if (!response || typeof response.status !== "number") {
        throw new ProtocolError("host returned a malformed response");
      }
      if (response.status < 200 || response.status >= 300) {
        const error = new ProtocolError(
          `host file request failed with status ${response.status}`,
        );
        error.status = response.status;
        throw error;
      }
      return readReply(response);
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
