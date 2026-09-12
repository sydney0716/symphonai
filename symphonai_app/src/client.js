import {
  PROTOCOL_VERSION,
  ProtocolError,
  decodeFrame,
  encodeRequest,
} from "./protocol.js";

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function parseHandshake(line) {
  let handshake;
  try {
    handshake = JSON.parse(line);
  } catch {
    throw new ProtocolError("handshake is not valid JSON");
  }
  if (
    !isObject(handshake) ||
    !Number.isInteger(handshake.port) ||
    handshake.port < 1 ||
    handshake.port > 65535 ||
    typeof handshake.token !== "string" ||
    handshake.token.length === 0
  ) {
    throw new ProtocolError("handshake must contain a valid port and token");
  }
  return { port: handshake.port, token: handshake.token };
}

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

  async function request(method, path, body) {
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
      throw new Error(`host request failed with status ${response.status}`);
    }
    return readReply(response);
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

    async prompt(text) {
      const encoded = encodeRequest("prompt", { prompt: text });
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

    approve(id, allowed, reason = "") {
      return post("approval", { approval_id: id, allowed, reason });
    },

    openSession(runId) {
      return post("session/open", { run_id: runId });
    },

    approvals() {
      return request("GET", "/approvals");
    },

    sessions() {
      return request("GET", "/sessions");
    },

    survey() {
      return request("GET", "/survey");
    },

    health() {
      return request("GET", "/health");
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
