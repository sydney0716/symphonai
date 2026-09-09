export const PROTOCOL_VERSION = 1;

export const KNOWN_EVENT_TYPES = Object.freeze([
  "RunStarted",
  "RunFinished",
  "RunFailed",
  "TurnStarted",
  "TurnFinished",
  "AssistantTextDelta",
  "ToolCallStarted",
  "ToolCallFinished",
  "PromptSubmitted",
  "ToolCallFailed",
  "PermissionRequested",
  "PermissionDenied",
  "SessionStarted",
  "SessionEnded",
  "SubagentSpawned",
  "SubagentStopped",
  "CompactionApplied",
]);

const FRAME_KINDS = new Set([
  "event",
  "reply",
  "error",
  "approval_requested",
]);
const KNOWN_EVENTS = new Set(KNOWN_EVENT_TYPES);

export class ProtocolError extends Error {
  constructor(message) {
    super(message);
    this.name = "ProtocolError";
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function decodeFrame(text) {
  let frame;
  try {
    frame = JSON.parse(text);
  } catch {
    throw new ProtocolError("frame is not valid JSON");
  }
  if (!isObject(frame)) {
    throw new ProtocolError("frame must be an object");
  }
  if (!Number.isInteger(frame.protocol_version)) {
    throw new ProtocolError("frame protocol_version must be an integer");
  }
  if (frame.protocol_version > PROTOCOL_VERSION) {
    throw new ProtocolError(
      `peer protocol version ${frame.protocol_version} exceeds supported version ${PROTOCOL_VERSION}`,
    );
  }
  if (!FRAME_KINDS.has(frame.kind)) {
    throw new ProtocolError("frame kind is not supported");
  }
  if (!Object.hasOwn(frame, "payload") || !isObject(frame.payload)) {
    throw new ProtocolError("frame payload must be an object");
  }
  return { kind: frame.kind, payload: frame.payload };
}

export function decodeEvent(payload) {
  if (!isObject(payload) || typeof payload.type !== "string") {
    throw new ProtocolError("event must be an object with a string type");
  }
  const { type, ...fields } = payload;
  return { type, known: KNOWN_EVENTS.has(type), fields };
}

export function encodeRequest(kind, payload) {
  const paths = {
    prompt: "/prompt",
    stop: "/stop",
    approval: "/approval",
    "session/open": "/session/open",
  };
  if (!Object.hasOwn(paths, kind)) {
    throw new ProtocolError("request kind is not supported");
  }
  if (!isObject(payload)) {
    throw new ProtocolError("request payload must be an object");
  }
  return { path: paths[kind], body: payload };
}
