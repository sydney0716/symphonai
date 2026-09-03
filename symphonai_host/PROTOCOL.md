# SymphonAI host protocol

This is the transport-neutral boundary between a SymphonAI runtime host and a
client. It is intentionally small enough to implement in another language.
The runtime may emit events faster than a client can consume them; events may
be dropped and clients must not treat the stream as a transcript.

## Frames

One message is one JSON object:

```json
{"protocol_version": 1, "kind": "event", "payload": {}}
```

`kind` is one of `event`, `reply`, `error`, or `approval_requested`; `payload`
is always an object.
A peer must reject a `protocol_version` greater than 1 and report both its
version and the supported version. Older versions may be accepted where their
shape remains compatible.

## Events

An event payload is a flat JSON object with `type` equal to the class name and
every field below. Field names use JSON strings; `int` is a JSON number, `bool`
is a JSON boolean, and `str | null` accepts a string or JSON null. A client
that receives an unknown `type` must preserve its complete event object as an
unknown event, rather than dropping it or failing the entire stream. Unknown
fields on a known event are ignored for forward compatibility.

| Event type | Fields |
| --- | --- |
| `RunStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `agent_name: str` |
| `RunFinished` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `agent_name: str`, `stopped_reason: str` |
| `RunFailed` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `agent_name: str`, `error: str` |
| `TurnStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `index: int` |
| `TurnFinished` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `index: int` |
| `AssistantTextDelta` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `text: str` |
| `ToolCallStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str` |
| `ToolCallFinished` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `ok: bool` |
| `SubagentSpawned` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `CompactionApplied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `before_tokens: int`, `after_tokens: int`, `dropped_messages: int` |

## Client requests

Requests are validated separately from frame decoding so a transport only needs
to pass a `kind` and object payload to the host. Required fields must have the
listed type; `reason` is optional and defaults to `""`.

| Request kind | Payload |
| --- | --- |
| `prompt` | `prompt: str` |
| `approval` | `approval_id: str`, `allowed: bool`, `reason: str` |
| `stop` | `reason: str` |

Unknown request kinds and malformed fields are protocol errors. This document
defines encoding only; it does not define a socket, HTTP endpoint, client, or
authentication mechanism.

## Approvals

An `approval_requested` frame carries `approval_id`, `operation`, `target`,
and `details`. A client answers with the `approval` request above. An approval
id is single-use; unknown or expired replies are rejected. After any `error`
frame carrying `dropped`, a client re-reads `GET /approvals`, because a dropped
frame may have been a question.

## Writing a client

Read the host handshake line into its `port` and bearer `token`, send that
token only in the `Authorization` header, and decode every SSE `data:` frame
with the protocol decoder. Keep unknown events and dropped notices visible.

## HTTP transport

The reference host binds only to `127.0.0.1` on an ephemeral port. On startup
it prints exactly one JSON line containing `port` and the process-local token.
All routes except `GET /health` require `Authorization: Bearer <token>`;
missing or invalid credentials receive `401` with an empty body. The token is
never placed in a URL, log, or error response.

`GET /events` returns `text/event-stream`. Each event is one `data:` line
whose content is an `event` frame containing the event payload above. A client
that has fallen behind receives an `error` frame with `{"dropped": n}` before
its next event, and a silent stream emits `: keepalive` comments every 15
seconds. `POST /prompt` accepts a `prompt` request and returns an accepted run
id. This is a host control-plane handle; events retain the id allocated by the
runtime, including distinct ids for subagent runs. Only one run may be active,
so a concurrent prompt receives `409` with the active host id. `POST /stop` is
idempotent. `GET /health` returns protocol version, `idle` or `active` state,
the active host `run_id`, and the root runtime `runtime_run_id` once its
`RunStarted` event has been seen. Both ids are `null` while idle. Other paths
return JSON `404` responses.
