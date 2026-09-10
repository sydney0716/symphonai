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
| `PromptSubmitted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `text: str`, `message_count: int` |
| `ToolCallFailed` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `error: str` |
| `PermissionRequested` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `mode: str` |
| `PermissionDenied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `reason: str` |
| `SessionStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SessionEnded` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SubagentSpawned` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `SubagentStopped` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `CompactionApplied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `before_tokens: int`, `after_tokens: int`, `dropped_messages: int` |

`ToolCallFailed` is emitted in addition to `ToolCallFinished` with `ok: false`,
never instead of it; clients that treat these events as alternatives will
double-count some failures or miss others.

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
Receive the handshake through a pipe or stdin, never on a command line.

## Sessions

`GET /sessions` returns persisted session metadata newest first. `POST
/session/open` takes a `run_id`, replays `HistoryMessage` event frames before
its reply, and holds that conversation for the next prompt. The continuation
is written as a new descendant session; the opened transcript is never changed.

## Packaged sidecar

The packaged host is launched directly by its parent; its first stdout line is
the handshake JSON and the parent is responsible for terminating the sidecar.

## HTTP transport

The reference host binds only to `127.0.0.1` on an ephemeral port. On startup
it prints exactly one JSON line containing `port` and the process-local token.
All routes except `GET /health` require `Authorization: Bearer <token>`;
missing or invalid credentials receive `401` with an empty body. The sole
exception is the browser's initial `GET /app?token=<token>` navigation, which
cannot set a request header. Query authentication is accepted only on exact
`GET /app`, never on `/app/<path>`, `/file`, `/events`, or any POST. The host
does not log request URLs.

An authenticated `GET /app` response sets exactly one development session
cookie:

```text
Set-Cookie: symphonai_app=<token>; Path=/app/; HttpOnly; SameSite=Strict
```

It has no `Max-Age` or `Expires`. `GET /app/<path>` accepts that cookie or the
ordinary bearer header so native browser stylesheet and module requests can
load. No other route accepts the cookie: `/file`, `/events`, `/health`, and all
POST requests retain their existing authentication behavior. `Path=/app/`
limits where the browser sends it, `HttpOnly` prevents app JavaScript from
reading it, and `SameSite=Strict` plus the loopback bind limits ambient use.
The `?token=` exception remains confined to exact `/app`; the cookie is the only
subresource exception.

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

`GET /file?path=<repository-relative path>` returns a UTF-8 text file as
`{"path": <path as requested>, "text": <contents>}`. It serves regular files
only from the resolved repository's `specs/` and `docs/` directories. Absolute
paths, traversal outside the resolved repository, and symlinks escaping it
receive `403` with an empty body. Missing files receive `404`, non-UTF-8 files
receive `415`, and files larger than 1 MiB receive `413`. The route requires the
same bearer token as every non-health endpoint and never returns the token or an
absolute filesystem path.

`GET /app` returns the browser shell with one inline handshake script setting
`window.__symphonai` to the running host's `port` and `token`. `GET
/app/<path>` serves only `.js`, `.css`, and `.html` files resolved beneath
`symphonai_app/`; other extensions, absolute paths, traversal, and escaping
symlinks receive `403` with an empty body. JavaScript is served as
`text/javascript`, CSS as `text/css`, and HTML as `text/html`. Static assets do
not contain the token or absolute filesystem paths.

The app directory is resolved beside the installed host package, not beneath
the repository selected by `--repo-root`. If that sibling `symphonai_app/`
directory is absent, `/app` returns `404` with
`{"error": "app is not installed"}` and never falls back to a directory in the
user's repository.
