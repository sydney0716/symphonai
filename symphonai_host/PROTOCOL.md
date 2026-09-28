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
| `ToolCallStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `target: str` |
| `ToolCallFinished` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `ok: bool`, `result_kind: str`, `result_path: str`, `lines_added: int`, `lines_removed: int`, `diff: str`, `truncated: bool` |
| `PromptSubmitted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `text: str`, `message_count: int` |
| `ToolCallFailed` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `error: str` |
| `PermissionRequested` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `mode: str` |
| `PermissionDenied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `reason: str` |
| `SessionStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SessionEnded` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SubagentSpawned` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `SubagentStopped` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `CompactionApplied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `before_tokens: int`, `after_tokens: int`, `dropped_messages: int` |

`target` is a bounded display string derived from the call, not the argument
itself. For a shell call it is the program name, for a fetch it is the origin,
and it is `""` when the runtime has no safe derivation.

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
`details`, and `tool_call_id`. The `tool_call_id` is `""` when the approval
belongs to no tool call. A client answers with the `approval` request above. An
approval id is single-use; unknown or expired replies are rejected. After any
`error` frame carrying `dropped`, a client re-reads `GET /approvals`, because a
dropped frame may have been a question.

## Writing a client

Read the host handshake line into its `port` and bearer `token`, send that
token only in the `Authorization` header, and decode every SSE `data:` frame
with the protocol decoder. Keep unknown events and dropped notices visible.
Receive the handshake through a pipe or stdin, never on a command line.

When started for interactive use, the host writes its unchanged JSON handshake
as the first stdout line, then writes `http://127.0.0.1:<port>/app/?token=<token>`
to stderr for the operator to open. The URL is on stderr so the first stdout
line remains machine-readable for a parent process.

## Sessions

`GET /sessions` returns a bare array of persisted session metadata newest first.
An optional positive `limit` query parameter bounds the entries returned and
classified; missing, empty, non-numeric, zero, and negative values return the
full listing. `POST /session/open` takes a `run_id`, replays `HistoryMessage` event frames before
its reply, and makes that conversation current. Later prompts append to the
same session directory and transcript, while each prompt still has a distinct
run id. `POST /session/new` takes an empty JSON object, ends the current
conversation, and returns `{"ended": true}`. The next prompt creates a new
session directory. It returns `409` if a run is active.

Each replayed `HistoryMessage` also carries an opaque `record_id` for the
message. `POST /session/fork` takes `{"run_id": str, "record_id": str}` and
copies the source conversation through that message into a new session. It
replays the fork's messages, makes the fork current, and returns the same
reply shape as `/session/open`, with the new `run_id`. The original session is
unchanged. A fork inherits the source provider choice and seeded instructions;
later prompts continue the fork without reloading instruction files. The new
session's `parent_session_id` identifies its source in `GET /sessions`;
`parent_run_id` retains its runtime meaning. Any current message may be
selected if its prefix has no unanswered tool call. A missing session or
message returns `404`, and an active run returns `409`.

At a new conversation's first prompt, the host loads the user, project, and
working-directory `.symphonai/INSTRUCTIONS.md` hierarchy. It seeds the
leader with the existing system prompt first, then a separate system message
containing the loader's rendered instruction blocks. Each block names its
scope and source path. The working directory is the host's current directory
when it is inside `repo_root`, or `repo_root` otherwise. Loader warnings go to
host stderr with an `instruction warning:` prefix; a warned file can still be
loaded in full. If the hierarchy is empty, no instruction message is added.
`POST /session/open` restores the recorded messages and does not reload files.

Scoped configuration can set per-run limits for the leader and each dispatched
subagent:

```toml
[budgets]
price_table = "prices.json"

[budgets.leader]
max_turns = 8
wall_seconds = 60
max_total_tokens = 100000
max_cost = "1.50"

[budgets.subagent]
max_turns = 4
```

Each role may set any subset of the four limit keys. Turn and token limits are
positive integers, wall time is a positive finite number of seconds, and cost
is a quoted non-negative decimal. `price_table` names a version-1 JSON price
table, relative to the config file containing that key (or the repository root
for a session override). A host-supplied price table is used when no path is
configured. A cost limit without either table is a configuration error. A
subagent definition's own budget is narrowed by the configured subagent
ceiling. The leader budget's `max_turns` takes precedence over the host's
`--max-turns`; without a leader budget, the launch limit still applies.
Limits reset for each prompt. `RunFinished.stopped_reason` reports `max_turns`,
`budget_wall_time`, `budget_tokens`, or `budget_cost` when a limit ends a run;
the conversation remains available for another prompt.

Authenticated `POST /provider` takes `{"name": "anthropic" | "gemini" |
"openai", "model": str?, "base_url": str?, "effort": str?}` and returns
`{"selected": true}`. `model`, `base_url`, and `effort` are optional non-empty
strings. The model and effort apply to the next leader request in the current
conversation. They do not replace a dispatched agent's own settings, and a
`leader` definition's explicit model or effort takes precedence over this choice.
`openai` with a
`base_url` selects an OpenAI-compatible endpoint. Unknown vendors, malformed
options, and vendors without a configured API key return `400`. A provider
change rebuilds the conversation's leader for the next turn while preserving
the session and message history; provider-specific tool-call metadata from the
previous provider is discarded. A selection while a run is active returns
`409`. Reopening a selected conversation restores its provider choice from
session metadata. Both Leader providers use the conversation's selected
provider. When no selection was sent, the host uses
the first vendor with a key in Settings order (anthropic, gemini, openai).
If no vendor has a key, the first prompt returns `400` until a key is added or
a valid selection is sent. Launch flags do not select a provider.

Authenticated `GET /models?provider=<name>` lists the models for a known
provider, with optional `base_url` using the same meaning as `POST /provider`.
It returns `provider`, `state`, `models`, and `detail`. Each model is
`{"id": str, "efforts": [str, ...]}`. A successful lookup has `state:
"available"`, the filtered models from live model discovery, their declared
effort identifiers, and an empty detail. A model absent from the capability
table has an empty effort list. A lookup that cannot be attempted or fails has
`state: "unknown"`, an empty models array, and a key-safe explanation in detail.
Successful vendor results are cached for the host process by provider and base
URL; adding efforts does not perform another vendor call. Unknown results are
not cached. A listing is advisory: clients may still submit any model id to
`POST /provider`, without an effort when none was offered.

Authenticated `POST /mode` takes exactly `{"mode": "ask" | "plan" | "allow"}`
and returns `{"mode": <current mode>}`. The available values are intersected
with `agents.ceiling.modes`; an unknown or excluded value returns `400` with
the permitted values and leaves the current mode unchanged. A successful
change updates the open conversation immediately, so its next permission check
and every later subagent dispatch use the new mode. A tool call already in
progress continues, and an approval request already waiting remains pending;
its eventual answer still resolves that request. A host starts in `ask`; if
`agents.ceiling.modes` excludes ask, it starts in the tightest permitted mode,
preferring `plan` to `allow`. An empty modes ceiling is a configuration error.
Starting a new chat or reopening a session resets to that starting mode.

Authenticated `GET /conversation` returns `{"conversation": null}` before a
conversation is open. Otherwise `conversation` contains the live `provider`,
`model`, and `effort`, plus an `agents` array whose
entries identify an agent by opaque id and name, plus `parent_agent_id` (an
opaque id or `null` for a root), and `mode` reports the mode currently in force.
Each agent appears once; the first run in the
timestamp-ordered run graph determines its parent. After a run completes in
this process, the payload also contains `context` (`used_tokens`, `budget_tokens`,
`remaining_tokens`, and `by_source`), aggregate `usage`, and per-agent
`input_tokens`, `output_tokens`, `calls`, and `total_tokens`. Immediately after
reopening a conversation, `context`, aggregate `usage`, and per-agent usage and
`cost` are omitted until another run is accounted in this process.
When every used model has a configured price, usage objects also contain
`cost` with decimal-string `amount` and `currency`; `cost` is omitted when no
price table exists or any used model is unpriced. The payload contains no
repository paths, credentials, or model request content.

## Packaged sidecar

The packaged host is launched directly by its parent; its first stdout line is
the handshake JSON and the parent is responsible for terminating the sidecar.

## HTTP transport

The reference host binds only to `127.0.0.1` on an ephemeral port. On startup
it prints exactly one JSON line containing `port` and the process-local token.
All routes except `GET /health` and the canonicalizing `GET /app` redirect
require credentials; missing or invalid credentials receive `401` with an
empty body. `GET /app` redirects to `/app/` without authenticating and preserves
the query string exactly, so `/app?token=<token>` redirects to
`/app/?token=<token>`. The browser's initial `GET /app/?token=<token>` page
navigation cannot set a request header. `GET /app` and `GET /app/` are the only
query-token routes: the former preserves the query during the redirect and the
latter authenticates it. Query authentication is never accepted on
`/app/<asset path>`, `/file`, `/events`, or any POST. The host does not log
request URLs.

An authenticated `GET /app/` response sets exactly one development session
cookie:

```text
Set-Cookie: symphonai_app=<token>; Path=/app/; HttpOnly; SameSite=Strict
```

It has no `Max-Age` or `Expires`. `GET /app/` accepts that cookie on later page
loads. `GET /app/<asset path>` accepts the cookie or the ordinary bearer header
so native browser stylesheet and module requests can load. No other route
accepts the cookie: `/file`, `/events`, `/health`, and all POST requests retain
their existing authentication behavior. `Path=/app/` limits where the browser
sends it, `HttpOnly` prevents app JavaScript from reading it, and
`SameSite=Strict` plus the loopback bind limits ambient use. The `?token=`
authentication exception is confined to exact `/app/`; the cookie is the only
subresource exception.

An authenticated `GET /survey` returns a bounded, permission-gated repository
survey under the `survey` key. Its fields are `root`, `languages`,
`by_directory`, `entry_points`, `docs`, `tests`, `tree_summary`,
`truncated_directories`, `stopped`, and `file_count`; paths are
repository-relative strings. The route accepts only the
ordinary bearer header; query tokens and app cookies do not authorize it.

An authenticated `GET /project` returns the host repository's absolute,
resolved path as `repo_root` and its basename as `name`. It accepts only the
ordinary bearer header; an app query token or app cookie does not authorize it.

An authenticated `GET /settings` returns a `settings` object with the current
permission `mode`, `config`
entries (`key`, `value`, `scope`), `ceiling`, `trust`, `hooks` (`event`, `command`),
`mcp_servers` (`name`, `command`, `started`), sorted `agents`, `skills`, and
`plugins` entries (`name`, `path`), `withheld` entries (`scope`, `directory`, `names`, `reason`),
and `providers` entries (`name`, `env_var`, `key_present`). A withheld
`directory` and roster `path` are repository-relative within the repository
and absolute otherwise; an unknown roster path is `""`. Provider key presence
is a boolean; key values never appear in the
response. It is assembled from already loaded host state and makes no network
requests. Only the ordinary bearer header authorizes this route.

Provider keys can be stored with authenticated `POST /credentials` and a JSON
body containing `name` and `value`. `name` must be one of the host's known
`API_KEY_ENV_VAR` names. An empty `value` deletes the stored entry and unsets
the process variable. A successful response is `{"stored": true, "name":
<name>}` and never includes the value. Query tokens and app cookies do not
authorize this route. The host loads `~/.symphonai/credentials.json` at startup
(or `SYMPHONAI_CREDENTIALS_FILE` when set); a non-empty launch environment
value takes precedence over the stored value. The file is private to the OS
user and a file wider than mode `0600` prevents startup.

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

`GET /agent?name=<name>&scope=<user|project>` returns one agent definition as
`{"name": <stem>, "scope": <scope>, "text": <contents>}`. `POST /agent`
takes `{"name": <name>, "scope": <user|project>, "text": <contents>}` and
validates the text with the runtime's `load_agent_file` before atomically
replacing or creating the definition. A name may be a single stem or that
stem with a `.toml` suffix; separators, `..`, and other extensions are
rejected before filesystem access. Project reads and writes require the
repository's `agents` trust capability, while user scope uses the user's
`.symphonai/agents` directory without repository trust. Validation errors
preserve the loader's file-and-key message. A successful write returns
`written`, the normalized `name` and `scope`, the target `path`, and a
message explaining that the definition applies on the next run. Both routes
require the ordinary bearer token.

`GET /app/` returns the browser shell with one inline handshake script setting
`window.__symphonai` to the running host's `port` and `token`. `GET
/app/<asset path>` serves only `.js`, `.css`, and `.html` files resolved beneath
`symphonai_app/`; other extensions, absolute paths, traversal, and escaping
symlinks receive `403` with an empty body. JavaScript is served as
`text/javascript`, CSS as `text/css`, and HTML as `text/html`. Static assets do
not contain the token or absolute filesystem paths.

The app directory is resolved beside the installed host package, not beneath
the repository selected by `--repo-root`. If that sibling `symphonai_app/`
directory is absent, `/app/` returns `404` with
`{"error": "app is not installed"}` and never falls back to a directory in the
user's repository.
