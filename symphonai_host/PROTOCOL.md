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
| `GoalChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `change: str` (`set`, `check`, `pause`, `resume`, `clear`, `update`, or `round`), `phase: str`, `rounds: int`, `max_rounds: int`, `reason: str`, `last_check: dict | null`, `session_id: str` |

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
An approval request's optional `remember` field must be a boolean and defaults
to `false`.

| Request kind | Payload |
| --- | --- |
| `prompt` | `prompt: str`, `attachments?: [{data: str, filename?: str}]` |
| `approval` | `approval_id: str`, `allowed: bool`, `reason: str`, `remember: bool` (optional) |
| `stop` | `reason: str` |

Unknown request kinds and malformed fields are protocol errors. This document
defines encoding only; it does not define a socket, HTTP endpoint, client, or
authentication mechanism.

The HTTP `POST /prompt` request may include up to 10 attachments. Each `data`
value is strict standard base64 for a PNG, JPEG, GIF, WEBP, or PDF no larger
than 5,000,000 bytes. Invalid base64, unsupported content, oversized content,
or malformed attachment entries return `400` with the attachment index in the
error. An empty prompt is allowed when an attachment is present. Replay
`HistoryMessage` events include `attachments`, an array of `{kind, media_type,
filename}` metadata for non-text blocks; attachment data is never included.

## Approvals

An `approval_requested` frame carries `approval_id`, `operation`, `target`,
`details`, `tool_call_id`, `remember`, and `session_id`. The `tool_call_id` is `""` when the
approval belongs to no tool call. `remember` is the shell command prefix the
client may grant for the current chat, or `""` when no grant applies. A client
answers with the `approval` request above; `remember: true` grants that prefix
for its conversation. An approval id is single-use; unknown or expired replies are rejected. After any
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

### Spec runs

`[worktree] symlink = [".venv"]` may list relative paths that should be
symlinked from the main repository into newly created worktrees. Authenticated
`POST /spec/run` accepts `{"path":"specs/<phase>/<file>.md"}` and starts an
isolated implementation conversation. `GET /spec/runs` lists those runs,
their worktree changes, state, and whether an ignored report was copied into
the main tree. A spec validation block runs in the worktree; specs without one
use a checkless goal.

`POST /spec/review` accepts a spec run's `session_id`. The reviewer reads its
spec, report and patch, and the final non-empty answer line determines `passed`,
`follow-ups`, or `no-verdict`. A changed implementation tree overrides the
answer with `tree-changed`. Follow-up specs under the same phase directory are
copied to the main tree when new. `POST /spec/commit` accepts `session_id` and
a non-blank `message`; it applies the worktree, stages only its changed paths,
and commits them after a person requests it. It refuses with `409` when a
changed path has unstaged edits in the main tree, or when a path the run adds
is already untracked there. If commit fails after the patch was applied, its
worktree is gone and the applied files remain in the main tree for the person
to handle. Run entries expose `review`,
`committed`, and the default `commit_message`.

`POST /spec/plan` accepts a roadmap `phase` id and zero-based `item` index.
Planner sessions are listed as `kind: "plan"` with `phase`, `item`, and
`bound`. A single new Markdown spec under that phase binds the roadmap item.
After a successful spec commit, every roadmap item bound to the spec is marked
done, and the phase is marked `done` only when every item is done.

Authenticated `GET /files?query=<text>&limit=<n>` searches readable regular
files under the repository root and returns `{"files": [...], "truncated":
bool}`. The default limit is 20; valid limits are 1–50. The search skips
`.git`, checks each candidate with the read policy, and stops after 20,000
files. Matching is case-insensitive, ranked by basename prefix, basename
substring, path substring, then ordered-character subsequence; ties use shorter
paths and alphabetical order. An empty query returns paths in walk order.
New conversations tell the agent that `@<path>` references a repository file
and that it must read it with `read_file` before relying on its contents.

Authenticated `GET /history?limit=<n>` returns `{"prompts": [...]}` in newest
first order for sessions whose `repo_root` matches this host. The default limit
is 100; valid limits are 1–500. It excludes empty prompts and host-written goal
round prompts, collapses consecutive duplicates, skips damaged sessions, and
stops reading sessions after it has enough prompts.

`GET /sessions` returns a bare array of persisted session metadata newest first.
Each item includes `activity`: `working` during a run or goal check, `waiting`
while an approval is pending, and `idle` otherwise.
An optional positive `limit` query parameter bounds the entries returned and
classified; missing, empty, non-numeric, zero, and negative values return the
full listing. `POST /session/open` takes a `run_id`, replays `HistoryMessage` event frames before
its reply, and makes that conversation current. Each history and runtime event
frame includes the originating `session_id`. Later prompts append to the same
session directory and transcript, while each prompt still has a distinct run
id. `POST /session/new` takes an empty JSON object, selects a new conversation
for the next prompt, and returns `{"ended": true}`. Opening, forking, and
starting another session leave other runs going. Opening a session that is
already open reuses its leader. At most four conversations may run at once; a
fifth start returns `409` with `{"error": "4 conversations are already running"}`.
Busy checks concern the current conversation. `POST /stop` stops only the
current conversation; open another one first to stop its run.

Each replayed `HistoryMessage` also carries an opaque `record_id` for the
message. `POST /session/fork` takes `{"run_id": str, "record_id": str}` and
optionally `"force": true`. It copies the source conversation through that
message into a new session and restores files changed by later prompts to the
state at the fork point. If a file changed outside the agent, it returns `409`
with the changed `paths`; `force: true` allows the restore to proceed. It
replays the fork's messages, makes the fork current, and returns the same
reply shape as `/session/open`, with the new `run_id`. The original session is
unchanged. A fork inherits the kept prompts' checkpoints, source provider choice, and seeded instructions;
later prompts continue the fork without reloading instruction files. The new
session's `parent_session_id` identifies its source in `GET /sessions`;
`parent_run_id` retains its runtime meaning. Any current message may be
selected if its prefix has no unanswered tool call. A missing session or
message returns `404`.

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
It returns `provider`, `state`, `models`, `detail`, and `filter`. Each model is
`{"id": str, "efforts": [str, ...]}`. A successful lookup has `state:
"available"`, models from live model discovery, their declared effort
identifiers, and an empty detail. `filter` is `{"applied": bool, "hidden": int}`;
when a TOML `[models]` table has an array for that provider (for example,
`openai = ["gpt-5", "gpt-5-mini"]`), only matching discovered IDs are returned
and `hidden` counts discovered IDs omitted from the response. Without a
provider list, all discovered models are returned with `applied: false` and
`hidden: 0`. Configured IDs absent from discovery are ignored. A lookup that
cannot be attempted or fails has `state: "unknown"`, an empty models array,
and a key-safe explanation in detail. Successful vendor results are cached for the host process by
provider and base URL; adding efforts does not perform another vendor call.
Failed or unavailable listings are not cached. For each model,
`efforts` is an array when the capability table has a row (including an empty
array when the model accepts no effort) and `null` when the table has no row
and effort support is unknown. A listing is advisory: clients may still submit
any model id to `POST /provider`; an effort for an unlisted model is passed
through as supplied.

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

Permission modes have these effects: `ask` requests approval for writes, shell
commands, and non-preapproved fetches; `plan` refuses writes and shell commands
while fetches continue to follow the configured fetch rules; `allow` permits
writes anywhere inside the repository, any command, and public URL fetches
without asking. Repository boundaries, forbidden files, always-denied
commands, and local or private fetch hosts remain refused in every mode. In
`allow`, shell commands run in the configured sandbox when it is available;
otherwise they run unconfined. `sandbox.shell = true` requires the sandbox and
refuses shell commands when it is unavailable. Shell network access remains
controlled by `sandbox.network`. The machine owner's `agents.ceiling` limits
the conversation's writes, shell commands, and fetches before it starts; all
per-conversation and subagent policies inherit those limits. In `ask`, the
person may still approve an operation outside those static limits.

Authenticated `POST /compact` takes `{}` or `{"instructions": str}` and
returns `{"changed": bool, "before_tokens": int, "after_tokens": int,
"dropped_messages": int}`. It forces compaction of the current conversation,
preserving its system messages, first user message, and latest user turn. A
summary may include the optional user instructions. Other keys or a
non-string `instructions` value return `400`; an active run returns `409` with
its `run_id`; no open conversation returns `400` with
`{"error": "no conversation to compact"}`.

Authenticated `GET /conversation` returns `{"conversation": null}` before a
conversation is open. Otherwise `conversation` contains the live `provider`,
`model`, and `effort`, plus an `agents` array whose
entries identify an agent by opaque id and name, plus `parent_agent_id` (an
opaque id or `null` for a root), and `mode` reports the mode currently in force.
Each agent appears once; the first run in the
timestamp-ordered run graph determines its parent. After a run completes in
this process, the payload also contains `context` (`used_tokens`, `budget_tokens`,
`remaining_tokens`, `window_tokens`, and `by_source`), aggregate `usage`, and per-agent
`input_tokens`, `output_tokens`, `calls`, `total_tokens`, `cache_read_tokens`,
and `cache_write_tokens`. Immediately after
reopening a conversation, `context`, aggregate `usage`, and per-agent usage and
`cost` are omitted until another run is accounted in this process.
When every used model has a configured price, usage objects also contain
`cost` with decimal-string `amount` and `currency`; `cost` is omitted when no
price table exists or any used model is unpriced. The payload contains no
repository paths, credentials, or model request content.

## Goals

Authenticated `POST /goal` takes `{"objective": str, "check": [str, ...]?,
"max_rounds": int?}`. An omitted or empty check stores as `[]`. A check is an
argv run by the host in the repository root, outside the agent permission
policy. The default is 10 rounds; the limit is 1–100. A final response runs a
configured check with a 600-second timeout. Exit 0 completes the goal. A failed
check starts another round with its output as feedback until the limit, then
the goal becomes blocked. Without a check, a final response starts another
round unless the leader used `update_goal` to report completion or being
blocked. Non-final round ends pause the goal. Check output in state is limited
to its last 4,000 characters.

The leader alone receives two tools. `get_goal` takes no arguments and returns
the current goal as JSON, or `No goal is set.` `update_goal` takes
`{"status": "blocked" | "complete", "message": str}`. The message must be
non-blank and at most 2,000 characters. It can update only an active goal.
`blocked` sets the goal phase and reason to the message. `complete` completes a
goal without a check; with a configured check, it leaves the goal active and
the check decides at the end of the round. Goal updates publish
`GoalChanged(change="update")` with the message in `reason`.

For an unchecked goal, each final response starts the next round with a prompt
reminding the leader to report `complete` or `blocked`. The host publishes
`GoalChanged(change="round")` when it starts that next round and when reaching
`max_rounds` blocks the goal with reason `rounds exhausted`.

`POST /goal/state` takes `{"action": "pause" | "resume" | "clear"}`. Resume
while a goal round is running reactivates the goal; its round-end check or
check-less continuation then proceeds normally. Resume while a check is
running returns `409`. With no run active, resume runs the check immediately
and continues if needed. `GET /conversation`
includes `goal`, either the current goal state or `null`. Goal state is stored
in session metadata; reopening an active goal pauses it with reason `reopened`,
and a fork starts without a goal. `GoalChanged` reports state transitions.
When a check fails, its `check` event arrives after the round's terminal event
and before the next prompt.

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

Authenticated `GET /spec/files` returns `{"paths": [...]}` with sorted paths
for readable Markdown specs under `specs/<directory>/`. It excludes
`specs/report/`, files named `*-PLAN.md`, and Markdown files directly under
`specs/`.

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

Authenticated `POST /agent/control` takes
`{"agent_id": <id>, "action": <pause|resume|redirect|stop>, "text"?: <text>}`.
Pause and resume take effect at the agent's next turn boundary; redirect queues
text as the next user message. Stop on a subagent cancels only that subagent,
while stop on the leader uses the ordinary host stop path. Success returns
`{"agent_id": <id>, "state": <paused|running|stopping>}`. Invalid requests
receive `400`, an absent active run or invalid transition receives `409`, and
an unknown or finished agent receives `404`.

Authenticated `GET /changes` returns the current conversation's checkpointed
file changes and durable session worktrees as `{"turns": [...], "files": [...], "worktrees": [...]}`. Each turn contains its
checkpoint `key`, the first 80 characters of its prompt, and the paths first
written in that prompt. Each file contains its repository-relative `path`,
`status` (`modified`, `added`, or `deleted`), `changed_outside`, a unified
`diff`, and `truncated`. With no open conversation it returns empty arrays.
An active run returns `409`.

Authenticated `POST /changes/revert` takes exactly one of `{"path": <path>}`
or `{"key": <checkpoint key>}`, with optional `"force": true`. A path restores
its earliest checkpoint; a key restores each affected path to its earliest
checkpoint at or after that prompt. Success returns `{"reverted": [paths]}`.
If any target differs from the agent's last write, the host returns `409` with
the affected `paths` and makes no changes unless `force` is true. An active run
returns `409`, unknown paths or keys return `404`, and malformed requests return
`400`.

Each worktree contains its `name`, changed `files`, a unified `diff`, and a
`truncated` flag. `POST /worktree/apply` and `POST /worktree/discard` each take
`{"name": <worktree name>}`. Apply checks the patch against the main tree,
records the applied files as a checkpoint labeled `Applied worktree <name>`,
then removes the worktree. A patch conflict returns `409` with git's error and
keeps the worktree. Discard removes it without changing the main tree. Both
actions return `409` during an active run and `404` for an unknown name.

`GET /app/` returns the browser shell with one inline handshake script setting
`window.__symphonai` to the running host's `port` and `token`. `GET
/app/<asset path>` serves only `.js`, `.json`, `.css`, and `.html` files
resolved beneath `symphonai_app/`; other extensions, absolute paths, traversal,
and escaping symlinks receive `403` with an empty body. JavaScript is served as
`text/javascript`, JSON as `application/json`, CSS as `text/css`, and HTML as
`text/html`. Static assets do not contain the token or absolute filesystem
paths.

The app directory is resolved beside the installed host package, not beneath
the repository selected by `--repo-root`. If that sibling `symphonai_app/`
directory is absent, `/app/` returns `404` with
`{"error": "app is not installed"}` and never falls back to a directory in the
user's repository.
