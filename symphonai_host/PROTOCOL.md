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
| `AssistantReasoningDelta` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `text: str` |
| `ModelFallback` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `from_model: str`, `to_model: str`, `reason: str` |
| `ToolCallStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `target: str` |
| `ToolCallFinished` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `ok: bool`, `result_kind: str`, `result_path: str`, `lines_added: int`, `lines_removed: int`, `diff: str`, `truncated: bool`, `output: str`, `output_truncated: bool` |
| `PromptSubmitted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `text: str`, `message_count: int`, `checkpoint_key: str | null` |
| `ToolCallFailed` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `error: str` |
| `PermissionRequested` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `mode: str` |
| `PermissionDenied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `tool_name: str`, `tool_call_id: str`, `reason: str` |
| `SessionStarted` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SessionEnded` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_run_id: str` |
| `SessionNotesUpdated` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_id: str`; emitted after atomic replacement of the conversation leader's notes.md |
| `SubagentSpawned` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `SubagentStopped` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str` |
| `SubagentNotice` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `subagent_name: str`, `subagent_agent_id: str`, `kind: "finished" | "progress" | "question" | "message"`, `text: str` (finished: the delivered blocking result as JSON; otherwise the message text) |
| `CompactionApplied` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `before_tokens: int`, `after_tokens: int`, `dropped_messages: int` |
| `GoalChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `change: str` (`set`, `check`, `pause`, `resume`, `clear`, `update`, or `round`), `phase: str`, `rounds: int`, `max_rounds: int`, `reason: str`, `last_check: dict | null`, `session_id: str` |
| `PlanChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `steps: list`, `explanation: str`, `session_id: str` |
| `BackgroundUsageRecorded` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `session_id: str` |
| `JobChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `job_id: str`, `status: str`, `session_id: str | null` |
| `LoopChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `loop_id: str`, `state: str`, `change: str`, `card_id: str`, `reason: str` |
| `TerminalOutput` | `terminal_id: str`, `data: str`, `offset: int`, `exited: bool`, `exit_code: int | null`, plus inherited event fields; ordered per terminal and independent of conversations |
| `FilesChanged` | `agent_id: str`, `run_id: str`, `turn_id: str | null`, `schema_version: int`, `paths: list[str]`, `truncated: bool` |

`target` is a bounded display string derived from the call, not the argument
itself. For a shell call it is the program name, for a fetch it is the origin,
and it is `""` when the runtime has no safe derivation.

`ToolCallFailed` is emitted in addition to `ToolCallFinished` with `ok: false`,
never instead of it; clients that treat these events as alternatives will
double-count some failures or miss others.

Finished tool events include up to 4,000 characters of the tool result in
`output`, with `output_truncated` set when the text was cut. File diffs and
tools with their own transcript entries have empty output. Replayed ordinary
tool calls include their safe `target`, `ok`, `output`, and `output_truncated`
fields from the paired result; call arguments are not sent to the client.

## Client requests

### Projects

`POST /project/open` accepts `{"path": "/absolute/project/path"}` and returns
`{"repo_root": "/resolved/path", "name": "directory-name"}`. The host keeps
work in a project running after switching to another project. Events from
background projects continue on the shared event stream with their
`session_id`; `/sessions` and `/approvals` include every open project, and an
approval can be resolved while another project is current. A project that is
not current closes after its last turn, approval, question, goal check, or
command ends. Session operations on a conversation from another project return
409 with `{"error":"that conversation belongs to another project"}`; clients
switch to its project first.
When project or private config is skipped, `POST /project/open` includes
`config_warnings` naming each file and reason. `GET /settings` includes the
same list under `settings.config_warnings` when non-empty.
The host may start with no project. In that state `GET /project` returns
`{"repo_root":null,"name":null}` and project-dependent routes return 409 with
`{"error":"open a project first"}`. `GET /projects` lists the current and open
projects plus recent roots; `GET /directories?path=<absolute-path>` returns
visible child directories for a folder picker. `/health`, `/events`, `/project`,
`/projects`, `/directories`, `/settings`, `/models`, `/app`, `POST /project/open`,
`/credentials`, and `/provider` remain available without a current project.
When no explicit `--repo-root` is supplied, the host uses its current directory
unless that is `/` or the user's home directory; then it uses the first recent
directory that still exists, or starts without a project.

Requests are validated separately from frame decoding so a transport only needs
to pass a `kind` and object payload to the host. Required fields must have the
listed type; `reason` is optional and defaults to `""`.
An approval request's optional `remember` field must be a boolean and defaults
to `false`; optional `answer` must be a string and defaults to `""`. The HTTP
`POST /approval` endpoint additionally accepts `remember: "project"`.

| Request kind | Payload |
| --- | --- |
| `prompt` | `prompt: str`, `attachments?: [{data: str, filename?: str}]` |
| `approval` | `approval_id: str`, `allowed: bool`, `reason: str`, `remember: bool` (optional; HTTP also accepts `"project"`), `answer: str` (optional) |
| `stop` | `reason: str` |

Unknown request kinds and malformed fields are protocol errors. This document
defines encoding only; it does not define a socket, HTTP endpoint, client, or
authentication mechanism.

The HTTP `POST /prompt` request may include up to 10 attachments. Each `data`
value is strict standard base64 for a PNG, JPEG, GIF, WEBP, or PDF no larger
than 5,000,000 bytes. Invalid base64, unsupported content, oversized content,
or malformed attachment entries return `400` with the attachment index in the
error. An empty prompt is allowed when an attachment is present. Replay
`HistoryMessage` events include `reasoning`, the assistant's readable reasoning
text (empty when none was returned), and `attachments`, an array of `{kind, media_type,
filename}` metadata for non-text blocks; attachment data is never included.
They also include `checkpoint_key: str | null`, `edits: [{path: str, added: int,
removed: int}]`, and `undone: bool`. On replay, a user message identifies its prompt
checkpoint and the turn's recorded distinct edited paths, sorted by path, with net
line counts. If the turn has no recorded totals, its own transcript's tool-result
counts are used instead; replay never recalculates totals from today's working tree.
Live, an empty assistant `HistoryMessage` carries that checkpoint key and the same
recorded edit list before the leader's terminal event, updating the existing turn's
card. `undone` is true on replay when that checkpoint was reverted, including when a
later checkpoint was reverted and therefore undid this one. Other messages carry a
null key, an empty edits list, and false. These fields contain
repository-relative paths only; they do not include file contents or diffs.
Replayed tool calls carry `id` and `name`; `dispatch_subagent` calls also carry
their `subagent_name` and `task` so clients can show the dispatch after reopen.
When a saved dispatch result exists, its replay also includes `ok`. Successful
dispatches include secret-masked `output`, limited to 4,000 characters, and
`output_truncated`; failed dispatches include the masked `error` reason.
Root user messages carry `turn_started_at` as an ISO-8601 UTC string. Exactly
one replayed record for each root run carries `turn_ended_at` and
`turn_outcome` (`finished`, `cancelled`, or `failed`): the final assistant
message when the run has one, or its final user message when it has no
assistant message. A client can place the turn-time line after that turn's
last replayed message and any edit card.

Authenticated `GET /memory` returns the project and global memory sets. Entries
contain `id`, `text`, `written_at` (ISO-8601 UTC), and `agent_name`; no storage
paths are exposed. If a set cannot be read, it returns `{"entries": [],
"error": "unreadable"}` while the other set is still returned normally.
Authenticated `POST /memory/update` accepts `{ "id", "text" }` and returns
`{ "entry" }`; `POST /memory/delete` accepts `{ "id" }` and returns
`{ "deleted" }`. Unknown ids return 404. Blank or over-500-character memory
text returns 400. An update or delete targeting an unreadable set returns 503
with an empty body.

## Approvals

An `approval_requested` frame carries `approval_id`, `operation`, `target`,
`details`, `tool_call_id`, `remember`, `session_id`, `kind`, `options`, and
`diff`. `diff` is an empty string when the request has no proposed file change.
`kind` is `approval`, `question`, or `plan`; `options` is empty for an approval and has
the offered choices for a question or plan. A question uses `operation: "ask_user"`,
with its prompt in `target`. The `tool_call_id` is `""` when the
approval belongs to no tool call. `remember` is the shell command prefix or
fetch host the client may grant for the current chat, or `""` when no grant
applies. A client answers with the `approval` request above; `remember: true`
grants it for the current conversation, while HTTP `remember: "project"` stores
it in the project's private config. An approval id is single-use; unknown or expired replies are rejected. After any
`error` frame carrying `dropped`, a client re-reads `GET /approvals`, because a
dropped frame may have been a question.
For `kind: "question"`, an allowed reply includes non-blank `answer`; a denied
reply skips the question. A blank allowed answer returns `400`.
For `kind: "plan"`, `target` contains the proposed Markdown plan and `options`
contains the permitted approval modes. An allowed reply must include one of
those modes in `answer`; a denied reply sends feedback in `reason` and keeps
the conversation in plan mode.

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

### Roadmap

Each project keeps its ordered goals in `.symphonai/roadmap.json`. Authenticated
`GET /roadmap` returns the goals and their linked loop state. The Roadmap tab
is read-only progress: it has no goal input or Add/Move/Remove/Start controls.
Goals with a loop open it in the Loop tab. Empty roadmaps say "No goals yet.
Ask the leader to add one." The former POST routes `/roadmap/add`,
`/roadmap/remove`, `/roadmap/move` and `/roadmap/start` return 404.

App conversation leaders receive `list_roadmap`, `add_roadmap_goal(goal)`,
`start_roadmap_goal(goal_id)` and `remove_roadmap_goal(goal_id)`; headless run
and review do not. Tools change the roadmap only when the person asks. Each
returns the GET response shape; start also returns a top-level `loop_id`.
Unknown ids and already-started goals return tool errors with a reason.
Removing a goal never removes its loop. Add/remove emit `LoopChanged` with
`change: "roadmap"`; ordinary loop changes also refresh the tab. Every loop
started in the app is recorded with its goal before its initial event, including
loops started through the chat composer's spec mode. A goal can be started again only
when its linked loop was cancelled. Approving a loop at its commit checkpoint
marks its linked roadmap goal done. Invalid roadmap files return `409` and are
left unchanged.

### Spec loops

`[worktree] symlink = [".venv"]` may list relative paths that should be
symlinked from the main repository into newly created worktrees.

Authenticated `POST /loop/start` accepts `{"goal": str}` and returns the new
loop plus its planner `session_id`. `GET /loops` lists loop summaries newest
first; `GET /loop?id=<loop_id>` returns a loop record. `POST /loop/plan`
accepts `approve`, `drop` (with `cards`), or `revise` (with a `note`) while the
loop is `awaiting_plan`. Approval moves it to `running`; dropping marks the
selected cards `dropped`; revision starts a new planner on the same loop.
`POST /loop/cancel` accepts a `loop_id` and cancels a non-terminal loop. These
routes publish `LoopChanged` events when loop state or cards change. Invalid
loop ids return `404`, plan actions in another state return `409`, and invalid
drop requests or approval of an empty plan return `400`.

After plan approval, the host runs ready cards in lane order, with one active
card per lane and at most four active conversations. A finished implementation
is reviewed before it is integrated into the loop branch. Review follow-ups
become cards in the same lane, with no more than two rounds; failed runs,
review findings beyond that limit, and integration conflicts move the loop to
`needs_attention`. When every non-dropped card is done, the loop moves to
`validating`. `LoopChanged` announces each card transition with `change: "card"`
and its `card_id`.

`GET /loop` includes usage totals for planning, each card's implementation and
review, and the loop total. Each total reports input tokens, output tokens,
calls, cache-read tokens and cache-write tokens; stages without usage are
omitted. The headless `loop --json` result includes the same object, and its
text output prints the input, output and call totals.

Authenticated `POST /loop/commit` accepts `{"loop_id": str, "action":
"approve" | "decline"}` only while the loop is `awaiting_commit`. Approval
keeps the loop branch and finishes the loop; decline deletes the branch and
moves the loop to `cancelled`. The reply includes the loop, branch, and
committed cards. `POST /loop/resume` accepts `{"loop_id": str}` only while the
loop is `needs_attention`; it retries failed cards or combined validation and
returns the updated loop. A project runtime resumes interrupted loops when it
starts: running cards without live conversations are reset to `planned`, and
loops in `validating` rerun combined validation.

Authenticated `GET /loop/card?loop=<loop_id>&card=<card_id>` returns the card,
its spec and report text, the card commit diff (or its current running-worktree
diff), review verdict and answer, follow-up card ids, and implementer session
id. Invalid loop or card ids return `404`; text and diffs are bounded by the
host's `MAX_READ_BYTES` limit.

Authenticated `GET /files?query=<text>&limit=<n>` searches readable regular
files under the repository root and returns `{"files": [...], "truncated":
bool}`. The default limit is 20; valid limits are 1–50. The search skips
`.git`, checks each candidate with the read policy, and stops after 20,000
files. Matching is case-insensitive, ranked by basename prefix, basename
substring, path substring, then ordered-character subsequence; ties use shorter
paths and alphabetical order. An empty query returns paths in walk order.
New conversations tell the agent that `@<path>` references a repository file
and that it must read it with `read_file` before relying on its contents.

Authenticated `GET /tree?path=<repository-relative directory>` returns the
readable direct children as `{"path": str, "entries": [{"name": str,
"kind": "dir" | "file"}], "truncated": bool}`. An empty or missing path
selects the repository root. Entries sort with directories first and then by
name; non-symlink directories and regular files are included only when the
read policy allows them. The listing is capped at 2,000 entries and reports
whether more were omitted. Authenticated `GET /source?path=<repository-relative
file>` returns `{"path": str, "text": str}` for a file allowed by the read
policy. Absolute paths, traversal, symlink escapes, and denied paths return
403; missing paths return 404, files over 1 MiB return 413, and non-UTF-8
files return 415. Error responses have empty bodies. Neither route returns an
absolute path.

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
full listing. An optional `query` filters by a case-insensitive substring in
the title or a user or assistant message, before applying `limit`.
Archived sessions are omitted by default; `archived=1` lists only archived
sessions, with `query` and `limit` applied to that set.
`GET /session/export?id=<run_id>` returns `{"filename": "<title or run_id>.md",
"markdown": str}` for an open or closed session, or `404` for an unknown id.
The Markdown contains the title, creation date, user prompts, assistant text,
attachment names, and tool names, display-safe targets, and outcomes. It omits
reasoning, tool arguments, tool output, and image data.
`POST /session/rename` takes `{"run_id": str, "title": str}` and trims a
non-blank title of at most 200 characters; it returns `{"renamed": true}` or
`404` for an unknown session. `POST /session/delete` takes `{"run_id": str}`;
it returns `{"deleted": true}`, `404` when unknown, or `409` while a turn,
approval, question, or goal check is pending. An idle open session is closed
before its directory is removed. `POST /session/archive` takes
`{"run_id": str, "archived": bool}`, updates that metadata flag, and returns
`{"archived": bool}` or `404` for an unknown session. Archiving an open
conversation leaves it open. `POST /session/open` takes a `run_id` and makes that
conversation current. History is returned over HTTP, never published to the live
stream for this request. The reply includes an opaque `replay_id`, `history_frames`
(the total including waited-subagent notices), `offset: 0`, `history` (up to 250
`{kind: "event", payload: HistoryMessage}` frames), and `next_offset: int | null`.
`replayed` still counts saved records only. A zero count returns an empty history
and null next_offset.

Fetch subsequent pages with `POST /session/history`, body
`{"replay_id": str, "offset": int}`. Each reply has the same replay_id and
history_frames, its offset, history, and next_offset. Pages come from an immutable
snapshot; collect all frames and verify offsets/counts before painting once at the
bottom. The host retains the eight most recent snapshots; an expired replay returns
404 and the client can reopen to obtain a new one. Invalid offsets return 400.
Frames from different replay IDs must never be mixed. Live events received during
an open are held separately until the HTTP history is complete. Each history and runtime event
frame includes the originating `session_id`. Later prompts append to the same
session directory and transcript, while each prompt still has a distinct run
id. `POST /session/new` takes an empty JSON object, selects a new conversation
for the next prompt, and returns `{"ended": true}`. Opening, forking, and
starting another session leave other runs going. Opening a session that is
already open reuses its leader. At most four conversations may run at once; a
fifth start returns `409` with `{"error": "4 conversations are already running"}`.
Busy checks concern the current conversation. `POST /stop` stops only the
current conversation; open another one first to stop its run.

`GET /session/notes?session_id=<id>` returns
`{"text": <Markdown notes>, "updated_at": <ISO UTC time>}` for an existing
session, or `{"text": "", "updated_at": null}` before notes exist. Unknown
sessions and malformed session ids return `404` (`not_found`). Notes are
read-only. The conversation leader publishes `SessionNotesUpdated` after an
atomic notes write; clients may reload this route for that session.

`GET /session/agent?session_id=<id>&agent_id=<id>` returns
`{"session_id": <id>, "agent_id": <id>, "messages": [...]}` for one
subagent transcript. Each message has the `HistoryMessage` payload shape,
including tool calls with their results. This read-only route works for open
and closed sessions and publishes no event frames. Missing sessions, unknown
agents, and malformed agent ids return `404`.

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
for a session override). The host always loads its shipped USD estimates
(`symphonai_api/data/model_prices.json`, read 2026-11-05), then overlays
`~/.symphonai/prices.json` if present (respecting `SYMPHONAI_HOME`), then a
configured `budgets.price_table`. Later layers replace earlier prices per model;
a different currency replaces the entire table, so currencies never mix.
A host-supplied table overlays the shipped prices before file overrides.
Invalid override files report their path. Lookup tries exact model names first,
then a `-preview` or eight-digit dated suffix, as for model capabilities.
Conversation usage includes leader and subagent totals, with `unpriced_model`
naming the first unresolved model when no complete cost is available. Shipped
rates are estimates; the table includes its read date and sources.
A cost limit with no price for the requested model stops before
the first request with `budget_cost_unpriced`. A subagent definition's own budget is narrowed by the configured subagent
ceiling. There is no turn limit unless one is configured. The leader budget's
`max_turns` takes precedence over the host's `--max-turns`; without a leader
budget, the launch limit applies when set. Limits reset for each prompt.
`RunFinished.stopped_reason` reports `refused` when a model declines the request, or `max_turns`,
`budget_wall_time`, `budget_tokens`, `budget_cost`, or
`budget_cost_unpriced` when a limit ends a run;
the conversation remains available for another prompt.

Authenticated `POST /provider` takes `{"name": str, "model": str?,
"base_url": str?, "effort": str?}` and returns
`{"selected": true}`. `model`, `base_url`, and `effort` are optional non-empty
strings. The model and effort apply to the next leader request in the current
conversation. They do not replace a dispatched agent's own settings, and a
`leader` definition's explicit model or effort takes precedence over this choice.
The known providers are `anthropic`, `gemini`, `openai`, and the catalog keys
`deepseek`, `grok`, `groq`, `kimi`, `ollama`, `openrouter`, `qwen`, and
`together`. `openai` with a `base_url` selects an OpenAI-compatible endpoint;
catalog providers use their preset URL and reject `base_url`. Unknown providers,
malformed options, and providers without a configured API key return `400`. A provider
change rebuilds the conversation's leader for the next turn while preserving
the session and message history; provider-specific tool-call metadata from the
previous provider is discarded. A selection while a run is active returns
`409`. Reopening a selected conversation restores its provider choice from
session metadata. Both Leader providers use the conversation's selected
provider. When no selection was sent, the host uses
the first configured provider in Settings order (anthropic, gemini, openai,
then the catalog keys listed above).
If no provider has a key, the first prompt returns `400` until a key is added or
a valid selection is sent. Launch flags do not select a provider.

Authenticated `GET /models?provider=<name>` lists the models for a known
provider, with optional `base_url` for vendors that permit endpoint selection.
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
opaque id or `null` for a root). When known, each entry also has `state`:
`running`, `finished`, `failed`, or `stopped`; unknown state is omitted. Folded
agents report the state of their most recent run. `mode` reports the mode
currently in force. `instruction_warnings` is a list of instruction loading
warnings for conversations started by this host, and is empty for clean or
reopened conversations.
When project or private config was skipped, `config_warnings` contains the
file and reason; clean conversations omit that field.
Each agent appears once; the first run in the
timestamp-ordered run graph determines its parent. After a run completes in
this process, the payload also contains `context` (`used_tokens`, `budget_tokens`,
`remaining_tokens`, `window_tokens`, and `by_source`), aggregate `usage`, and per-agent
`input_tokens`, `output_tokens`, `calls`, `total_tokens`, `cache_read_tokens`,
and `cache_write_tokens`. Aggregate and per-agent token usage are rebuilt from
session usage records on reopen; `context` remains omitted until another run is
accounted in this process.
`plan` is the latest leader plan (`steps` and `explanation`), or `null` when
none has been set.
When every used model has a configured price, usage objects also contain
`cost` with decimal-string `amount`, `currency`, and `estimated: true`.
Every conversation cost is an estimate computed from token usage and a price
table, whether shipped, user-supplied, or configured; it is not a vendor bill.
Vendor-reported Usage figures remain separate. `cost` is omitted when no
price table exists or any used model is unpriced. The payload contains no
repository paths, credentials, or model request content.

Replayed assistant `HistoryMessage` tool calls include their `plan` and
`explanation` arguments for `update_plan`.

Replayed `ask_user` tool calls include `question` and `options`, plus
`answer` when answered or `skipped: true` when skipped. Replayed `propose_plan`
calls include `plan`, plus `decision` (`ask`, `allow`, or `keep_planning`) and
optional `feedback` when the person kept planning. Calls with no result or an
`ok: false` result have no decision fields. Tool result text is never sent to
the client.

## Goals

Authenticated `POST /goal` takes `{"objective": str, "check": [str, ...]?,
"max_rounds": int?}`. An omitted or empty check stores as `[]`. A check is an
argv run by the host in the repository root, outside the agent permission
policy. Checks supplied through `/goal` are the person's commands and run
unsandboxed. A spec run's Validation check runs in its worktree sandbox, with
network access following that run's sandbox policy. The default is 10 rounds; the limit is 1–100. A final response runs a
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
`/app/<asset path>`, `/events`, or any POST. The host does not log
request URLs.

An authenticated `GET /app/` response sets exactly one development session
cookie:

```text
Set-Cookie: symphonai_app=<token>; Path=/app/; HttpOnly; SameSite=Strict
```

It has no `Max-Age` or `Expires`. `GET /app/` accepts that cookie on later page
loads. `GET /app/<asset path>` accepts the cookie or the ordinary bearer header
so native browser stylesheet and module requests can load. No other route
accepts the cookie: `/events`, `/health`, and all POST requests retain
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
permission `mode`, the next-turn `selection` (`provider`, `model`, and
`effort`, or `null` when no provider is selected), `config`
entries (`key`, `value`, `scope`), `ceiling`, `trust`, `hooks` (`event`, `command`),
`mcp_servers` (`name`, `command`, `started`), sorted `agents`, `skills`, and
`plugins` entries (`name`, `path`), `withheld` entries (`scope`, `directory`, `names`, `reason`),
and `providers` entries (`name`, `env_var`, `key_present`). A withheld
`directory` and roster `path` are repository-relative within the repository
and absolute otherwise; an unknown roster path is `""`. Provider key presence
is a boolean; key values never appear in the
response. It is assembled from already loaded host state and makes no network
requests. Only the ordinary bearer header authorizes this route.

Authenticated `POST /extensions/reload` takes `{}` and reloads the current
project's config, trust, hooks, skills, agents, plugins and MCP definitions.
Preparation and MCP startup occur outside the host/runtime state locks; the
result applies to newly started conversations. The reply is
`{"changes": {"skills": delta, "agents": delta, "plugins": delta,
"mcp_servers": delta}, "error": null}`, where each delta contains sorted
`added`, `removed`, `changed` name lists and a `failed` list of error objects.
Config/definition errors return 400 with the file-naming `error` and retain the
previous extensions. MCP connection failures are reported per server without
discarding successfully loaded extensions. Unchanged MCP definitions reuse
their pool. A changed roster starts a replacement pool; removed pools close
immediately if no conversations reference them, otherwise at project shutdown
so existing conversations retain working clients. `/reload` and Reload in
Settings refresh the settings and the slash/mention menus.

Authenticated `POST /config/set` takes `{"key": "memory.extract", "value":
false, "scope": "user"}`. Scope is `user` or `project`; only scalar booleans,
finite numbers and strings may be set. Tables/lists and unknown sections are
refused, as are project writes to roles, permissions, always_deny and
sandbox.readable. Existing single-line assignments preserve surrounding text,
including comments and line endings; missing keys are placed under their table
or a new final header. Multiline scalar assignments require editing the file.
The candidate is parsed and validated with the layered config loader before an
atomic write. Validation errors return 400 without changing the file. A saved
value invokes the reload above and returns its result. General settings exposes
scalar editors with a destination scope and per-row errors; arrays/tables keep
an Open file hint.

`GET /memory` additionally includes each scope's `last_consolidation` (time,
merged groups with prior/replacement texts, removed ids/texts/reasons),
`has_previous`, and any `consolidation_error`. Authenticated `POST
/memory/restore` takes `{"scope": "project"}` or `{"scope": "global"}` and
atomically restores that scope's `.bak`, logging the restoration. It returns
`{"restored": scope}`; an absent or invalid backup returns 400. Settings shows
the consolidation and offers Restore previous for scopes with a backup.

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

Authenticated `POST /steer` accepts exactly `{"text": <non-blank string>}`.
It delivers the text to the open conversation's running leader for its next
tool call and returns `{"steered": true}`. With no running turn, or if the
leader has already finished, it returns `409 {"error": "no turn is running"}`.
Malformed requests return `400`.

Authenticated `POST /command` accepts exactly `{"command": <non-blank string>}`.
It runs the command in the repository with the run_shell environment and output
limit, a closed stdin, and a 120-second timeout. It returns
`{"exit_code": <int or null>, "output": <str>, "timed_out": <bool>, "stopped": <bool>}`. A
running turn returns `409 {"error": "wait for the turn to finish"}`; a running
command returns `409 {"error": "a command is already running"}`. While a command
runs, `POST /prompt` returns `409 {"error": "wait for the command to finish"}`.
Malformed requests return `400`. `POST /stop` stops the command's process group;
the reply has a null `exit_code`, `timed_out: false`, and `stopped: true`. The
host adds the command and result to the open conversation as user context, or
holds it for the next prompt when no conversation is open.

`GET /agent?name=<name>&scope=<user|project>` returns one agent definition as
`{"name": <stem>, "scope": <scope>, "text": <contents>, "fields": <fields>}`.
The host parses `fields` with `tomllib`: `prompt`, `provider`, `model`, and
`effort` are strings; `tools` and `deny_tools` are string arrays. Missing fields
have empty defaults. `POST /agent` takes `name`, `scope`, and exactly one of
`fields` (a partial field patch) or `text` (a complete TOML document). Field
patches preserve unchanged lines, comments and multi-line strings; changed
values are encoded as TOML strings or arrays by the host. The host
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

Authenticated `GET /git/status` returns the current project's working-tree
changes as `{"repo": true, "branch": str | null, "files": [...]}`. Each
file has its repository-relative `path`, `status` (`modified`, `added`,
`deleted`, `untracked`, or `renamed`), whether the change is staged, a diff
against `HEAD`, and `diff_truncated`. Untracked text files are shown as added
lines; binary diffs are omitted. Diffs are capped at 20,000 characters and
files sort by path. A directory outside a Git repository returns
`{"repo": false}`. This route reads Git state and does not modify the index.
The first 200 files include diffs; later files have `diff: null` and
`diff_deferred: true`. Authenticated `GET /git/diff?path=<repository-relative
path>` returns one changed file's diff and truncation flag. It returns `404`
when the path has no uncommitted change and `400` when the path leaves the
repository.

Authenticated `POST /git/commit` takes `{"message": <non-blank string>,
"paths": [<repository-relative path>, ...]}`. It commits the current contents
of exactly those changed paths, runs configured commit hooks, and returns
`{"commit": <sha>, "paths": [...]}`. Unchanged or outside paths are rejected
with `400`. A running turn or a failed commit returns `409`; commit failure
includes Git's last 4,000 output characters and leaves the existing index
unchanged. Unselected staged and unstaged changes are excluded from the commit.

Authenticated `GET /git/pull-request` returns the current branch, `origin`'s
default branch, latest commit subject, ahead count, and an optional `problem`
that explains why a pull request cannot be opened. `POST /git/pull-request`
takes `{"title": <non-blank string>, "body": <string>}`. It pushes the current
branch to `origin`, then runs `gh pr create` with the supplied title and body,
the default base, and the current head. Success returns `{"url": <URL>}`.
Missing `gh`, missing authentication or remote, detached `HEAD`, a base branch
with no commits ahead, a running turn, or a failed command returns `409` with
the problem or command output. Working-tree changes are not pushed.

Authenticated `GET /jobs` lists the current project's background jobs, newest
first, with owner, session id, label, status, exit code, start and end times,
the last 4,000 output characters, and whether earlier output was omitted.
`POST /jobs/start` takes `{"command": <non-blank string>}` and starts the
person's unsandboxed project command with closed stdin and no timeout. At most
eight of the person's jobs may run; another start returns `409`.
`POST /jobs/stop` takes `{"id": <job id>}` and returns its final status, or
`404` for an unknown id. A `JobChanged` event is published when a job starts
and ends; its `session_id` is null for a person's job.

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

### POST /agent/draft

Takes `{"description": "…"}`: a non-blank string of at most 2,000 characters.
Makes one tool-free model request using the current provider, model and effort.
Returns 200 with `name`, `prompt`, `tools`, `deny_tools`, optional `provider`,
`model`, `effort`, and `dropped` (unknown tool names removed from either list).
Other model fields are omitted. Name rules match saved agent definitions.
The draft is unsaved; POST /agent remains the save operation.
Invalid descriptions return 400, no selected provider returns 409, and invalid
model replies or provider failures return 502 with a short, credential-free reason.


## Person's terminal

These authenticated, project-scoped routes control the person's own unsandboxed
login shell, separate from agent commands and jobs. Its PTY starts in the
project root with the person's environment and TERM=dumb, PAGER=cat and
GIT_PAGER=cat. At most four shells may be open.

- `POST /terminal/start {}` returns `{id}`.
- `POST /terminal/input {id, data}` writes UTF-8 keystrokes and returns `{written: true}`.
- `POST /terminal/resize {id, cols, rows}` sets a size from 1 to 1000 in each dimension and returns `{resized: true}`.
- `POST /terminal/stop {id}` stops the shell and its job process groups and returns `{stopped: true}`.
- `GET /terminals` returns `{terminals: [{id, exited, exit_code, output}]}` with a bounded output tail.

Unknown terminal ids return 404; invalid input or a fifth running terminal
returns 400. TerminalOutput events carry incremental output, followed by one
exit event. Closing the Terminal tab stops its shell; closing the host stops
all shells. The plain terminal view handles colours and basic line/screen
editing, not full-screen applications such as vim or top.


The `files` array in `GET /files?query=<text>&limit=<n>` includes readable
folders as well as files. Folder paths end in `/`, including empty folders;
files and folders share the same query ranking and bounded search. Choosing a
folder inserts its path as an @ mention; the agent lists the folder itself.
The composer's @ menu additionally offers matching configured skills after
paths. Skill mentions add one `Use the <name> skill.` line per distinct skill,
a blank line, then the original typed text. The app displays that typed text.

### MCP sign-in

Settings `mcp_servers` rows include `name`, `command` (or HTTP URL), `started`,
`state` (`started`, `needs_sign_in`, `failed`, or `disabled`), and a safe `error`.
A failed server is skipped independently; other servers and the project remain usable.

`POST /mcp/sign-in {"server":"name"}` returns `{"authorization_url":"..."}`.
Open that URL in a browser. OAuth uses PKCE S256, dynamic registration and the
MCP resource indicator. `GET /mcp/callback?state=...&code=...` is the only
unauthenticated OAuth endpoint: state is single-use and expires in ten minutes.
Its response says `Signed in to <server>. You can close this tab.` on success.
Tokens are stored only in `<SymphonAI home>/credentials/mcp-<server>.json`,
written atomically with mode 0600 inside a mode-0700 directory. There is no OS
credential-store lookup or migration; credentials saved there by an older build
are ignored, requiring sign-in again if no guarded file exists. No token is
passed in a subprocess argument. The callback restarts only that server. New conversations
receive its tools; existing conversations retain their tool registries.
Access and refresh tokens never appear in these replies. Expired tokens refresh
before requests, with one refresh retry after a 401; refresh failure requires sign-in.

### MCP resources and prompts

`GET /mcp/prompts` returns `{"prompts":[{"server":"name","name":"prompt",
"description":"...","arguments":[{"name":"topic","required":true}]}]}`
for started servers declaring prompt capabilities. Lists follow the server's
pagination cursors. `POST /mcp/prompt` takes `server`, `name` and an `arguments`
object of strings and returns `{"text":"..."}` from the prompt messages' text
blocks. Unknown servers/prompts, undeclared arguments and missing required
arguments are rejected. The app offers `/mcp.<server>.<prompt>` commands; one
argument receives the whole remaining text, while multiple arguments use
`key=value` pairs (quoted values may contain spaces). The rendered text becomes
the person's message.

When started servers advertise resources, new agent registries include the
read-only `list_mcp_resources(server?, cursor?)` and
`read_mcp_resource(server, uri)` tools. Lists return at most 100 entries with
`server`, `uri`, `name`, `description`, `mimeType`, and an optional opaque
`next_cursor`. Pass that cursor back with the same server selection to continue;
it preserves the underlying server cursor and position within oversized pages.
Reads return text and describe binary contents without inlining their base64.
Normal tool output limits/offload apply. Tools-only servers add no resource
tools and no prompt commands; MCP tool deferral is unchanged.

### Power

`[power] keep_awake = false` in user or project configuration disables idle-sleep
prevention for that project's host and headless runs. It defaults to true;
explicit values appear as `power.keep_awake` in Settings' existing config rows.
Configuration is read when the project starts. Host activity is polled every ten
seconds; approval waiting alone is idle. Headless run/loop commands hold through
completion. This prevents idle sleep, not lid closure, shutdown or quitting.

Terminal snapshots include `end`, the total Unicode code-point count read so far,
including output trimmed from the 200,000-character scrollback. Each
`TerminalOutput` includes `offset`, the count before that event's `data`.
Reattach writes saved output, then only buffered event data past the snapshot's
`end`; identical repeated text is not used to infer overlap. Output events are
coalesced with at least 30 ms between publications. Input arriving while a
terminal input request is in flight is batched into the next request.

### Viewed and unread conversations

`POST /session/viewed {"session_id":"…"}` records UTC `viewed_at` in the
current project's session metadata and returns `{session_id, viewed_at}`.
Unknown sessions return 404, foreign-project sessions 409, invalid IDs 400.
`GET /sessions` includes `unread: boolean`: a non-spec conversation never
viewed, or a session whose last run ended after its viewed time. Activity
`waiting` takes visual precedence over unread. The app records viewed on open
and when the visible conversation's root run finishes.

### Usage over time

`GET /usage?days=<n>` returns usage for the last `n` UTC calendar days including
today (default 30, integer 1–90). Invalid ranges return 400. It reads all saved
projects under the host's sessions root; it does not change per-call records.

Response:

- `days`, `starting_at`, `ending_at`: selected period, with UTC timestamps.
- `tokens`: rows `{date, provider, model, project, ...usage counters}`.
- `providers`: rows `{provider, models, ...usage counters, cost, balance, error}`.
  `models` contains model totals. Counters retain input, output, calls, cache
  read/write, one-hour writes and long-context fields from saved usage.
- `cost`: null or `{amount, currency, source, scope}`, with decimal-string
  amounts. `source` is `vendor` or `estimate`. Vendor cost covers the entire
  organization, whereas estimates cover local saved calls. A vendor failure
  leaves its error visible even if a price-file estimate is available.
  Anthropic includes `data_refreshed_at` when returned by the vendor.
- `balance`: null or an array of `{amount, currency}`. Moonshot uses provider
  ID `kimi`; its currency is null because the vendor does not state one.
  DeepSeek preserves each returned currency. OpenRouter credits minus usage
  are USD. Balances describe the current account, not the selected period.
- `projects`: totals by `repo_root`; `daily`: zero-filled UTC daily totals;
  `warnings`: unreadable/invalid saved-record warnings without transcript data.

The host's usage module alone reads OPENAI_ADMIN_KEY and ANTHROPIC_ADMIN_KEY.
It sends them only as Authorization headers to fixed vendor endpoints, never
stores the keys or exposes headers/errors containing them, and refuses HTTP
redirects. Cost reports and balances are cached in memory for 15 minutes;
credential changes invalidate their cached results. Failures stay in the
provider's row and do not fail other providers. No admin key is required for
local tokens or price-file estimates. Normal DEEPSEEK_API_KEY,
MOONSHOT_API_KEY and OPENROUTER_API_KEY enable their balances. Gemini has no
vendor cost-report request and can show only a price-file estimate.

Anthropic requests use the supplied analytics beta header, RFC3339 bounds,
31-day windows and pagination; fractional cents are divided by 100. OpenAI
costs use daily buckets and pagination; amounts are already dollars. All
amounts use decimal arithmetic. `/stats` opens the app's Usage main pane with
7/30/90-day controls and daily token bars; `/cost` remains conversation-only.

Saved usage predating provider fields uses the session's provider selection.
Sessions known to have switched providers group untagged calls as `unknown`
and include a warning rather than assigning them all to the latest provider.
Per-call recording is unchanged. Inherited usage in forked root transcripts
is excluded until the fork's first new run, avoiding counting the copied
prefix as new usage.

### Element marks in local web previews

`POST /preview/start {"url":"http://127.0.0.1:3000"}` starts the project's
preview and returns `{preview_url}` on its own loopback port. Omitting url
reuses the last URL from this project's host lifetime. Only http URLs whose
host is exactly localhost, 127.0.0.1 or [::1] are accepted; credentials and
fragments are rejected. The proxy connects only to literal loopback addresses,
never follows upstream redirects itself, and never substitutes the host's
bearer credentials. `/preview stop` calls authenticated `POST /preview/stop`.
Starting another preview invalidates the previous preview token; project
closure stops the proxy and its tunnels.

The proxy mirrors origin-relative paths, uses identity encoding, injects a
picker into HTML, and tunnels WebSocket traffic. HTML injection adjusts body
length and removes stale encoding, ETag and CSP headers; ordinary HTTP uses
connection-close framing while retaining body bytes. Compressed gzip/deflate
HTML is decoded before injection. The picker runs in a separate window with
no opener reference and receives only a per-preview token.

`POST /preview/mark` accepts the picker's `X-Preview-Token`, with
`{comment,url,selector,html,text,source_hint}`. It is also available directly
on the preview origin, avoiding cross-origin host access. Both paths use the
same token validation. Marks are accepted only for that preview's page
origin or its target origin. Invalid/missing tokens return 403. HTML is
limited to 2,000 characters; marks are secret-masked. Missing source metadata
uses a bounded, permission-checked project search and returns
`source_guess: true` with the best matching relative path or an empty hint.

Authenticated `GET /preview/marks` returns `{session_id,marks}` for the
current conversation (null for a not-yet-started chat). Each mark adds an
`id`. Creation emits `PreviewMarkAdded` with `{mark,session_id,repo_root}`
plus normal event identity fields. This event updates composer chips and is
not a chat entry. Pending marks remain in the host for its lifetime.

The same POST mark route accepts `{remove:[ids],session_id}` only with the
host's normal bearer authorization, for chip removal and clearing accepted
messages. The preview token grants creation only and cannot remove marks,
read marks, stop/start previews, change mode or call other host routes.
The app combines selected marks and the person's text into one prompt,
including comment, URL, selector, fenced HTML and a source hint labelled
"guess" when appropriate. Marks are cleared after the prompt is accepted.

### Leader notes

The leader-only `notify_person(text)` tool accepts nonblank text up to 500
characters, returns `sent` without waiting for a reply, and emits
`PersonNotice {agent_id,run_id,text}`. The text is secret-masked. The saved
assistant tool call and successful tool result carry the note; history replay
exposes its text rather than creating a person prompt or final answer.
The app shows `Leader: …` and uses its existing notification preference and
background-notification behavior. Headless runs print `note: …` to stderr and
include current-run notes in the result's `notes` array, including --json.

### Auto approval review

Permission modes are plan, ask, auto and allow, in increasing capability order.
`auto` has its own meaning and is no longer a renamed value for allow. The
app offers it and `/mode [ask|plan|auto|allow]` accepts an optional value;
without a value the picker opens. Headless accepts `--mode auto`.

Fixed permission rules still run first; existing grants bypass model review.
For other approvals the broker reviews using the conversation's own model,
with low effort only if its capabilities list it, and background call class.
The request contains operation, target, command(s)/diff, project root, the
latest person message and the calling agent's last text. These excerpts are
secret-masked and labelled as untrusted data. The required reply is
`{"decision":"approve"|"deny","reason":"<one sentence>"}`. A single JSON
fence is accepted; null optional fields are ignored. Unknown decisions fail.
A missing/null reason uses a neutral default rather than losing the decision.

Each decision emits `AutoReviewed` with operation, target, decision, reason,
tool_call_id and standard event identity. A denial names the auto reviewer
and its reason. Review calls are bounded to 30 seconds and cancellable; usage
is recorded as the leader's background usage even if parsing fails. An app
review failure falls back to its existing approval interaction; headless
review failure is denied without asking a connected client. A narrower
calling-agent mode retains its ordinary approval behavior.

Review decisions use the `auto_reviewed` transcript record and are joined to
their tool call by tool_call_id during history replay. The app renders
`Auto-approved: …` or `Auto-denied: …` beside the call. They are not person
prompts and are not sent as model messages. This separate record preserves
decisions for refused calls and failed actions without rewriting tool output.

Live `ToolCallFinished` for `dispatch_subagent` includes the text the leader
received as `output`, secret-masked before truncation to 4,000 characters,
with `output_truncated` indicating omitted text. This uses the same bound and
masking as replayed dispatch output. Plan/question/proposal and file-edit
outputs remain hidden as before; dispatch failures continue to use the
existing ToolCallFailed reason in chat.
