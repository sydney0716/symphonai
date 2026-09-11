export function element(document, tagName, { className = "", text = "" } = {}) {
  const value = document.createElement(tagName);
  value.className = className;
  value.textContent = text;
  return value;
}

export function append(parent, ...children) {
  parent.append(...children);
  return parent;
}

export function replace(parent, ...children) {
  parent.replaceChildren(...children);
  return parent;
}

export function listen(value, type, listener) {
  value.addEventListener(type, listener);
  return value;
}

const ACTIVITY_VERBS = Object.freeze({
  read_file: "read",
  grep: "searched for",
  glob: "searched for",
  list_files: "listed",
  edit_file: "edited",
  multi_edit_file: "edited",
  write_file: "wrote",
  run_shell: "ran",
});

function proseList(parts) {
  return parts.length < 2
    ? parts.join("")
    : parts.length === 2
      ? parts.join(" and ")
      : `${parts.slice(0, -1).join(", ")}, and ${parts.at(-1)}`;
}

function activityText(entry) {
  const parts = entry.calls.map((call) => {
    const verb = ACTIVITY_VERBS[call.name] ?? `used ${call.name}`;
    const target = call.target === null ? "" : ` \`${call.target}\``;
    const failure = call.status === "failed"
      ? ` (failed${call.error ? `: ${call.error}` : ""})`
      : "";
    return `${verb}${target}${failure}`;
  });
  const sentence = proseList(parts);
  return `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.`;
}

function renderPrompt(document, entry) {
  return element(document, "p", { className: "prompt", text: entry.text });
}

function renderText(document, entry) {
  return element(document, "p", { className: "assistant", text: entry.text });
}

function renderActivity(document, entry) {
  return element(document, "p", {
    className: "activity",
    text: activityText(entry),
  });
}

function renderEdit(document, entry) {
  const details = element(document, "details", { className: "edit" });
  append(
    details,
    element(document, "summary", {
      text: `Edited ${entry.path} (+${entry.added} −${entry.removed})${entry.truncated ? " · diff truncated" : ""}`,
    }),
    element(document, "pre", { text: entry.diff }),
  );
  return details;
}

function renderQuestion(document, entry) {
  const state = entry.resolved
    ? entry.allowed ? "allowed" : `denied${entry.reason ? `: ${entry.reason}` : ""}`
    : "waiting for an answer";
  return element(document, "p", {
    className: `question ${entry.resolved ? "resolved" : "open"}`,
    text: `${entry.toolName} asks for ${entry.mode} permission — ${state}.`,
  });
}

function renderCompaction(document, entry) {
  return element(document, "p", {
    className: "compaction",
    text: `Compacted ${entry.beforeTokens} to ${entry.afterTokens} tokens; ${entry.droppedMessages} messages dropped.`,
  });
}

function renderGap(document, entry) {
  return element(document, "p", {
    className: "gap",
    text: `${entry.dropped} events were dropped.`,
  });
}

function renderUnknown(document, entry) {
  return element(document, "p", {
    className: "unknown-event",
    text: `Received ${entry.event?.type ?? entry.type}.`,
  });
}

const TRANSCRIPT_RENDERERS = Object.freeze({
  prompt: renderPrompt,
  text: renderText,
  activity: renderActivity,
  edit: renderEdit,
  question: renderQuestion,
  compaction: renderCompaction,
  gap: renderGap,
  unknown: renderUnknown,
});

export function renderTranscript(document, root, model) {
  return replace(
    root,
    ...model.map((entry) => (
      TRANSCRIPT_RENDERERS[entry.type] ?? renderUnknown
    )(document, entry)),
  );
}
