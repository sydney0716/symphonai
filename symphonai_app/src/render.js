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

function renderInline(document, node) {
  return INLINE_RENDERERS[node.type](document, node);
}

function renderInlineBlock(document, tagName, children) {
  const renderer = children.length === 1 && children[0].type === "text"
    ? INLINE_BLOCK_RENDERERS.plain
    : INLINE_BLOCK_RENDERERS.rich;
  return renderer(document, tagName, children);
}

function renderPlainInlineBlock(document, tagName, children) {
  return element(document, tagName, { text: children[0].text });
}

function renderRichInlineBlock(document, tagName, children) {
  const block = element(document, tagName);
  append(block, ...children.map((child) => renderInline(document, child)));
  return block;
}

const INLINE_BLOCK_RENDERERS = Object.freeze({
  plain: renderPlainInlineBlock,
  rich: renderRichInlineBlock,
});

function renderText(document, entry, parseMarkdown) {
  const assistant = element(document, "div", { className: "assistant" });
  append(assistant, ...parseMarkdown(entry.text).map((block) => renderBlock(document, block)));
  return assistant;
}

function renderInlineCode(document, node) {
  return element(document, "code", { text: node.text });
}

function renderInlineChildren(document, tagName, node) {
  const block = element(document, tagName);
  append(block, ...node.children.map((child) => renderInline(document, child)));
  return block;
}

function renderLink(document, node) {
  const link = element(document, "a", { text: node.label });
  link.setAttribute("href", node.url);
  link.setAttribute("rel", "noopener noreferrer");
  link.setAttribute("target", "_blank");
  return link;
}

function renderLineBreak(document) {
  return element(document, "br");
}

const INLINE_RENDERERS = Object.freeze({
  text: (document, node) => element(document, "span", { text: node.text }),
  break: renderLineBreak,
  code: renderInlineCode,
  strong: (document, node) => renderInlineChildren(document, "strong", node),
  em: (document, node) => renderInlineChildren(document, "em", node),
  link: renderLink,
});

function renderParagraph(document, block) {
  return renderInlineBlock(document, "p", block.children);
}

function renderHeading(document, block) {
  return renderInlineBlock(document, `h${block.level}`, block.children);
}

function renderList(document, block) {
  const list = element(document, block.ordered ? "ol" : "ul");
  append(list, ...block.items.map((children) => renderInlineBlock(document, "li", children)));
  return list;
}

function renderCodeBlock(document, block) {
  const pre = element(document, "pre");
  const code = element(document, "code", { text: block.text });
  code.setAttribute("data-language", block.language ?? "");
  return append(pre, code);
}

function renderQuote(document, block) {
  return renderInlineBlock(document, "blockquote", block.children);
}

const BLOCK_RENDERERS = Object.freeze({
  paragraph: renderParagraph,
  heading: renderHeading,
  list: renderList,
  "code-block": renderCodeBlock,
  quote: renderQuote,
});

function renderBlock(document, block) {
  return BLOCK_RENDERERS[block.type](document, block);
}

function renderCommand(document, entry) {
  const command = element(document, "div", { className: "command" });
  append(
    command,
    element(document, "p", { className: "command-echo", text: entry.command }),
    ...entry.output.map((text) => element(document, "p", { className: "command-output", text })),
  );
  return command;
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
  command: renderCommand,
  activity: renderActivity,
  edit: renderEdit,
  question: renderQuestion,
  compaction: renderCompaction,
  gap: renderGap,
  unknown: renderUnknown,
});

export function renderTranscript(document, root, model, parseMarkdown) {
  return replace(
    root,
    ...model.map((entry) => (
      TRANSCRIPT_RENDERERS[entry.type] ?? renderUnknown
    )(document, entry, parseMarkdown)),
  );
}
