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
  dispatch_subagent: "dispatched",
  remember: "remembered",
  forget: "Forgot",
});

const RUNNING_ACTIVITY_VERBS = Object.freeze({
  read_file: "reading",
  grep: "searching for",
  glob: "searching for",
  list_files: "listing",
  edit_file: "editing",
  multi_edit_file: "editing",
  write_file: "writing",
  run_shell: "running",
  dispatch_subagent: "dispatching",
  remember: "remembering",
  forget: "forgetting",
});

const ACTIVITY_KINDS = Object.freeze({
  read_file: "activity-read",
  list_files: "activity-read",
  grep: "activity-search",
  glob: "activity-search",
  edit_file: "activity-edit",
  multi_edit_file: "activity-edit",
  write_file: "activity-edit",
  run_shell: "activity-shell",
  web_fetch: "activity-web",
  web_search: "activity-web",
  remember: "activity-memory",
  forget: "activity-memory",
});

function renderPrompt(document, entry) {
  const prompt = element(document, "p", { className: "prompt", text: entry.text });
  for (const attachment of entry.attachments ?? []) {
    const label = attachment.kind === "document"
      ? `PDF · ${attachment.filename || "document"}`
      : `Image${attachment.filename ? ` · ${attachment.filename}` : ""}`;
    append(prompt, element(document, "span", { className: "attachment", text: label }));
  }
  return prompt;
}

function renderSteer(document, entry) {
  const prompt = element(document, "div", { className: "prompt steer" });
  append(
    prompt,
    element(document, "span", { className: "steer-label", text: "Sent while working" }),
    element(document, "p", { text: entry.text }),
  );
  return prompt;
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

function renderText(document, entry, parseMarkdown, actions) {
  const assistant = element(document, "div", { className: "assistant" });
  append(assistant, ...parseMarkdown(entry.text).map((block) => renderBlock(document, block, actions)));
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
  block.ordered && list.setAttribute("start", String(block.start));
  append(list, ...block.items.map((item) => {
    const row = renderInlineBlock(document, "li", item.children);
    append(row, ...item.blocks.map((nested) => renderBlock(document, nested)));
    return row;
  }));
  return list;
}

function renderTableCell(document, tagName, children, alignment) {
  const cell = renderInlineBlock(document, tagName, children);
  alignment && cell.setAttribute("style", `text-align: ${alignment};`);
  return cell;
}

function renderTable(document, block) {
  const header = element(document, "thead");
  const headerRow = element(document, "tr");
  append(headerRow, ...block.header.map((cell, index) => renderTableCell(
    document,
    "th",
    cell,
    block.align[index],
  )));
  append(header, headerRow);

  const body = element(document, "tbody");
  append(body, ...block.rows.map((row) => {
    const tableRow = element(document, "tr");
    append(tableRow, ...row.map((cell, index) => renderTableCell(
      document,
      "td",
      cell,
      block.align[index],
    )));
    return tableRow;
  }));

  const table = element(document, "table");
  append(table, header, body);
  return append(element(document, "div", { className: "table-scroll" }), table);
}

function renderRule(document) {
  return element(document, "hr");
}

function renderCodeBlock(document, block, actions) {
  const pre = element(document, "pre");
  const code = element(document, "code", { text: block.text });
  code.setAttribute("data-language", block.language ?? "");
  const copy = element(document, "button", { className: "code-copy" });
  copy.type = "button";
  copy.setAttribute("aria-label", "Copy code");
  copy.setAttribute("title", "Copy code");
  copy.addEventListener("click", async () => {
    try {
      await actions?.copyText?.(block.text);
      copy.setAttribute("title", "Copied");
      setTimeout(() => copy.setAttribute("title", "Copy code"), 2000);
    } catch {
      copy.setAttribute("title", "Could not copy");
    }
  });
  return append(pre, code, copy);
}

function renderQuote(document, block) {
  return renderInlineBlock(document, "blockquote", block.children);
}

const BLOCK_RENDERERS = Object.freeze({
  paragraph: renderParagraph,
  heading: renderHeading,
  list: renderList,
  table: renderTable,
  rule: renderRule,
  "code-block": renderCodeBlock,
  quote: renderQuote,
});

function renderBlock(document, block, actions) {
  return BLOCK_RENDERERS[block.type](document, block, actions);
}

export function renderMarkdownBlocks(document, value, parseMarkdown) {
  return parseMarkdown(value).map((block) => renderBlock(document, block));
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

function renderGoal(document, entry) {
  const command = element(document, "div", { className: "command" });
  const details = element(document, "details", { className: "goal-check-output" });
  append(
    details,
    element(document, "summary", { text: "Check output" }),
    element(document, "pre", { text: entry.output }),
  );
  append(
    command,
    element(document, "p", { className: "command-output", text: entry.text }),
    ...(entry.output ? [details] : []),
  );
  return command;
}

function renderPlan(document, entry, parseMarkdown) {
  const plan = element(document, "section", { className: "plan-entry" });
  append(plan, ...(entry.explanation ? [
    append(
      element(document, "div", { className: "plan-explanation" }),
      ...parseMarkdown(entry.explanation).map((block) => renderBlock(document, block)),
    ),
  ] : []));
  const list = element(document, "ol", { className: "plan-steps" });
  for (const item of entry.steps) {
    const step = element(document, "li", { className: `plan-step ${item.status}` });
    const marker = item.status === "completed" ? "✓" : item.status === "in_progress" ? "●" : "○";
    append(step,
      element(document, "span", { className: "plan-marker", text: marker }),
      element(document, "span", { className: "plan-step-text", text: item.step }),
    );
    append(list, step);
  }
  append(plan, list);
  return plan;
}

function renderUserQuestion(document, entry) {
  const question = element(document, "div", { className: "user-question" });
  append(
    question,
    element(document, "p", { className: "user-question-text", text: entry.question }),
    ...(entry.answer
      ? [element(document, "p", { className: "user-question-answer", text: entry.answer })]
      : []),
  );
  return question;
}

function renderUserPlan(document, entry, parseMarkdown) {
  const plan = element(document, "section", { className: "user-plan" });
  append(
    plan,
    ...parseMarkdown(entry.plan).map((block) => renderBlock(document, block)),
    ...(entry.outcome ? [element(document, "p", { className: "user-plan-outcome", text: entry.outcome })] : []),
  );
  return plan;
}

function renderActivity(document, entry, _parseMarkdown, actions) {
  const failed = entry.calls.some((call) => call.status === "failed");
  const running = entry.calls.some((call) => call.status === "running");
  const kind = ACTIVITY_KINDS[entry.calls[0]?.name];
  const dispatches = entry.calls.filter((call) => call.name === "dispatch_subagent" && typeof call.task === "string");
  const outputCalls = entry.calls.filter((call) => typeof call.output === "string" && call.output.length > 0);
  const expandable = dispatches.length > 0 || outputCalls.length > 0;
  const line = element(document, expandable ? "div" : "p", {
    className: ["activity", kind, failed ? "failed" : ""].filter(Boolean).join(" "),
  });
  expandable && line.setAttribute("role", "button");
  expandable && line.setAttribute("tabindex", "0");
  expandable && line.setAttribute("aria-expanded", String(entry.expanded === true));
  const text = element(document, "span", { className: "activity-text" });
  const endsWithQuotedMemory = entry.calls.at(-1)?.target !== null
    && ["remember", "forget"].includes(entry.calls.at(-1)?.name);
  let sentence = "";
  append(
    text,
    ...entry.calls.flatMap((call, index, calls) => {
      const verb = call.status === "running"
        ? RUNNING_ACTIVITY_VERBS[call.name] ?? `using ${call.name}`
        : ACTIVITY_VERBS[call.name] ?? `used ${call.name}`;
      const label = index === 0 ? `${verb.charAt(0).toUpperCase()}${verb.slice(1)}` : verb;
      const separator = index === 0 ? "" : index === calls.length - 1
        ? calls.length === 2 ? " and " : ", and "
        : ", ";
      const failure = call.status === "failed"
        ? ` (failed${call.error ? `: ${call.error}` : ""})`
        : "";
      const memoryTarget = call.target !== null && ["remember", "forget"].includes(call.name);
      const targetText = call.target === null
        ? ""
        : memoryTarget ? ` "${call.target}"` : ` ${call.target}`;
      sentence += `${separator}${label}${targetText}${failure}`;
      return [
        ...(separator ? [element(document, "span", { text: separator })] : []),
        element(document, "span", { text: label }),
        ...(call.target === null ? [] : [
          element(document, "span", { text: memoryTarget ? ` "${call.target}"` : " " }),
          ...(memoryTarget ? [] : [element(document, "code", { text: call.target })]),
        ]),
        ...(failure ? [element(document, "span", { text: failure })] : []),
      ];
    }),
    ...(!endsWithQuotedMemory
      ? [element(document, "span", { text: running ? "…" : "." })]
      : []),
  );
  append(line, text);
  line.setAttribute("title", `${sentence}${endsWithQuotedMemory ? "" : running ? "…" : "."}`);
  const toggle = () => {
    entry.expanded = entry.expanded !== true;
    actions?.rerender?.();
  };
  expandable && line.addEventListener("click", toggle);
  expandable && line.addEventListener("keydown", (event) => (
    ["Enter", " "].includes(event.key) && (event.preventDefault(), toggle())
  ));
  const taskBlock = expandable && entry.expanded
    ? element(document, "div", { className: "dispatch-tasks" })
    : null;
  taskBlock && append(line, taskBlock);
  taskBlock && dispatches.forEach((call) => append(taskBlock, element(document, "p", {
    className: "dispatch-task",
    text: `${call.target}: ${call.task}`,
  })));
  taskBlock && outputCalls.forEach((call) => {
    const label = ACTIVITY_VERBS[call.name] ?? call.name;
    const output = element(document, "section", { className: "activity-output" });
    append(output, element(document, "p", {
      className: "activity-output-label",
      text: `${label.charAt(0).toUpperCase()}${label.slice(1)}${call.target ? ` ${call.target}` : ""}`,
    }));
    append(output, element(document, "pre", { text: call.output }));
    call.outputTruncated && append(output, element(document, "p", {
        className: "activity-output-truncated",
        text: "… output truncated",
      }));
    append(taskBlock, output);
  });
  return line;
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

function renderEditCard(document, entry, _parseMarkdown, actions) {
  const card = element(document, "section", { className: "edit-card" });
  const heading = element(document, "div", { className: "edit-card-heading" });
  append(heading,
    element(document, "strong", { text: `Edited ${entry.files.length} ${entry.files.length === 1 ? "file" : "files"}` }),
    element(document, "span", { text: `+${entry.added} −${entry.removed}` }),
  );
  const undo = element(document, "button", { text: "Undo" });
  undo.type = "button";
  undo.disabled = entry.undoing === true;
  undo.addEventListener("click", () => actions?.onUndo?.(entry));
  const view = element(document, "button", { text: "View changes" });
  view.type = "button";
  view.addEventListener("click", () => actions?.onViewChanges?.());
  append(heading,
    ...(entry.undone ? [element(document, "span", { text: "Undone" })] : []),
    ...(!entry.undone && entry.canUndo && actions?.onUndo ? [undo] : []),
    ...(actions?.onViewChanges ? [view] : []),
  );
  append(card, heading);
  const rows = element(document, "ul", { className: "edit-card-files" });
  const visible = entry.expanded ? entry.files : entry.files.slice(0, 3);
  for (const file of visible) {
    append(rows, element(document, "li", { text: `${file.path}  +${file.added} −${file.removed}` }));
  }
  append(card, rows);
  const more = element(document, "button", {
    className: "edit-card-more",
    text: entry.expanded ? "Show fewer files" : `Show ${entry.files.length - 3} more files`,
  });
  more.type = "button";
  more.addEventListener("click", () => {
    entry.expanded = !entry.expanded;
    actions?.rerender?.();
  });
  append(card, ...(entry.files.length > 3 ? [more] : []));
  const conflicts = element(document, "ul");
  for (const path of entry.conflicts ?? []) append(conflicts, element(document, "li", { text: path }));
  const force = element(document, "button", { text: "Revert anyway" });
  force.type = "button";
  force.addEventListener("click", () => actions?.onUndo?.(entry, true));
  append(card,
    ...(entry.conflicts?.length ? [
      element(document, "p", { text: "These files changed outside the agent:" }), conflicts, force,
    ] : []),
    ...(entry.error ? [element(document, "p", { className: "error", text: entry.error })] : []),
  );
  return card;
}

function renderQuestion(document, entry) {
  const state = entry.resolved
    ? entry.allowed ? "allowed" : `denied${entry.reason ? `: ${entry.reason}` : ""}`
    : "waiting for an answer";
  return element(document, "p", {
    className: `question ${entry.resolved ? "resolved" : "open"}`,
    text: `${entry.toolName} needs approval — ${state}.`,
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
  steer: renderSteer,
  text: renderText,
  command: renderCommand,
  goal: renderGoal,
  plan: renderPlan,
  "user-question": renderUserQuestion,
  "user-plan": renderUserPlan,
  activity: renderActivity,
  edit: renderEdit,
  "edit-card": renderEditCard,
  question: renderQuestion,
  compaction: renderCompaction,
  gap: renderGap,
  unknown: renderUnknown,
  "turn-time": (document, entry) => element(document, "p", {
    className: "turn-time",
    text: entry.text,
  }),
});

export function renderTranscript(document, root, model, parseMarkdown, actions) {
  return replace(
    root,
    ...model.map((entry) => (
      TRANSCRIPT_RENDERERS[entry.type] ?? renderUnknown
    )(document, entry, parseMarkdown, actions)),
  );
}
