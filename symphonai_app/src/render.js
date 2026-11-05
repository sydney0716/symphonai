export function button(document, text, onClick, options = {}) {
  const control = element(document, "button", { text, ...options });
  control.type = options.type ?? "button";
  onClick && listen(control, "click", onClick);
  return control;
}

export function errorText(error) {
  return error instanceof Error ? error.message : String(error);
}

export function element(document, tagName, { className = "", text = "", attributes = {} } = {}) {
  const value = document.createElement(tagName);
  value.className = className;
  value.textContent = text;
  for (const [name, attribute] of Object.entries(attributes)) value.setAttribute(name, attribute);
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
  forget: "forgot",
  read_tool_result: "read an earlier result",
  job_output: "checked a job",
  job_stop: "stopped a job",
  web_fetch: "fetched",
  web_search: "searched the web for",
  use_skill: "used skill",
  lsp: "looked up code",
  get_goal: "checked the goal",
  update_goal: "updated the goal",
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
  read_tool_result: "reading an earlier result",
  job_output: "checking a job",
  job_stop: "stopping a job",
  web_fetch: "fetching",
  web_search: "searching the web for",
  use_skill: "using skill",
  lsp: "looking up code",
  get_goal: "checking the goal",
  update_goal: "updating the goal",
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

function renderPrompt(document, entry, _parseMarkdown, actions) {
  const prompt = element(document, "p", { className: "prompt", text: actions?.promptText?.(entry.text) ?? entry.text });
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

function renderInline(document, node, actions) {
  return INLINE_RENDERERS[node.type](document, node, actions);
}

function renderInlineBlock(document, tagName, children, actions) {
  const renderer = children.length === 1 && children[0].type === "text"
    ? INLINE_BLOCK_RENDERERS.plain
    : INLINE_BLOCK_RENDERERS.rich;
  return renderer(document, tagName, children, actions);
}

function renderPlainInlineBlock(document, tagName, children) {
  return element(document, tagName, { text: children[0].text });
}

function renderRichInlineBlock(document, tagName, children, actions) {
  const block = element(document, tagName);
  append(block, ...children.map((child) => renderInline(document, child, actions)));
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

function sourcePath(value) {
  const valid = typeof value === "string" && !/\s|:\/\//.test(value) && !value.startsWith("/")
    && !/^[A-Za-z]:[\\/]/.test(value);
  const lineMatch = valid ? value.match(/^(.*):(\d+)$/) : null;
  const path = lineMatch ? lineMatch[1] : value;
  const safePath = valid && path && (!value.includes(":") || lineMatch);
  const basename = path.split("/").at(-1);
  return safePath && !path.split("/").some((part) => part === "..")
    && /^[^.].*\.[^.]+$/.test(basename) ? path : null;
}

function renderPathCode(document, value, actions) {
  const path = sourcePath(value);
  return path && typeof actions?.openSourceFile === "function"
    ? (() => {
      const link = element(document, "a", { className: "chat-file-path" });
      link.href = "#";
      link.setAttribute("aria-label", `Open ${path} in Files`);
      append(link, element(document, "code", { text: value }));
      link.addEventListener("click", (event) => {
        event.preventDefault();
        actions.openSourceFile(path);
      });
      return link;
    })()
    : element(document, "code", { text: value });
}

function renderInlineCode(document, node, actions) {
  return renderPathCode(document, node.text, actions);
}

function renderInlineChildren(document, tagName, node, actions) {
  const block = element(document, tagName);
  append(block, ...node.children.map((child) => renderInline(document, child, actions)));
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
  strong: (document, node, actions) => renderInlineChildren(document, "strong", node, actions),
  em: (document, node, actions) => renderInlineChildren(document, "em", node, actions),
  link: renderLink,
});

function renderParagraph(document, block, actions) {
  return renderInlineBlock(document, "p", block.children, actions);
}

function renderHeading(document, block, actions) {
  return renderInlineBlock(document, `h${block.level}`, block.children, actions);
}

function renderList(document, block, actions) {
  const list = element(document, block.ordered ? "ol" : "ul");
  block.ordered && list.setAttribute("start", String(block.start));
  append(list, ...block.items.map((item) => {
    const row = renderInlineBlock(document, "li", item.children, actions);
    append(row, ...item.blocks.map((nested) => renderBlock(document, nested, actions)));
    return row;
  }));
  return list;
}

function renderTableCell(document, tagName, children, alignment, actions) {
  const cell = renderInlineBlock(document, tagName, children, actions);
  alignment && cell.setAttribute("style", `text-align: ${alignment};`);
  return cell;
}

function renderTable(document, block, actions) {
  const header = element(document, "thead");
  const headerRow = element(document, "tr");
  append(headerRow, ...block.header.map((cell, index) => renderTableCell(
    document,
    "th",
    cell,
    block.align[index],
    actions,
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
      actions,
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
  const code = element(document, "code");
  code.setAttribute("data-language", block.language ?? "");
  const tokens = actions?.highlight?.(block.language, block.text)
    ?? [{ kind: "plain", text: block.text }];
  append(code, ...tokens.map((token) => token.kind === "plain"
    ? document.createTextNode(token.text)
    : element(document, "span", {
      className: `syntax-${token.kind}`,
      text: token.text,
    })));
  const copy = button(document, "", null, { className: "code-copy" });
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

function renderQuote(document, block, actions) {
  return renderInlineBlock(document, "blockquote", block.children, actions);
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

export function renderMarkdownBlocks(document, value, parseMarkdown, actions) {
  return parseMarkdown(value).map((block) => renderBlock(document, block, actions));
}

export function renderDiff(document, value) {
  const pre = element(document, "pre", { className: "diff" });
  let inHunk = false;
  let remaining = null;
  for (const line of String(value ?? "").split("\n")) {
    if (line.startsWith("diff --git ")) { inHunk = false; remaining = null; }
    if (line.startsWith("@@")) {
      inHunk = true;
      const header = line.match(/^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@/);
      remaining = header ? [Number(header[1] ?? 1), Number(header[2] ?? 1)] : null;
    }
    const span = element(document, "span", { text: line });
    span.className = line.startsWith("+") && (inHunk || !line.startsWith("+++")) ? "diff-add"
      : line.startsWith("-") && (inHunk || !line.startsWith("---")) ? "diff-remove"
        : line.startsWith("@@") ? "diff-hunk" : "";
    append(pre, span);
    if (inHunk && remaining && !line.startsWith("@@")) {
      if (line[0] === " " || line[0] === "-") remaining[0] -= 1;
      if (line[0] === " " || line[0] === "+") remaining[1] -= 1;
      if (remaining[0] <= 0 && remaining[1] <= 0) inHunk = false;
    }
  }
  return pre;
}

export function renderSplitDiff(document, value, actions) {
  const table = element(document, "div", { className: "diff-side-by-side" });
  const rows = actions?.splitDiff?.(value) ?? [];
  append(table, ...rows.flatMap((row) => row.kind === "hunk" || row.kind === "header"
    ? [element(document, "div", {
        className: `diff-split-header ${row.kind === "hunk" ? "diff-hunk" : ""}`,
        text: row.text,
      })]
    : [
      element(document, "span", {
        className: row.kind === "context" ? "diff-split-context" : "diff-remove",
        text: row.old,
      }),
      element(document, "span", {
        className: row.kind === "context" ? "diff-split-context" : "diff-add",
        text: row.new,
      }),
    ]));
  return table;
}

function renderCommand(document, entry) {
  const command = element(document, "div", { className: "command" });
  append(
    command,
    ...(entry.command ? [element(document, "p", { className: "command-echo", text: entry.command })] : []),
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

function renderPlan(document, entry, parseMarkdown, actions) {
  const plan = element(document, "section", { className: "plan-entry" });
  append(plan, ...(entry.explanation ? [
    append(
      element(document, "div", { className: "plan-explanation" }),
      ...parseMarkdown(entry.explanation).map((block) => renderBlock(document, block, actions)),
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

function renderUserPlan(document, entry, parseMarkdown, actions) {
  const plan = element(document, "section", { className: "user-plan" });
  append(
    plan,
    ...parseMarkdown(entry.plan).map((block) => renderBlock(document, block, actions)),
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
      const description = call.name === "run_shell" && !call.target ? `${verb} a command` : verb;
      const label = index === 0 ? `${description.charAt(0).toUpperCase()}${description.slice(1)}` : description;
      const separator = index === 0 ? "" : index === calls.length - 1
        ? calls.length === 2 ? " and " : ", and "
        : ", ";
      const sandboxBlocked = call.name === "run_shell"
        && call.error?.startsWith("blocked by the sandbox:");
      const failure = sandboxBlocked
        ? " (Blocked by the sandbox)"
        : call.status === "failed"
          ? ` (failed${call.error ? `: ${call.error}` : ""})`
          : "";
      const finished = call.name === "dispatch_subagent" && call.status === "succeeded"
        && call.output && call.output !== `subagent ${call.target} is running` ? " (finished)" : "";
      const memoryTarget = call.target !== null && ["remember", "forget"].includes(call.name);
      const targetText = call.target === null
        ? ""
        : memoryTarget ? ` "${call.target}"` : ` ${call.target}`;
      sentence += `${separator}${label}${targetText}${failure}${finished}`;
      return [
        ...(separator ? [element(document, "span", { text: separator })] : []),
        element(document, "span", { text: label }),
        ...(call.target === null ? [] : [
          element(document, "span", { text: memoryTarget ? ` "${call.target}"` : " " }),
          ...(memoryTarget ? [] : [renderPathCode(document, call.target, actions)]),
        ]),
        ...(failure ? [element(document, "span", { text: failure })] : []),
        ...(finished ? [element(document, "span", { text: finished })] : []),
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
  const undo = button(document, "Undo");
  undo.disabled = entry.undoing === true;
  undo.addEventListener("click", () => actions?.onUndo?.(entry));
  const view = button(document, "View changes", () => actions?.onViewChanges?.());
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
  const more = button(document, entry.expanded ? "Show fewer files" : `Show ${entry.files.length - 3} more files`, () => {
    entry.expanded = !entry.expanded;
    actions?.rerender?.();
  }, { className: "edit-card-more" });
  append(card, ...(entry.files.length > 3 ? [more] : []));
  const conflicts = element(document, "ul");
  for (const path of entry.conflicts ?? []) append(conflicts, element(document, "li", { text: path }));
  const force = button(document, "Revert anyway", () => actions?.onUndo?.(entry, true));
  append(card,
    ...(entry.conflicts?.length ? [
      element(document, "p", { text: "These files changed outside the agent:" }), conflicts, force,
    ] : []),
    ...(entry.error ? [element(document, "p", { className: "error", text: entry.error })] : []),
  );
  return card;
}

function renderCommitCard(document, entry, _parseMarkdown, actions) {
  const card = element(document, "section", { className: "commit-card" });
  append(card,
    element(document, "p", { className: "command-echo", text: entry.command }),
    element(document, "h3", { text: "Commit changes" }),
  );
  const files = element(document, "ul", { className: "commit-files" });
  for (const file of entry.files) {
    const checkbox = element(document, "input");
    checkbox.type = "checkbox";
    checkbox.checked = file.checked;
    checkbox.setAttribute("aria-label", `${file.path} · ${file.status}`);
    checkbox.addEventListener("change", () => {
      file.checked = checkbox.checked;
      commit.disabled = !entry.message.trim() || !entry.files.some((item) => item.checked);
    });
    const label = element(document, "label");
    append(label, checkbox, element(document, "span", { text: `${file.path} · ${file.status}` }));
    append(files, label);
  }
  const message = element(document, "input", { className: "commit-message" });
  message.type = "text";
  message.value = entry.message;
  message.setAttribute("aria-label", "Commit message");
  message.addEventListener("input", () => {
    entry.message = message.value;
    commit.disabled = !entry.message.trim() || !entry.files.some((item) => item.checked);
  });
  const commit = button(document, "Commit");
  commit.disabled = !entry.message.trim() || !entry.files.some((item) => item.checked);
  commit.addEventListener("click", () => actions?.onCommit?.(entry));
  const cancel = button(document, "Cancel", () => actions?.onCancelCommit?.(entry));
  append(card, files, message, commit, cancel);
  append(card, ...(entry.error ? [element(document, "p", { className: "error", text: entry.error })] : []));
  return card;
}

function renderPullRequestCard(document, entry, _parseMarkdown, actions) {
  const card = element(document, "section", { className: "pull-request-card" });
  const close = button(document, entry.problem ? "Close" : "Cancel", () => actions?.onClosePullRequest?.(entry));
  const problem = entry.problem
    ? [element(document, "p", { className: "error", text: entry.problem }), close]
    : [];
  const title = element(document, "input", { className: "pull-request-title" });
  title.type = "text";
  title.value = entry.title;
  title.setAttribute("aria-label", "Pull request title");
  title.addEventListener("input", () => {
    entry.title = title.value;
    submit.disabled = !entry.title.trim();
  });
  const body = element(document, "textarea", { className: "pull-request-body" });
  body.value = entry.body;
  body.setAttribute("aria-label", "Pull request body");
  body.addEventListener("input", () => { entry.body = body.value; });
  const submit = button(document, "Push and open pull request");
  submit.disabled = !entry.title.trim();
  submit.addEventListener("click", () => actions?.onOpenPullRequest?.(entry));
  const form = entry.problem
    ? []
    : [
      element(document, "p", { text: `This pushes ${entry.branch} to origin and opens a pull request into ${entry.base} on GitHub.` }),
      title,
      body,
      submit,
      close,
      ...(entry.error ? [element(document, "p", { className: "error", text: entry.error })] : []),
    ];
  append(card, element(document, "p", { className: "command-echo", text: entry.command }), ...problem, ...form);
  return card;
}

function renderPullRequestDone(document, entry) {
  const line = element(document, "p", { className: "pull-request-result" });
  const link = element(document, "a", { text: entry.url });
  link.href = entry.url;
  link.setAttribute("rel", "noopener noreferrer");
  link.setAttribute("target", "_blank");
  append(line, element(document, "span", { text: "Opened pull request: " }), link);
  return line;
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

function renderSubagentNotice(document, entry, _parseMarkdown, actions) {
  let fullText = entry.text;
  let failure = "";
  if (entry.kind === "finished") {
    try {
      const result = JSON.parse(fullText);
      fullText = [result.content, result.error].filter(Boolean).join("\n");
      if (result.ok === false) failure = result.error || "subagent did not finish";
    } catch { /* A peer may send the result as plain text. */ }
  }
  const preview = String(fullText).split("\n")[0].slice(0, 160);
  const label = entry.kind === "finished" ? failure
    ? `${entry.name} failed: ${String(failure).split("\n")[0].slice(0, 160)}` : `${entry.name} finished`
    : entry.kind === "progress" ? `${entry.name}: ${preview}`
      : entry.kind === "question" ? `${entry.name} asks: ${preview}`
        : entry.name ? `To ${entry.name}: ${preview}` : `From the leader: ${preview}`;
  const notice = element(document, "div", { className: "activity subagent-notice" });
  notice.tabIndex = 0;
  notice.setAttribute("role", "button");
  notice.setAttribute("aria-label", label);
  notice.setAttribute("aria-expanded", String(entry.expanded === true));
  const expand = () => {
    entry.expanded = entry.expanded !== true;
    actions?.rerender?.();
  };
  notice.addEventListener("click", expand);
  notice.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    expand();
  });
  append(notice, element(document, "span", { className: "activity-text", text: label }));
  if (entry.expanded) {
    const output = element(document, "section", { className: "activity-output" });
    append(output, element(document, "pre", { text: fullText }));
    append(notice, output);
  }
  return notice;
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
  "subagent-notice": renderSubagentNotice,
  "auto-reviewed": (document, entry) => element(document, "p", { className: "activity auto-reviewed", text:
    `Auto-${entry.decision === "approve" ? "approved" : "denied"}: ${entry.operation} ${entry.target} — ${entry.reason}` }),
  "person-notice": (document, entry) => element(document, "p", { className: "activity person-notice", text: `Leader: ${entry.text}` }),
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
  "commit-card": renderCommitCard,
  "pull-request-card": renderPullRequestCard,
  "pull-request-done": renderPullRequestDone,
  question: renderQuestion,
  compaction: renderCompaction,
  gap: renderGap,
  unknown: renderUnknown,
  "turn-time": (document, entry) => element(document, "p", {
    className: "turn-time",
    text: entry.text,
  }),
  "turn-error": (document, entry) => element(document, "p", {
    className: "turn-error",
    text: entry.text,
  }),
  "model-fallback": (document, entry) => element(document, "p", {
    className: "model-fallback",
    text: entry.text,
  }),
});

const transcriptRoots = new WeakMap();

function entrySnapshot(value) {
  if (Array.isArray(value)) return value.map(entrySnapshot);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, entrySnapshot(item)]));
  }
  return value;
}

function entryMatches(value, snapshot) {
  if (value === snapshot) return true;
  if (!value || !snapshot || typeof value !== "object" || typeof snapshot !== "object") return false;
  const keys = Object.keys(value);
  return keys.length === Object.keys(snapshot).length
    && keys.every(key => Object.hasOwn(snapshot, key) && entryMatches(value[key], snapshot[key]));
}

function updateTranscriptNode(node, next) {
  if (node.nodeType !== next.nodeType || node.tagName !== next.tagName
      || ["BUTTON", "INPUT", "TEXTAREA", "A"].includes(node.tagName)) {
    node.parentNode.replaceChild(next, node);
    return next;
  }
  if (node.nodeType === 3) {
    if (node.data !== next.data) {
      if (node.appendData && next.data.startsWith(node.data)) node.appendData(next.data.slice(node.data.length));
      else node.data = next.data;
    }
    return node;
  }
  const names = element => Array.from(element.attributes, attribute => attribute.name ?? attribute[0]);
  for (const name of names(node)) {
    if (name === "open" && node.tagName === "DETAILS") continue;
    if (next.getAttribute(name) === null) node.removeAttribute(name);
  }
  for (const name of names(next)) {
    if (name === "open" && node.tagName === "DETAILS") continue;
    if (node.getAttribute(name) !== next.getAttribute(name)) node.setAttribute(name, next.getAttribute(name));
  }
  if (node.className !== next.className) node.className = next.className;
  if (node.hidden !== next.hidden) node.hidden = next.hidden;
  const children = Array.from(next.childNodes);
  for (let index = 0; index < children.length; index += 1) {
    const child = children[index];
    if (node.childNodes[index]) updateTranscriptNode(node.childNodes[index], child);
    else node.append(child);
  }
  while (node.childNodes.length > children.length) node.childNodes[node.childNodes.length - 1].remove();
  return node;
}

export function renderTranscript(document, root, model, parseMarkdown, actions, options = {}) {
  let state = transcriptRoots.get(root);
  if (!state) {
    state = { records: new Map(), scheduled: false, pending: null, editing: null, viewKey: null };
    transcriptRoots.set(root, state);
  }
  state.pending = { model, parseMarkdown, actions, options };
  const paint = () => {
    state.scheduled = false;
    const pending = state.pending;
    if (!pending) return;
    state.pending = null;
    if (pending.options.isCurrent && !pending.options.isCurrent()) return;
    const records = new Map();
    const retained = new Set(pending.model);
    for (const [entry, record] of state.records) {
      if (!retained.has(entry)) record.node.remove();
    }
    pending.model.forEach((entry, index) => {
      let record = state.records.get(entry);
      if (!record || !entryMatches(entry, record.snapshot)) {
        const next = (TRANSCRIPT_RENDERERS[entry.type] ?? renderUnknown)(document, entry, pending.parseMarkdown, pending.actions);
        const node = record?.node.parentNode === root ? updateTranscriptNode(record.node, next) : next;
        record = { node, snapshot: entrySnapshot(entry) };
      }
      const current = root.childNodes[index];
      if (current !== record.node) {
        if (current) current.before(record.node);
        else root.append(record.node);
      }
      records.set(entry, record);
    });
    while (root.childNodes.length > pending.model.length) root.childNodes[root.childNodes.length - 1].remove();
    state.records = records;
    const editing = records.get(pending.options.editingEntry)?.node ?? null;
    if (state.editing && state.editing !== editing) {
      state.editing.className = state.editing.className.replace(/\s*editing-previous-prompt\b/g, "");
    }
    if (editing && !editing.className.includes("editing-previous-prompt")) editing.className += " editing-previous-prompt";
    state.editing = editing;
    if (state.viewKey != null && pending.options.viewKey == null) root.scrollTop = state.returnScrollTop;
    state.viewKey = pending.options.viewKey ?? null;
    if (pending.options.returnScrollTop !== undefined) state.returnScrollTop = pending.options.returnScrollTop;
    pending.options.afterRender?.();
  };
  if (options.requestAnimationFrame) {
    if (!state.scheduled) {
      state.scheduled = true;
      options.requestAnimationFrame(paint);
    }
  } else paint();
  return root;
}
