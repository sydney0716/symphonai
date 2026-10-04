function textNode(text) {
  return { type: "text", text };
}

function closingDelimiter(text, delimiter, start) {
  const index = text.indexOf(delimiter, start);
  return index < 0 ? null : index;
}

function characterBefore(text, index) {
  const previous = text.charCodeAt(index - 1);
  const beforePrevious = text.charCodeAt(index - 2);
  return previous >= 0xdc00 && previous <= 0xdfff && beforePrevious >= 0xd800 && beforePrevious <= 0xdbff
    ? text.slice(index - 2, index)
    : text.slice(index - 1, index);
}

function characterAt(text, index) {
  return index < text.length ? String.fromCodePoint(text.codePointAt(index)) : "";
}

function isLetterOrDigit(character) {
  return /[\p{L}\p{N}]/u.test(character);
}

function underscoreCanOpen(text, index) {
  const before = characterBefore(text, index);
  const after = characterAt(text, index + 1);
  return before !== "_" && after !== "_" && !isLetterOrDigit(before) && !/\s/u.test(after);
}

function underscoreCanClose(text, index) {
  const before = characterBefore(text, index);
  const after = characterAt(text, index + 1);
  return before !== "_" && after !== "_" && !/\s/u.test(before) && !isLetterOrDigit(after);
}

function linkAt(text, start) {
  const labelEnd = text.indexOf("](", start + 1);
  if (labelEnd < 0) return null;
  let depth = 1;
  for (let index = labelEnd + 2; index < text.length; index += 1) {
    if (text[index] === "(") depth += 1;
    if (text[index] === ")") depth -= 1;
    if (depth === 0) {
      const label = text.slice(start + 1, labelEnd);
      const url = text.slice(labelEnd + 2, index);
      return /^https?:\/\//i.test(url)
        ? { type: "link", label, url, next: index + 1 }
        : { ...textNode(label), next: index + 1 };
    }
  }
  return null;
}

function emphasisEnd(text, delimiter, start) {
  for (let index = start; index < text.length; index += 1) {
    if (text[index] !== delimiter) continue;
    if (delimiter === "*" && (text[index - 1] === "*" || text[index + 1] === "*")) continue;
    if (delimiter === "_" && !underscoreCanClose(text, index)) continue;
    return index;
  }
  return null;
}

function parseInline(text) {
  const nodes = [];
  let plain = "";
  const flush = () => {
    if (plain) nodes.push(textNode(plain));
    plain = "";
  };

  for (let index = 0; index < text.length;) {
    const link = text[index] === "[" ? linkAt(text, index) : null;
    if (link) {
      flush();
      const { next, ...node } = link;
      nodes.push(node);
      index = next;
      continue;
    }

    const delimiter = text.startsWith("**", index)
      ? "**"
      : text[index] === "`" || text[index] === "*"
        ? text[index]
        : text[index] === "_" && underscoreCanOpen(text, index)
          ? "_"
        : "";
    const close = delimiter
      ? delimiter === "*" || delimiter === "_"
        ? emphasisEnd(text, delimiter, index + 1)
        : closingDelimiter(text, delimiter, index + delimiter.length)
      : null;
    if (close !== null && close > index + delimiter.length) {
      flush();
      const content = text.slice(index + delimiter.length, close);
      const type = delimiter === "**" ? "strong" : delimiter === "`" ? "code" : "em";
      nodes.push(type === "code" ? { type, text: content } : { type, children: parseInline(content) });
      index = close + delimiter.length;
      continue;
    }

    plain += text[index];
    index += 1;
  }
  flush();
  return plainBreaks(nodes);
}

function plainBreaks(nodes) {
  return nodes.flatMap((node) => (
    node.type === "text"
      ? node.text.split("\n").flatMap((part, index) => [
        ...(index > 0 ? [{ type: "break" }] : []),
        ...(part ? [textNode(part)] : []),
      ])
      : [node]
  ));
}

function heading(line) {
  const match = line.match(/^(#{1,6}) (.*)$/);
  return match ? { type: "heading", level: match[1].length, children: parseInline(match[2]) } : null;
}

function fence(line) {
  return line.match(/^```([A-Za-z0-9_-]*)\s*$/);
}

function bullet(line) {
  return line.match(/^[-*+] (.*)$/);
}

function numbered(line) {
  return line.match(/^(\d+)\. (.*)$/);
}

function quote(line) {
  return line.startsWith("> ") ? line.slice(2) : null;
}

function startsBlock(line) {
  return Boolean(heading(line) || fence(line) || bullet(line) || numbered(line) || quote(line) !== null);
}

function parseList(lines, start, ordered) {
  const items = [];
  let index = start;
  while (index < lines.length) {
    const match = ordered ? numbered(lines[index]) : bullet(lines[index]);
    if (!match) break;
    items.push(parseInline(match[ordered ? 2 : 1]));
    index += 1;
  }
  return { block: { type: "list", ordered, items }, next: index };
}

export function parseMarkdown(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const openingFence = fence(line);
    if (!line.trim()) {
      index += 1;
    } else if (openingFence) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      index += Number(index < lines.length);
      blocks.push({
        type: "code-block",
        language: openingFence[1] || null,
        text: codeLines.join("\n"),
      });
    } else if (heading(line)) {
      blocks.push(heading(line));
      index += 1;
    } else if (bullet(line)) {
      const result = parseList(lines, index, false);
      blocks.push(result.block);
      index = result.next;
    } else if (numbered(line)) {
      const result = parseList(lines, index, true);
      blocks.push(result.block);
      index = result.next;
    } else if (quote(line) !== null) {
      const quoteLines = [];
      while (index < lines.length && quote(lines[index]) !== null) {
        quoteLines.push(quote(lines[index]));
        index += 1;
      }
      blocks.push({ type: "quote", children: parseInline(quoteLines.join("\n")) });
    } else {
      const paragraph = [];
      while (index < lines.length && lines[index].trim() && !startsBlock(lines[index])) {
        paragraph.push(lines[index]);
        index += 1;
      }
      blocks.push({ type: "paragraph", children: parseInline(paragraph.join("\n")) });
    }
  }
  return blocks;
}
