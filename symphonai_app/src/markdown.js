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

function lineIndent(line) {
  const prefix = line.match(/^[ \t]*/)?.[0] ?? "";
  return [...prefix].reduce((total, character) => total + (character === "\t" ? 4 : 1), 0);
}

function listMarker(line) {
  const prefix = line.match(/^[ \t]*/)?.[0] ?? "";
  const indent = lineIndent(line);
  const content = line.slice(prefix.length);
  const ordered = content.match(/^(\d+)\. (.*)$/);
  const bullet = content.match(/^[-*+] (.*)$/);
  return ordered
    ? {
      indent,
      textIndent: indent + ordered[0].indexOf(ordered[2]),
      ordered: true,
      number: Number(ordered[1]),
      text: ordered[2],
    }
    : bullet
      ? {
        indent,
        textIndent: indent + bullet[0].indexOf(bullet[1]),
        ordered: false,
        number: null,
        text: bullet[1],
      }
      : null;
}

function quote(line) {
  return line.startsWith("> ") ? line.slice(2) : null;
}

function horizontalRule(line) {
  const value = line.trim();
  return /^(?:-\s*){3,}$|^(?:\*\s*){3,}$|^(?:_\s*){3,}$/.test(value);
}

function tableCells(line) {
  const trimmed = line.trim();
  const cells = trimmed.split("|");
  if (trimmed.startsWith("|")) cells.shift();
  if (trimmed.endsWith("|")) cells.pop();
  return cells.map((cell) => cell.trim());
}

function tableAlignment(cell) {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  return left && right ? "center" : left ? "left" : right ? "right" : null;
}

function tableStart(lines, index) {
  if (index + 1 >= lines.length || !lines[index].includes("|") || !lines[index + 1].includes("|")) {
    return null;
  }
  const header = tableCells(lines[index]);
  const delimiters = tableCells(lines[index + 1]);
  return header.length === delimiters.length && delimiters.every((cell) => /^:?-+:?$/.test(cell))
    ? { header, align: delimiters.map(tableAlignment) }
    : null;
}

function startsBlock(lines, index) {
  const line = lines[index];
  const marker = listMarker(line);
  return Boolean(
    heading(line)
    || fence(line)
    || (marker && marker.indent <= 3)
    || quote(line) !== null
    || horizontalRule(line)
    || tableStart(lines, index),
  );
}

function sameList(marker, indent, ordered) {
  return marker && marker.indent === indent && marker.ordered === ordered;
}

function parseList(lines, start) {
  const first = listMarker(lines[start]);
  const items = [];
  const { indent, ordered } = first;
  let index = start;

  while (index < lines.length) {
    const marker = listMarker(lines[index]);
    if (!sameList(marker, indent, ordered)) break;
    const item = { text: [marker.text], blocks: [] };
    let paragraphLines = [];
    const flushParagraph = () => {
      if (!paragraphLines.length) return;
      item.blocks.push({ type: "paragraph", children: parseInline(paragraphLines.join("\n")) });
      paragraphLines = [];
    };
    index += 1;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        let next = index;
        while (next < lines.length && !lines[next].trim()) next += 1;
        const nextMarker = listMarker(lines[next] ?? "");
        if (nextMarker && nextMarker.indent > indent) {
          index = next;
          continue;
        }
        break;
      }

      const nestedMarker = listMarker(line);
      if (nestedMarker) {
        if (nestedMarker.indent > indent) {
          flushParagraph();
          const nested = parseList(lines, index);
          item.blocks.push(nested.block);
          index = nested.next;
          continue;
        }
        break;
      }

      if (lineIndent(line) >= marker.textIndent) {
        const openingFence = fence(line.trimStart());
        if (openingFence) {
          flushParagraph();
          const codeLines = [];
          index += 1;
          while (index < lines.length && !/^```\s*$/.test(lines[index].trim())) {
            const codeLine = lines[index];
            const prefixLength = codeLine.match(/^[ \t]*/)?.[0].length ?? 0;
            const content = codeLine.slice(Math.min(marker.textIndent, prefixLength));
            codeLines.push(content);
            index += 1;
          }
          index += Number(index < lines.length);
          item.blocks.push({
            type: "code-block",
            language: openingFence[1] || null,
            text: codeLines.join("\n"),
          });
          continue;
        }
        if (item.blocks.length) paragraphLines.push(line.slice(Math.min(marker.textIndent, line.match(/^[ \t]*/)?.[0].length ?? 0)));
        else item.text.push(line.trimStart());
        index += 1;
        continue;
      }
      break;
    }
    flushParagraph();
    items.push({
      children: parseInline(item.text.join("\n")),
      blocks: item.blocks,
    });

    if (index < lines.length && !lines[index].trim()) {
      let next = index;
      while (next < lines.length && !lines[next].trim()) next += 1;
      if (sameList(listMarker(lines[next] ?? ""), indent, ordered)) index = next;
    }
  }

  return {
    block: {
      type: "list",
      ordered,
      start: ordered ? first.number : null,
      items,
    },
    next: index,
  };
}

function parseTable(lines, start, table) {
  const rows = [];
  let index = start + 2;
  while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
    const cells = tableCells(lines[index]).slice(0, table.header.length);
    rows.push([...cells, ...Array(Math.max(0, table.header.length - cells.length)).fill("")].map(parseInline));
    index += 1;
  }
  return {
    block: {
      type: "table",
      align: table.align,
      header: table.header.map(parseInline),
      rows,
    },
    next: index,
  };
}

export function parseMarkdown(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const openingFence = fence(line);
    const marker = listMarker(line);
    const table = tableStart(lines, index);
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
    } else if (horizontalRule(line)) {
      blocks.push({ type: "rule" });
      index += 1;
    } else if (marker && marker.indent <= 3) {
      const result = parseList(lines, index);
      blocks.push(result.block);
      index = result.next;
    } else if (table) {
      const result = parseTable(lines, index, table);
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
      while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
        paragraph.push(lines[index]);
        index += 1;
      }
      blocks.push({ type: "paragraph", children: parseInline(paragraph.join("\n")) });
    }
  }
  return blocks;
}
