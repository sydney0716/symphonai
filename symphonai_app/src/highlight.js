const ALIASES = Object.freeze({
  py: "python",
  js: "javascript",
  ts: "javascript",
  typescript: "javascript",
  sh: "shell",
  bash: "shell",
  zsh: "shell",
  rs: "rust",
});

const KEYWORDS = Object.freeze({
  python: new Set("and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" ")),
  javascript: new Set("async await break case catch class const continue debugger default delete do else export extends false finally for from function if import in instanceof let new null of return static super switch this throw true try typeof undefined var void while with yield interface type enum implements package private protected public readonly satisfies as".split(" ")),
  shell: new Set("case coproc do done elif else esac fi for function if in select then time until while".split(" ")),
  json: new Set(["true", "false", "null"]),
  rust: new Set("as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while".split(" ")),
  go: new Set("break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var true false iota nil".split(" ")),
  toml: new Set(["true", "false"]),
});

const LANGUAGES = new Set(["python", "javascript", "shell", "json", "diff", "rust", "go", "toml"]);
const NUMBER = /^(?:0[xX][\da-fA-F]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/;
const IDENTIFIER = /^[A-Za-z_$][\w$]*/;
const MAX_HIGHLIGHT_LENGTH = 200_000;

function plain(text) {
  return text ? [{ kind: "plain", text }] : [];
}

function add(tokens, kind, text) {
  if (!text) return;
  const previous = tokens.at(-1);
  if (previous?.kind === kind) previous.text += text;
  else tokens.push({ kind, text });
}

function commentStart(language, text, index) {
  if ((language === "python" || language === "shell" || language === "toml") && text[index] === "#") return 1;
  if (["javascript", "json", "rust", "go"].includes(language) && text.startsWith("//", index)) return 2;
  if (["javascript", "json", "rust", "go"].includes(language) && text.startsWith("/*", index)) return 0;
  return -1;
}

function stringEnd(text, start, quote, language) {
  const triple = language === "python" && text.startsWith(quote.repeat(3), start);
  const delimiter = triple ? quote.repeat(3) : quote;
  for (let index = start + delimiter.length; index < text.length; index += 1) {
    if (text[index] === "\\") {
      index += 1;
      continue;
    }
    if (text.startsWith(delimiter, index)) return index + delimiter.length;
    if (!triple && quote !== "`" && text[index] === "\n") break;
  }
  return text.length;
}

function highlightDiff(text) {
  const tokens = [];
  let inHunk = false;
  let remaining = null;
  for (const line of text.match(/[^\n]*\n|[^\n]+$/g) ?? []) {
    if (line.startsWith("diff --git ")) { inHunk = false; remaining = null; }
    if (line.startsWith("@@")) {
      inHunk = true;
      const header = line.match(/^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@/);
      remaining = header ? [Number(header[1] ?? 1), Number(header[2] ?? 1)] : null;
    }
    if (line.startsWith("+") && (inHunk || !line.startsWith("+++"))) {
      add(tokens, "keyword", "+");
      for (const token of highlight("javascript", line.slice(1))) add(tokens, token.kind, token.text);
    } else if (line.startsWith("-") && (inHunk || !line.startsWith("---"))) {
      add(tokens, "comment", "-");
      for (const token of highlight("javascript", line.slice(1))) add(tokens, token.kind, token.text);
    } else {
      add(tokens, "plain", line);
    }
    if (inHunk && remaining && !line.startsWith("@@")) {
      if (line[0] === " " || line[0] === "-") remaining[0] -= 1;
      if (line[0] === " " || line[0] === "+") remaining[1] -= 1;
      if (remaining[0] <= 0 && remaining[1] <= 0) inHunk = false;
    }
  }
  return tokens;
}

export function highlight(language, text) {
  const source = String(text ?? "");
  const requested = String(language ?? "").trim().toLowerCase();
  const canonical = ALIASES[requested] ?? requested;
  if (!LANGUAGES.has(canonical) || source.length > MAX_HIGHLIGHT_LENGTH) return plain(source);
  if (canonical === "diff") return highlightDiff(source);

  const tokens = [];
  let index = 0;
  while (index < source.length) {
    const character = source[index];
    if (character === "'" || character === '"' || (character === "`" && canonical === "javascript")) {
      const end = stringEnd(source, index, character, canonical);
      add(tokens, "string", source.slice(index, end));
      index = end;
      continue;
    }

    const comment = commentStart(canonical, source, index);
    if (comment > 0) {
      const newline = source.indexOf("\n", index);
      const end = newline < 0 ? source.length : newline;
      add(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }
    if (comment === 0) {
      const close = source.indexOf("*/", index + 2);
      const end = close < 0 ? source.length : close + 2;
      add(tokens, "comment", source.slice(index, end));
      index = end;
      continue;
    }

    const identifier = /[A-Za-z_$]/.test(character)
      ? source.slice(index).match(IDENTIFIER)?.[0]
      : null;
    if (identifier) {
      add(tokens, KEYWORDS[canonical]?.has(identifier) ? "keyword" : "plain", identifier);
      index += identifier.length;
      continue;
    }
    const number = /[\d]/.test(character) ? source.slice(index).match(NUMBER)?.[0] : null;
    if (number) {
      add(tokens, "number", number);
      index += number.length;
      continue;
    }
    add(tokens, "plain", character);
    index += 1;
  }
  return tokens;
}
