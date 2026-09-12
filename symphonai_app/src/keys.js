const MODIFIER_ORDER = Object.freeze(["ctrl", "meta", "alt", "shift", "mod"]);
const MODIFIERS = new Set(MODIFIER_ORDER);


export class KeymapError extends Error {
  constructor(message) {
    super(message);
    this.name = "KeymapError";
  }
}


function normalizeChord(value) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new KeymapError("keymap chord must not be empty");
  }
  const parts = value.split("+").map((part) => part.trim().toLowerCase());
  if (parts.some((part) => part === "")) {
    throw new KeymapError(`invalid chord ${JSON.stringify(value)}`);
  }
  const key = parts.at(-1);
  const modifiers = parts.slice(0, -1);
  for (const modifier of modifiers) {
    if (!MODIFIERS.has(modifier)) {
      throw new KeymapError(
        `unknown modifier ${JSON.stringify(modifier)} in chord ${JSON.stringify(value)}`,
      );
    }
  }
  if (MODIFIERS.has(key)) {
    throw new KeymapError(`chord ${JSON.stringify(value)} has no key`);
  }
  if (new Set(modifiers).size !== modifiers.length) {
    throw new KeymapError(`chord ${JSON.stringify(value)} repeats a modifier`);
  }
  const ordered = MODIFIER_ORDER.filter((modifier) => modifiers.includes(modifier));
  return [...ordered, key].join("+");
}


function chordSpellings(chord) {
  const normalized = normalizeChord(chord);
  const parts = normalized.split("+");
  if (!parts.includes("mod")) {
    return new Set([normalized]);
  }
  return new Set(["ctrl", "meta"].map((primary) => normalizeChord(
    parts.map((part) => part === "mod" ? primary : part).join("+"),
  )));
}


function chordsConflict(first, second) {
  const secondSpellings = chordSpellings(second);
  return [...chordSpellings(first)].some((spelling) => secondSpellings.has(spelling));
}


function skipWhitespace(text, start) {
  let cursor = start;
  while (/\s/.test(text[cursor] ?? "")) {
    cursor += 1;
  }
  return cursor;
}


function stringEnd(text, start) {
  let escaped = false;
  for (let cursor = start + 1; cursor < text.length; cursor += 1) {
    if (!escaped && text[cursor] === '"') {
      return cursor + 1;
    }
    if (!escaped && text[cursor] === "\\") {
      escaped = true;
    } else {
      escaped = false;
    }
  }
  return text.length;
}


function objectEntries(text) {
  const entries = [];
  let cursor = skipWhitespace(text, 0) + 1;
  while (cursor < text.length) {
    cursor = skipWhitespace(text, cursor);
    if (text[cursor] === "}") {
      break;
    }
    const keyStart = cursor;
    const keyEnd = stringEnd(text, keyStart);
    const key = JSON.parse(text.slice(keyStart, keyEnd));
    cursor = skipWhitespace(text, keyEnd) + 1;
    const valueStart = skipWhitespace(text, cursor);
    let depth = 0;
    let inString = false;
    let escaped = false;
    let valueEnd = valueStart;
    for (; valueEnd < text.length; valueEnd += 1) {
      const character = text[valueEnd];
      if (inString) {
        if (!escaped && character === '"') {
          inString = false;
        }
        if (!escaped && character === "\\") {
          escaped = true;
        } else {
          escaped = false;
        }
        continue;
      }
      if (character === '"') {
        inString = true;
      } else if (character === "[" || character === "{") {
        depth += 1;
      } else if (character === "]" || character === "}") {
        if (character === "}" && depth === 0) {
          break;
        }
        depth -= 1;
      } else if (character === "," && depth === 0) {
        break;
      }
    }
    entries.push([key, JSON.parse(text.slice(valueStart, valueEnd))]);
    cursor = text[valueEnd] === "," ? valueEnd + 1 : valueEnd;
  }
  return entries;
}


export const RESERVED = Object.freeze(["escape", "mod+.", "mod+l"]);

const RESERVED_REASONS = new Map([
  ["escape", "cancel must always remain reachable"],
  ["mod+.", "stop run must always remain reachable"],
  ["mod+l", "focus input must always remain reachable"],
]);


export function parseKeymap(text) {
  if (typeof text !== "string") {
    throw new KeymapError("keymap file must be text");
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new KeymapError("keymap file is not valid JSON");
  }
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new KeymapError("keymap file must contain an object");
  }

  const bindings = new Map();
  const sourceChords = new Map();
  for (const [rawChord, action] of objectEntries(text)) {
    if (typeof action !== "string" && action !== null) {
      throw new KeymapError(
        `binding ${JSON.stringify(rawChord)} must name an action or be null`,
      );
    }
    const chord = normalizeChord(rawChord);
    const conflict = [...bindings].find(([existing]) => (
      chordsConflict(existing, chord)
    ));
    if (conflict) {
      const [existing, previous] = conflict;
      throw new KeymapError(
        `chords ${JSON.stringify(sourceChords.get(existing))} and ${JSON.stringify(rawChord)} bind both ${JSON.stringify(previous)} and ${JSON.stringify(action)}`,
      );
    }
    bindings.set(chord, action);
    sourceChords.set(chord, rawChord);
  }
  return bindings;
}


export function mergeKeymap(defaults, user) {
  const bindings = new Map(defaults);
  const rejected = [];
  for (const [rawChord, action] of user) {
    const chord = normalizeChord(rawChord);
    const reserved = RESERVED.find((candidate) => chordsConflict(candidate, chord));
    if (reserved) {
      rejected.push({
        chord,
        action,
        reason: `collides with reserved chord ${reserved}: ${RESERVED_REASONS.get(reserved)}`,
      });
      continue;
    }
    if (!bindings.has(chord)) {
      const conflict = [...bindings].find(([existing]) => (
        chordsConflict(existing, chord)
      ));
      if (conflict) {
        const [existing, existingAction] = conflict;
        rejected.push({
          chord,
          action,
          reason: `collides with ${existing}, bound to ${existingAction}`,
        });
        continue;
      }
    }
    if (action === null) {
      bindings.delete(chord);
    } else {
      bindings.set(chord, action);
    }
  }
  return { bindings, rejected };
}


export function primaryModifier(platform) {
  return platform === "darwin" ? "meta" : "ctrl";
}


function resolvedChord(chord, platform) {
  const resolved = chord.split("+").map((part) => (
    part.trim().toLowerCase() === "mod" ? primaryModifier(platform) : part
  ));
  return normalizeChord(resolved.join("+"));
}


function eventChord(event) {
  if (!event || typeof event.key !== "string" || event.key === "") {
    return null;
  }
  const modifiers = [];
  if (event.ctrlKey) modifiers.push("ctrl");
  if (event.metaKey) modifiers.push("meta");
  if (event.altKey) modifiers.push("alt");
  if (event.shiftKey) modifiers.push("shift");
  return [...modifiers, event.key.toLowerCase()].join("+");
}


export function lookup(map, event, { platform }) {
  const chord = eventChord(event);
  if (chord === null) {
    return null;
  }
  for (const [binding, action] of map) {
    if (action !== null && resolvedChord(binding, platform) === chord) {
      return action;
    }
  }
  return null;
}
