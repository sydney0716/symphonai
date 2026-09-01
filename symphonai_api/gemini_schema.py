"""Sanitize JSON Schema into the restricted dialect Gemini accepts.

Anthropic and OpenAI accept fairly ordinary JSON Schema for tool
parameters, so `symphonai_api.tool_schema` can pass a `LocalTool`'s
`parameters` straight through to them. Gemini does not: it accepts only a
narrow subset, and rejects the rest with an opaque "invalid input_schema"
error rather than ignoring it. This module is the translation step that
gap requires.

Constraints implemented here (see `docs/symphonai-external-references.md`
for where these came from):

- Only a small keyword allowlist survives; everything else is dropped.
- Nullability is a `nullable: true` flag, not a `type: ["string", "null"]`
  union.
- Local `$ref` pointers must be inlined; circular refs abort rather than
  loop forever.
- The root must be `type: "object"` -- no `anyOf`/`oneOf` at the top.
- A union that cannot be collapsed becomes `{}` rather than invalid output.
- Depth and node budgets bound the work so a pathological schema cannot
  hang the caller.

Pure functions only: no network, no subprocess, no I/O.
"""

from __future__ import annotations

from typing import Any

# Keywords Gemini understands. Anything else is dropped.
_ALLOWED_KEYWORDS = frozenset(
    {
        "type",
        "nullable",
        "description",
        "format",
        "enum",
        "required",
        "properties",
        "items",
        "anyOf",
    }
)

# Primitive type names Gemini accepts (lowercase JSON Schema spelling).
_ALLOWED_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object"})

# Gemini's own nesting ceiling is around 32; stay under it so its
# validator, which counts differently than you would expect, has headroom.
MAX_DEPTH = 24

# Cap on how many schema nodes we will visit, so a pathological or
# maliciously deep schema cannot hang the caller.
MAX_NODES = 1024

# How far to follow local $ref chains before giving up.
MAX_REF_DEPTH = 16


class _Budget:
    """Mutable visit budget shared across one sanitize_for_gemini() call."""

    def __init__(self, max_nodes: int = MAX_NODES) -> None:
        self.remaining = max_nodes

    def spend(self) -> bool:
        """Consume one node. Returns False once the budget is exhausted."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _resolve_ref(ref: str, root: dict[str, Any], depth: int = 0) -> dict[str, Any] | None:
    """Resolve a local '#/a/b' JSON pointer against `root`.

    Returns None for external refs, malformed pointers, missing targets, or
    chains deeper than MAX_REF_DEPTH (which is how a circular $ref ends up
    aborting instead of looping forever).
    """
    if depth > MAX_REF_DEPTH or not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if not isinstance(node, dict):
        return None
    if "$ref" in node:
        return _resolve_ref(node["$ref"], root, depth + 1)
    return node


def _normalize_type(raw: Any) -> tuple[str | None, bool]:
    """Return (type_name, nullable) for a `type` value.

    Gemini has no union types: `["string", "null"]` must become
    `type: "string"` plus `nullable: true`.
    """
    if isinstance(raw, str):
        return (raw if raw in _ALLOWED_TYPES else None), False
    if isinstance(raw, list):
        nullable = "null" in raw
        candidates = [t for t in raw if isinstance(t, str) and t in _ALLOWED_TYPES]
        return (candidates[0] if candidates else None), nullable
    return None, False


def _collapse_any_of(
    branches: list[Any], root: dict[str, Any], depth: int, budget: _Budget
) -> dict[str, Any]:
    """Collapse an anyOf into a single schema where possible.

    Gemini tolerates anyOf below the root, but an unresolvable union
    produces an opaque error, so anything that cannot be collapsed becomes
    `{}` -- permissive, but valid.
    """
    cleaned = [
        s
        for s in (_sanitize_node(b, root, depth + 1, budget) for b in branches if isinstance(b, dict))
        if s
    ]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return cleaned[0]

    types = {s.get("type") for s in cleaned}
    if len(types) == 1 and None not in types:
        # Same type across all branches: merge their enums if they all have
        # one, otherwise keep just the shared type.
        merged: dict[str, Any] = {"type": next(iter(types))}
        if all("enum" in s for s in cleaned):
            values: list[Any] = []
            for s in cleaned:
                for v in s["enum"]:
                    if v not in values:
                        values.append(v)
            merged["enum"] = values
        if any(s.get("nullable") for s in cleaned):
            merged["nullable"] = True
        return merged

    # Genuinely heterogeneous union: Gemini has no way to express it.
    return {}


def _sanitize_node(
    node: Any, root: dict[str, Any], depth: int, budget: _Budget
) -> dict[str, Any]:
    """Sanitize one schema node. Returns {} for anything unrepresentable."""
    if not isinstance(node, dict) or depth > MAX_DEPTH or not budget.spend():
        return {}

    if "$ref" in node:
        resolved = _resolve_ref(node["$ref"], root)
        if resolved is None:
            return {}
        return _sanitize_node(resolved, root, depth + 1, budget)

    # oneOf/allOf are not supported; treat oneOf like anyOf and drop allOf.
    branches = node.get("anyOf") or node.get("oneOf")
    if isinstance(branches, list) and branches:
        return _collapse_any_of(branches, root, depth, budget)

    out: dict[str, Any] = {}
    type_name, nullable_from_type = _normalize_type(node.get("type"))
    if type_name:
        out["type"] = type_name
    if nullable_from_type or node.get("nullable") is True:
        out["nullable"] = True

    for key in ("description", "format"):
        value = node.get(key)
        if isinstance(value, str):
            out[key] = value

    if isinstance(node.get("enum"), list) and node["enum"]:
        out["enum"] = list(node["enum"])

    if type_name == "object" or "properties" in node:
        raw_props = node.get("properties")
        if isinstance(raw_props, dict):
            props: dict[str, Any] = {}
            for prop_name, prop_schema in raw_props.items():
                sanitized = _sanitize_node(prop_schema, root, depth + 1, budget)
                # An empty schema is still a valid, maximally permissive
                # property -- keep the key so `required` stays coherent.
                props[prop_name] = sanitized
            out["properties"] = props
            out.setdefault("type", "object")
            required = node.get("required")
            if isinstance(required, list):
                kept = [r for r in required if isinstance(r, str) and r in props]
                if kept:
                    out["required"] = kept

    if type_name == "array" or "items" in node:
        items = node.get("items")
        if isinstance(items, dict):
            out["items"] = _sanitize_node(items, root, depth + 1, budget)
            out.setdefault("type", "array")

    return {k: v for k, v in out.items() if k in _ALLOWED_KEYWORDS}


def sanitize_for_gemini(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a JSON Schema into the restricted dialect Gemini accepts.

    The result is always a `type: "object"` schema, since Gemini requires
    that at the root of a function declaration's parameters.
    """
    budget = _Budget()
    root = schema if isinstance(schema, dict) else {}
    sanitized = _sanitize_node(root, root, 0, budget)

    # Root must be an object; a collapsed/empty/non-object root becomes an
    # empty object schema (a function taking no arguments).
    if sanitized.get("type") != "object":
        properties = sanitized.get("properties")
        sanitized = {"type": "object", "properties": properties if isinstance(properties, dict) else {}}
        return sanitized

    sanitized.setdefault("properties", {})
    return sanitized
