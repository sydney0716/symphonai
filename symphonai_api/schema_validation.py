"""Shared JSON Schema subset for tool arguments and agent I/O contracts."""

from collections.abc import Mapping


def validate_schema(value: object, schema: Mapping[str, object], path: str) -> str | None:
    expected = schema.get("type")
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool), "null": value is None,
    }
    if expected in valid and not valid[expected]:
        return f"{path} must be {expected}"
    if expected in ("number", "integer") and isinstance(value, (int, float)):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return f"{path} must be >= {minimum}"
        exclusive = schema.get("exclusiveMinimum")
        if exclusive is not None and not value > exclusive:
            return f"{path} must be > {exclusive}"
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            return f"{path} must be <= {maximum}"
        exclusive = schema.get("exclusiveMaximum")
        if exclusive is not None and not value < exclusive:
            return f"{path} must be < {exclusive}"
    for kind, keyword in (("array", "minItems"), ("string", "minLength")):
        if expected == kind and keyword in schema and len(value) < schema[keyword]:
            return f"{path} must have at least {schema[keyword]} {'items' if kind == 'array' else 'characters'}"
    if expected == "string" and "maxLength" in schema and len(value) > schema["maxLength"]:
        return f"{path} exceeds maxLength {schema['maxLength']}; actual length is {len(value)}"
    if expected == "array" and "maxItems" in schema and len(value) > schema["maxItems"]:
        return f"{path} must have at most {schema['maxItems']} items"
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and value not in enum:
        return f"{path} must be one of the allowed enum values"
    if expected == "object" and isinstance(value, dict):
        required = schema.get("required", ())
        if isinstance(required, (list, tuple)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path} is missing required property {key!r}"
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            if schema.get("additionalProperties") is False:
                for key in value:
                    if key not in properties:
                        return f"{path} has unexpected property {key!r}"
            for key, child in properties.items():
                if key in value and isinstance(child, Mapping):
                    error = validate_schema(value[key], child, f"{path}.{key}")
                    if error:
                        return error
    if expected == "array" and isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            error = validate_schema(item, schema["items"], f"{path}[{index}]")
            if error:
                return error
    return None

