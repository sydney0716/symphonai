def parse_pair(text: str) -> tuple[str, str]:
    left, separator, right = text.partition(":")
    if not separator or not left or not right:
        raise ValueError("expected two values separated by ':'")
    return left, right
