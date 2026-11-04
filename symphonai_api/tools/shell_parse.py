"""Parse the shell syntax SymphonAI can check safely."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ShellCommand:
    argv: tuple[str, ...]
    writes: bool = False
    stdin_scripts: tuple[str, ...] = ()
    input_paths: tuple[str, ...] = ()


_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL = {"sh", "bash", "zsh", "dash"}
_SHELL_C = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")
_SEPARATORS = {"|", "||", "&&", ";", "\n"}
_WRAPPERS = {"env", "nohup", "time", "command", "exec", "nice", "timeout", "xargs", "stdbuf", "find"}


def _lex(text: str) -> list[list[tuple[str, str, bool, bool, bool]]] | None:
    segments: list[list[tuple[str, str, bool, bool, bool]]] = []
    current: list[tuple[str, str, bool, bool, bool]] = []
    pending_heredocs: list[tuple[list, str, bool, bool]] = []
    word: list[str] = []
    active = False
    expands = False
    pattern = False
    brace = False
    state = "plain"
    index = 0

    def flush_word() -> None:
        nonlocal active, expands, pattern, brace
        if active:
            value = "".join(word)
            current.append(("brace" if brace and value in {"{", "}"} else "word", value, expands, pattern, brace))
            word.clear()
            active = False
            expands = pattern = brace = False

    def flush_segment() -> bool:
        nonlocal current
        flush_word()
        if not current:
            return False
        segments.append(current)
        current = []
        return True

    def heredoc_delimiter(start: int) -> tuple[str, int, bool] | None:
        while start < len(text) and text[start] in " \t":
            start += 1
        value: list[str] = []
        state = "plain"
        quoted = False
        while start < len(text):
            char = text[start]
            if state == "single":
                if char == "'":
                    state = "plain"
                else:
                    value.append(char)
                start += 1
                continue
            if state == "double":
                if char == '"':
                    state = "plain"
                elif char == "\\" and start + 1 < len(text):
                    start += 1
                    value.append(text[start])
                else:
                    if char == "$" or char == "`":
                        return None
                    value.append(char)
                start += 1
                continue
            if char in " \t\r\n|;&<>()":
                break
            if char == "'":
                quoted = True
                state = "single"
            elif char == '"':
                quoted = True
                state = "double"
            elif char == "\\" and start + 1 < len(text):
                quoted = True
                start += 1
                value.append(text[start])
            elif char == "$" or char == "`":
                return None
            else:
                value.append(char)
            start += 1
        if state != "plain" or not value:
            return None
        return "".join(value), start, quoted

    def consume_heredocs(start: int) -> int | None:
        nonlocal pending_heredocs
        for owner, delimiter, strip_tabs, quoted in pending_heredocs:
            body: list[str] = []
            found = False
            while start <= len(text):
                end = text.find("\n", start)
                if end < 0:
                    line, next_start = text[start:], len(text)
                else:
                    line, next_start = text[start:end], end + 1
                compare = line.lstrip("\t") if strip_tabs else line
                if compare.endswith("\r"):
                    compare = compare[:-1]
                if compare == delimiter:
                    start = next_start
                    found = True
                    break
                body.append(line.lstrip("\t") if strip_tabs else line)
                if end < 0:
                    break
                start = next_start
            if not found:
                return None
            body_text = "\n".join(body)
            # Here-doc contents are opaque data; only command substitutions
            # make an unquoted body impossible to check safely.
            if not quoted and ("$(" in body_text or "`" in body_text):
                return None
            owner.append(("heredoc", body_text, False, False, False))
        pending_heredocs = []
        return start

    while index < len(text):
        char = text[index]
        if state == "single":
            if char == "'":
                state = "plain"
            else:
                word.append(char)
            index += 1
            continue
        if state == "double":
            if char == '"':
                state = "plain"
                index += 1
                continue
            if char == "`" or text.startswith("$(", index):
                return None
            if char == "$":
                expands = True
            if char == "\\" and index + 1 < len(text):
                next_char = text[index + 1]
                if next_char == "\n":
                    index += 2
                    continue
                if next_char in {'"', "\\", "$", "`"}:
                    word.append(next_char)
                    index += 2
                    continue
            word.append(char)
            index += 1
            continue

        if char == "#" and not active:
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        if char in " \t\r":
            flush_word()
            index += 1
            continue
        if char == "'":
            active = True
            state = "single"
            index += 1
            continue
        if char == '"':
            active = True
            state = "double"
            index += 1
            continue
        if char == "\\":
            if index + 1 >= len(text):
                return None
            if text[index + 1] != "\n":
                word.append(text[index + 1])
                active = True
            index += 2
            continue
        if char == "`" or text.startswith("$(", index):
            return None
        if text.startswith("<(", index) or text.startswith(">(", index):
            return None
        if char in "()":
            return None
        if char in "{}" and not active and (
            index + 1 == len(text)
            or text[index + 1].isspace()
            or text[index + 1] in "|;&<>"
        ):
            word.append(char)
            active = True
            brace = True
            index += 1
            continue
        if char in "|;&<>" or char == "\n":
            flush_word()
            if char == "&":
                if text.startswith("&&", index):
                    operator, width = "&&", 2
                elif text.startswith("&>>", index):
                    current.append(("redir", "&>>", False, False, False))
                    index += 3
                    continue
                elif text.startswith("&>", index):
                    current.append(("redir", "&>", False, False, False))
                    index += 2
                    continue
                else:
                    return None
            elif char == "|":
                if text.startswith("||", index):
                    operator, width = "||", 2
                else:
                    operator, width = "|", 1
            elif char == ";" or char == "\n":
                operator, width = char, 1
            else:
                if text.startswith("<<<", index):
                    current.append(("redir", "<<<", False, False, False))
                    index += 3
                    continue
                if text.startswith("<<-", index) or text.startswith("<<", index):
                    strip_tabs = text.startswith("<<-", index)
                    op_width = 3 if strip_tabs else 2
                    delimiter = heredoc_delimiter(index + op_width)
                    if delimiter is None:
                        return None
                    delimiter_word, index, quoted = delimiter
                    owner = current
                    current.append(("redir", "<<-" if strip_tabs else "<<", False, False, False))
                    current.append(("word", delimiter_word, False, False, False))
                    pending_heredocs.append((owner, delimiter_word, strip_tabs, quoted))
                    continue
                fd = ""
                if text.startswith(">>", index):
                    redir, width = ">>", 2
                elif text.startswith(">&", index):
                    redir, width = ">&", 2
                elif text.startswith("<&", index):
                    redir, width = "<&", 2
                elif text.startswith(">|", index):
                    redir, width = ">|", 2
                else:
                    redir, width = char, 1
                current.append(("redir", f"{fd}{redir}", False, False, False))
                index += width
                continue
            if not flush_segment() and char != "\n":
                return None
            index += width
            if char == "\n" and pending_heredocs:
                consumed = consume_heredocs(index)
                if consumed is None:
                    return None
                index = consumed
            continue

        # A descriptor number belongs to an adjacent redirection, not argv.
        if char.isdigit() and not active:
            end = index
            while end < len(text) and text[end].isdigit():
                end += 1
            if end < len(text) and text[end] in "<>":
                if text.startswith("<<-", end) or text.startswith("<<", end):
                    strip_tabs = text.startswith("<<-", end)
                    op_width = 3 if strip_tabs else 2
                    delimiter = heredoc_delimiter(end + op_width)
                    if delimiter is None:
                        return None
                    delimiter_word, next_index, quoted = delimiter
                    owner = current
                    current.append(("redir", text[index:end] + ("<<-" if strip_tabs else "<<"), False, False, False))
                    current.append(("word", delimiter_word, False, False, False))
                    pending_heredocs.append((owner, delimiter_word, strip_tabs, quoted))
                    index = next_index
                    continue
                if text.startswith(">>", end):
                    redir, width = ">>", 2
                elif text.startswith(">&", end):
                    redir, width = ">&", 2
                elif text.startswith("<&", end):
                    redir, width = "<&", 2
                elif text.startswith(">|", end):
                    redir, width = ">|", 2
                else:
                    redir, width = text[end], 1
                current.append(("redir", text[index:end] + redir, False, False, False))
                index = end + width
                continue
        if char == "$":
            expands = True
        elif char in "*?[":
            pattern = True
        elif char == "{":
            brace = True
        word.append(char)
        active = True
        index += 1

    if state != "plain":
        return None
    if pending_heredocs:
        return None
    flush_word()
    if current:
        segments.append(current)
    elif not segments:
        return None
    return segments


def _nested_script(argv: tuple[str, ...]) -> tuple[int, str | None] | None:
    if not argv:
        return None
    program = os.path.basename(argv[0])
    if program in _SHELL:
        shell_index = 0
    elif program in _WRAPPERS:
        if program == "command" and len(argv) > 1 and argv[1] in {"-v", "-V"}:
            return None
        candidates = range(1, len(argv))
        if program == "find":
            exec_index = next(
                (index for index, word in enumerate(argv) if word in {"-exec", "-execdir", "-ok", "-okdir"}),
                None,
            )
            candidates = () if exec_index is None else range(exec_index + 1, len(argv))
        shell_index = next(
            (index for index in candidates if os.path.basename(argv[index]) in _SHELL),
            None,
        )
        if shell_index is None:
            return None
    else:
        return None

    index = shell_index + 1
    while index < len(argv):
        option = argv[index]
        if option in {"--version", "--help"}:
            return None
        if _SHELL_C.fullmatch(option):
            return shell_index, argv[index + 1] if index + 1 < len(argv) else None
        if option in {"-o", "+o", "-O", "+O"}:
            index += 2
            continue
        if option.startswith(("-", "+")):
            index += 1
            continue
        return None  # A script file operand is checked as an ordinary command.
    return shell_index, None  # Without a script or file operand, the shell reads stdin.


def _expand(command: ShellCommand, depth: int) -> tuple[ShellCommand, ...] | None:
    nested = _nested_script(command.argv)
    if nested is None:
        return (command,)
    shell_index, script = nested
    if script is None:
        if not command.stdin_scripts:
            return None
        scripts = command.stdin_scripts
    else:
        scripts = (script,)
    if depth >= 3:
        return None
    parsed_commands: list[ShellCommand] = []
    for script in scripts:
        parsed = _parse_command_line(script, depth + 1)
        if parsed is None:
            return None
        parsed_commands.extend(parsed)
    prefix = command.argv[:shell_index]
    return tuple(
        ShellCommand(prefix + item.argv, item.writes or command.writes,
                     input_paths=command.input_paths + item.input_paths)
        for item in parsed_commands
    )


def _parse_segment(tokens: list[tuple[str, str, bool, bool, bool]]) -> ShellCommand | None:
    argv: list[str] = []
    dollar_words: list[bool] = []
    pattern_words: list[bool] = []
    brace_words: list[bool] = []
    stdin_scripts: list[str] = []
    input_paths: list[str] = []
    writes = False
    index = 0
    while index < len(tokens):
        kind, value, expands, pattern, _brace = tokens[index]
        if kind == "heredoc":
            stdin_scripts.append(value)
            index += 1
            continue
        if kind == "brace":
            return None
        if kind == "word":
            argv.append(value)
            dollar_words.append(expands)
            pattern_words.append(pattern)
            brace_words.append(_brace)
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1][0] != "word":
            return None
        operator = value.lstrip("0123456789")
        target = tokens[index + 1][1]
        if operator == "<<<":
            stdin_scripts.append(target)
        if operator == "<":
            input_paths.append(target)
        if operator.startswith(">") or operator.startswith("&>"):
            duplicate = operator.endswith("&") and target.isdigit()
            if target != "/dev/null" and not duplicate:
                writes = True
        index += 2
    while argv and _ASSIGNMENT.fullmatch(argv[0]):
        writes = True  # Environment overrides can make a nominal read execute code.
        argv.pop(0)
        dollar_words.pop(0)
        pattern_words.pop(0)
        brace_words.pop(0)
    if not argv:
        return None
    if os.path.basename(argv[0]) == "env" and any(_ASSIGNMENT.fullmatch(word) for word in argv[1:]):
        writes = True
    keywords = {"if", "then", "elif", "else", "do", "while", "until", "!"}
    while argv and argv[0] in keywords:
        argv.pop(0)
        dollar_words.pop(0)
        pattern_words.pop(0)
        brace_words.pop(0)
    if not argv or argv[0] in {"fi", "done", "esac"}:
        return ShellCommand((":" ,), writes)
    if (
        os.path.basename(argv[0]) == "eval"
        or dollar_words[0]
        or (pattern_words[0] and argv[0] not in {"[", "[["})
        or brace_words[0]
        or (os.path.basename(argv[0]) in _WRAPPERS and any(dollar_words[1:]))
    ):
        return None
    return ShellCommand(tuple(argv), writes, tuple(stdin_scripts), tuple(input_paths))


def _parse_command_line(text: str, depth: int) -> tuple[ShellCommand, ...] | None:
    if not isinstance(text, str) or not text:
        return None
    segments = _lex(text)
    if segments is None:
        return None
    commands: list[ShellCommand] = []
    for segment in segments:
        command = _parse_segment(segment)
        if command is None:
            return None
        expanded = _expand(command, depth)
        if expanded is None:
            return None
        commands.extend(expanded)
    return tuple(commands) if commands else None


def parse_command_line(text: str) -> tuple[ShellCommand, ...] | None:
    """Parse a shell line, returning None when its effects cannot be checked."""
    return _parse_command_line(text, 0)


def commands_for_argv(argv: list[str] | tuple[str, ...]) -> tuple[ShellCommand, ...] | None:
    """Expand nested shell -c scripts; otherwise treat argv as one command."""
    if not argv or not all(isinstance(word, str) for word in argv):
        return None
    command = ShellCommand(tuple(argv))
    return _expand(command, 0)
