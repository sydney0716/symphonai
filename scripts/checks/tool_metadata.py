"""Fixture-free checks for tool metadata."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from symphonai_api import FAIL_CLOSED, InterruptBehavior, ResultHint, SCHEMA_VERSION, ToolEffect, ToolMetadata, safe_metadata
from symphonai_api.cancellation import OperationCancelled
from symphonai_api.models import ToolResult
from symphonai_api.runner import standard_tool_registry
from symphonai_api.tool_schema import to_provider_tool_schema
from symphonai_api.tools.base import LocalTool
from scripts.checks.harness import check, fail


class _ToolContractStub(LocalTool):
    @property
    def name(self) -> str:
        return "contract_stub"

    @property
    def description(self) -> str:
        return "Exercise the LocalTool contract."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}


class _MissingMetadataTool(_ToolContractStub):
    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult(tool_call_id=tool_call.id, ok=True)


class _MissingExecuteTool(_ToolContractStub):
    def metadata(self, arguments: dict) -> ToolMetadata:
        return ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
        )


class _RaisingMetadataTool(_ToolContractStub):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def metadata(self, arguments: dict) -> ToolMetadata:
        raise self._error

    def _execute(
        self,
        tool_call: ToolCall,
        policy: PermissionPolicy,
        cancel: CancellationToken | None = None,
    ) -> ToolResult:
        return ToolResult(tool_call_id=tool_call.id, ok=True)

@check("tools.localtool_contract")
def check_tools_localtool_contract() -> None:
    for incomplete_tool, missing_member in (
        (_MissingMetadataTool, "metadata"),
        (_MissingExecuteTool, "_execute"),
    ):
        try:
            incomplete_tool()
        except TypeError:
            pass
        else:
            fail(f"LocalTool subclass missing {missing_member} was instantiable")

@check("tools.metadata_contract")
def check_tools_metadata_contract() -> None:
    metadata_tools = standard_tool_registry()
    execute_overrides = [
        name
        for name, tool in metadata_tools.items()
        if type(tool).execute is not LocalTool.execute
    ]
    if execute_overrides:
        fail(f"tools override the base validation pipeline: {execute_overrides!r}")

    metadata_arguments = {
        "read_file": {"path": "sample.txt"},
        "write_file": {"path": "sample.txt", "content": "replacement"},
        "edit_file": {
            "path": "sample.txt",
            "old_string": "before",
            "new_string": "after",
        },
        "multi_edit_file": {
            "path": "sample.txt",
            "edits": [{"old_string": "before", "new_string": "after"}],
        },
        "list_files": {"path": "sample-dir"},
        "glob": {"pattern": "**/*.py", "path": "sample-dir"},
        "grep": {"pattern": "needle", "path": "sample-dir"},
        "run_shell": {"argv": ["ls"]},
        "web_fetch": {"url": "https://docs.python.org/3/"},
    }
    expected_metadata = {
        "read_file": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample.txt",),
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "write_file": ToolMetadata(
            effect=ToolEffect.DESTRUCTIVE,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "edit_file": ToolMetadata(
            effect=ToolEffect.MUTATING,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.DIFF,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "multi_edit_file": ToolMetadata(
            effect=ToolEffect.MUTATING,
            concurrency_safe=False,
            paths=("sample.txt",),
            result_hint=ResultHint.DIFF,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "list_files": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "glob": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "grep": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=("sample-dir",),
            result_hint=ResultHint.FILE_LIST,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        "run_shell": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=None,
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
        # paths=() and not None: a fetch provably touches no path, which is a
        # different answer from run_shell's "not derivable from the arguments".
        "web_fetch": ToolMetadata(
            effect=ToolEffect.READ_ONLY,
            concurrency_safe=True,
            paths=(),
            result_hint=ResultHint.TEXT,
            interrupt_behavior=InterruptBehavior.CANCEL,
        ),
    }
    actual_metadata = {
        name: tool.metadata(metadata_arguments[name])
        for name, tool in metadata_tools.items()
    }
    if actual_metadata != expected_metadata:
        fail(
            "standard tool metadata did not match the literal contract: "
            f"actual={actual_metadata!r}, expected={expected_metadata!r}"
        )
    if any(item.schema_version != SCHEMA_VERSION for item in actual_metadata.values()):
        fail(f"tool metadata schema version drifted: {actual_metadata!r}")

    raw_path = "../outside.txt"
    if metadata_tools["read_file"].metadata({"path": raw_path}).paths != (raw_path,):
        fail("read_file metadata resolved or discarded its raw path")
    if metadata_tools["write_file"].metadata({"path": raw_path}).paths != (raw_path,):
        fail("write_file metadata resolved or discarded its raw path")
    if metadata_tools["list_files"].metadata({}).paths != (".",):
        fail("list_files metadata did not expose its default path")
    for name in ("glob", "grep"):
        if metadata_tools[name].metadata({}).paths != (".",):
            fail(f"{name} metadata did not expose its default path")
        if metadata_tools[name].metadata({"path": 3}).paths is not None:
            fail(f"{name} metadata accepted a non-string path")
    if metadata_tools["grep"].metadata({"output_mode": "content"}).result_hint != ResultHint.TEXT:
        fail("grep content metadata did not declare a text result")
    for name in ("read_file", "write_file", "edit_file", "multi_edit_file"):
        if metadata_tools[name].metadata({}).paths is not None:
            fail(f"{name} metadata treated a missing path as an empty path set")
        if metadata_tools[name].metadata({"path": 3}).paths is not None:
            fail(f"{name} metadata accepted a non-string path")
    if metadata_tools["list_files"].metadata({"path": 3}).paths is not None:
        fail("list_files metadata accepted a non-string path")
    if metadata_tools["run_shell"].metadata({"argv": ["ls"]}).paths is not None:
        fail("run_shell inferred paths from ambiguous argv operands")

    if safe_metadata(_RaisingMetadataTool(ValueError("bad metadata")), {}) is not FAIL_CLOSED:
        fail("safe_metadata did not fail closed after a metadata exception")
    try:
        safe_metadata(_RaisingMetadataTool(OperationCancelled()), {})
    except OperationCancelled:
        pass
    else:
        fail("safe_metadata swallowed OperationCancelled")

    for name, item in actual_metadata.items():
        if item.concurrency_safe and item.effect != ToolEffect.READ_ONLY:
            fail(f"concurrency-safe registry call was not read-only: {name}={item!r}")
    if FAIL_CLOSED.concurrency_safe and FAIL_CLOSED.effect != ToolEffect.READ_ONLY:
        fail(f"FAIL_CLOSED violated the concurrency invariant: {FAIL_CLOSED!r}")
    if FAIL_CLOSED.schema_version != SCHEMA_VERSION:
        fail(f"FAIL_CLOSED schema version drifted: {FAIL_CLOSED!r}")

    frozen_metadata = expected_metadata["read_file"]
    try:
        frozen_metadata.effect = ToolEffect.DESTRUCTIVE
    except FrozenInstanceError:
        pass
    else:
        fail("ToolMetadata fields were mutable")

@check("tools.metadata_absent_from_schemas")
def check_tools_metadata_absent_from_schemas() -> None:
    metadata_tools = standard_tool_registry()
    metadata_field_names = (
        "effect",
        "concurrency_safe",
        "paths",
        "result_hint",
        "interrupt_behavior",
        "schema_version",
    )
    read_parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative to the allowed root.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based first line to read.",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read; 0 reads to end of file. Defaults to 2000.",
                "default": 2000,
            },
        },
        "required": ["path"],
    }
    gemini_read_parameters = json.loads(json.dumps(read_parameters))
    gemini_read_parameters["properties"]["offset"].pop("default")
    gemini_read_parameters["properties"]["limit"].pop("default")
    read_description = "Read a numbered line range from a text file inside the allowed scope."
    literal_read_schemas = {
        1: {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": read_description,
                "parameters": read_parameters,
            },
        },
        2: {
            "name": "read_file",
            "description": read_description,
            "input_schema": read_parameters,
        },
        3: {
            "name": "read_file",
            "description": read_description,
            "parameters": gemini_read_parameters,
        },
        4: {
            "name": "read_file",
            "description": read_description,
            "properties": read_parameters,
        },
    }
    for wire_format in (1, 2, 3, 4):
        for name, tool in metadata_tools.items():
            schema = to_provider_tool_schema(tool, wire_format)
            serialized = json.dumps(schema)
            leaked = [field for field in metadata_field_names if field in serialized]
            if leaked:
                fail(
                    f"metadata leaked into {name} wire format {wire_format}: "
                    f"fields={leaked!r}, schema={schema!r}"
                )
        read_schema = to_provider_tool_schema(
            metadata_tools["read_file"], wire_format
        )
        if read_schema != literal_read_schemas[wire_format]:
            fail(
                f"read_file wire format {wire_format} changed: "
                f"actual={read_schema!r}, expected={literal_read_schemas[wire_format]!r}"
            )
