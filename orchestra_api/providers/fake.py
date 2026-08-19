"""A fully deterministic ModelProvider for tests and local development.

`FakeModelProvider` never makes a network call. It replays a pre-configured
scripted sequence of `ModelResponse` objects, one per call to
`create_response()`, so an `ApiAgent` run -- including tool-call turns --
can be exercised end-to-end without any real model or network access.
"""

from __future__ import annotations

from orchestra_api.models import Message, ModelRequest, ModelResponse, Role
from orchestra_api.providers.base import ModelProvider

_NO_SCRIPT_RESPONSE = ModelResponse(
    message=Message(role=Role.ASSISTANT, content="(no scripted response configured)")
)


class FakeModelProvider(ModelProvider):
    """Deterministic provider that replays a scripted list of responses.

    Each call to `create_response()` returns the next response in the
    script, in order. If more calls are made than there are scripted
    responses, the last one is repeated -- so a caller can always end a run
    on a final (non-tool-call) answer without over-scripting exact call
    counts.
    """

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._responses: list[ModelResponse] = list(responses) if responses else [_NO_SCRIPT_RESPONSE]
        self._call_count = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def wire_format(self) -> int:
        return 4

    @property
    def call_count(self) -> int:
        return self._call_count

    def create_response(self, request: ModelRequest) -> ModelResponse:
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[index]
