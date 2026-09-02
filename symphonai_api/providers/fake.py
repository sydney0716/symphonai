"""A fully deterministic ModelProvider for tests and local development.

`FakeModelProvider` never makes a network call. It replays a pre-configured
scripted sequence of `ModelResponse` objects, one per call to
`create_response()`, so an `ApiAgent` run -- including tool-call turns --
can be exercised end-to-end without any real model or network access.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterator

from symphonai_api.cancellation import CancellationToken
from symphonai_api.models import Message, ModelRequest, ModelResponse, Role
from symphonai_api.providers.base import ModelProvider

if TYPE_CHECKING:
    from symphonai_api.streaming import StreamChunk

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

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        *,
        streams: list[Sequence[StreamChunk]] | None = None,
    ) -> None:
        self._responses: list[ModelResponse] = (
            list(responses) if responses else [_NO_SCRIPT_RESPONSE]
        )
        self._streams = (
            None if not streams else [tuple(stream) for stream in streams]
        )
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

    def create_response(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> ModelResponse:
        if cancel is not None:
            cancel.raise_if_cancelled()
        index = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[index]

    def create_response_stream(
        self, request: ModelRequest, *, cancel: CancellationToken | None = None
    ) -> Iterator[StreamChunk]:
        if self._streams is None:
            yield from super().create_response_stream(request, cancel=cancel)
            return
        index = min(self._call_count, len(self._streams) - 1)
        self._call_count += 1
        for chunk in self._streams[index]:
            if cancel is not None:
                cancel.raise_if_cancelled()
            yield chunk
