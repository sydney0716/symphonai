"""Host-side protocol boundary for the SymphonAI runtime."""

from symphonai_host.protocol import (
    PROTOCOL_VERSION,
    ApprovalReply,
    PromptRequest,
    ProtocolError,
    StopRequest,
    UnknownEvent,
    decode_event,
    decode_frame,
    decode_request,
    encode_event,
    encode_frame,
    event_type_name,
)
from symphonai_host.server import HostServer

__all__ = [
    "PROTOCOL_VERSION",
    "ApprovalReply",
    "PromptRequest",
    "ProtocolError",
    "StopRequest",
    "UnknownEvent",
    "decode_event",
    "decode_frame",
    "decode_request",
    "encode_event",
    "encode_frame",
    "event_type_name",
    "HostServer",
]
