"""Pure-ASGI request admission before framework JSON parsing.

The middleware deliberately buffers only a bounded raw request body and then
replays it to FastAPI.  It rejects ambiguous lengths and compressed bodies so
the Pydantic model limit cannot be bypassed by chunking or decompression.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Mapping, MutableMapping


AsgiMessage = MutableMapping[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiScope = MutableMapping[str, Any]
AsgiApp = Callable[[AsgiScope, AsgiReceive, AsgiSend], Awaitable[None]]


_APPROVAL_PATH = re.compile(
    r"^/api/v1/runs/[^/]+/approvals/[^/]+$"
)
_EMPTY_POST_PATH = re.compile(r"^/api/v1/runs/[^/]+/(?:cancel|resume)$")


def _body_policy(scope: Mapping[str, Any]) -> str:
    if str(scope.get("method", "")).upper() != "POST":
        return "optional"
    path = str(scope.get("path", ""))
    if path == "/api/v1/runs" or _APPROVAL_PATH.fullmatch(path):
        return "json"
    if _EMPTY_POST_PATH.fullmatch(path):
        return "empty"
    return "optional"


def _error_document(message: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "code": "INVALID_REQUEST",
                "message": message,
                "retryable": False,
                "field_path": None,
                "dependency": None,
                "details": {},
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")


async def _send_error(
    send: AsgiSend,
    *,
    status_code: int,
    message: str,
) -> None:
    body = _error_document(message)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _header_values(scope: Mapping[str, Any], name: bytes) -> list[bytes]:
    headers = scope.get("headers", ())
    return [
        value
        for key, value in headers
        if isinstance(key, bytes)
        and isinstance(value, bytes)
        and key.lower() == name
    ]


class RawBodyAdmissionMiddleware:
    """Reject unsafe or oversized bodies before the downstream parser runs."""

    def __init__(self, app: AsgiApp, *, max_body_bytes: int = 1024 * 1024) -> None:
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes < 1
            or max_body_bytes > 64 * 1024 * 1024
        ):
            raise ValueError("max_body_bytes must be between 1 byte and 64 MiB")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: AsgiScope,
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_lengths = _header_values(scope, b"content-length")
        if len(content_lengths) > 1:
            await _send_error(
                send,
                status_code=400,
                message="request body length is invalid",
            )
            return
        declared_length: int | None = None
        if content_lengths:
            raw_length = content_lengths[0]
            if (
                not raw_length
                or len(raw_length) > 20
                or not raw_length.isdigit()
            ):
                await _send_error(
                    send,
                    status_code=400,
                    message="request body length is invalid",
                )
                return
            declared_length = int(raw_length)
            if declared_length > self.max_body_bytes:
                await _send_error(
                    send,
                    status_code=413,
                    message="request body exceeds the configured limit",
                )
                return

        content_encodings = _header_values(scope, b"content-encoding")
        if len(content_encodings) > 1:
            await _send_error(
                send,
                status_code=415,
                message="request content encoding is not supported",
            )
            return
        if content_encodings:
            try:
                content_encoding = content_encodings[0].decode("ascii").strip().lower()
            except UnicodeDecodeError:
                content_encoding = ""
            if content_encoding != "identity":
                await _send_error(
                    send,
                    status_code=415,
                    message="request content encoding is not supported",
                )
                return

        body_policy = _body_policy(scope)
        if body_policy == "json":
            content_types = _header_values(scope, b"content-type")
            if len(content_types) != 1:
                await _send_error(
                    send,
                    status_code=415,
                    message="request content type is not supported",
                )
                return
            try:
                content_type = (
                    content_types[0]
                    .decode("ascii")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
            except UnicodeDecodeError:
                content_type = ""
            if content_type != "application/json":
                await _send_error(
                    send,
                    status_code=415,
                    message="request content type is not supported",
                )
                return

        buffered = bytearray()
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                await _send_error(
                    send,
                    status_code=400,
                    message="request body was interrupted",
                )
                return
            if message_type != "http.request":
                await _send_error(
                    send,
                    status_code=400,
                    message="request body is invalid",
                )
                return
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                await _send_error(
                    send,
                    status_code=400,
                    message="request body is invalid",
                )
                return
            if len(chunk) > self.max_body_bytes - len(buffered):
                await _send_error(
                    send,
                    status_code=413,
                    message="request body exceeds the configured limit",
                )
                return
            buffered.extend(chunk)
            if not message.get("more_body", False):
                break

        if declared_length is not None and declared_length != len(buffered):
            await _send_error(
                send,
                status_code=400,
                message="request body length is invalid",
            )
            return
        if body_policy == "empty" and buffered:
            await _send_error(
                send,
                status_code=422,
                message="request body must be empty",
            )
            return

        replayed = False

        async def replay_receive() -> AsgiMessage:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)


__all__ = ["RawBodyAdmissionMiddleware"]
