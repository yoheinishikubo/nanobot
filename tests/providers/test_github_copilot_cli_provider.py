"""Tests for the GitHub Copilot CLI provider (ACP-based GitHubCopilotCLIProvider)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.providers.github_copilot_cli_provider import (
    GitHubCopilotCLIProvider,
    _JsonRpcConnection,
)


# ---------------------------------------------------------------------------
# Helpers – fake ACP subprocess
# ---------------------------------------------------------------------------

def _encode_rpc(msg: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with LSP Content-Length header."""
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


class _FakeStdin:
    """Captures writes and parses JSON-RPC requests to auto-respond."""

    def __init__(self, auto_responses: dict[str, Any] | None = None, notifications: list[dict[str, Any]] | None = None) -> None:
        self.written = bytearray()
        self._auto_responses = auto_responses or {}
        self._notifications = notifications or []
        self._stdout_ref: _FakeStdout | None = None
        self._buf = b""

    def write(self, data: bytes) -> None:
        self.written.extend(data)
        self._buf += data
        # Parse incoming requests and auto-respond
        while True:
            header_end = self._buf.find(b"\r\n\r\n")
            if header_end == -1:
                break
            header_part = self._buf[:header_end].decode("ascii", "ignore")
            content_length = 0
            for line in header_part.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":", 1)[1].strip())
            body_start = header_end + 4
            if len(self._buf) < body_start + content_length:
                break
            body = self._buf[body_start : body_start + content_length]
            self._buf = self._buf[body_start + content_length:]
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "id" in msg and "method" in msg:
                self._handle_request(msg)

    def _handle_request(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        msg_id = msg["id"]
        if method in self._auto_responses:
            result = self._auto_responses[method]
            if callable(result):
                result = result(msg.get("params", {}))
            response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            if self._stdout_ref:
                self._stdout_ref.inject(_encode_rpc(response))
                # If this is session.send, also inject notifications
                if method == "session.send":
                    for notif in self._notifications:
                        self._stdout_ref.inject(_encode_rpc(notif))


class _FakeStdout:
    """Async readable that allows injecting data dynamically."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._closed = False

    def inject(self, data: bytes) -> None:
        """Inject data to be read."""
        self._queue.put_nowait(data)

    def close(self) -> None:
        self._closed = True
        self._queue.put_nowait(b"")

    async def read(self, n: int) -> bytes:
        try:
            data = await asyncio.wait_for(self._queue.get(), timeout=5)
            return data
        except asyncio.TimeoutError:
            return b""


class _FakeProc:
    """Fake asyncio.subprocess.Process with auto-response capabilities."""

    returncode: int | None = None

    def __init__(
        self,
        auto_responses: dict[str, Any] | None = None,
        notifications: list[dict[str, Any]] | None = None,
    ) -> None:
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(auto_responses, notifications)
        self.stdin._stdout_ref = self.stdout
        self.stderr = None

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()

    async def wait(self) -> int:
        return self.returncode or 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_PING_RESULT = {"message": "nanobot", "timestamp": 0, "protocolVersion": 2}


@pytest.mark.asyncio
async def test_json_rpc_connection_send_and_receive() -> None:
    """The JSON-RPC connection sends requests and resolves futures from responses."""
    proc = _FakeProc(auto_responses={"ping": {"message": "pong", "timestamp": 123, "protocolVersion": 2}})
    conn = _JsonRpcConnection(proc)
    conn.start()

    result = await asyncio.wait_for(conn.send_request("ping", {"message": "hello"}), timeout=2)
    assert result["message"] == "pong"

    await conn.close()
    proc.terminate()


@pytest.mark.asyncio
async def test_json_rpc_connection_handles_notifications() -> None:
    """Notifications from the server are dispatched to registered handlers."""
    proc = _FakeProc()
    conn = _JsonRpcConnection(proc)

    received: list[Any] = []
    conn.on_notification("session.event", lambda p: received.append(p))
    conn.start()

    # Inject a notification
    proc.stdout.inject(_encode_rpc({
        "jsonrpc": "2.0",
        "method": "session.event",
        "params": {"type": "session.idle", "data": {}},
    }))

    await asyncio.sleep(0.1)
    assert len(received) == 1
    assert received[0]["type"] == "session.idle"

    await conn.close()
    proc.terminate()


@pytest.mark.asyncio
async def test_json_rpc_connection_handles_server_requests() -> None:
    """Server→client requests are handled and responses sent back."""
    proc = _FakeProc()
    conn = _JsonRpcConnection(proc)
    conn.on_request("permission.request", lambda p: {"result": "allow"})
    conn.start()

    # Inject a server→client request
    proc.stdout.inject(_encode_rpc({
        "jsonrpc": "2.0", "id": 42,
        "method": "permission.request",
        "params": {"kind": "shell"},
    }))

    await asyncio.sleep(0.1)

    # Check that a response was written back
    written = bytes(proc.stdin.written)
    assert b'"id": 42' in written or b'"id":42' in written
    assert b'"allow"' in written

    await conn.close()
    proc.terminate()


def _make_provider_proc(
    notifications: list[dict[str, Any]] | None = None,
) -> _FakeProc:
    """Create a _FakeProc pre-configured for provider tests."""
    return _FakeProc(
        auto_responses={
            "ping": _PING_RESULT,
            "session.create": {},
            "session.send": {"messageId": "msg-1"},
        },
        notifications=notifications,
    )


@pytest.mark.asyncio
async def test_provider_chat_via_acp(monkeypatch, tmp_path: Path) -> None:
    """Full end-to-end: provider sends prompt via ACP and receives streamed response."""
    notifications = [
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "assistant.message_delta", "data": {"messageId": "msg-1", "deltaContent": "Hello "}},
        }},
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "assistant.message_delta", "data": {"messageId": "msg-1", "deltaContent": "world"}},
        }},
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "session.idle", "data": {}},
        }},
    ]

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _: "/usr/bin/copilot")

    async def _fake_subprocess(*args, **kwargs):
        return _make_provider_proc(notifications)

    monkeypatch.setattr(
        "nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec",
        _fake_subprocess,
    )

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    response = await provider.chat([{"role": "user", "content": "Say hi"}])

    assert response.content == "Hello world"
    assert response.finish_reason == "stop"


async def _async_append(lst: list[str], item: str) -> None:
    lst.append(item)


@pytest.mark.asyncio
async def test_provider_chat_stream_via_acp(monkeypatch, tmp_path: Path) -> None:
    """Provider streams deltas via on_content_delta callback."""
    notifications = [
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "assistant.message_delta", "data": {"messageId": "msg-1", "deltaContent": "hello "}},
        }},
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "assistant.message_delta", "data": {"messageId": "msg-1", "deltaContent": "from "}},
        }},
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "assistant.message_delta", "data": {"messageId": "msg-1", "deltaContent": "copilot"}},
        }},
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "session.idle", "data": {}},
        }},
    ]

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _: "/usr/bin/copilot")

    async def _fake_subprocess(*args, **kwargs):
        return _make_provider_proc(notifications)

    monkeypatch.setattr(
        "nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec",
        _fake_subprocess,
    )

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    deltas: list[str] = []
    response = await provider.chat_stream(
        [{"role": "user", "content": "Say hi"}],
        on_content_delta=lambda chunk: _async_append(deltas, chunk),
    )

    assert deltas == ["hello ", "from ", "copilot"]
    assert response.content == "hello from copilot"


@pytest.mark.asyncio
async def test_provider_handles_acp_error(monkeypatch, tmp_path: Path) -> None:
    """Provider returns error when ACP session reports an error."""
    notifications = [
        {"jsonrpc": "2.0", "method": "session.event", "params": {
            "event": {"type": "session.error", "data": {"message": "Rate limit exceeded"}},
        }},
    ]

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _: "/usr/bin/copilot")

    async def _fake_subprocess(*args, **kwargs):
        return _make_provider_proc(notifications)

    monkeypatch.setattr(
        "nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec",
        _fake_subprocess,
    )

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    response = await provider.chat([{"role": "user", "content": "test"}])

    assert response.finish_reason == "error"
    assert "Rate limit exceeded" in (response.content or "")


@pytest.mark.asyncio
async def test_provider_cli_not_found(monkeypatch, tmp_path: Path) -> None:
    """Provider returns error when copilot CLI is not found."""
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _: None)

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    response = await provider.chat([{"role": "user", "content": "test"}])

    assert response.finish_reason == "error"
    assert "not installed" in (response.content or "")
