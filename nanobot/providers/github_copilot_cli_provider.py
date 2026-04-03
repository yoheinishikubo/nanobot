"""GitHub Copilot CLI provider using ACP (Agent Client Protocol).

Spawns a persistent ``copilot --acp`` process and communicates via JSON-RPC
over stdio, following the Copilot SDK protocol (session.create → session.send
→ session.event notifications).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


class _JsonRpcConnection:
    """Minimal JSON-RPC 2.0 client over stdin/stdout of a subprocess."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._notification_handlers: dict[str, list[Any]] = {}
        self._request_handlers: dict[str, Any] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._buf = b""

    def start(self) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())

    def on_notification(self, method: str, handler: Any) -> None:
        self._notification_handlers.setdefault(method, []).append(handler)

    def on_request(self, method: str, handler: Any) -> None:
        """Register a handler for server→client requests (e.g. permission.request)."""
        self._request_handlers[method] = handler

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future
        self._write(msg)
        return await future

    def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def _write(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        assert self._proc.stdin is not None
        self._proc.stdin.write(header + body)

    async def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        try:
            while True:
                chunk = await self._proc.stdout.read(8192)
                if not chunk:
                    break
                self._buf += chunk
                self._process_buffer()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("JSON-RPC reader error: {}", exc)
        finally:
            # Reject all pending requests
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("ACP process exited"))
            self._pending.clear()

    def _process_buffer(self) -> None:
        while True:
            # Parse LSP-style header: Content-Length: N\r\n\r\n
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
            self._buf = self._buf[body_start + content_length :]
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON-RPC message: {}", body[:200])
                continue
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and "method" in msg:
            # Server→client request
            self._handle_server_request(msg)
        elif "id" in msg:
            # Response to our request
            msg_id = msg["id"]
            future = self._pending.pop(msg_id, None)
            if future and not future.done():
                if "error" in msg:
                    future.set_exception(
                        RuntimeError(f"RPC error: {msg['error'].get('message', msg['error'])}")
                    )
                else:
                    future.set_result(msg.get("result"))
        elif "method" in msg:
            # Notification
            method = msg["method"]
            handlers = self._notification_handlers.get(method, [])
            for handler in handlers:
                try:
                    handler(msg.get("params"))
                except Exception as exc:
                    logger.warning("Notification handler error ({}): {}", method, exc)

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        handler = self._request_handlers.get(method)
        if handler:
            asyncio.create_task(self._respond_to_request(msg["id"], handler, msg.get("params")))
        else:
            # Send error response for unhandled requests
            self._write({
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

    async def _respond_to_request(self, msg_id: Any, handler: Any, params: Any) -> None:
        try:
            result = handler(params)
            if asyncio.iscoroutine(result):
                result = await result
            self._write({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except Exception as exc:
            self._write({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": str(exc)},
            })

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass


class GitHubCopilotCLIProvider(LLMProvider):
    """Provider using ``copilot --acp`` as a persistent JSON-RPC server."""

    def __init__(
        self,
        default_model: str = "github-copilot/gpt-5-mini",
        copilot_model: str = "gpt-5-mini",
        copilot_force: bool = False,
        copilot_allow_all: bool = True,
        copilot_continue: bool = True,
        copilot_autopilot: bool = False,
        copilot_no_ask_user: bool = False,
        copilot_max_autopilot_continues: int | None = None,
        copilot_available_tools: list[str] | None = None,
        copilot_excluded_tools: list[str] | None = None,
        copilot_no_custom_instructions: bool = False,
        copilot_experimental: bool = False,
        working_dir: Path | None = None,
        cli_command: str = "copilot",
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.copilot_model = copilot_model
        self.copilot_force = copilot_force
        self.copilot_allow_all = copilot_allow_all
        self.copilot_continue = copilot_continue
        self.copilot_autopilot = copilot_autopilot
        self.copilot_no_ask_user = copilot_no_ask_user
        self.copilot_max_autopilot_continues = copilot_max_autopilot_continues
        self.copilot_available_tools = copilot_available_tools
        self.copilot_excluded_tools = copilot_excluded_tools
        self.copilot_no_custom_instructions = copilot_no_custom_instructions
        self.copilot_experimental = copilot_experimental
        self.working_dir = working_dir
        self.cli_command = cli_command

        self._conn: _JsonRpcConnection | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._lock = asyncio.Lock()

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # ACP process lifecycle
    # ------------------------------------------------------------------

    async def _ensure_connection(self) -> _JsonRpcConnection:
        """Start the ACP process if not already running, return the connection."""
        if self._conn is not None and self._proc is not None and self._proc.returncode is None:
            return self._conn

        async with self._lock:
            # Double-check after acquiring lock
            if self._conn is not None and self._proc is not None and self._proc.returncode is None:
                return self._conn

            await self._start_acp()
            assert self._conn is not None
            return self._conn

    async def _start_acp(self) -> None:
        command = shutil.which(self.cli_command)
        if not command:
            raise RuntimeError(
                "GitHub Copilot CLI is not installed. Install `@github/copilot` "
                f"and ensure `{self.cli_command}` is on PATH."
            )

        args = [command, "--headless", "--stdio", "--no-auto-update", "--log-level", "error"]
        if self.copilot_experimental:
            args.append("--experimental")

        logger.debug("Starting Copilot CLI (headless stdio): {}", " ".join(args))

        self._proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self.working_dir) if self.working_dir else None,
            env=os.environ.copy(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._conn = _JsonRpcConnection(self._proc)
        self._conn.start()

        # Verify connectivity
        try:
            result = await asyncio.wait_for(
                self._conn.send_request("ping", {"message": "nanobot"}),
                timeout=30,
            )
            logger.debug("ACP ping response: {}", result)
        except Exception as exc:
            await self._cleanup()
            raise RuntimeError(f"Failed to connect to Copilot ACP: {exc}") from exc

    async def _ensure_session(self) -> str:
        """Create or reuse an ACP session, return session ID."""
        conn = await self._ensure_connection()

        if self._session_id is not None:
            return self._session_id

        async with self._lock:
            if self._session_id is not None:
                return self._session_id

            session_id = str(uuid.uuid4())
            create_params: dict[str, Any] = {
                "sessionId": session_id,
                "model": self.copilot_model,
                "requestPermission": True,
                "streaming": True,
            }
            if self.copilot_available_tools:
                create_params["availableTools"] = self.copilot_available_tools
            if self.copilot_excluded_tools:
                create_params["excludedTools"] = self.copilot_excluded_tools
            if self.working_dir:
                create_params["workingDirectory"] = str(self.working_dir)
            if self.copilot_no_custom_instructions:
                create_params["systemMessage"] = {"mode": "replace", "content": "You are a helpful assistant."}

            # Auto-approve all permissions when copilot_allow_all is set
            if self.copilot_allow_all:
                conn.on_request("permission.request", self._handle_permission_approve_all)

            logger.debug("Creating ACP session: {}", session_id)
            await asyncio.wait_for(
                conn.send_request("session.create", create_params),
                timeout=30,
            )
            self._session_id = session_id
            logger.debug("ACP session created: {}", session_id)
            return session_id

    @staticmethod
    def _handle_permission_approve_all(params: Any) -> dict[str, Any]:
        """Approve all permission requests (equivalent to --allow-all)."""
        return {"result": "allow"}

    async def _cleanup(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._proc.kill()
                    await self._proc.wait()
                except Exception:
                    pass
        self._proc = None
        self._session_id = None

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert nanobot messages to a single prompt string for session.send."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            text = self._content_to_text(content)
            if text:
                parts.append(f"[{role}]\n{text}")
        return "\n\n".join(parts)

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if not isinstance(item, dict):
                    chunks.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                elif item_type == "image_url":
                    meta = item.get("_meta") or {}
                    path = meta.get("path") if isinstance(meta, dict) else None
                    chunks.append(f"[image: {path}]" if path else "[image]")
                else:
                    chunks.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(chunks)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._send_and_wait(messages, reasoning_effort=reasoning_effort)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta=None,
    ) -> LLMResponse:
        return await self._send_and_wait(
            messages, reasoning_effort=reasoning_effort, on_content_delta=on_content_delta
        )

    async def _send_and_wait(
        self,
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
        on_content_delta=None,
        timeout: float = 180,
    ) -> LLMResponse:
        """Send a prompt via ACP and wait for the assistant to finish."""
        try:
            session_id = await self._ensure_session()
            conn = self._conn
            assert conn is not None
        except Exception as exc:
            return LLMResponse(content=f"Failed to start Copilot ACP: {exc}", finish_reason="error")

        prompt = self._build_prompt(messages)

        # Set up event collection
        content_parts: list[str] = []
        delta_tasks: list[asyncio.Task[None]] = []
        error_msg: str | None = None
        idle_event = asyncio.Event()

        def on_event(params: Any) -> None:
            nonlocal error_msg
            if not params:
                return
            event_session = params.get("sessionId")
            if event_session and event_session != session_id:
                return
            # Events are nested: params.event.type / params.event.data
            event_obj = params.get("event", params)
            event_type = event_obj.get("type")
            data = event_obj.get("data", {})

            if event_type == "assistant.message_delta":
                delta = data.get("deltaContent", "")
                if delta:
                    content_parts.append(delta)
                    if on_content_delta:
                        coro = on_content_delta(delta)
                        if asyncio.iscoroutine(coro):
                            delta_tasks.append(asyncio.create_task(coro))

            elif event_type == "assistant.message":
                # Final message — may contain full content
                full = data.get("content")
                if full and not content_parts:
                    content_parts.append(full)

            elif event_type == "session.idle":
                idle_event.set()

            elif event_type == "session.error":
                error_msg = data.get("message", "Unknown ACP error")
                idle_event.set()

        conn.on_notification("session.event", on_event)

        try:
            # Switch model if needed
            if reasoning_effort:
                try:
                    await conn.send_request(
                        "session.model.switchTo",
                        {"sessionId": session_id, "modelId": self.copilot_model, "reasoningEffort": reasoning_effort},
                    )
                except Exception as exc:
                    logger.warning("Failed to set reasoning effort: {}", exc)

            # Send the prompt
            logger.debug("ACP session.send (session={})", session_id)
            await conn.send_request(
                "session.send",
                {"sessionId": session_id, "prompt": prompt},
            )

            # Wait for session.idle or timeout
            try:
                await asyncio.wait_for(idle_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return LLMResponse(content="GitHub Copilot ACP timed out.", finish_reason="error")

        except Exception as exc:
            # Connection broken, reset
            await self._cleanup()
            return LLMResponse(content=f"Copilot ACP error: {exc}", finish_reason="error")
        finally:
            # Remove our handler
            handlers = conn._notification_handlers.get("session.event", [])
            if on_event in handlers:
                handlers.remove(on_event)

        # Ensure all delta callbacks have completed
        if delta_tasks:
            await asyncio.gather(*delta_tasks, return_exceptions=True)

        if error_msg:
            return LLMResponse(content=f"Copilot ACP error: {error_msg}", finish_reason="error")

        content = "".join(content_parts).strip() or None
        logger.debug("ACP response ({} chars)", len(content) if content else 0)
        return LLMResponse(content=content)
