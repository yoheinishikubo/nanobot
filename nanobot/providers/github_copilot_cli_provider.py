"""GitHub Copilot CLI provider that shells out to the Copilot CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from nanobot.auth.github_copilot import get_stored_token
from nanobot.providers.base import LLMProvider, LLMResponse


class GitHubCopilotCLIProvider(LLMProvider):
    """Provider implementation backed by the `copilot` CLI."""

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
        # When True, skip --continue on the next call to avoid resuming a broken session.
        # Reset to False after a successful call.
        self._session_broken: bool = False

    def get_default_model(self) -> str:
        return self.default_model

    # Errors that indicate the --continue session is broken and should be retried fresh.
    _BROKEN_SESSION_MARKERS = ("timed out", "no tool output found", "capiError: 400", "400 no tool")

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
        prompt = self._build_prompt(messages)
        cli_model = self._resolve_cli_model(model)
        # Skip --continue if the previous session was broken to avoid hanging on ghost state.
        use_continue = self.copilot_continue and not self._session_broken
        content, error = await self._run_cli(
            prompt, cli_model, reasoning_effort=reasoning_effort, use_continue=use_continue
        )
        if error and self.copilot_continue and not self._session_broken and self._is_broken_session(error):
            # Session was broken (timeout / CAPIError). Mark it and retry fresh.
            self._session_broken = True
            content, error = await self._run_cli(
                prompt, cli_model, reasoning_effort=reasoning_effort, use_continue=False
            )
        if error:
            self._session_broken = self._is_broken_session(error)
            return LLMResponse(content=error, finish_reason="error")
        self._session_broken = False
        return LLMResponse(content=content)

    @classmethod
    def _is_broken_session(cls, error: str) -> bool:
        err = error.lower()
        return any(marker.lower() in err for marker in cls._BROKEN_SESSION_MARKERS)


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
        if not on_content_delta:
            return await self.chat(
                messages=messages, tools=tools, model=model,
                max_tokens=max_tokens, temperature=temperature,
                reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            )

        prompt = self._build_prompt(messages)
        cli_model = self._resolve_cli_model(model)
        use_continue = self.copilot_continue and not self._session_broken

        content, error = await self._run_cli_stream(
            prompt, cli_model, reasoning_effort=reasoning_effort,
            use_continue=use_continue, on_content_delta=on_content_delta,
        )
        if error and self.copilot_continue and not self._session_broken and self._is_broken_session(error):
            self._session_broken = True
            content, error = await self._run_cli_stream(
                prompt, cli_model, reasoning_effort=reasoning_effort,
                use_continue=False, on_content_delta=on_content_delta,
            )
        if error:
            self._session_broken = self._is_broken_session(error)
            return LLMResponse(content=error, finish_reason="error")
        self._session_broken = False
        return LLMResponse(content=content)

    def _resolve_cli_model(self, model: str | None) -> str:
        return self.copilot_model

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = [
            "You are nanobot, a helpful assistant.",
            "Return only the next assistant response as plain text.",
        ]
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

    async def _run_cli_stream(
        self,
        prompt: str,
        cli_model: str,
        reasoning_effort: str | None = None,
        use_continue: bool | None = None,
        on_content_delta=None,
    ) -> tuple[str | None, str | None]:
        """Run CLI and stream stdout chunks via on_content_delta."""
        command = shutil.which(self.cli_command)
        if not command:
            return None, (
                "GitHub Copilot CLI is not installed. Install `@github/copilot` "
                f"and ensure `{self.cli_command}` is on PATH."
            )

        token = get_stored_token()
        if not token:
            return None, (
                "GitHub Copilot is not authenticated. Run `nanobot provider login github-copilot` first."
            )

        args = self._build_cli_args(command, cli_model, reasoning_effort, use_continue)

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.working_dir) if self.working_dir else None,
                env={**os.environ, "COPILOT_GITHUB_TOKEN": token},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Send prompt and close stdin
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # Stream stdout chunks as they arrive
            chunks: list[str] = []
            deadline = asyncio.get_event_loop().time() + 180
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    return None, "GitHub Copilot CLI timed out."
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(4096), timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return None, "GitHub Copilot CLI timed out."
                if not chunk:
                    break
                text = chunk.decode("utf-8", "ignore")
                chunks.append(text)
                if on_content_delta:
                    await on_content_delta(text)

            await proc.wait()
        except asyncio.CancelledError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            raise
        except Exception as exc:
            return None, f"Error calling GitHub Copilot CLI: {exc}"

        stderr_text = ""
        if proc.stderr:
            raw = await proc.stderr.read()
            stderr_text = raw.decode("utf-8", "ignore").strip()

        if proc.returncode != 0:
            detail = stderr_text or "".join(chunks).strip() or f"exit code {proc.returncode}"
            return None, f"GitHub Copilot CLI failed: {detail}"

        full_content = "".join(chunks).strip()
        return full_content or None, None

    def _build_cli_args(
        self, command: str, cli_model: str,
        reasoning_effort: str | None = None,
        use_continue: bool | None = None,
    ) -> list[str]:
        """Build CLI argument list (shared by _run_cli and _run_cli_stream)."""
        args = [command, "--model", cli_model, "--no-color", "--output-format", "text"]
        if self.copilot_allow_all:
            args.append("--allow-all")
        if use_continue is None:
            use_continue = self.copilot_continue
        if use_continue:
            args.append("--continue")
        if self.copilot_autopilot:
            args.append("--autopilot")
        if self.copilot_no_ask_user:
            args.append("--no-ask-user")
        if self.copilot_max_autopilot_continues is not None:
            args += ["--max-autopilot-continues", str(self.copilot_max_autopilot_continues)]
        if self.copilot_available_tools:
            args += ["--available-tools", ",".join(self.copilot_available_tools)]
        if self.copilot_excluded_tools:
            args += ["--excluded-tools", ",".join(self.copilot_excluded_tools)]
        if self.copilot_no_custom_instructions:
            args.append("--no-custom-instructions")
        if self.copilot_experimental:
            args.append("--experimental")
        if reasoning_effort:
            args += ["--reasoning-effort", reasoning_effort]
        return args

    async def _run_cli(self, prompt: str, cli_model: str, reasoning_effort: str | None = None, use_continue: bool | None = None) -> tuple[str | None, str | None]:
        command = shutil.which(self.cli_command)
        if not command:
            return None, (
                "GitHub Copilot CLI is not installed. Install `@github/copilot` "
                f"and ensure `{self.cli_command}` is on PATH."
            )

        token = get_stored_token()
        if not token:
            return None, (
                "GitHub Copilot is not authenticated. Run `nanobot provider login github-copilot` first."
            )

        args = self._build_cli_args(command, cli_model, reasoning_effort, use_continue)

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.working_dir) if self.working_dir else None,
                env={
                    **os.environ,
                    "COPILOT_GITHUB_TOKEN": token,
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=180,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return None, "GitHub Copilot CLI timed out."
        except asyncio.CancelledError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            raise
        except Exception as exc:
            return None, f"Error calling GitHub Copilot CLI: {exc}"

        stdout_text = stdout.decode("utf-8", "ignore").strip()
        stderr_text = stderr.decode("utf-8", "ignore").strip()

        if proc.returncode != 0:
            detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
            return None, f"GitHub Copilot CLI failed: {detail}"

        return stdout_text or None, None
