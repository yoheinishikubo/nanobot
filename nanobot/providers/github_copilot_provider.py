"""GitHub Copilot provider that shells out to the Copilot CLI."""

from __future__ import annotations

import asyncio
import json
import shutil
import os
from pathlib import Path
from typing import Any

from nanobot.auth.github_copilot import get_stored_token
from nanobot.providers.base import LLMProvider, LLMResponse


class GitHubCopilotProvider(LLMProvider):
    """Provider implementation backed by the `copilot` CLI."""

    def __init__(
        self,
        default_model: str = "github-copilot/gpt-5-mini",
        copilot_model: str = "gpt-5-mini",
        copilot_force: bool = False,
        working_dir: Path | None = None,
        cli_command: str = "copilot",
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.copilot_model = copilot_model
        self.copilot_force = copilot_force
        self.working_dir = working_dir
        self.cli_command = cli_command

    def get_default_model(self) -> str:
        return self.default_model

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
        content, error = await self._run_cli(prompt, cli_model)
        if error:
            return LLMResponse(content=error, finish_reason="error")
        return LLMResponse(content=content)

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
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

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

    async def _run_cli(self, prompt: str, cli_model: str) -> tuple[str | None, str | None]:
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

        args = [
            command,
            "-i",
            prompt,
            "--model",
            cli_model,
            "--no-color",
            "--output-format",
            "text",
        ]
        if self.copilot_force:
            args.append("--yolo")

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.working_dir) if self.working_dir else None,
                env={
                    **os.environ,
                    "COPILOT_GITHUB_TOKEN": token,
                },
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            return None, "GitHub Copilot CLI timed out."
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return None, f"Error calling GitHub Copilot CLI: {exc}"

        stdout_text = stdout.decode("utf-8", "ignore").strip()
        stderr_text = stderr.decode("utf-8", "ignore").strip()

        if proc.returncode != 0:
            detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
            return None, f"GitHub Copilot CLI failed: {detail}"

        return stdout_text or None, None
