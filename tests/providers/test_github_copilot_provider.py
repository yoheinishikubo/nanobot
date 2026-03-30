"""Tests for the GitHub Copilot CLI provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.providers.github_copilot_provider import GitHubCopilotProvider


@pytest.mark.asyncio
async def test_github_copilot_provider_invokes_cli_with_model_and_token(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            seen["input"] = input
            return b"hello from copilot", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotProvider(working_dir=tmp_path)
    response = await provider.chat([{"role": "user", "content": "Say hi"}])

    assert response.content == "hello from copilot"
    assert seen["args"][0] == "/usr/bin/copilot"
    assert seen["args"][1:3] == ("--model", "gpt-5-mini")
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert seen["kwargs"]["env"]["COPILOT_GITHUB_TOKEN"] == "gho-token"
    assert seen["kwargs"]["stdin"] is not None
    assert seen["input"].startswith(b"You are nanobot")


@pytest.mark.asyncio
async def test_github_copilot_provider_ignores_agent_model_for_cli_model(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            return b"ok", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotProvider(copilot_model="gpt-5-mini", working_dir=tmp_path)
    await provider.chat([{"role": "user", "content": "Say hi"}], model="gpt-4o-mini")

    assert seen["args"][1:3] == ("--model", "gpt-5-mini")


@pytest.mark.asyncio
async def test_github_copilot_provider_adds_force_flag_when_enabled(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            return b"ok", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotProvider(copilot_model="gpt-5-mini", copilot_force=True, working_dir=tmp_path)
    await provider.chat([{"role": "user", "content": "Say hi"}])

    assert "--yolo" in seen["args"]
