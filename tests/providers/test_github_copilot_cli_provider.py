"""Tests for the GitHub Copilot CLI provider (GitHubCopilotCLIProvider)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.providers.github_copilot_cli_provider import GitHubCopilotCLIProvider


@pytest.mark.asyncio
async def test_github_copilot_cli_provider_invokes_cli_with_model_and_token(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            seen["input"] = input
            return b"hello from copilot", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    response = await provider.chat([{"role": "user", "content": "Say hi"}])

    assert response.content == "hello from copilot"
    assert seen["args"][0] == "/usr/bin/copilot"
    assert seen["args"][1:3] == ("--model", "gpt-5-mini")
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert seen["kwargs"]["env"]["COPILOT_GITHUB_TOKEN"] == "gho-token"
    assert seen["kwargs"]["stdin"] is not None
    assert seen["input"].startswith(b"You are nanobot")


@pytest.mark.asyncio
async def test_github_copilot_cli_provider_ignores_agent_model_for_cli_model(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            return b"ok", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotCLIProvider(copilot_model="gpt-5-mini", working_dir=tmp_path)
    await provider.chat([{"role": "user", "content": "Say hi"}], model="gpt-4o-mini")

    assert seen["args"][1:3] == ("--model", "gpt-5-mini")


@pytest.mark.asyncio
async def test_github_copilot_cli_provider_adds_force_flag_when_enabled(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.get_stored_token", lambda: "gho-token")

    class _Proc:
        returncode = 0

        async def communicate(self, input=None):
            return b"ok", b""

    async def _fake_create_subprocess_exec(*args, **kwargs):
        seen["args"] = args
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotCLIProvider(copilot_model="gpt-5-mini", copilot_force=True, working_dir=tmp_path)
    await provider.chat([{"role": "user", "content": "Say hi"}])

    assert "--yolo" in seen["args"]


async def _async_append(lst, item):
    lst.append(item)


@pytest.mark.asyncio
async def test_github_copilot_cli_provider_streams_stdout_chunks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.shutil.which", lambda _cmd: "/usr/bin/copilot")
    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.get_stored_token", lambda: "gho-token")

    class _FakeStdin:
        def write(self, data):
            pass

        async def drain(self):
            pass

        def close(self):
            pass

    class _FakeStdout:
        def __init__(self):
            self._chunks = [b"hello ", b"from ", b"copilot"]
            self._index = 0

        async def read(self, n):
            if self._index >= len(self._chunks):
                return b""
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

    class _FakeStderr:
        async def read(self):
            return b""

    class _Proc:
        returncode = 0
        stdin = _FakeStdin()
        stdout = _FakeStdout()
        stderr = _FakeStderr()

        async def wait(self):
            pass

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr("nanobot.providers.github_copilot_cli_provider.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)

    provider = GitHubCopilotCLIProvider(working_dir=tmp_path)
    deltas: list[str] = []
    response = await provider.chat_stream(
        [{"role": "user", "content": "Say hi"}],
        on_content_delta=lambda chunk: _async_append(deltas, chunk),
    )

    assert deltas == ["hello ", "from ", "copilot"]
    assert response.content == "hello from copilot"
