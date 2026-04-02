"""Tests for unified authentication manager."""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.auth.manager import (
    AuthManager,
    TokenEntry,
    ApiKeyEntry,
    get_auth_manager,
    reset_auth_manager,
    DEFAULT_CREDENTIALS_PATH,
)


@pytest.fixture
def temp_credentials_path():
    """Provide a temporary credentials file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def isolated_auth_manager(temp_credentials_path):
    """Provide an isolated AuthManager instance for testing."""
    reset_auth_manager()
    manager = AuthManager(temp_credentials_path)
    yield manager
    reset_auth_manager()


class TestAuthManager:
    """Test suite for AuthManager."""

    def test_initial_state(self, isolated_auth_manager):
        """Test that a new AuthManager starts with empty state."""
        assert isolated_auth_manager.list_tokens() == {}
        assert isolated_auth_manager.list_api_keys() == []

    def test_set_and_get_token(self, isolated_auth_manager):
        """Test storing and retrieving OAuth tokens."""
        manager = isolated_auth_manager

        # Store a token
        manager.set_token("github_copilot", "github.com", "test_token_123")

        # Retrieve it
        entry = manager.get_token("github_copilot", "github.com")
        assert entry is not None
        assert entry.access_token == "test_token_123"
        assert entry.updated_at is not None

    def test_set_token_with_optional_fields(self, isolated_auth_manager):
        """Test storing tokens with refresh_token and expires_at."""
        manager = isolated_auth_manager

        manager.set_token(
            "openai_codex",
            "default",
            "access_123",
            refresh_token="refresh_456",
            expires_at="2024-12-31T23:59:59Z",
        )

        entry = manager.get_token("openai_codex", "default")
        assert entry.access_token == "access_123"
        assert entry.refresh_token == "refresh_456"
        assert entry.expires_at == "2024-12-31T23:59:59Z"

    def test_get_nonexistent_token(self, isolated_auth_manager):
        """Test retrieving a token that doesn't exist."""
        entry = isolated_auth_manager.get_token("unknown_provider", "default")
        assert entry is None

    def test_delete_token(self, isolated_auth_manager):
        """Test deleting a stored token."""
        manager = isolated_auth_manager

        manager.set_token("github_copilot", "github.com", "token123")
        assert manager.get_token("github_copilot", "github.com") is not None

        result = manager.delete_token("github_copilot", "github.com")
        assert result is True
        assert manager.get_token("github_copilot", "github.com") is None

    def test_delete_nonexistent_token(self, isolated_auth_manager):
        """Test deleting a token that doesn't exist."""
        result = isolated_auth_manager.delete_token("unknown", "key")
        assert result is False

    def test_api_key_management(self, isolated_auth_manager):
        """Test storing and retrieving API keys."""
        manager = isolated_auth_manager

        # Store API key with all fields
        manager.set_api_key(
            "openai",
            api_key="sk-test123",
            api_base="https://api.openai.com/v1",
            extra_headers={"X-Custom-Header": "value"},
        )

        entry = manager.get_api_key("openai")
        assert entry is not None
        assert entry.api_key == "sk-test123"
        assert entry.api_base == "https://api.openai.com/v1"
        assert entry.extra_headers == {"X-Custom-Header": "value"}

    def test_simple_api_key(self, isolated_auth_manager):
        """Test storing a simple API key without optional fields."""
        manager = isolated_auth_manager

        manager.set_api_key("anthropic", api_key="sk-ant-123")

        entry = manager.get_api_key("anthropic")
        assert entry.api_key == "sk-ant-123"
        assert entry.api_base is None
        assert entry.extra_headers == {}

    def test_list_api_keys(self, isolated_auth_manager):
        """Test listing all stored API key providers."""
        manager = isolated_auth_manager

        manager.set_api_key("openai", "key1")
        manager.set_api_key("anthropic", "key2")
        manager.set_api_key("deepseek", "key3")

        keys = manager.list_api_keys()
        assert sorted(keys) == ["anthropic", "deepseek", "openai"]

    def test_persistence(self, temp_credentials_path):
        """Test that credentials are persisted to disk."""
        # Create manager and store data
        manager1 = AuthManager(temp_credentials_path)
        manager1.set_token("github_copilot", "github.com", "persisted_token")
        manager1.set_api_key("openai", "sk-persisted")

        # Create new manager instance pointing to same file
        manager2 = AuthManager(temp_credentials_path)

        # Verify data is loaded
        token_entry = manager2.get_token("github_copilot", "github.com")
        assert token_entry.access_token == "persisted_token"

        api_entry = manager2.get_api_key("openai")
        assert api_entry.api_key == "sk-persisted"

    def test_migrate_from_legacy_github_copilot(self, isolated_auth_manager, temp_credentials_path):
        """Test migration from legacy github_copilot.json format."""
        # Create legacy format file
        legacy_path = temp_credentials_path.parent / "legacy_github_copilot.json"
        legacy_data = {
            "github.com": {"access_token": "legacy_token_123"},
            "enterprise.github.com": {"access_token": "enterprise_token_456"},
        }
        legacy_path.write_text(json.dumps(legacy_data))

        # Migrate
        manager = isolated_auth_manager
        result = manager.migrate_from_legacy(legacy_path, "github_copilot", "github.com")

        assert result is True
        token = manager.get_token("github_copilot", "github.com")
        assert token.access_token == "legacy_token_123"

        # Cleanup
        legacy_path.unlink()

    def test_migrate_legacy_single_token_format(self, isolated_auth_manager, temp_credentials_path):
        """Test migration from legacy format with just access_token at root."""
        legacy_path = temp_credentials_path.parent / "legacy_single.json"
        legacy_data = {"access_token": "single_token_789"}
        legacy_path.write_text(json.dumps(legacy_data))

        manager = isolated_auth_manager
        result = manager.migrate_from_legacy(legacy_path, "github_copilot", "default")

        assert result is True
        token = manager.get_token("github_copilot", "default")
        assert token.access_token == "single_token_789"

        legacy_path.unlink()

    def test_migrate_nonexistent_file(self, isolated_auth_manager):
        """Test migration from a file that doesn't exist."""
        result = isolated_auth_manager.migrate_from_legacy(
            Path("/nonexistent/path.json"), "github_copilot", "default"
        )
        assert result is False

    def test_file_permissions(self, isolated_auth_manager, temp_credentials_path):
        """Test that credentials file has restrictive permissions."""
        manager = isolated_auth_manager
        manager.set_api_key("test", "secret_key")

        import os
        stat = temp_credentials_path.stat()
        # Check owner has read/write (0o600)
        assert stat.st_mode & 0o777 == 0o600


class TestGlobalAuthManager:
    """Tests for global auth manager instance."""

    def test_get_auth_manager_singleton(self):
        """Test that get_auth_manager returns the same instance."""
        reset_auth_manager()

        manager1 = get_auth_manager()
        manager2 = get_auth_manager()

        assert manager1 is manager2

    def test_reset_auth_manager(self):
        """Test that reset_auth_manager clears the singleton."""
        reset_auth_manager()

        manager1 = get_auth_manager()
        reset_auth_manager()
        manager2 = get_auth_manager()

        assert manager1 is not manager2
