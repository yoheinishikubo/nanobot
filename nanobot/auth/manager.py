"""Unified authentication manager for all credentials.

Credentials are stored in ~/.nanobot/auth/credentials.json with the following structure:
{
    "version": 1,
    "tokens": {
        "github_copilot": {
            "github.com": {"access_token": "...", "updated_at": "..."},
            "enterprise.github.com": {"access_token": "...", "updated_at": "..."}
        },
        "openai_codex": {
            "default": {"access_token": "...", "refresh_token": "...", "updated_at": "..."}
        }
    },
    "api_keys": {
        "openai": {"api_key": "..."},
        "anthropic": {"api_key": "..."},
        "azure_openai": {"api_key": "...", "api_base": "..."},
        ...
    }
}
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CREDENTIALS_PATH = Path.home() / ".nanobot" / "auth" / "credentials.json"
CURRENT_VERSION = 1


@dataclass
class TokenEntry:
    """OAuth token entry."""

    access_token: str
    updated_at: str | None = None
    refresh_token: str | None = None
    expires_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "access_token": self.access_token,
            "updated_at": self.updated_at,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenEntry:
        return cls(
            access_token=data.get("access_token", ""),
            updated_at=data.get("updated_at"),
            refresh_token=data.get("refresh_token"),
            expires_at=data.get("expires_at"),
        )


@dataclass
class ApiKeyEntry:
    """API key entry for providers."""

    api_key: str
    api_base: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"api_key": self.api_key}
        if self.api_base:
            result["api_base"] = self.api_base
        if self.extra_headers:
            result["extra_headers"] = self.extra_headers
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApiKeyEntry:
        return cls(
            api_key=data.get("api_key", ""),
            api_base=data.get("api_base"),
            extra_headers=data.get("extra_headers", {}),
        )


class AuthManager:
    """Unified manager for all authentication credentials."""

    def __init__(self, credentials_path: Path | None = None):
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load credentials from disk."""
        if not self.credentials_path.exists():
            self._data = {"version": CURRENT_VERSION, "tokens": {}, "api_keys": {}}
            return

        try:
            content = self.credentials_path.read_text(encoding="utf-8")
            self._data = json.loads(content)
            if not isinstance(self._data, dict):
                self._data = {"version": CURRENT_VERSION, "tokens": {}, "api_keys": {}}
        except (json.JSONDecodeError, IOError):
            self._data = {"version": CURRENT_VERSION, "tokens": {}, "api_keys": {}}

        # Ensure required sections exist
        self._data.setdefault("tokens", {})
        self._data.setdefault("api_keys", {})

    def _save(self) -> None:
        """Save credentials to disk with secure permissions."""
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        tmp_path = self.credentials_path.with_suffix(f".{secrets.token_hex(8)}.tmp")
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")

        # Set restrictive permissions (owner read/write only)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass

        # Atomic rename
        os.replace(tmp_path, self.credentials_path)
        try:
            os.chmod(self.credentials_path, 0o600)
        except OSError:
            pass

    # Token management (OAuth)
    def get_token(
        self, provider: str, key: str = "default"
    ) -> TokenEntry | None:
        """Get OAuth token for a provider.

        Args:
            provider: Provider name (e.g., "github_copilot", "openai_codex")
            key: Token key (e.g., host for GitHub Copilot)
        """
        tokens = self._data.get("tokens", {}).get(provider, {})
        entry = tokens.get(key)
        if not entry:
            return None
        return TokenEntry.from_dict(entry)

    def set_token(
        self,
        provider: str,
        key: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: str | None = None,
    ) -> None:
        """Set OAuth token for a provider."""
        if "tokens" not in self._data:
            self._data["tokens"] = {}
        if provider not in self._data["tokens"]:
            self._data["tokens"][provider] = {}

        entry = TokenEntry(
            access_token=access_token,
            updated_at=datetime.now(UTC).isoformat(),
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
        self._data["tokens"][provider][key] = entry.to_dict()
        self._save()

    def delete_token(self, provider: str, key: str = "default") -> bool:
        """Delete OAuth token for a provider."""
        tokens = self._data.get("tokens", {}).get(provider, {})
        if key in tokens:
            del tokens[key]
            self._save()
            return True
        return False

    def list_tokens(self, provider: str | None = None) -> dict[str, Any]:
        """List all stored tokens."""
        tokens = self._data.get("tokens", {})
        if provider:
            return tokens.get(provider, {})
        return tokens

    # API key management
    def get_api_key(self, provider: str) -> ApiKeyEntry | None:
        """Get API key entry for a provider."""
        entry = self._data.get("api_keys", {}).get(provider)
        if not entry:
            return None
        return ApiKeyEntry.from_dict(entry)

    def set_api_key(
        self,
        provider: str,
        api_key: str,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """Set API key for a provider."""
        if "api_keys" not in self._data:
            self._data["api_keys"] = {}

        entry = ApiKeyEntry(
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers or {},
        )
        self._data["api_keys"][provider] = entry.to_dict()
        self._save()

    def delete_api_key(self, provider: str) -> bool:
        """Delete API key for a provider."""
        api_keys = self._data.get("api_keys", {})
        if provider in api_keys:
            del api_keys[provider]
            self._save()
            return True
        return False

    def list_api_keys(self) -> list[str]:
        """List all providers with stored API keys."""
        return list(self._data.get("api_keys", {}).keys())

    # Migration helpers
    def migrate_from_legacy(
        self,
        legacy_path: Path,
        provider: str,
        key: str = "default",
    ) -> bool:
        """Migrate token from legacy file format.

        Args:
            legacy_path: Path to legacy auth file
            provider: Target provider name
            key: Token key (e.g., host for GitHub Copilot)
        """
        if not legacy_path.exists():
            return False

        try:
            content = legacy_path.read_text(encoding="utf-8")
            data = json.loads(content)

            # Handle github_copilot.json format {"github.com": {"access_token": "..."}}
            if isinstance(data, dict):
                # Try to extract token from various formats
                token = None

                # Format 1: {"github.com": {"access_token": "..."}}
                if key in data and isinstance(data[key], dict):
                    token = data[key].get("access_token")
                # Format 2: {"access_token": "..."}
                elif "access_token" in data:
                    token = data["access_token"]

                if token:
                    self.set_token(
                        provider=provider,
                        key=key,
                        access_token=token,
                    )
                    return True
        except (json.JSONDecodeError, IOError):
            pass

        return False


# Global instance for convenience
_default_manager: AuthManager | None = None


def get_auth_manager(credentials_path: Path | None = None) -> AuthManager:
    """Get or create the default AuthManager instance."""
    global _default_manager
    if _default_manager is None or credentials_path is not None:
        _default_manager = AuthManager(credentials_path)
    return _default_manager


def reset_auth_manager() -> None:
    """Reset the global AuthManager instance (useful for testing)."""
    global _default_manager
    _default_manager = None
