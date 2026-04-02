"""GitHub Copilot OAuth/device-flow helpers.

Uses the unified AuthManager for credential storage.
Legacy github_copilot.json is automatically migrated.
"""

from __future__ import annotations

import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from nanobot.auth.manager import get_auth_manager

COPILOT_CLIENT_ID = "Ov23ctDVkRmgkPke0Mmm"
COPILOT_SCOPES = "read:user,read:org,repo,gist"
DEFAULT_HOST = "github.com"
LEGACY_AUTH_STATE_PATH = Path.home() / ".nanobot" / "auth" / "github_copilot.json"
PROVIDER_NAME = "github_copilot"


class GitHubCopilotAuthError(RuntimeError):
    """Raised when GitHub Copilot OAuth/device flow fails."""


@dataclass(frozen=True)
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


def _normalize_host(host: str | None = None) -> str:
    value = (host or DEFAULT_HOST).strip()
    if not value:
        value = DEFAULT_HOST
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        return parsed.netloc or parsed.path
    return value


def _base_url(host: str | None = None) -> str:
    host_value = host or DEFAULT_HOST
    if host_value.startswith("http://") or host_value.startswith("https://"):
        return host_value.rstrip("/")
    return f"https://{host_value.strip('/')}"


def _api_base_url(host: str | None = None) -> str:
    """Return the GitHub API base URL for Copilot user endpoints."""

    host_value = _normalize_host(host)
    if host_value == "github.com":
        return "https://api.github.com"
    if host_value.startswith("api."):
        return f"https://{host_value}"
    return f"https://api.{host_value}"


def _maybe_migrate_legacy_token(host: str | None = None) -> str | None:
    """Migrate token from legacy github_copilot.json if present."""
    host_key = _normalize_host(host)
    auth_manager = get_auth_manager()

    # Check if already migrated
    existing = auth_manager.get_token(PROVIDER_NAME, host_key)
    if existing:
        return existing.access_token

    # Try to migrate from legacy file
    if LEGACY_AUTH_STATE_PATH.exists():
        if auth_manager.migrate_from_legacy(
            LEGACY_AUTH_STATE_PATH, PROVIDER_NAME, host_key
        ):
            # Return the migrated token
            migrated = auth_manager.get_token(PROVIDER_NAME, host_key)
            if migrated:
                return migrated.access_token

    return None


def get_stored_token(host: str | None = None) -> str | None:
    """Get stored Copilot token, migrating from legacy if needed."""
    host_key = _normalize_host(host)
    auth_manager = get_auth_manager()

    # Try unified storage first
    entry = auth_manager.get_token(PROVIDER_NAME, host_key)
    if entry:
        return entry.access_token

    # Try legacy migration
    return _maybe_migrate_legacy_token(host)


def get_runtime_token(host: str | None = None) -> str | None:
    """Resolve a Copilot token for API requests from stored login state."""
    return get_stored_token(host)


def _request_device_code(base_url: str) -> DeviceCodeResponse:
    response = httpx.post(
        f"{base_url}/login/device/code",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "client_id": COPILOT_CLIENT_ID,
            "scope": COPILOT_SCOPES,
        },
        timeout=10.0,
    )
    if not response.is_success:
        raise GitHubCopilotAuthError(
            f"Failed to request GitHub device code: {response.status_code} {response.reason_phrase}"
        )

    payload = response.json()
    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    verification_uri = payload.get("verification_uri")
    expires_in = payload.get("expires_in")
    interval = payload.get("interval") or 5

    if not all(isinstance(v, str) and v for v in (device_code, user_code, verification_uri)):
        raise GitHubCopilotAuthError("GitHub returned an invalid device code response")
    if not isinstance(expires_in, int) or expires_in <= 0:
        raise GitHubCopilotAuthError("GitHub returned an invalid device code expiration")
    if not isinstance(interval, int) or interval <= 0:
        interval = 5

    return DeviceCodeResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        expires_in=expires_in,
        interval=interval,
    )


def _exchange_device_code(base_url: str, device_code: str, interval: int, expires_in: int) -> str:
    deadline = time.monotonic() + expires_in
    poll_interval = max(interval, 1)

    while True:
        if time.monotonic() >= deadline:
            raise GitHubCopilotAuthError("GitHub device code expired before authorization completed")

        response = httpx.post(
            f"{base_url}/login/oauth/access_token",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "client_id": COPILOT_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=10.0,
        )
        if not response.is_success:
            raise GitHubCopilotAuthError(
                f"GitHub rejected the device token exchange: {response.status_code} {response.reason_phrase}"
            )

        payload = response.json()
        token = payload.get("access_token")
        if isinstance(token, str) and token:
            return token

        error = payload.get("error")
        error_description = payload.get("error_description")
        if error == "authorization_pending":
            time.sleep(poll_interval)
            continue
        if error == "slow_down":
            poll_interval += 5
            time.sleep(poll_interval)
            continue
        if error in {"expired_token", "access_denied", "incorrect_device_code"}:
            raise GitHubCopilotAuthError(
                error_description or f"GitHub device flow failed: {error}"
            )
        raise GitHubCopilotAuthError(
            error_description or f"Unexpected GitHub device flow response: {payload}"
        )


def validate_token(token: str, host: str | None = None) -> dict[str, object]:
    """Validate a Copilot token against GitHub's Copilot user endpoint."""

    base_url = _api_base_url(host)
    response = httpx.get(
        f"{base_url}/copilot_internal/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        timeout=10.0,
    )
    if not response.is_success:
        raise GitHubCopilotAuthError(
            f"GitHub rejected the Copilot token: {response.status_code} {response.reason_phrase}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise GitHubCopilotAuthError("GitHub returned an invalid Copilot user payload")
    return payload


def store_token(token: str, host: str | None = None) -> None:
    """Store Copilot token using unified AuthManager."""
    host_key = _normalize_host(host)
    auth_manager = get_auth_manager()
    auth_manager.set_token(PROVIDER_NAME, host_key, token)


def login_device_flow(
    host: str | None = None,
    *,
    print_fn=print,
    open_browser: bool = True,
) -> str:
    """Run the GitHub device flow, persist the token, and return it."""

    base_url = _base_url(host)
    device = _request_device_code(base_url)

    print_fn("[cyan]Open this URL in your browser:[/cyan]")
    print_fn(device.verification_uri)
    print_fn(f"[cyan]Then enter this code:[/cyan] [bold]{device.user_code}[/bold]")

    if open_browser:
        try:
            webbrowser.open(device.verification_uri, new=2)
        except Exception:
            pass

    token = _exchange_device_code(base_url, device.device_code, device.interval, device.expires_in)
    validate_token(token, host=host)
    store_token(token, host=host)
    return token
