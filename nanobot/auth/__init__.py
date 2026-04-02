"""Authentication helpers for OAuth-backed providers and API key management."""

from nanobot.auth.github_copilot import (
    GitHubCopilotAuthError,
    DeviceCodeResponse,
    get_stored_token as get_copilot_token,
    get_runtime_token,
    store_token as store_copilot_token,
    validate_token as validate_copilot_token,
    login_device_flow,
)
from nanobot.auth.manager import (
    AuthManager,
    TokenEntry,
    ApiKeyEntry,
    get_auth_manager,
    reset_auth_manager,
)

__all__ = [
    # GitHub Copilot
    "GitHubCopilotAuthError",
    "DeviceCodeResponse",
    "get_copilot_token",
    "get_runtime_token",
    "store_copilot_token",
    "validate_copilot_token",
    "login_device_flow",
    # Unified auth manager
    "AuthManager",
    "TokenEntry",
    "ApiKeyEntry",
    "get_auth_manager",
    "reset_auth_manager",
]
