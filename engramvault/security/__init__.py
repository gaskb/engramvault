"""Security & Authentication module for EngramVault (Issue #15)."""

from engramvault.security.auth import (
    SecurityManager,
    verify_agent_request,
    verify_web_session,
)

__all__ = [
    "SecurityManager",
    "verify_agent_request",
    "verify_web_session",
]
