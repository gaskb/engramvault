"""
Security & Authentication Manager (Issues #15 & #22):
- Bearer Token (`MEMORIA_API_TOKEN` / DB setting) for `/api/v1/agent/*` and `/mcp`
- Optional Web UI Password (`MEMORIA_WEB_PASSWORD` / DB setting) with HttpOnly HMAC session cookie
  with random per-password PBKDF2 salt (S-02) and cryptographic timestamp expiration (S-06/S-07).
"""

import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import Request

from engramvault.config import API_TOKEN as ENV_API_TOKEN
from engramvault.config import WEB_PASSWORD as ENV_WEB_PASSWORD
from engramvault.db.engine import Database

COOKIE_NAME = "memoria_session"
SESSION_MAX_AGE_SECONDS = 86400 * 30  # 30 days


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique 16-byte random salt (S-02)."""
    active_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), active_salt.encode("utf-8"), 100_000
    ).hex()
    return f"pbkdf2_sha256${active_salt}${digest}"


def _verify_password_hash(candidate: str, stored_hash: str) -> bool:
    """Verifies candidate password against pbkdf2_sha256$salt$hash (or legacy hex hash)."""
    if not candidate or not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        parts = stored_hash.split("$", 2)
        if len(parts) != 3:
            return False
        _, salt, _ = parts
        expected = _hash_password(candidate, salt=salt)
        return hmac.compare_digest(expected, stored_hash)
    # Legacy fallback for hashes created before v2.1.0
    legacy = hashlib.pbkdf2_hmac(
        "sha256", candidate.encode("utf-8"), b"memoria_v1_salt", 100_000
    ).hex()
    return hmac.compare_digest(legacy, stored_hash)


class SecurityManager:
    """Manages API Bearer Tokens and optional Web UI password protection."""

    def __init__(self, db: Database):
        self.db = db
        self._signing_secret: Optional[str] = None
        self._ensure_signing_secret()

    def _ensure_signing_secret(self) -> str:
        if self._signing_secret:
            return self._signing_secret
        secret = self.db.get_setting("security_signing_secret")
        if not secret:
            secret = secrets.token_hex(32)
            self.db.set_setting("security_signing_secret", secret)
        self._signing_secret = secret
        return secret

    def get_active_api_token(self) -> str:
        """Returns the active API token (env var takes precedence over DB setting)."""
        if ENV_API_TOKEN:
            return ENV_API_TOKEN
        tok = self.db.get_setting("security_api_token")
        if not tok:
            tok = f"mem_{secrets.token_urlsafe(24)}"
            self.db.set_setting("security_api_token", tok)
        return tok

    def is_api_token_required(self) -> bool:
        """True if Bearer token enforcement is active for /api/v1/agent/* and /mcp."""
        if ENV_API_TOKEN:
            return True
        val = self.db.get_setting("security_api_token_required")
        return str(val).lower() in ("true", "1", "yes")

    def generate_new_api_token(self) -> str:
        tok = f"mem_{secrets.token_urlsafe(24)}"
        self.db.set_setting("security_api_token", tok)
        return tok

    def is_web_password_enabled(self) -> bool:
        if ENV_WEB_PASSWORD:
            return True
        pw_hash = self.db.get_setting("security_web_password_hash")
        return bool(pw_hash)

    def verify_web_password(self, candidate: str) -> bool:
        if not candidate:
            return False
        if ENV_WEB_PASSWORD:
            return hmac.compare_digest(candidate, ENV_WEB_PASSWORD)
        pw_hash = self.db.get_setting("security_web_password_hash")
        if not pw_hash:
            return True
        return _verify_password_hash(candidate, pw_hash)

    def create_session_cookie_value(self, issued_at: Optional[int] = None) -> str:
        secret = self._ensure_signing_secret()
        ts = int(issued_at if issued_at is not None else time.time())
        payload = f"authenticated:{ts}"
        sig = hmac.HMAC(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify_session_cookie(
        self, cookie_val: Optional[str], max_age: int = SESSION_MAX_AGE_SECONDS
    ) -> bool:
        if not self.is_web_password_enabled():
            return True
        if not cookie_val or "." not in cookie_val:
            return False
        payload, sig = cookie_val.rsplit(".", 1)
        secret = self._ensure_signing_secret()
        expected = hmac.HMAC(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        if ":" in payload:
            try:
                _, ts_str = payload.split(":", 1)
                issued_at = int(ts_str)
                now = int(time.time())
                if now - issued_at > max_age or issued_at > now + 300:
                    return False
            except ValueError:
                return False
        return True

    def _extract_bearer_or_header_token(self, request: Request) -> str:
        auth_header = request.headers.get("authorization", "") or request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        if request.headers.get("x-api-key"):
            return request.headers.get("x-api-key", "").strip()
        if os.getenv("MEMORIA_ALLOW_QUERY_API_KEY", "false").lower() in ("true", "1", "yes"):
            if request.query_params.get("api_key"):
                return request.query_params.get("api_key", "").strip()
        return ""

    def verify_bearer_token(self, request: Request) -> bool:
        if not self.is_api_token_required():
            return True
        expected = self.get_active_api_token()
        if not expected:
            return True
        token = self._extract_bearer_or_header_token(request)
        if not token:
            return False
        return hmac.compare_digest(token, expected)

    def is_request_authorized(self, request: Request) -> bool:
        """
        Checks authorization for administrative/mutating routes (S-04 & S-05):
        - If neither Web Password nor Bearer Token is enabled (Open LAN Mode), returns True.
        - If security is enabled, accepts either a valid Web Session Cookie OR a valid API Bearer Token.
        """
        web_pw_on = self.is_web_password_enabled()
        token_on = self.is_api_token_required()
        if not web_pw_on and not token_on:
            return True

        # Check explicit Bearer / X-API-Key header first
        token = self._extract_bearer_or_header_token(request)
        expected_tok = self.get_active_api_token()
        if token and expected_tok and hmac.compare_digest(token, expected_tok):
            return True

        # Check web session cookie if present
        cookie = request.cookies.get(COOKIE_NAME)
        if cookie and "." in cookie:
            payload, sig = cookie.rsplit(".", 1)
            secret = self._ensure_signing_secret()
            expected_sig = hmac.HMAC(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, expected_sig):
                if ":" in payload:
                    try:
                        _, ts_str = payload.split(":", 1)
                        if int(time.time()) - int(ts_str) <= SESSION_MAX_AGE_SECONDS:
                            return True
                    except ValueError:
                        pass
                else:
                    return True

        # If only Bearer token is required (no web password enabled) and request comes from the Web UI same-origin
        # with an active session or referer, allow web UI interaction only if web_pw_on is False and path is not /api/security/*
        return False

    def update_config(
        self,
        api_token_required: Optional[bool] = None,
        custom_api_token: Optional[str] = None,
        web_password: Optional[str] = None,
        clear_web_password: bool = False,
    ) -> Dict[str, Any]:
        if api_token_required is not None:
            self.db.set_setting("security_api_token_required", "true" if api_token_required else "false")
        if custom_api_token and custom_api_token.strip():
            self.db.set_setting("security_api_token", custom_api_token.strip())
        if clear_web_password:
            self.db.set_setting("security_web_password_hash", "")
        elif web_password and web_password.strip():
            self.db.set_setting("security_web_password_hash", _hash_password(web_password.strip()))
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        tok = self.get_active_api_token()
        masked = f"{tok[:8]}••••{tok[-4:]}" if len(tok) > 12 else "••••"
        return {
            "api_token": tok,
            "api_token_masked": masked,
            "api_token_required": self.is_api_token_required(),
            "env_api_token_set": bool(ENV_API_TOKEN),
            "web_password_enabled": self.is_web_password_enabled(),
            "env_web_password_set": bool(ENV_WEB_PASSWORD),
        }


def _get_mgr(request: Request, db: Database) -> SecurityManager:
    mgr = getattr(getattr(request, "app", None), "state", None)
    if mgr and hasattr(mgr, "security_mgr") and mgr.security_mgr is not None:
        return mgr.security_mgr
    return SecurityManager(db)


def verify_agent_request(request: Request, db: Database) -> bool:
    return _get_mgr(request, db).verify_bearer_token(request)


def verify_web_session(request: Request, db: Database) -> bool:
    mgr = _get_mgr(request, db)
    if not mgr.is_web_password_enabled():
        return True
    cookie = request.cookies.get(COOKIE_NAME)
    return mgr.verify_session_cookie(cookie)
