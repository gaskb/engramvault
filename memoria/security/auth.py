"""
Security & Authentication Manager (Issue #15):
- Bearer Token (`MEMORIA_API_TOKEN` / DB setting) for `/api/v1/agent/*` and `/mcp`
- Optional Web UI Password (`MEMORIA_WEB_PASSWORD` / DB setting) with HttpOnly HMAC session cookie
"""

import hashlib
import hmac
import secrets
from typing import Any, Dict, Optional

from fastapi import Request

from memoria.config import API_TOKEN as ENV_API_TOKEN
from memoria.config import WEB_PASSWORD as ENV_WEB_PASSWORD
from memoria.db.engine import Database

COOKIE_NAME = "memoria_session"


def _hash_password(password: str, salt: str = "memoria_v1_salt") -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


class SecurityManager:
    """Manages API Bearer Tokens and optional Web UI password protection."""

    def __init__(self, db: Database):
        self.db = db
        self._ensure_signing_secret()

    def _ensure_signing_secret(self) -> str:
        secret = self.db.get_setting("security_signing_secret")
        if not secret:
            secret = secrets.token_hex(32)
            self.db.set_setting("security_signing_secret", secret)
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
        return hmac.compare_digest(_hash_password(candidate), pw_hash)

    def create_session_cookie_value(self) -> str:
        secret = self._ensure_signing_secret()
        payload = "authenticated_memoria_user"
        sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify_session_cookie(self, cookie_val: Optional[str]) -> bool:
        if not self.is_web_password_enabled():
            return True
        if not cookie_val or "." not in cookie_val:
            return False
        payload, sig = cookie_val.rsplit(".", 1)
        secret = self._ensure_signing_secret()
        expected = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)

    def verify_bearer_token(self, request: Request) -> bool:
        if not self.is_api_token_required():
            return True
        expected = self.get_active_api_token()
        if not expected:
            return True

        auth_header = request.headers.get("authorization", "") or request.headers.get("Authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        elif request.headers.get("x-api-key"):
            token = request.headers.get("x-api-key", "").strip()
        elif request.query_params.get("api_key"):
            token = request.query_params.get("api_key", "").strip()

        if not token:
            return False
        return hmac.compare_digest(token, expected)

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


def verify_agent_request(request: Request, db: Database) -> bool:
    return SecurityManager(db).verify_bearer_token(request)


def verify_web_session(request: Request, db: Database) -> bool:
    mgr = SecurityManager(db)
    if not mgr.is_web_password_enabled():
        return True
    cookie = request.cookies.get(COOKIE_NAME)
    return mgr.verify_session_cookie(cookie)
