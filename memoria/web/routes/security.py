"""
Security, Authentication, Theme Studio & Onboarding Routes (Issue #26 / Q-01, S-04, S-05, A-02).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from memoria.config import API_VERSION, APP_VERSION, MCP_PROTOCOL_VERSION
from memoria.security.auth import COOKIE_NAME, SecurityManager
from memoria.themes import ThemeManager


class AuthLoginRequest(BaseModel):
    password: str = Field(default="", max_length=256)


class UpdateSecurityConfigRequest(BaseModel):
    api_token_required: Optional[bool] = None
    custom_api_token: Optional[str] = Field(default=None, max_length=256)
    web_password: Optional[str] = Field(default=None, max_length=256)
    clear_web_password: bool = False


class SetActiveThemeRequest(BaseModel):
    palette: Optional[str] = Field(default=None, max_length=64)
    mode: Optional[str] = Field(default=None, max_length=16)


class CustomThemeRequest(BaseModel):
    id: Optional[str] = Field(default=None, max_length=64)
    name: str = Field(default="Custom Theme", max_length=128)
    primary: str = Field(default="#10b981", max_length=32)
    primary_hover: Optional[str] = Field(default=None, max_length=32)
    secondary: str = Field(default="#6366f1", max_length=32)
    dark_bg: str = Field(default="#0c0f17", max_length=32)
    dark_card: str = Field(default="#161b26", max_length=32)
    dark_inner: str = Field(default="#10141d", max_length=32)
    dark_border: str = Field(default="#232d3f", max_length=32)
    light_bg: str = Field(default="#f4f7fb", max_length=32)
    light_card: str = Field(default="#ffffff", max_length=32)
    light_inner: str = Field(default="#f8fafc", max_length=32)
    light_border: str = Field(default="#dbe3ee", max_length=32)


def get_security_mgr(request: Request) -> SecurityManager:
    """Returns the singleton SecurityManager attached to app.state (A-02)."""
    return request.app.state.security_mgr


def require_mutation_auth(request: Request) -> None:
    """
    Explicit dependency protecting /api/security/* and mutating CRUD routes (S-04 & S-05).
    Enforced whenever Web Password OR API Bearer Token is enabled.
    """
    mgr: SecurityManager = request.app.state.security_mgr
    if not mgr.is_request_authorized(request):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: valid Web Session or Authorization: Bearer <MEMORIA_API_TOKEN> required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def build_security_router(
    security_mgr: SecurityManager,
    theme_mgr: ThemeManager,
    templates: Jinja2Templates,
) -> APIRouter:
    router = APIRouter()

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "app_version": APP_VERSION,
                "api_version": API_VERSION,
                "mcp_version": MCP_PROTOCOL_VERSION,
            },
        )

    @router.post("/api/auth/login")
    async def auth_login(payload: AuthLoginRequest):
        password = str(payload.password or "")
        if not security_mgr.verify_web_password(password):
            raise HTTPException(status_code=401, detail="Invalid password")
        cookie_val = security_mgr.create_session_cookie_value()
        resp = JSONResponse({"status": "success"})
        resp.set_cookie(
            key=COOKIE_NAME,
            value=cookie_val,
            httponly=True,
            samesite="lax",
            max_age=86400 * 30,
        )
        return resp

    @router.post("/api/auth/logout")
    async def auth_logout():
        resp = JSONResponse({"status": "success"})
        resp.delete_cookie(key=COOKIE_NAME)
        return resp

    @router.get("/api/security/status", dependencies=[Depends(require_mutation_auth)])
    async def get_security_status():
        return security_mgr.get_status()

    @router.post("/api/security/config", dependencies=[Depends(require_mutation_auth)])
    async def update_security_config(payload: UpdateSecurityConfigRequest):
        return security_mgr.update_config(
            api_token_required=payload.api_token_required,
            custom_api_token=payload.custom_api_token,
            web_password=payload.web_password,
            clear_web_password=bool(payload.clear_web_password),
        )

    @router.post("/api/security/token/generate", dependencies=[Depends(require_mutation_auth)])
    async def generate_security_token():
        tok = security_mgr.generate_new_api_token()
        return {"status": "success", "api_token": tok, **security_mgr.get_status()}

    # --- Multi-Theme Engine Endpoints (Issue #20) ---

    @router.get("/api/themes")
    async def list_themes():
        return theme_mgr.get_active_theme_info()

    @router.post("/api/themes/active", dependencies=[Depends(require_mutation_auth)])
    async def set_active_theme(payload: SetActiveThemeRequest):
        return theme_mgr.set_active_theme(
            palette_id=payload.palette,
            mode=payload.mode,
        )

    @router.post("/api/themes", dependencies=[Depends(require_mutation_auth)])
    async def save_custom_theme(payload: CustomThemeRequest):
        return theme_mgr.save_custom_theme(
            name=payload.name,
            primary=payload.primary,
            secondary=payload.secondary,
            dark_bg=payload.dark_bg,
            dark_card=payload.dark_card,
            light_bg=payload.light_bg,
            light_card=payload.light_card,
            theme_id=payload.id,
        )

    @router.delete("/api/themes/{theme_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_custom_theme(theme_id: str):
        return theme_mgr.delete_custom_theme(theme_id)

    @router.get("/api/onboarding/mcp-config")
    async def get_onboarding_mcp_config(request: Request):
        base_url = str(request.base_url).rstrip("/")
        sec = security_mgr.get_status()
        mcp_url = f"{base_url}/mcp"
        headers: Dict[str, str] = {}
        if sec.get("api_token_required") and sec.get("api_token"):
            headers["Authorization"] = f"Bearer {sec['api_token']}"
        server_entry: Dict[str, Any] = {"serverUrl": mcp_url}
        if headers:
            server_entry["headers"] = headers
        return {
            "mcp_url": mcp_url,
            "api_token_required": sec.get("api_token_required"),
            "config": {"mcpServers": {"memoria": server_entry}},
        }

    return router
