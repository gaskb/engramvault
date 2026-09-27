"""Tests for Security/Bearer Auth (Issue #15), Onboarding/MCP Config (Issue #17), and Multi-Theme Engine (Issue #20)."""

from pathlib import Path
from fastapi.testclient import TestClient

from memoria.web.app import create_app


def test_security_bearer_token_and_web_password(tmp_path: Path):
    db_file = tmp_path / "test_sec.db"
    app = create_app(db_file)
    client = TestClient(app)

    # 1. Default LAN mode: agent search works without token
    res = client.get("/api/v1/agent/search?q=test")
    assert res.status_code == 200

    # 2. Check security status and generate new token
    gen_res = client.post("/api/security/token/generate")
    assert gen_res.status_code == 200
    token = gen_res.json()["api_token"]
    assert token.startswith("mem_")

    # 3. Enable Bearer token enforcement
    cfg_res = client.post("/api/security/config", json={"api_token_required": True})
    assert cfg_res.status_code == 200
    assert cfg_res.json()["api_token_required"] is True

    # 4. Request without token is rejected with 401
    unauth = client.get("/api/v1/agent/search?q=test")
    assert unauth.status_code == 401

    mcp_unauth = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert mcp_unauth.status_code == 401

    # 5. Request with Authorization: Bearer <token> succeeds
    auth_ok = client.get("/api/v1/agent/search?q=test", headers={"Authorization": f"Bearer {token}"})
    assert auth_ok.status_code == 200

    mcp_ok = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert mcp_ok.status_code == 200

    # 6. Enable Web UI password protection (now requires Bearer token since api_token_required=True)
    unauth_cfg = client.post("/api/security/config", json={"web_password": "secret-homelab-pw"})
    assert unauth_cfg.status_code == 401
    pw_cfg = client.post(
        "/api/security/config",
        json={"web_password": "secret-homelab-pw"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert pw_cfg.status_code == 200
    home_locked = client.get("/")
    assert "Unlock Vault" in home_locked.text

    # 7. Regression Test (review.md / Issue #32):
    # When Web Password is enabled, ALL /api/* read and write data endpoints must return 401 without auth!
    unauth_client = TestClient(app)
    for protected_path in (
        "/api/conversations",
        "/api/entities",
        "/api/decisions",
        "/api/relations",
        "/api/stats",
        "/api/settings",
    ):
        r = unauth_client.get(protected_path)
        assert r.status_code == 401, f"{protected_path} should be 401 Unauthorized without session or token"

    # Bearer token unlocks read endpoints
    for protected_path in ("/api/conversations", "/api/entities", "/api/stats"):
        r_auth = unauth_client.get(protected_path, headers={"Authorization": f"Bearer {token}"})
        assert r_auth.status_code == 200, f"{protected_path} should be 200 with Bearer token"

    # Login with wrong password fails
    bad_login = client.post("/api/auth/login", json={"password": "wrong"})
    assert bad_login.status_code == 401

    # Login with right password sets cookie and unlocks / AND /api/* data endpoints
    good_login = client.post("/api/auth/login", json={"password": "secret-homelab-pw"})
    assert good_login.status_code == 200
    home_unlocked = client.get("/")
    assert "tab-btn-ask" in home_unlocked.text

    # Cookie unlocks data endpoints
    cookie_res = client.get("/api/conversations")
    assert cookie_res.status_code == 200
    assert client.get("/api/stats").status_code == 200


def test_multi_theme_engine_and_custom_theme(tmp_path: Path):
    db_file = tmp_path / "test_themes.db"
    app = create_app(db_file)
    client = TestClient(app)

    # 1. List built-in presets
    res = client.get("/api/themes")
    assert res.status_code == 200
    data = res.json()
    theme_ids = {t["id"] for t in data["themes"]}
    assert {"emerald", "amber", "indigo", "ruby", "slate"}.issubset(theme_ids)

    # 2. Switch active preset
    act_res = client.post("/api/themes/active", json={"palette": "indigo", "mode": "light"})
    assert act_res.status_code == 200
    assert act_res.json()["active_palette"] == "indigo"
    assert act_res.json()["active_mode"] == "light"

    # 3. Create custom theme
    create_res = client.post(
        "/api/themes",
        json={
            "name": "Cyber Ocean",
            "primary": "#06b6d4",
            "secondary": "#3b82f6",
            "dark_bg": "#09111e",
            "dark_card": "#122036",
        },
    )
    assert create_res.status_code == 200
    created_data = create_res.json()
    assert created_data["active_palette"] == "cyber-ocean"

    # 4. Delete custom theme
    del_res = client.delete("/api/themes/cyber-ocean")
    assert del_res.status_code == 200
    assert del_res.json()["active_palette"] == "emerald"


def test_onboarding_mcp_config_generator(tmp_path: Path):
    db_file = tmp_path / "test_onboard.db"
    app = create_app(db_file)
    client = TestClient(app)

    # 1. Before token is required: accessible without headers
    res_open = client.get("/api/onboarding/mcp-config")
    assert res_open.status_code == 200
    cfg_open = res_open.json()["config"]
    assert "memoria" in cfg_open["mcpServers"]
    assert "Authorization" not in cfg_open["mcpServers"]["memoria"].get("headers", {})

    # 2. Generate token and enable token enforcement
    gen_res = client.post("/api/security/token/generate")
    assert gen_res.status_code == 200
    token = gen_res.json()["api_token"]
    client.post("/api/security/config", json={"api_token_required": True})

    # 3. Unauthenticated request is rejected with 401 (prevents leaking token)
    res_unauth = client.get("/api/onboarding/mcp-config")
    assert res_unauth.status_code == 401

    # 4. Authenticated request succeeds and provides bearer token in config snippet
    res = client.get("/api/onboarding/mcp-config", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    cfg = res.json()["config"]
    assert "memoria" in cfg["mcpServers"]
    assert "Authorization" in cfg["mcpServers"]["memoria"]["headers"]
    assert token in cfg["mcpServers"]["memoria"]["headers"]["Authorization"]
