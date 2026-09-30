from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from engramvault.db.engine import Database
from engramvault.web.app import STATIC_DIR, create_app


@pytest.fixture
def test_client(tmp_path):
    db_file = tmp_path / "test_pwa_vault.db"
    Database(db_file)
    app = create_app(db_file)
    return TestClient(app)


def test_manifest_endpoint_and_schema(test_client):
    response = test_client.get("/manifest.json")
    assert response.status_code == 200
    assert "application/manifest+json" in response.headers.get("content-type", "")

    data = response.json()
    assert data["name"] == "EngramVault — Personal AI Memory"
    assert data["short_name"] == "EngramVault"
    assert data["display"] == "standalone"
    assert data["start_url"] == "/?pwa=true"
    assert data["theme_color"] == "#0c0f17"
    assert data["background_color"] == "#0c0f17"

    # Icons validation
    icons = data.get("icons", [])
    assert len(icons) >= 3
    icon_sizes = [i.get("sizes") for i in icons]
    assert "192x192" in icon_sizes
    assert "512x512" in icon_sizes
    purposes = [i.get("purpose") for i in icons]
    assert "maskable" in purposes

    # Web Share Target validation
    share_target = data.get("share_target")
    assert share_target is not None
    assert share_target["action"] == "/?share=true"
    assert share_target["method"] == "GET"
    assert share_target["params"] == {
        "title": "title",
        "text": "text",
        "url": "url",
    }


def test_service_worker_endpoint_and_headers(test_client):
    response = test_client.get("/service-worker.js")
    assert response.status_code == 200
    assert "javascript" in response.headers.get("content-type", "").lower()
    assert response.headers.get("service-worker-allowed") == "/"

    content = response.text
    assert "engramvault-shell" in content
    assert "addEventListener('install'" in content
    assert "addEventListener('activate'" in content
    assert "addEventListener('fetch'" in content


def test_pwa_icons_static_delivery(test_client):
    expected_icons = [
        "/static/icons/icon-192x192.png",
        "/static/icons/icon-512x512.png",
        "/static/icons/icon-maskable-512x512.png",
        "/static/icons/apple-touch-icon.png",
    ]
    for icon_path in expected_icons:
        res = test_client.get(icon_path)
        assert res.status_code == 200, f"Icon {icon_path} returned {res.status_code}"
        assert "image/png" in res.headers.get("content-type", "")
        assert len(res.content) > 1000  # Non-empty valid binary image


def test_pwa_meta_tags_in_authenticated_view(test_client):
    res = test_client.get("/")
    assert res.status_code == 200
    html = res.text

    assert '<link rel="manifest" href="/manifest.json">' in html
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/static/icons/apple-touch-icon.png">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' in html
    assert '<meta name="apple-mobile-web-app-title" content="EngramVault">' in html
    assert 'id="share-target-banner"' in html
    assert "registerServiceWorker()" in html
    assert "checkWebShareTarget()" in html


def test_pwa_share_query_parameters_safe_rendering(test_client):
    # Simulates Android Web Share Target invoking GET /?share=true&title=...&text=...&url=...
    res = test_client.get(
        "/?share=true&title=Homelab+Backup&text=Remember+Proxmox+snapshot+config&url=https%3A%2F%2Fexample.com"
    )
    assert res.status_code == 200
    assert 'id="share-target-banner"' in res.text


def test_pwa_meta_tags_in_login_page(tmp_path):
    db_file = tmp_path / "test_pwa_login_vault.db"
    db = Database(db_file)
    db.set_setting("web_password_hash", "hashed_secret_test")
    app = create_app(db_file)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    # Should render login.html with PWA tags
    assert '<link rel="manifest" href="/manifest.json">' in html
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/static/icons/apple-touch-icon.png">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html


def test_pwa_missing_files_error_paths(test_client):
    # When manifest or service worker file does not exist on disk, ensure clean 404 without 500 error
    with patch.object(Path, "exists", return_value=False):
        res_manifest = test_client.get("/manifest.json")
        assert res_manifest.status_code == 404
        assert res_manifest.json()["detail"] == "Manifest not found"

        res_sw = test_client.get("/service-worker.js")
        assert res_sw.status_code == 404
        assert res_sw.json()["detail"] == "Service worker not found"
