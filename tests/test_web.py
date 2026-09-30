import pytest
from starlette.testclient import TestClient
from engramvault.db.engine import Database
from engramvault.web.app import create_app


@pytest.fixture
def test_client(tmp_path):
    db_file = tmp_path / "test_web_vault.db"
    db = Database(db_file)
    app = create_app(db_file)
    return TestClient(app)


def test_index_page(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Memoria AI" in response.text
    assert "site-header-centered" in response.text
    assert "tabs tabs-centered" in response.text
    assert "/mcp" in response.text
    assert "gaskb_style" in response.text


def test_static_css_served(test_client):
    res_css = test_client.get("/static/css/gaskb.css")
    assert res_css.status_code == 200
    assert "tokens.css" in res_css.text

    res_theme = test_client.get("/static/css/themes/emerald.css")
    assert res_theme.status_code == 200
    assert "--md-primary" in res_theme.text


def test_api_stats(test_client):
    response = test_client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert "messages" in data
