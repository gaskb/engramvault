import pytest
from starlette.testclient import TestClient
from memoria.db.engine import Database
from memoria.web.app import create_app


@pytest.fixture
def test_client(tmp_path):
    db_file = tmp_path / "test_web_vault.db"
    db = Database(db_file)
    app = create_app(db_file)
    return TestClient(app)


def test_index_page(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Memoria AI Anti" in response.text


def test_api_stats(test_client):
    response = test_client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert "messages" in data
