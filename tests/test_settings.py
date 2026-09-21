from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient
from memoria.db.engine import Database
from memoria.knowledge.enricher import KnowledgeEnricher
from memoria.knowledge.query_planner import QueryPlanner
from memoria.settings.manager import SettingsManager, mask_key
from memoria.web.app import create_app


def test_mask_key():
    assert mask_key("") == ""
    assert mask_key("dummy") == ""
    assert mask_key("12345678") == "••••••••"
    masked = mask_key("sk-proj-1234567890abcdef")
    assert masked.startswith("sk-")
    assert "••••••••" in masked
    assert masked.endswith("cdef")


def test_settings_manager_defaults_and_updates(tmp_path):
    db_path = tmp_path / "test_settings.db"
    db = Database(db_path)
    mgr = SettingsManager(db)

    # Defaults
    assert mgr.get_active_provider() in ("qwen", "openai")
    models = mgr.get_models_for_provider("openai")
    assert "gpt-4o" in models

    # Add custom model
    updated_models = mgr.add_model_to_provider("openai", "gpt-4.5-preview")
    assert "gpt-4.5-preview" in updated_models
    assert "gpt-4.5-preview" in mgr.get_models_for_provider("openai")

    # Update active provider and model
    mgr.update_settings(
        active_provider="openai",
        active_model="gpt-4.5-preview",
        api_keys={"openai": "sk-test-secret-12345678"},
    )
    assert mgr.get_active_provider() == "openai"
    assert mgr.get_active_model("openai") == "gpt-4.5-preview"
    assert mgr.get_api_key("openai") == "sk-test-secret-12345678"
    assert mgr.is_provider_configured("openai") is True

    # Check masking in full settings
    full = mgr.get_full_settings()
    assert full["active_provider"] == "openai"
    assert full["active_model"] == "gpt-4.5-preview"
    assert "••••••••" in full["providers"]["openai"]["api_key_masked"]


def test_settings_api_endpoints(tmp_path):
    db_path = tmp_path / "test_web_settings.db"
    app = create_app(db_path)
    client = TestClient(app)

    # 1. GET /api/settings
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "active_provider" in data
    assert "providers" in data
    assert "openai" in data["providers"]

    # 2. POST /api/settings - Update active provider & model & keys
    payload = {
        "active_provider": "anthropic",
        "active_model": "claude-sonnet-4-5-20250929",
        "api_keys": {"anthropic": "sk-ant-test-99887766"},
    }
    post_res = client.post("/api/settings", json=payload)
    assert post_res.status_code == 200
    new_data = post_res.json()["settings"]
    assert new_data["active_provider"] == "anthropic"
    assert new_data["active_model"] == "claude-sonnet-4-5-20250929"
    assert new_data["providers"]["anthropic"]["is_configured"] is True

    # 3. GET /api/settings/models
    models_res = client.get("/api/settings/models?provider=anthropic")
    assert models_res.status_code == 200
    assert "claude-sonnet-4-5-20250929" in models_res.json()["models"]

    # 4. POST /api/settings/models - Add custom model
    add_res = client.post(
        "/api/settings/models",
        json={"provider": "anthropic", "model_name": "claude-custom-agent"},
    )
    assert add_res.status_code == 200
    assert "claude-custom-agent" in add_res.json()["models"]

    # 5. POST /api/settings/test
    with patch("memoria.llm.client.LLMClient.chat_completion", return_value="OK"):
        test_res = client.post(
            "/api/settings/test",
            json={"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        )
        assert test_res.status_code == 200
        assert test_res.json()["status"] == "success"
        assert test_res.json()["response"] == "OK"


def test_planner_and_enricher_use_active_settings(tmp_path):
    db_path = tmp_path / "test_planner_settings.db"
    db = Database(db_path)
    mgr = SettingsManager(db)

    # Configure active settings to Anthropic
    mgr.update_settings(
        active_provider="anthropic",
        active_model="claude-sonnet-4-5-20250929",
        api_keys={"anthropic": "sk-ant-mock-key-1234"},
    )

    # QueryPlanner resolves active LLM client
    planner = QueryPlanner(db)
    assert planner.llm.provider_name == "anthropic"
    assert planner.llm.model == "claude-sonnet-4-5-20250929"
    assert planner.llm.api_key == "sk-ant-mock-key-1234"

    # Switch to DeepSeek
    mgr.update_settings(
        active_provider="deepseek",
        active_model="deepseek-reasoner",
        api_keys={"deepseek": "sk-deepseek-mock-5678"},
    )

    # QueryPlanner dynamically reflects the change without recreation
    assert planner.llm.provider_name == "deepseek"
    assert planner.llm.model == "deepseek-reasoner"
    assert planner.llm.api_key == "sk-deepseek-mock-5678"

    # KnowledgeEnricher dynamically reflects the change
    enricher = KnowledgeEnricher(db)
    assert enricher.llm.provider_name == "deepseek"
    assert enricher.llm.model == "deepseek-reasoner"
