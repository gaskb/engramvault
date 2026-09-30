from fastapi.testclient import TestClient
from engramvault.web.app import create_app


def test_agent_api_search_and_remember(tmp_path):
    db_file = tmp_path / "test_agent.db"
    app = create_app(db_file)
    client = TestClient(app)

    # Seed data first so we have conversations, facts, and decisions
    seed_resp = client.post("/api/seed")
    assert seed_resp.status_code == 200

    # 1. Test GET /api/v1/agent/search
    res = client.get("/api/v1/agent/search", params={"q": "TDK SA-X audiocassette rock"})
    assert res.status_code == 200
    data = res.json()
    assert "markdown_brief" in data
    assert "TDK" in data["markdown_brief"]
    assert data["counts"]["statements"] >= 1

    # 2. Test POST /api/v1/agent/remember (fact)
    rem_fact = client.post(
        "/api/v1/agent/remember",
        json={
            "record_type": "fact",
            "subject": "Neumann TLM 102",
            "predicate": "owned_by_user",
            "object_value": "Microfono condensatore studio principale",
            "category": "audio",
            "notes": "Test agent remember fact",
        },
    )
    assert rem_fact.status_code == 200
    assert rem_fact.json()["status"] == "success"

    # 3. Test POST /api/v1/agent/remember (decision)
    rem_dec = client.post(
        "/api/v1/agent/remember",
        json={
            "record_type": "decision",
            "topic": "Protocollo MCP Homelab",
            "decision_text": "Esporre /mcp direttamente sul container FastAPI .220:5006",
            "status": "ADOPTED",
        },
    )
    assert rem_dec.status_code == 200
    assert rem_dec.json()["record_type"] == "decision"

    # Verify newly remembered fact and decision show up in agent search
    res2 = client.get("/api/v1/agent/search", params={"q": "Neumann Protocollo MCP"})
    assert res2.status_code == 200
    brief2 = res2.json()["markdown_brief"]
    assert "Neumann TLM 102" in brief2
    assert "Protocollo MCP Homelab" in brief2

    # 4. Strict validation: invalid status or record_type must return 422 (Issue #32)
    invalid_status = client.post(
        "/api/v1/agent/remember",
        json={
            "record_type": "decision",
            "topic": "Test Invalid",
            "decision_text": "Should fail validation",
            "status": "NOT_AN_ENUM",
        },
    )
    assert invalid_status.status_code == 422

    invalid_record_type = client.post(
        "/api/v1/agent/remember",
        json={
            "record_type": "invalid_type",
            "subject": "Foo",
            "predicate": "bar",
            "object_value": "baz",
        },
    )
    assert invalid_record_type.status_code == 422


def test_mcp_jsonrpc_and_sse(tmp_path):
    db_file = tmp_path / "test_mcp.db"
    app = create_app(db_file)
    client = TestClient(app)
    client.post("/api/seed")

    # 1. GET /mcp info
    info = client.get("/mcp")
    assert info.status_code == 200
    assert info.json()["server"]["name"] == "memoria-ai-mcp"

    # 2. POST /mcp initialize (JSON)
    init_resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["result"]["serverInfo"]["name"] == "memoria-ai-mcp"

    # 3. POST /mcp tools/list
    tools_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    assert tools_resp.status_code == 200
    tool_names = [t["name"] for t in tools_resp.json()["result"]["tools"]]
    assert "memoria_search" in tool_names
    assert "memoria_get_conversation" in tool_names
    assert "memoria_remember_fact" in tool_names
    assert "memoria_remember_decision" in tool_names

    # 4. POST /mcp tools/call -> memoria_search
    call_resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memoria_search",
                "arguments": {"query": "Zen Go interfaccia audio"},
            },
        },
    )
    assert call_resp.status_code == 200
    text_out = call_resp.json()["result"]["content"][0]["text"]
    assert "Memoria AI" in text_out

    # 5. POST /mcp with SSE Accept header (text/event-stream)
    sse_resp = client.post(
        "/mcp",
        headers={"Accept": "text/event-stream"},
        json={"jsonrpc": "2.0", "id": 4, "method": "ping"},
    )
    assert sse_resp.status_code == 200
    assert "event: message" in sse_resp.text
