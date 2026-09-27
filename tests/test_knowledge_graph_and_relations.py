import json
import uuid
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from memoria.db.engine import Database
from memoria.db.models import RelationRecord, StatementRecord
from memoria.knowledge.enricher import KnowledgeEnricher
from memoria.knowledge.query_planner import QueryPlanner
from memoria.llm.client import LLMClient
from memoria.mcp_server import handle_mcp_jsonrpc, build_agent_search_response
from memoria.seed.seed_data import seed_database
from memoria.web.app import create_app


@pytest.fixture
def graph_db(tmp_path):
    db_file = tmp_path / "test_graph.db"
    db = Database(db_file)
    seed_database(db)
    return db


def test_relation_crud_and_traversal(graph_db):
    # 1. Upsert entities
    e1 = graph_db.upsert_entity("Yaesu FT-891", "ham_radio")
    e2 = graph_db.upsert_entity("Antenna Dipolo 40m", "ham_radio")
    e3 = graph_db.upsert_entity("Accordatore AT-100", "ham_radio")

    # 2. Insert relations
    rel1 = RelationRecord(
        id=str(uuid.uuid4()),
        from_entity_id=e1,
        to_entity_id=e2,
        relation_type="Connected To / Coax Cable",
    )
    rel_id1 = graph_db.insert_relation(rel1)
    assert rel_id1 is not None

    rel2 = RelationRecord(
        id=str(uuid.uuid4()),
        from_entity_id=e2,
        to_entity_id=e3,
        relation_type="paired_with",
    )
    rel_id2 = graph_db.insert_relation(rel2)

    # 3. get_relations
    all_rels = graph_db.get_relations()
    assert len(all_rels) >= 2

    filtered = graph_db.get_relations(entity_id=e1)
    assert len(filtered) == 1
    assert filtered[0]["relation_type"] == "connected_to_coax_cable"
    assert filtered[0]["from_name"] == "Yaesu FT-891"
    assert filtered[0]["to_name"] == "Antenna Dipolo 40m"

    # 4. find_related_entities
    related = graph_db.find_related_entities([e1])
    assert len(related) == 1
    assert related[0]["to_name"] == "Antenna Dipolo 40m"

    # 5. get_entity_graph (1-hop vs 2-hop)
    g_1hop = graph_db.get_entity_graph("Yaesu FT-891", depth=1)
    assert g_1hop["root"]["name"] == "Yaesu FT-891"
    assert len(g_1hop["nodes"]) == 2
    assert len(g_1hop["edges"]) == 1

    g_2hop = graph_db.get_entity_graph("Yaesu FT-891", depth=2)
    assert len(g_2hop["nodes"]) == 3
    assert len(g_2hop["edges"]) == 2

    # 6. delete_relation
    ok = graph_db.delete_relation(rel_id1)
    assert ok is True
    assert len(graph_db.get_relations(entity_id=e1)) == 0


def test_enricher_relation_extraction(graph_db):
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_configured.return_value = True
    mock_llm.model = "mock-graph-extractor"

    llm_payload = {
        "entities": [
            {"name": "EngramVault", "category": "project"},
            {"name": "Docker", "category": "software"},
            {"name": "SQLite", "category": "software"},
        ],
        "facts": [
            {"subject": "EngramVault", "predicate": "runs_on", "object": "Docker"},
        ],
        "claims": [],
        "decisions": [],
        "relations": [
            {"from_entity": "EngramVault", "relation": "runs_on", "to_entity": "Docker"},
            {"from_entity": "EngramVault", "relation": "depends_on", "to_entity": "SQLite"},
        ],
    }
    mock_llm.chat_completion.return_value = json.dumps(llm_payload)

    enricher = KnowledgeEnricher(graph_db, llm=mock_llm, delay_between_calls=0)
    summary = enricher.run_batch(limit=1, rebuild=True)

    assert summary["processed"] == 1
    assert summary["new_relations"] >= 2

    rels = graph_db.get_relations()
    assert any(r["from_name"] == "EngramVault" and r["to_name"] == "Docker" for r in rels)
    assert any(r["from_name"] == "EngramVault" and r["to_name"] == "SQLite" for r in rels)


def test_agent_api_and_mcp_graph(graph_db):
    # Setup test entities & relations
    e1 = graph_db.upsert_entity("Proxmox VE", "homelab_infra")
    e2 = graph_db.upsert_entity("LXC Container 103", "homelab_infra")
    rel = RelationRecord(
        id=str(uuid.uuid4()),
        from_entity_id=e1,
        to_entity_id=e2,
        relation_type="hosts",
    )
    graph_db.insert_relation(rel)

    app = create_app(graph_db.db_path)
    client = TestClient(app)

    # 1. REST GET /api/v1/agent/graph/{entity}
    res_g = client.get("/api/v1/agent/graph/Proxmox VE")
    assert res_g.status_code == 200
    g_data = res_g.json()
    assert g_data["root"]["name"] == "Proxmox VE"
    assert len(g_data["edges"]) == 1

    # 2. REST POST /api/v1/agent/relation
    res_post_rel = client.post("/api/v1/agent/relation", json={
        "from_entity": "LXC Container 103",
        "relation_type": "runs_service",
        "to_entity": "Nginx Proxy Manager"
    })
    assert res_post_rel.status_code == 200
    assert res_post_rel.json()["status"] == "success"

    # 3. Curation API /api/relations
    res_cur_rels = client.get("/api/relations")
    assert res_cur_rels.status_code == 200
    assert len(res_cur_rels.json()) >= 2

    # 4. MCP Tools
    planner = QueryPlanner(graph_db)

    # memoria_get_entity_graph tool
    mcp_call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "memoria_get_entity_graph",
            "arguments": {"entity": "Proxmox VE", "depth": 2}
        }
    }
    mcp_res = handle_mcp_jsonrpc(mcp_call, db=graph_db, planner=planner)
    assert "error" not in mcp_res
    text_out = mcp_res["result"]["content"][0]["text"]
    assert "Knowledge Graph: `Proxmox VE`" in text_out
    assert "hosts" in text_out

    # memoria_search with graph connections
    search_res = build_agent_search_response(db=graph_db, planner=planner, query="Proxmox")
    assert "relations" in search_res
    assert len(search_res["relations"]) >= 1
    assert "Relazioni Semantiche nel Knowledge Graph" in search_res["markdown_brief"]
