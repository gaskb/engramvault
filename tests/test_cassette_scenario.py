from pathlib import Path
import pytest

from memoria.db.engine import Database
from memoria.knowledge.query_planner import QueryPlanner
from memoria.seed.seed_data import seed_database


@pytest.fixture
def cassette_planner(tmp_path: Path):
    db = Database(tmp_path / "full_vault.db")
    seed_database(db)
    return QueryPlanner(db)


def test_audiocassette_synthesis_query(cassette_planner: QueryPlanner):
    query = "quali sono le migliori audiocassette per registrare un brano rock tra le audiocassette che ho ?"
    res = cassette_planner.route_query(query, mode="synthetic")

    assert res["mode"].startswith("synthetic")
    answer = res["answer"]

    # Verify that the user's owned cassettes are cited
    assert "TDK SA-X" in answer
    assert "Maxell XLII" in answer
    assert "Sony HF" in answer

    # Verify recommendation
    assert "TDK SA-X" in answer
    assert "Rock" in answer or "rock" in answer
    assert "Type II" in answer

    # Verify facts and sources
    assert len(res["facts"]) >= 3
    assert len(res["sources"]) > 0


def test_structured_query_inventory(cassette_planner: QueryPlanner):
    res = cassette_planner.route_query("Quali cassette possiedo?", mode="structured")

    assert res["mode"] == "structured"
    assert "TDK SA-X" in res["answer"]
    assert "Maxell XLII" in res["answer"]
    assert "Sony HF" in res["answer"]


def test_structured_query_decisions(cassette_planner: QueryPlanner):
    res = cassette_planner.route_query("Cosa ho deciso di fare?", mode="structured")

    assert res["mode"] == "structured"
    assert "FastAPI" in res["answer"]
    assert "Zen Go" in res["answer"]

