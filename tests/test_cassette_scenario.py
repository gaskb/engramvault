import unittest
import tempfile
from pathlib import Path
from memoria.db.engine import Database
from memoria.seed.seed_data import seed_database
from memoria.knowledge.query_planner import QueryPlanner


class TestCassetteScenario(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_file = Path(self.tmp_dir.name) / "full_vault.db"
        self.db = Database(self.db_file)
        seed_database(self.db)
        self.planner = QueryPlanner(self.db)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_audiocassette_synthesis_query(self):
        query = "quali sono le migliori audiocassette per registrare un brano rock tra le audiocassette che ho ?"
        res = self.planner.route_query(query, mode="synthetic")

        self.assertTrue(res["mode"].startswith("synthetic"))
        answer = res["answer"]

        # Verify that the user's owned cassettes are cited
        self.assertIn("TDK SA-X", answer)
        self.assertIn("Maxell XLII", answer)
        self.assertIn("Sony HF", answer)

        # Verify recommendation
        self.assertIn("TDK SA-X", answer)
        self.assertTrue("Rock" in answer or "rock" in answer)
        self.assertIn("Type II", answer)

        # Verify facts and sources
        self.assertGreaterEqual(len(res["facts"]), 3)
        self.assertGreater(len(res["sources"]), 0)

    def test_structured_query_inventory(self):
        res = self.planner.route_query("Quali cassette possiedo?", mode="structured")

        self.assertEqual(res["mode"], "structured")
        self.assertIn("TDK SA-X", res["answer"])
        self.assertIn("Maxell XLII", res["answer"])
        self.assertIn("Sony HF", res["answer"])

    def test_structured_query_decisions(self):
        res = self.planner.route_query("Cosa ho deciso di fare?", mode="structured")

        self.assertEqual(res["mode"], "structured")
        self.assertIn("FastAPI", res["answer"])
        self.assertIn("Zen Go", res["answer"])


if __name__ == "__main__":
    unittest.main()
