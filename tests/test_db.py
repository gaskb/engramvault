import unittest
import tempfile
from pathlib import Path
from memoria.db.engine import Database
from memoria.adapters.base import NormalizedConversation, NormalizedMessage
from memoria.db.models import StatementRecord, DecisionRecord


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_file = Path(self.tmp_dir.name) / "test_vault.db"
        self.db = Database(self.db_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_db_init(self):
        stats = self.db.get_stats()
        self.assertEqual(stats["conversations"], 0)
        self.assertEqual(stats["messages"], 0)
        self.assertEqual(stats["chunks"], 0)

    def test_insert_and_fts_sync(self):
        conv = NormalizedConversation(
            assistant="chatgpt",
            title="Test Titolo",
            date="2024-01-01T00:00:00Z",
            external_ref="ref-1",
            messages=[
                NormalizedMessage(role="user", text="Vorrei registrare delle chitarre"),
                NormalizedMessage(role="assistant", text="Usa un microfono a nastro.")
            ]
        )
        chunks = [
            (0, "Vorrei registrare delle chitarre", None),
            (1, "Usa un microfono a nastro.", None)
        ]

        conv_id = self.db.insert_conversation_with_chunks(conv, chunks)
        self.assertIsNotNone(conv_id)

        # Verify duplicate prevention by hash
        self.assertTrue(self.db.conversation_exists_by_hash(conv.compute_hash()))

        # Verify FTS5 query directly
        with self.db.get_connection() as conn:
            res = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'chitarre'").fetchall()
            self.assertEqual(len(res), 1)
            res_nastro = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'nastro'").fetchall()
            self.assertEqual(len(res_nastro), 1)

    def test_statements_and_decisions(self):
        e_id = self.db.upsert_entity("Audient iD22", "audio_interface")
        self.assertIsNotNone(e_id)

        # Add Fact
        stmt_id = self.db.insert_statement(StatementRecord(
            id=None,
            entity_id=e_id,
            statement_type="FACT",
            subject="Audient iD22",
            predicate="owned_by_user",
            object="true",
            author="user",
            valid_from="2024-01-01",
            is_current=1
        ))
        self.assertIsNotNone(stmt_id)

        # Invalidate
        self.db.invalidate_statement(stmt_id, valid_until="2026-03-01")
        curr_stmts = self.db.get_statements(entity_id=e_id, is_current=1)
        self.assertEqual(len(curr_stmts), 0)

        all_stmts = self.db.get_statements(entity_id=e_id)
        self.assertEqual(len(all_stmts), 1)
        self.assertEqual(all_stmts[0]["is_current"], 0)
        self.assertEqual(all_stmts[0]["valid_until"], "2026-03-01")

        # Add Decision
        dec_id = self.db.insert_decision(DecisionRecord(
            id=None,
            topic="Audio Interface",
            decision_text="Passato a Zen Go",
            decided_at="2026-03-01"
        ))
        self.assertIsNotNone(dec_id)
        decisions = self.db.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["topic"], "Audio Interface")


if __name__ == "__main__":
    unittest.main()
