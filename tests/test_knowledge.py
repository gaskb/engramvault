import unittest
import tempfile
from pathlib import Path
from memoria.db.engine import Database
from memoria.knowledge.extractor import KnowledgeExtractor
from memoria.adapters.base import NormalizedConversation, NormalizedMessage


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp_dir.name) / "test_k.db")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_fact_vs_claim_separation(self):
        extractor = KnowledgeExtractor(self.db)

        # Conversation 1: User states fact, AI makes claim
        c1 = NormalizedConversation(
            assistant="chatgpt",
            title="Audio setup",
            date="2024-01-10T12:00:00Z",
            external_ref="c1",
            messages=[
                NormalizedMessage(role="user", text="Nel mio setup ho Audient iD22."),
                NormalizedMessage(role="assistant", text="Audient iD22 possiede ottimi preamplificatori e convertitori Burr-Brown.")
            ]
        )

        conv_id = self.db.insert_conversation_with_chunks(c1, [(0, m.text, None) for idx, m in enumerate(c1.messages)])
        c_rec = self.db.get_conversation(conv_id)
        extractor.extract_from_conversation(c1, conv_id, c_rec["messages"])

        facts = self.db.get_statements(statement_type="FACT")
        claims = self.db.get_statements(statement_type="CLAIM")

        self.assertGreaterEqual(len(facts), 1)
        self.assertTrue(any("Audient iD22" in f["subject"] and f["predicate"] == "owned_by_user" for f in facts))
        self.assertTrue(any(f["author"] == "user" for f in facts))

        self.assertGreaterEqual(len(claims), 1)
        self.assertTrue(any(c["author"] == "assistant" for c in claims))

    def test_temporal_state_and_decisions(self):
        extractor = KnowledgeExtractor(self.db)

        # In 2024, user buys/owns iD22
        c1 = NormalizedConversation(
            assistant="chatgpt",
            title="Setup 2024",
            date="2024-01-10T12:00:00Z",
            external_ref="c1",
            messages=[
                NormalizedMessage(role="user", text="Nel mio setup ho Audient iD22.")
            ]
        )
        cid1 = self.db.insert_conversation_with_chunks(c1, [(0, c1.messages[0].text, None)])
        extractor.extract_from_conversation(c1, cid1, self.db.get_conversation(cid1)["messages"])

        current_items = self.db.get_statements(predicate="owned_by_user", is_current=1)
        self.assertEqual(len(current_items), 1)
        self.assertIn("Audient iD22", current_items[0]["subject"])

        # In 2026, user sells iD22 and decides to use Zen Go
        c2 = NormalizedConversation(
            assistant="claude",
            title="Setup 2026",
            date="2026-03-01T10:00:00Z",
            external_ref="c2",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Ho venduto Audient iD22 e ho deciso di usare Zen Go come interfaccia principale."
                )
            ]
        )
        cid2 = self.db.insert_conversation_with_chunks(c2, [(0, c2.messages[0].text, None)])
        extractor.extract_from_conversation(c2, cid2, self.db.get_conversation(cid2)["messages"])

        # Verify iD22 is no longer current
        current_after = self.db.get_statements(predicate="owned_by_user", is_current=1)
        self.assertFalse(any("Audient iD22" in s["subject"] for s in current_after))
        self.assertTrue(any("Zen Go" in s["subject"] for s in current_after))

        # Verify decision was logged
        decisions = self.db.get_decisions()
        self.assertGreaterEqual(len(decisions), 1)
        self.assertTrue(any("Zen Go" in d["decision_text"] for d in decisions))


if __name__ == "__main__":
    unittest.main()
