import unittest
import tempfile
from pathlib import Path
from memoria.db.engine import Database
from memoria.search.embeddings import EmbeddingEngine
from memoria.search.hybrid import HybridSearcher, clean_fts_query
from memoria.adapters.base import NormalizedConversation, NormalizedMessage


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_file = Path(self.tmp_dir.name) / "search_test.db"
        self.db = Database(self.db_file)
        self.embedder = EmbeddingEngine()

        c1 = NormalizedConversation(
            assistant="chatgpt",
            title="Cassette TDK e Maxell",
            date="2024-03-01T00:00:00Z",
            external_ref="1",
            messages=[
                NormalizedMessage(role="user", text="Ho comprato le TDK SA-X e le Maxell XLII."),
                NormalizedMessage(role="assistant", text="Ottimi nastri al cromo Type II per la registrazione.")
            ]
        )

        c2 = NormalizedConversation(
            assistant="claude",
            title="Registrare chitarre rock",
            date="2024-04-01T00:00:00Z",
            external_ref="2",
            messages=[
                NormalizedMessage(role="user", text="Come si registra un brano rock su nastro?"),
                NormalizedMessage(role="assistant", text="Per il rock serve saturazione calda, usa nastri ad alto bias.")
            ]
        )

        for c in [c1, c2]:
            chunks = []
            for idx, m in enumerate(c.messages):
                blob = self.embedder.embed_to_blob(m.text)
                chunks.append((idx, m.text, blob))
            self.db.insert_conversation_with_chunks(c, chunks)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_embedding_cosine(self):
        engine = EmbeddingEngine()
        v1 = engine.embed_text("registrazione brano rock chitarre")
        v2 = engine.embed_text("registrare chitarra elettrica rock nastro")
        v3 = engine.embed_text("ricetta torta alle mele e farina")

        sim1_2 = engine.cosine_similarity(v1, v2)
        sim1_3 = engine.cosine_similarity(v1, v3)

        self.assertGreater(sim1_2, sim1_3)
        self.assertGreater(sim1_2, 0.25)

    def test_clean_fts_query(self):
        self.assertNotEqual(clean_fts_query('TDK "SA-X"!'), "")
        self.assertIn('"TDK"*', clean_fts_query("TDK SA-X"))

    def test_hybrid_search(self):
        searcher = HybridSearcher(self.db, self.embedder)

        # Search exact model name
        results = searcher.hybrid_search("TDK SA-X")
        self.assertGreater(len(results), 0)
        self.assertIn("TDK SA-X", results[0]["chunk_text"])
        self.assertIsNotNone(results[0]["bm25_rank"])

        # Search topic
        rock_results = searcher.hybrid_search("musica rock e chitarre")
        self.assertGreater(len(rock_results), 0)
        self.assertTrue(any("rock" in r["chunk_text"].lower() for r in rock_results))


if __name__ == "__main__":
    unittest.main()
