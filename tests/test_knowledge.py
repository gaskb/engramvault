from pathlib import Path

from engramvault.adapters.base import NormalizedConversation, NormalizedMessage
from engramvault.db.engine import Database
from engramvault.knowledge.extractor import KnowledgeExtractor


def test_fact_vs_claim_separation(tmp_path: Path):
    db = Database(tmp_path / "test_k.db")
    extractor = KnowledgeExtractor(db)

    # Conversation 1: User states fact, AI makes claim
    c1 = NormalizedConversation(
        assistant="chatgpt",
        title="Audio setup",
        date="2024-01-10T12:00:00Z",
        external_ref="c1",
        messages=[
            NormalizedMessage(role="user", text="Nel mio setup ho Audient iD22."),
            NormalizedMessage(
                role="assistant",
                text="Audient iD22 possiede ottimi preamplificatori e convertitori Burr-Brown.",
            ),
        ],
    )

    conv_id = db.insert_conversation_with_chunks(c1, [(0, m.text, None) for idx, m in enumerate(c1.messages)])
    c_rec = db.get_conversation(conv_id)
    extractor.extract_from_conversation(c1, conv_id, c_rec["messages"])

    facts = db.get_statements(statement_type="FACT")
    claims = db.get_statements(statement_type="CLAIM")

    assert len(facts) >= 1
    assert any("Audient iD22" in f["subject"] and f["predicate"] == "owned_by_user" for f in facts)
    assert any(f["author"] == "user" for f in facts)

    assert len(claims) >= 1
    assert any(c["author"] == "assistant" for c in claims)


def test_temporal_state_and_decisions(tmp_path: Path):
    db = Database(tmp_path / "test_k.db")
    extractor = KnowledgeExtractor(db)

    # In 2024, user buys/owns iD22
    c1 = NormalizedConversation(
        assistant="chatgpt",
        title="Setup 2024",
        date="2024-01-10T12:00:00Z",
        external_ref="c1",
        messages=[NormalizedMessage(role="user", text="Nel mio setup ho Audient iD22.")],
    )
    cid1 = db.insert_conversation_with_chunks(c1, [(0, c1.messages[0].text, None)])
    extractor.extract_from_conversation(c1, cid1, db.get_conversation(cid1)["messages"])

    current_items = db.get_statements(predicate="owned_by_user", is_current=1)
    assert len(current_items) == 1
    assert "Audient iD22" in current_items[0]["subject"]

    # In 2026, user sells iD22 and decides to use Zen Go
    c2 = NormalizedConversation(
        assistant="claude",
        title="Setup 2026",
        date="2026-03-01T10:00:00Z",
        external_ref="c2",
        messages=[
            NormalizedMessage(
                role="user",
                text="Ho venduto Audient iD22 e ho deciso di usare Zen Go come interfaccia principale.",
            )
        ],
    )
    cid2 = db.insert_conversation_with_chunks(c2, [(0, c2.messages[0].text, None)])
    extractor.extract_from_conversation(c2, cid2, db.get_conversation(cid2)["messages"])

    # Verify iD22 is no longer current
    current_after = db.get_statements(predicate="owned_by_user", is_current=1)
    assert not any("Audient iD22" in s["subject"] for s in current_after)
    assert any("Zen Go" in s["subject"] for s in current_after)

    # Verify decision was logged
    decisions = db.get_decisions()
    assert len(decisions) >= 1
    assert any("Zen Go" in d["decision_text"] for d in decisions)
