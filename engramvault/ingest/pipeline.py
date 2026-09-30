from pathlib import Path
from typing import Any, Dict, Optional
from engramvault.adapters.registry import parse_source, detect_provider
from engramvault.db.engine import Database
from engramvault.knowledge.extractor import KnowledgeExtractor
from engramvault.search.embeddings import EmbeddingEngine
from engramvault.seed.seed_data import chunk_text


def ingest_source(
    source_path: Path | str,
    db: Database,
    provider: str = "auto",
    embedder: Optional[EmbeddingEngine] = None,
    extractor: Optional[KnowledgeExtractor] = None,
) -> Dict[str, Any]:
    """
    Unified ingestion pipeline for archives, single files, or folders.
    Extracts conversations, chunks text, creates embeddings, and distills knowledge into the DB.
    """
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"Sorgente non trovata: {p}")

    detected_provider = provider
    if provider == "auto":
        detected_provider = detect_provider(p)

    conversations = parse_source(p, provider=provider)
    if not conversations and p.is_file() and p.stat().st_size > 0:
        raise ValueError(f"Nessuna conversazione valida trovata in '{p.name}'. Formato non riconosciuto o corrotto.")

    embedder = embedder or EmbeddingEngine()
    extractor = extractor or KnowledgeExtractor(db)

    imported = 0
    skipped = 0

    for conv in conversations:
        chash = conv.compute_hash()
        if db.conversation_exists_by_hash(chash):
            skipped += 1
            continue

        chunk_payloads = []
        for idx, msg in enumerate(conv.messages):
            for c_str in chunk_text(msg.text):
                blob = embedder.embed_to_blob(c_str)
                chunk_payloads.append((idx, c_str, blob))

        conv_id = db.insert_conversation_with_chunks(conv, chunk_payloads)
        conv_record = db.get_conversation(conv_id)
        if conv_record:
            extractor.extract_from_conversation(conv, conv_id, conv_record["messages"])

        imported += 1

    return {
        "status": "success",
        "provider": detected_provider,
        "imported": imported,
        "skipped": skipped,
        "total": len(conversations),
        "source": str(p),
    }
