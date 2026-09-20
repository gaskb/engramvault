import argparse
import sys
from pathlib import Path
from memoria.config import DEFAULT_DB_PATH
from memoria.db.engine import Database
from memoria.adapters.registry import parse_source, detect_provider
from memoria.search.embeddings import EmbeddingEngine
from memoria.search.hybrid import HybridSearcher
from memoria.knowledge.extractor import KnowledgeExtractor
from memoria.knowledge.query_planner import QueryPlanner
from memoria.seed.seed_data import seed_database, chunk_text


def cmd_init(args, db: Database):
    print(f"✓ Database inizializzato correttamente in: {db.db_path}")


def cmd_seed(args, db: Database):
    print("Popolamento database con conversazioni di esempio...")
    count = seed_database(db)
    print(f"✓ Inserite {count} nuove conversazioni di esempio.")
    stats = db.get_stats()
    print(f"Statistiche attuali: {stats}")


def cmd_ingest(args, db: Database):
    source_path = Path(args.path)
    if not source_path.exists():
        print(f"Errore: Percorso '{source_path}' non trovato.", file=sys.stderr)
        sys.exit(1)

    provider = args.provider
    if provider == "auto":
        detected = detect_provider(source_path)
        print(f"Rilevato provider: '{detected}' per '{source_path.name}'")

    print(f"Caricamento conversazioni da: {source_path}...")
    conversations = parse_source(source_path, provider=provider)
    print(f"Trovate {len(conversations)} conversazioni. Indicizzazione in corso...")

    embedder = EmbeddingEngine()
    extractor = KnowledgeExtractor(db)
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

    print(f"✓ Ingestione completata: {imported} importate, {skipped} saltate (già presenti).")


def cmd_ask(args, db: Database):
    planner = QueryPlanner(db)
    res = planner.route_query(args.query, mode=args.mode)

    print("\n" + "="*70)
    print(f"DOMANDA: {args.query}")
    print(f"MODALITÀ ESEGUITA: [{res['mode'].upper()}]")
    print("="*70 + "\n")

    if res.get("answer"):
        print(res["answer"])
    elif res.get("results"):
        print(f"Trovati {len(res['results'])} passaggi pertinenti:\n")
        for idx, hit in enumerate(res["results"], 1):
            prov = hit.get("provider", "ai").upper()
            title = hit.get("conv_title", "")
            date = hit.get("created_at", "")[:10]
            role = hit.get("message_role", "")
            text = hit.get("chunk_text", "").strip()
            print(f"[{idx}] {prov} | '{title}' ({date}) - Autore: {role}")
            print(f"    RRF Score: {hit.get('rrf_score')} (BM25 rank: {hit.get('bm25_rank')}, Vec rank: {hit.get('vector_rank')})")
            print(f"    \"{text[:200]}...\"\n")


def cmd_search(args, db: Database):
    searcher = HybridSearcher(db)
    hits = searcher.hybrid_search(args.query, limit=args.limit, provider=args.provider)
    print(f"\nRisultati della Ricerca Ibrida per: \"{args.query}\" ({len(hits)} trovati)\n")
    for idx, hit in enumerate(hits, 1):
        print(f"[{idx}] {hit['provider'].upper()}: {hit['conv_title']} ({hit['created_at'][:10]})")
        print(f"    Punteggio RRF: {hit['rrf_score']} | Match: \"{hit['chunk_text'][:180].replace(chr(10), ' ')}...\"\n")


def cmd_entities(args, db: Database):
    entities = db.get_entities(category=args.category)
    print(f"\nCatalogo Entità ({len(entities)}):\n")
    for e in entities:
        stmts = db.get_statements(entity_id=e["id"])
        print(f"• [{e['category'].upper()}] {e['name']}")
        for s in stmts:
            badge = "FACT (Utente)" if s["statement_type"] == "FACT" else "CLAIM (AI)"
            curr = "ATTUALE" if s["is_current"] else "STORICO/DISMESSO"
            print(f"    └─ [{badge} - {curr}] {s['predicate']}: {s['object']} (dal {s['valid_from'][:10] if s['valid_from'] else 'N/D'})")


def cmd_decisions(args, db: Database):
    decisions = db.get_decisions()
    print(f"\nRegistro delle Decisioni ({len(decisions)}):\n")
    for d in decisions:
        print(f"• [{d['status']}] {d['topic']} ({d['decided_at'][:10]}):")
        print(f"    \"{d['decision_text']}\"")
        if d.get("conversation_title"):
            print(f"    Fonte: {d.get('conversation_provider')} - '{d.get('conversation_title')}'")
        print()


def cmd_timeline(args, db: Database):
    timeline = db.get_timeline(subject=args.subject)
    print(f"\nTimeline Eventi & Setup ({len(timeline)}):\n")
    for ev in timeline:
        dt = (ev.get("event_date") or "")[:10] or "N/D"
        st = ev.get("statement_type", "INFO")
        curr = " [ATTIVO]" if ev.get("is_current") else " [PASSATO]"
        print(f"• [{dt}] [{st}]{curr} {ev['subject']} -> {ev['predicate']}: {ev['object']}")


def cmd_stats(args, db: Database):
    stats = db.get_stats()
    print(f"\nStatistiche Vault '{db.db_path.name}':")
    for k, v in stats.items():
        print(f"  - {k:20s}: {v}")
    print()


def cmd_serve(args, db: Database):
    import uvicorn
    from memoria.web.app import create_app
    app = create_app(db.db_path)
    print(f"Avvio Dashboard Web su http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Percorso file SQLite vault.db")

    parser = argparse.ArgumentParser(
        prog="memoria",
        parents=[base_parser],
        description="Memoria AI: Conversation Vault & Knowledge Layer"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", parents=[base_parser], help="Inizializza il database SQLite")
    p_init.set_defaults(func=cmd_init)

    # seed
    p_seed = subparsers.add_parser("seed", parents=[base_parser], help="Carica dati di esempio realistici")
    p_seed.set_defaults(func=cmd_seed)

    # ingest
    p_ingest = subparsers.add_parser("ingest", parents=[base_parser], help="Importa conversazioni da file o directory")
    p_ingest.add_argument("path", help="Percorso del file export (JSON/ZIP/JSONL/MD) o cartella")
    p_ingest.add_argument("--provider", default="auto", choices=["auto", "chatgpt", "claude", "claude_code", "gemini", "markdown", "normalized"])
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = subparsers.add_parser("ask", parents=[base_parser], help="Fai una domanda alla tua memoria AI")
    p_ask.add_argument("query", help="Testo della domanda")
    p_ask.add_argument("--mode", default="auto", choices=["auto", "extractive", "structured", "synthetic"])
    p_ask.set_defaults(func=cmd_ask)

    # search
    p_search = subparsers.add_parser("search", parents=[base_parser], help="Ricerca ibrida nei messaggi")
    p_search.add_argument("query", help="Termini di ricerca")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--provider", default=None)
    p_search.set_defaults(func=cmd_search)

    # entities
    p_entities = subparsers.add_parser("entities", parents=[base_parser], help="Elenca le entità estratte")
    p_entities.add_argument("--category", default=None)
    p_entities.set_defaults(func=cmd_entities)

    # decisions
    p_dec = subparsers.add_parser("decisions", parents=[base_parser], help="Elenca le decisioni registrate")
    p_dec.add_argument("--status", default=None)
    p_dec.set_defaults(func=cmd_decisions)

    # timeline
    p_time = subparsers.add_parser("timeline", parents=[base_parser], help="Visualizza cronologia evolutiva")
    p_time.add_argument("--subject", default=None)
    p_time.set_defaults(func=cmd_timeline)

    # stats
    p_stats = subparsers.add_parser("stats", parents=[base_parser], help="Mostra statistiche del database")
    p_stats.set_defaults(func=cmd_stats)

    # serve
    p_serve = subparsers.add_parser("serve", parents=[base_parser], help="Avvia il server web dashboard")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    db = Database(args.db)
    args.func(args, db)


if __name__ == "__main__":
    main()
