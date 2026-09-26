import argparse
import sys
from pathlib import Path
from memoria.config import DEFAULT_DB_PATH, DEFAULT_INCOMING_DIR, WATCH_INTERVAL
from memoria.db.engine import Database
from memoria.ingest.pipeline import ingest_source
from memoria.search.hybrid import HybridSearcher
from memoria.knowledge.query_planner import QueryPlanner
from memoria.seed.seed_data import seed_database


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

    print(f"Avvio ingestione da: {source_path}...")
    res = ingest_source(source_path, db, provider=args.provider)
    print(
        f"✓ Ingestione completata ({res['provider']}): "
        f"{res['imported']} importate, {res['skipped']} saltate (totale: {res['total']})."
    )


def cmd_watch(args, db: Database):
    from memoria.ingest.watcher import DropFolderWatcher
    watcher = DropFolderWatcher(
        db,
        incoming_dir=args.incoming_dir,
        poll_interval=args.interval,
    )
    print(f"Avvio monitoraggio cartella drop: '{watcher.incoming_dir}' (intervallo {args.interval}s)...")
    print("I file (.zip, .json, .jsonl, .md) verranno importati ed eliminati automaticamente.")
    watcher.run_forever()


def cmd_sync_claude_code(args, db: Database):
    from memoria.sync.claude_code import ClaudeCodeSyncer
    syncer = ClaudeCodeSyncer(
        projects_dir=args.projects_dir,
        state_file=args.state_file,
        remote_url=args.remote,
        db=db if not args.remote else None,
    )
    dest_desc = f"server remoto '{args.remote}'" if args.remote else f"database locale '{db.db_path}'"
    print(f"Sincronizzazione log Claude Code verso {dest_desc}...")
    print(f"Sorgente: {syncer.projects_dir}")

    res = syncer.sync(dry_run=args.dry_run)
    if args.dry_run:
        print(f"[DRY-RUN] Trovate {res['changed_count']} sessioni nuove o modificate:")
        for f in res.get("files", []):
            print(f"  - {f}")
    elif res.get("status") == "up_to_date":
        print("✓ Nessuna nuova sessione da sincronizzare (tutto aggiornato).")
    else:
        print(
            f"✓ Sincronizzazione completata: {res.get('synced_count', 0)} sessioni sincronizzate, "
            f"{res.get('error_count', 0)} errori."
        )


def cmd_sync_antigravity(args, db: Database):
    from memoria.sync.antigravity import AntigravitySyncer
    syncer = AntigravitySyncer(
        antigravity_dir=args.antigravity_dir,
        state_file=args.state_file,
        remote_url=args.remote,
        db=db if not args.remote else None,
    )
    dest_desc = f"server remoto '{args.remote}'" if args.remote else f"database locale '{db.db_path}'"
    print(f"Sincronizzazione sessioni Antigravity verso {dest_desc}...")
    print(f"Sorgente: {syncer.antigravity_dir}")

    res = syncer.sync(dry_run=args.dry_run)
    if args.dry_run:
        print(f"[DRY-RUN] Trovate {res['changed_count']} sessioni nuove o modificate:")
        for f in res.get("files", []):
            print(f"  - {f}")
    elif res.get("status") == "up_to_date":
        print("✓ Nessuna nuova sessione Antigravity da sincronizzare (tutto aggiornato).")
    else:
        print(
            f"✓ Sincronizzazione completata: {res.get('synced_count', 0)} sessioni sincronizzate, "
            f"{res.get('error_count', 0)} errori."
        )


def cmd_enrich(args, db: Database):
    from memoria.knowledge.enricher import KnowledgeEnricher
    from memoria.llm.client import LLMClient

    llm = LLMClient(provider=args.provider, model=args.model)
    enricher = KnowledgeEnricher(db, llm=llm, delay_between_calls=args.delay)
    mode_desc = "ricostruzione completa (--rebuild)" if args.rebuild else "incrementale"
    print(f"Avvio pipeline di estrazione conoscenza [{mode_desc}]...")
    print(f"Modello: {enricher.status['extractor_model']} (delay: {args.delay}s)")

    res = enricher.run_batch(limit=args.limit, rebuild=args.rebuild, delay=args.delay)
    print(f"\n✓ Batch enrichment completato:")
    print(f"  - Elaborate       : {res['processed']} / {res['total_to_process']}")
    print(f"  - Nuovi fatti     : {res['new_facts']}")
    print(f"  - Nuove decisioni : {res['new_decisions']}")
    if res.get("error"):
        print(f"  - Errore          : {res['error']}")


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
    base_parser.add_argument("--db", default=argparse.SUPPRESS, help="Percorso file SQLite vault.db")

    parser = argparse.ArgumentParser(
        prog="memoria",
        parents=[base_parser],
        description="Memoria AI: Conversation Vault & Knowledge Layer",
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
    p_ingest.add_argument("--provider", default="auto", choices=["auto", "antigravity", "chatgpt", "claude", "claude_code", "gemini", "markdown", "normalized"])
    p_ingest.set_defaults(func=cmd_ingest)

    # watch (Issue #4)
    p_watch = subparsers.add_parser("watch", parents=[base_parser], help="Monitora cartella drop incoming per ingest ed eliminazione automatica")
    p_watch.add_argument("--incoming-dir", default=str(DEFAULT_INCOMING_DIR), help="Cartella monitorata")
    p_watch.add_argument("--interval", type=int, default=WATCH_INTERVAL, help="Intervallo di scansione in secondi")
    p_watch.set_defaults(func=cmd_watch)

    # sync (Issue #3)
    p_sync = subparsers.add_parser("sync", parents=[base_parser], help="Comandi di sincronizzazione log")
    sync_sub = p_sync.add_subparsers(dest="sync_target", required=True)
    p_sync_claude = sync_sub.add_parser("claude-code", parents=[base_parser], help="Sincronizza sessioni Claude Code (~/.claude/projects/)")
    p_sync_claude.add_argument("--remote", default=None, help="URL dell'istanza remota (es. http://localhost:5006)")
    p_sync_claude.add_argument("--projects-dir", default=None, help="Percorso alternativo ~/.claude/projects")
    p_sync_claude.add_argument("--state-file", default=None, help="Percorso file di stato tracking")
    p_sync_claude.add_argument("--dry-run", action="store_true", help="Mostra solo le sessioni da sincronizzare senza inviarle")
    p_sync_claude.set_defaults(func=cmd_sync_claude_code)

    p_sync_anti = sync_sub.add_parser("antigravity", parents=[base_parser], help="Sincronizza sessioni Antigravity (~/.gemini/antigravity/)")
    p_sync_anti.add_argument("--remote", default=None, help="URL dell'istanza remota (es. http://localhost:5006)")
    p_sync_anti.add_argument("--antigravity-dir", default=None, help="Percorso alternativo ~/.gemini/antigravity")
    p_sync_anti.add_argument("--state-file", default=None, help="Percorso file di stato tracking")
    p_sync_anti.add_argument("--dry-run", action="store_true", help="Mostra solo le sessioni da sincronizzare senza inviarle")
    p_sync_anti.set_defaults(func=cmd_sync_antigravity)

    # enrich (Issue #2)
    p_enrich = subparsers.add_parser("enrich", parents=[base_parser], help="Pipeline batch di estrazione e arricchimento conoscenza")
    p_enrich.add_argument("--limit", type=int, default=None, help="Numero massimo di conversazioni da elaborare")
    p_enrich.add_argument("--provider", default=None, choices=["auto", "anthropic", "openai", "gemini", "grok", "deepseek", "qwen"], help="Provider LLM da usare")
    p_enrich.add_argument("--model", default=None, help="Modello LLM specifico per estrazione")
    p_enrich.add_argument("--rebuild", action="store_true", help="Azzera e ri-estrae da zero la conoscenza su tutte le conversazioni")
    p_enrich.add_argument("--delay", type=float, default=0.1, help="Ritardo in secondi tra le chiamate LLM")
    p_enrich.set_defaults(func=cmd_enrich)

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
    db_path = getattr(args, "db", str(DEFAULT_DB_PATH))
    db = Database(db_path)
    args.func(args, db)


if __name__ == "__main__":
    main()
