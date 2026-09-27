#!/usr/bin/env python3
"""
Knowledge Graph Relations Extractor for vault.db.
Extracts semantic, functional, and physical relations between entities
across conversations, facts, and claims:
- Stage 1: Deterministic Heuristic Extraction from existing facts and structural links
- Stage 2: LLM-powered Multi-Hop Relation Extraction from high-density conversations
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memoria.db.engine import Database
from memoria.db.models import RelationRecord
from memoria.knowledge.extractor import guess_category
from memoria.llm.client import parse_llm_json
from memoria.settings.manager import SettingsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("relations_extractor")

ALLOWED_RELATIONS = [
    "connected_to",
    "runs_on",
    "part_of",
    "depends_on",
    "compatible_with",
    "subproject_of",
    "uses",
    "created_by",
    "configured_with",
    "installed_on",
]


def sanitize_relation_type(rel_type: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (rel_type or "relates_to").strip().lower()).strip("_")
    return cleaned or "relates_to"


def run_heuristic_relations(db: Database, dry_run: bool = False) -> int:
    logger.info("--- Fase 1: Estrazione Euristica delle Relazioni ---")
    created = 0

    with db.get_connection() as conn:
        # 1. Statements with explicit relational predicates
        cur = conn.execute(
            """
            SELECT s.id, s.subject, s.predicate, s.object, s.source_message_id, e.category AS subject_category
            FROM statements s
            JOIN entities e ON s.entity_id = e.id
            WHERE s.predicate IN ('installed_on', 'runs_on', 'connected_to', 'part_of', 'depends_on', 'uses', 'configured_with', 'subproject_of')
            """
        )
        explicit_stmts = [dict(r) for r in cur.fetchall()]

    logger.info(f"Dichiarazioni con predicati relazionali espliciti: {len(explicit_stmts)}")
    for s in explicit_stmts:
        subj = s["subject"].strip()
        obj = s["object"].strip()
        pred = sanitize_relation_type(s["predicate"])
        if len(obj) < 80 and not obj.lower().startswith(("http", "{", "[")) and subj.lower() != obj.lower():
            if not dry_run:
                f_id = db.upsert_entity(subj, s["subject_category"])
                t_id = db.upsert_entity(obj, guess_category(obj))
                rel = RelationRecord(
                    id=str(uuid.uuid4()),
                    from_entity_id=f_id,
                    to_entity_id=t_id,
                    relation_type=pred,
                    source_message_id=s.get("source_message_id"),
                )
                db.insert_relation(rel)
            created += 1

    # 2. Structural pairings: Projects -> Software/Infra dependencies
    with db.get_connection() as conn:
        cur = conn.execute(
            """
            SELECT c.id AS conv_id, GROUP_CONCAT(DISTINCT e.name) AS entity_names
            FROM conversations c
            JOIN statements s ON s.source_conversation_id = c.id
            JOIN entities e ON s.entity_id = e.id
            GROUP BY c.id
            HAVING COUNT(DISTINCT e.id) >= 2
            """
        )
        conv_clusters = [dict(r) for r in cur.fetchall()]

    logger.info(f"Conversazioni con cluster di entità correlate: {len(conv_clusters)}")
    logger.info(f"Relazioni create in Fase 1: {created}")
    return created


def run_llm_relations(
    db: Database,
    limit_convs: int = 50,
    dry_run: bool = False,
    delay: float = 0.2,
) -> int:
    logger.info(f"--- Fase 2: Estrazione Relazioni con LLM (limite={limit_convs} convs) ---")
    settings = SettingsManager(db)
    llm = settings.get_llm_client()
    if not llm.is_configured():
        logger.warning("Nessun LLM configurato. Salto Fase 2.")
        return 0

    logger.info(f"Modello LLM attivo: {llm.model}")

    with db.get_connection() as conn:
        cur = conn.execute(
            """
            SELECT c.id, c.title, c.provider, c.created_at,
                   COUNT(DISTINCT s.id) AS stmts_count,
                   GROUP_CONCAT(DISTINCT e.name) AS entities_list
            FROM conversations c
            JOIN statements s ON s.source_conversation_id = c.id
            JOIN entities e ON s.entity_id = e.id
            GROUP BY c.id
            HAVING COUNT(DISTINCT e.id) >= 2
            ORDER BY stmts_count DESC
            LIMIT ?
            """,
            (limit_convs,),
        )
        target_convs = [dict(r) for r in cur.fetchall()]

    logger.info(f"Conversazioni ad alta densità selezionate per analisi relazioni: {len(target_convs)}")
    total_extracted = 0

    system_prompt = (
        "Sei un Knowledge Graph Engineer esperto.\n"
        "Analizza il dialogo e l'elenco delle entità menzionate ed estrai TUTTE le relazioni dirette, certe e strutturate.\n"
        "Usa prioritariamente questi tipi di relazione:\n"
        "- connected_to: collegamento fisico/audio/RF tra due apparati o componenti (es. radio -> antenna, interfaccia -> microfono)\n"
        "- runs_on: applicazione o servizio che viene eseguito su un host, hypervisor o container (es. engramvault -> docker, vm -> proxmox)\n"
        "- part_of: componente che fa parte di un insieme più grande o modulo di un sistema\n"
        "- depends_on: dipendenza architetturale o software (es. fastapi -> python, servizio -> postgres)\n"
        "- compatible_with: compatibilità tra due formati o apparati (es. deck -> audiocassetta type ii)\n"
        "- subproject_of: progetto secondario che fa parte di un progetto ombrello\n"
        "- uses: utilizzo esplicito e verificato di uno strumento o libreria\n"
        "- created_by: progetto sviluppato dall'utente\n\n"
        "Rispondi ESCLUSIVAMENTE con un JSON con questo schema esatto:\n"
        "{\n"
        "  \"relations\": [\n"
        "    {\"from_entity\": \"NomeEntita1\", \"relation\": \"connected_to|runs_on|...\", \"to_entity\": \"NomeEntita2\"}\n"
        "  ]\n"
        "}\n"
        "Se non trovi relazioni certe, restituisci {\"relations\": []}."
    )

    for idx, c in enumerate(target_convs, 1):
        cid = c["id"]
        title = c.get("title", "")
        entities_hint = c.get("entities_list", "")

        conv_record = db.get_conversation(cid)
        if not conv_record or not conv_record.get("messages"):
            continue

        dialogue_text = "\n\n".join(
            [f"{m['role'].upper()}: {(m.get('content') or '').strip()}" for m in conv_record["messages"][:12]]
        )[:4000]

        user_prompt = (
            f"TITOLO: {title}\n"
            f"ENTITÀ IDENTIFICATE NEL TESTO: {entities_hint}\n\n"
            f"DIALOGO:\n{dialogue_text}"
        )

        try:
            t0 = time.time()
            resp = llm.chat_completion(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=1000,
            )
            elapsed = time.time() - t0
            parsed = parse_llm_json(resp)
            rels = parsed.get("relations", []) if parsed and isinstance(parsed, dict) else []

            batch_count = 0
            for r in rels:
                f_name = (r.get("from_entity") or "").strip()
                t_name = (r.get("to_entity") or "").strip()
                rel_type = sanitize_relation_type(r.get("relation") or "relates_to")

                if f_name and t_name and f_name.lower() != t_name.lower():
                    if not dry_run:
                        f_id = db.upsert_entity(f_name, guess_category(f_name))
                        t_id = db.upsert_entity(t_name, guess_category(t_name))
                        rel_rec = RelationRecord(
                            id=str(uuid.uuid4()),
                            from_entity_id=f_id,
                            to_entity_id=t_id,
                            relation_type=rel_type,
                            source_message_id=conv_record["messages"][0]["id"] if conv_record["messages"] else None,
                        )
                        db.insert_relation(rel_rec)
                    batch_count += 1
                    total_extracted += 1

            logger.info(
                f"[{idx}/{len(target_convs)}] '{title[:35]}' -> {batch_count} relazioni estratte in {elapsed:.2f}s "
                f"(Totale: {total_extracted})"
            )

        except Exception as exc:
            logger.error(f"[{idx}/{len(target_convs)}] Errore estrazione su '{title[:35]}': {exc}")

        if delay > 0:
            time.sleep(delay)

    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract entity relations for Knowledge Graph in vault.db")
    parser.add_argument("--db-path", default="vault.db", help="Path to SQLite database file")
    parser.add_argument("--limit-convs", type=int, default=60, help="Number of dense conversations to analyze with LLM")
    parser.add_argument("--only-heuristics", action="store_true", help="Run only heuristic extraction")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing to database")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between LLM calls")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Connessione al database: {db_path}")
    db = Database(db_path)

    stats_before = db.get_stats()
    logger.info(f"Relazioni presenti prima dell'esecuzione: {stats_before.get('relations', 0)}")

    h_count = run_heuristic_relations(db, dry_run=args.dry_run)
    llm_count = 0
    if not args.only_heuristics:
        llm_count = run_llm_relations(db, limit_convs=args.limit_convs, dry_run=args.dry_run, delay=args.delay)

    stats_after = db.get_stats()
    logger.info("=========================================")
    logger.info("       REPORT ESTRAZIONE RELAZIONI       ")
    logger.info("=========================================")
    logger.info(f"Relazioni estratte via Euristica: {h_count}")
    logger.info(f"Relazioni estratte via LLM:       {llm_count}")
    logger.info(f"Totale relazioni nel Knowledge Graph: {stats_after.get('relations', 0)}")
    logger.info("=========================================")


if __name__ == "__main__":
    main()
