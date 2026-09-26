#!/usr/bin/env python3
"""
Re-clustering Script for Entities in vault.db.
Migrates entities categorized as 'general' to specific taxonomy categories
using:
1. Fast Keyword Heuristics (zero latency, zero cost)
2. Batch LLM Classification with index-based JSON payload (fast & token-efficient)
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memoria.db.engine import Database
from memoria.knowledge.extractor import guess_category
from memoria.llm.client import parse_llm_json
from memoria.settings.manager import SettingsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recluster")

TAXONOMY_DESCRIPTION = """
Categorie ammesse e descrizioni:
- audiocassette: cassette a nastro, formulazioni, nastri magnetici (TDK, Maxell, Type I, II, IV, Avilyn, ecc.)
- audio_interface: schede audio USB/Thunderbolt esterne (Audient, Scarlett, Focusrite, Zen Go, Motu, RME, ecc.)
- hardware_deck: piastre di registrazione a cassette, tape deck, Walkman, componenti meccanici (capstan, cinghie, ecc.)
- audio_pro_studio: microfoni da studio, mixer audio, monitor, preamplificatori, compressori, outboard, cuffie monitor
- ham_radio: apparati radioamatoriali, ricetrasmettitori, antenne, balun, dipoli, accordatori, bande HF/VHF/UHF, nominativi (Yaesu, Icom, Kenwood, Baofeng, ecc.)
- homelab_infra: server domestici, Proxmox, hypervisor, Docker, container, reverse proxy, reti, switch, router, NAS, storage, RAID, ZFS, Wireguard, Tailscale
- hardware_pc: componenti hardware per PC/computer (CPU, GPU, RAM, SSD, schede madri, case, alimentatori, Ryzen, RTX, ThinkPad, ecc.)
- software: linguaggi di programmazione, librerie, framework web, tool di sviluppo, script, database, sistemi operativi Linux (Python, SQLite, FastAPI, Git, ecc.)
- project: progetti personali e software proprietari di Gas (Livellatore, Quattrotracce, OpenStoryBox, CozyTabs, Digestly, Eggonomics, PingSentry, EngramVault, Memoria, CheckDNS, HyphenCheck, TruckSaveKit)
- cooking: cucina, ricette, lieviti, farine, impasti, pizza, tecniche culinarie
- personal_life: vita privata, viaggi, vacanze, luoghi, salute, famiglia, impegni quotidiani
- general: concetti generici, parole comuni, aggettivi o termini astratti che non appartengono a nessuna categoria sopra.

Se un'entità appartiene a un altro dominio specialistico ben definito e ricorrente, puoi usare una nuova categoria sintetica in snake_case (es. 'fotografia', 'giardinaggio').
"""


def get_category_counts(db: Database) -> Dict[str, int]:
    with db.get_connection() as conn:
        cur = conn.execute(
            "SELECT COALESCE(category, 'general') AS cat, COUNT(*) AS c FROM entities GROUP BY cat ORDER BY c DESC"
        )
        return {r["cat"]: r["c"] for r in cur.fetchall()}


def sanitize_cat(cat_str: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (cat_str or "general").strip().lower()).strip("_")
    return cleaned or "general"


def run_heuristics_stage(db: Database, dry_run: bool = False) -> Tuple[int, Dict[str, int], set]:
    logger.info("--- Fase 1: Fast Keyword Heuristics ---")
    with db.get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name FROM entities WHERE category = 'general' OR category IS NULL OR category = ''"
        )
        candidates = [dict(r) for r in cur.fetchall()]

    logger.info(f"Entità candidate per l'euristica: {len(candidates)}")
    updates = []
    cat_stats: Dict[str, int] = {}
    reclassified_ids = set()

    for row in candidates:
        guessed = guess_category(row["name"])
        if guessed != "general":
            updates.append((guessed, row["id"]))
            cat_stats[guessed] = cat_stats.get(guessed, 0) + 1
            reclassified_ids.add(row["id"])

    logger.info(f"Entità riclassificate con euristica keyword: {len(updates)}")
    for cat, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  + {cat}: {count}")

    if not dry_run and updates:
        with db.get_connection() as conn:
            conn.executemany("UPDATE entities SET category = ? WHERE id = ?", updates)
            conn.commit()
        logger.info("Salvataggio Fase 1 completato nel database.")

    return len(updates), cat_stats, reclassified_ids


def run_llm_stage(
    db: Database,
    batch_size: int = 50,
    limit: int = 0,
    dry_run: bool = False,
    delay_between_batches: float = 0.2,
    exclude_ids: set = None,
) -> Tuple[int, Dict[str, int]]:
    logger.info(f"--- Fase 2: LLM Batch Classification (batch size={batch_size}) ---")
    settings = SettingsManager(db)
    llm = settings.get_llm_client()

    if not llm.is_configured():
        logger.warning("Nessun LLM configurato in SettingsManager! Salto Fase 2.")
        return 0, {}

    logger.info(f"Modello LLM attivo: {llm.model}")

    with db.get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name FROM entities WHERE category = 'general' OR category IS NULL OR category = '' ORDER BY id ASC"
        )
        all_candidates = [dict(r) for r in cur.fetchall()]

    if exclude_ids:
        candidates = [c for c in all_candidates if c["id"] not in exclude_ids]
    else:
        candidates = all_candidates

    if limit > 0:
        candidates = candidates[:limit]

    total_candidates = len(candidates)
    logger.info(f"Entità candidate da classificare con LLM: {total_candidates}")
    if total_candidates == 0:
        return 0, {}

    system_prompt = (
        "Sei un tassonomista esperto di conoscenza personale per uno sviluppatore, audiofilo e appassionato di homelab.\n"
        "Il tuo compito è classificare un elenco indicizzato di entità nella categoria più appropriata.\n"
        f"{TAXONOMY_DESCRIPTION}\n"
        "Regole Tassative:\n"
        "1. Rispondi ESCLUSIVAMENTE con un oggetto JSON valido che mappa ciascun indice numerico alla categoria scelta.\n"
        "2. Formato: {\"0\": \"categoria\", \"1\": \"categoria\", ...}\n"
        "3. Usa 'general' solo ed esclusivamente se l'entità è un concetto astratto o irrilevante.\n"
        "4. Non aggiungere spiegazioni o testo markdown prima o dopo il JSON."
    )

    total_reclassified = 0
    cat_stats: Dict[str, int] = {}
    batches = [candidates[i : i + batch_size] for i in range(0, total_candidates, batch_size)]
    total_batches = len(batches)

    for b_idx, batch in enumerate(batches, 1):
        batch_map = {str(i): item["name"] for i, item in enumerate(batch)}
        user_prompt = f"Classifica queste entità:\n{json.dumps(batch_map, ensure_ascii=False, indent=1)}"

        # Attempt with 1 retry on error
        parsed = None
        elapsed = 0.0
        for attempt in range(2):
            try:
                t0 = time.time()
                resp = llm.chat_completion(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
                elapsed = time.time() - t0
                parsed = parse_llm_json(resp)
                if parsed and isinstance(parsed, dict):
                    break
                logger.warning(f"[Batch {b_idx}/{total_batches}] Parse JSON fallito (tentativo {attempt+1}/2)")
            except Exception as exc:
                logger.warning(f"[Batch {b_idx}/{total_batches}] Errore chiamata LLM (tentativo {attempt+1}/2): {exc}")
            time.sleep(1.0)

        if not parsed or not isinstance(parsed, dict):
            logger.error(f"[Batch {b_idx}/{total_batches}] Impossibile processare il batch dopo i tentativi.")
            continue

        batch_updates = []
        for str_idx, cat in parsed.items():
            try:
                num_idx = int(str_idx)
                if 0 <= num_idx < len(batch):
                    target_item = batch[num_idx]
                    cleaned_cat = sanitize_cat(cat)
                    if cleaned_cat != "general":
                        batch_updates.append((cleaned_cat, target_item["id"]))
                        cat_stats[cleaned_cat] = cat_stats.get(cleaned_cat, 0) + 1
            except (ValueError, TypeError):
                continue

        if not dry_run and batch_updates:
            with db.get_connection() as conn:
                conn.executemany("UPDATE entities SET category = ? WHERE id = ?", batch_updates)
                conn.commit()

        total_reclassified += len(batch_updates)
        logger.info(
            f"[Batch {b_idx}/{total_batches}] {len(batch_updates)}/{len(batch)} riclassificate in {elapsed:.2f}s "
            f"(Totale aggiornato: {total_reclassified})"
        )

        if delay_between_batches > 0:
            time.sleep(delay_between_batches)

    return total_reclassified, cat_stats


def main():
    parser = argparse.ArgumentParser(description="Re-cluster entities in vault.db")
    parser.add_argument("--db-path", default="vault.db", help="Path to SQLite database file")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for LLM classification")
    parser.add_argument("--limit", type=int, default=0, help="Limit total entities to process with LLM (0 = all)")
    parser.add_argument("--only-heuristics", action="store_true", help="Run only fast keyword heuristics stage")
    parser.add_argument("--dry-run", action="store_true", help="Do not write updates to database")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between LLM calls")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)

    logger.info(f"Connessione al database: {db_path}")
    db = Database(db_path)

    initial_counts = get_category_counts(db)
    total_entities = sum(initial_counts.values())
    logger.info(f"Totale entità nel database: {total_entities}")
    logger.info("Distribuzione iniziale per categoria:")
    for cat, count in sorted(initial_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_entities * 100) if total_entities else 0
        logger.info(f"  - {cat:20}: {count:6d} ({pct:5.1f}%)")

    # Stage 1: Heuristics
    h_count, h_stats, h_ids = run_heuristics_stage(db, dry_run=args.dry_run)

    # Stage 2: LLM
    llm_count = 0
    llm_stats = {}
    if not args.only_heuristics:
        llm_count, llm_stats = run_llm_stage(
            db,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            delay_between_batches=args.delay,
            exclude_ids=h_ids if args.dry_run else None,
        )

    # Final Report
    logger.info("=========================================")
    logger.info("           REPORT FINALE                 ")
    logger.info("=========================================")
    logger.info(f"Riclassificate con Heuristics: {h_count}")
    logger.info(f"Riclassificate con LLM:        {llm_count}")
    logger.info(f"Totale riclassificate:         {h_count + llm_count}")
    logger.info("-----------------------------------------")

    if args.dry_run:
        # Simulate final counts
        sim_counts = dict(initial_counts)
        total_migrated = h_count + llm_count
        sim_counts["general"] = max(0, sim_counts.get("general", 0) - total_migrated)
        for cat, cnt in {**h_stats, **llm_stats}.items():
            sim_counts[cat] = sim_counts.get(cat, 0) + cnt
        final_counts = sim_counts
        logger.info("Nuova distribuzione categorie (SIMULATA - DRY RUN):")
    else:
        final_counts = get_category_counts(db)
        logger.info("Nuova distribuzione categorie (EFFETTIVA):")

    for cat, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_entities * 100) if total_entities else 0
        init = initial_counts.get(cat, 0)
        diff = count - init
        diff_str = f"(+{diff})" if diff > 0 else (f"({diff})" if diff < 0 else "")
        logger.info(f"  - {cat:20}: {count:6d} ({pct:5.1f}%) {diff_str}")
    logger.info("=========================================")


if __name__ == "__main__":
    main()
