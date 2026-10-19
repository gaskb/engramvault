"""
Contradiction Detection Engine for EngramVault (Issue #34).

Provides fast, lightweight (<5ms) conflict detection between incoming
claims/facts and existing active FACT statements or architectural DECISIONS.
"""

from typing import Any, Dict, List, Optional
from engramvault.db.engine import Database

TRUE_VALS = {"true", "1", "yes", "si", "sì", "attivo", "active", "enabled", "presente", "posseduto"}
FALSE_VALS = {"false", "0", "no", "disattivato", "spento", "venduto", "dismesso", "disabled", "assente"}

SINGLETON_PREDICATES = {
    "primary_db", "main_db", "database", "default_database",
    "version", "os", "operating_system", "architecture",
    "ip_address", "port", "primary_interface", "status",
    "model", "framework", "default_theme", "host",
}


def _is_bool_polarity_conflict(val1: str, val2: str) -> bool:
    v1 = val1.strip().lower()
    v2 = val2.strip().lower()
    return (v1 in TRUE_VALS and v2 in FALSE_VALS) or (v1 in FALSE_VALS and v2 in TRUE_VALS)


def detect_contradiction(
    db: Database,
    subject: str,
    predicate: str,
    object_value: str,
    statement_type: str = "CLAIM",
    entity_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyzes an incoming statement or claim against active FACTs and DECISIONs.
    Returns a structured conflict report:
    {
        "has_conflict": bool,
        "conflict_type": "CONTRADICTION" | "SUPERSEDES" | "NONE",
        "conflicting_id": Optional[str],
        "conflicting_type": Optional[str],  # "FACT" | "DECISION"
        "conflicting_summary": Optional[str],
        "explanation": str
    }
    """
    clean_subj = (subject or "").strip()
    clean_pred = (predicate or "").strip()
    clean_obj = (object_value or "").strip()

    if not clean_subj or not clean_obj:
        return {
            "has_conflict": False,
            "conflict_type": "NONE",
            "conflicting_id": None,
            "conflicting_type": None,
            "conflicting_summary": None,
            "explanation": "",
        }

    norm_subj = clean_subj.lower()
    norm_pred = clean_pred.lower()
    norm_obj = clean_obj.lower()

    # 1. Check against active FACT statements
    active_facts = db.get_statements(statement_type="FACT", is_current=1)
    for fact in active_facts:
        f_subj = (fact.get("subject") or "").strip().lower()
        f_pred = (fact.get("predicate") or "").strip().lower()
        f_obj = (fact.get("object") or "").strip()
        norm_f_obj = f_obj.lower()
        fact_id = fact.get("id")

        # Case A: Same subject and same predicate
        if f_subj == norm_subj and f_pred == norm_pred:
            # Check boolean polarity conflict
            if _is_bool_polarity_conflict(clean_obj, f_obj):
                return {
                    "has_conflict": True,
                    "conflict_type": "CONTRADICTION",
                    "conflicting_id": fact_id,
                    "conflicting_type": "FACT",
                    "conflicting_summary": f"FACT: {fact.get('subject')} -> {fact.get('predicate')}: {f_obj}",
                    "explanation": (
                        f"Polarità opposta su '{clean_subj}.{clean_pred}': "
                        f"esistente '{f_obj}' vs nuovo '{clean_obj}'."
                    ),
                }

            # Check value mismatch for singleton/incompatible attributes
            if norm_obj != norm_f_obj:
                # Check for update/migration intent
                is_migration = any(
                    kw in norm_obj or kw in clean_pred.lower()
                    for kw in ("migrat", "sostituit", "aggiornat", "nuov", "next", "v2", "v3", "upgrade")
                )
                conflict_type = "SUPERSEDES" if is_migration else "CONTRADICTION"
                return {
                    "has_conflict": True,
                    "conflict_type": conflict_type,
                    "conflicting_id": fact_id,
                    "conflicting_type": "FACT",
                    "conflicting_summary": f"FACT: {fact.get('subject')} -> {fact.get('predicate')}: {f_obj}",
                    "explanation": (
                        f"{'Aggiornamento/migrazione' if is_migration else 'Discrepanza di valore'} "
                        f"su '{clean_subj}.{clean_pred}': esistente '{f_obj}' vs nuovo '{clean_obj}'."
                    ),
                }

        # Case B: Cross-attribute conflict (e.g., subject matches singleton predicate in another fact)
        if norm_pred in SINGLETON_PREDICATES and f_pred == norm_pred and norm_obj != norm_f_obj:
            if f_subj == norm_subj:
                return {
                    "has_conflict": True,
                    "conflict_type": "CONTRADICTION",
                    "conflicting_id": fact_id,
                    "conflicting_type": "FACT",
                    "conflicting_summary": f"FACT: {fact.get('subject')} -> {fact.get('predicate')}: {f_obj}",
                    "explanation": (
                        f"Conflitto su attributo univoco '{clean_pred}': "
                        f"esistente '{f_obj}' vs nuovo '{clean_obj}'."
                    ),
                }

    # 2. Check against active architectural DECISION records
    active_decisions = db.get_decisions(limit=100)
    for dec in active_decisions:
        dec_status = (dec.get("status") or "").upper()
        if dec_status not in ("ADOPTED", "EVALUATING", "ABANDONED"):
            continue

        topic = (dec.get("topic") or "").lower()
        d_text = (dec.get("decision_text") or "").lower()
        dec_id = dec.get("id")

        # If decision abandoned this subject, but new statement asserts active adoption/ownership
        if dec_status == "ABANDONED":
            if (norm_subj in topic or norm_subj in d_text) and clean_obj.lower() in TRUE_VALS:
                return {
                    "has_conflict": True,
                    "conflict_type": "CONTRADICTION",
                    "conflicting_id": dec_id,
                    "conflicting_type": "DECISION",
                    "conflicting_summary": f"[{dec_status}] {dec.get('topic')}: {dec.get('decision_text')[:80]}",
                    "explanation": (
                        f"La decisione architetturale #{dec_id} ha contrassegnato '{dec.get('topic')}' come ABANDONED, "
                        f"in contrasto con l'asserzione attiva di '{clean_subj}'."
                    ),
                }

        # If decision adopted a competing choice for the same topic
        if dec_status in ("ADOPTED", "EVALUATING"):
            if norm_subj in topic or (norm_pred in topic and norm_subj in d_text):
                # Check if decision text explicitly excludes or contradicts this statement
                if any(neg in d_text for neg in (f"scartato {norm_subj}", f"non {norm_subj}", f"abbandonato {norm_subj}")):
                    return {
                        "has_conflict": True,
                        "conflict_type": "CONTRADICTION",
                        "conflicting_id": dec_id,
                        "conflicting_type": "DECISION",
                        "conflicting_summary": f"[{dec_status}] {dec.get('topic')}: {dec.get('decision_text')[:80]}",
                        "explanation": (
                            f"In contrasto con la decisione attiva #{dec_id} ({dec.get('topic')}): "
                            f"'{dec.get('decision_text')[:100]}'."
                        ),
                    }

    return {
        "has_conflict": False,
        "conflict_type": "NONE",
        "conflicting_id": None,
        "conflicting_type": None,
        "conflicting_summary": None,
        "explanation": "",
    }
