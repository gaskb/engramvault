"""
Curation, Ingestion, Seed & Knowledge Enrichment Routes (Issue #26 / Q-01, S-05, R-05).
"""

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from engramvault.config import MAX_UPLOAD_MB
from engramvault.db.engine import Database, utc_now_iso
from engramvault.db.models import DecisionRecord, StatementRecord
from engramvault.ingest.pipeline import ingest_source
from engramvault.knowledge.contradiction import detect_contradiction
from engramvault.knowledge.enricher import KnowledgeEnricher
from engramvault.knowledge.extractor import guess_category
from engramvault.seed.seed_data import seed_database
from engramvault.web.routes.security import require_mutation_auth

ALLOWED_UPLOAD_EXTENSIONS = {".json", ".jsonl", ".zip", ".md", ".markdown", ".txt"}


class CreateStatementRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=512)
    predicate: str = Field(..., min_length=1, max_length=256)
    object: str = Field(..., min_length=1, max_length=4000)
    statement_type: Literal["FACT", "CLAIM", "DECISION"] = Field(default="FACT")
    notes: Optional[str] = Field(default=None, max_length=2000)
    entity_category: Optional[str] = Field(default=None, max_length=128)


class CheckConflictRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=512)
    predicate: str = Field(..., min_length=1, max_length=256)
    object: str = Field(..., min_length=1, max_length=4000)
    statement_type: Literal["FACT", "CLAIM", "DECISION"] = Field(default="CLAIM")
    entity_category: Optional[str] = Field(default=None, max_length=128)


class UpdateStatementRequest(BaseModel):
    statement_type: Optional[Literal["FACT", "CLAIM", "DECISION"]] = None
    is_current: Optional[int] = None
    notes: Optional[str] = Field(default=None, max_length=2000)
    conflict_json: Optional[str] = None


class CreateDecisionRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=512)
    decision_text: str = Field(..., min_length=1, max_length=4000)
    status: Literal["ADOPTED", "EVALUATING", "SUPERSEDED", "ABANDONED"] = Field(default="ADOPTED")
    decided_at: Optional[str] = Field(default=None, max_length=64)


class UpdateDecisionRequest(BaseModel):
    status: Optional[Literal["ADOPTED", "EVALUATING", "SUPERSEDED", "ABANDONED"]] = None
    decision_text: Optional[str] = Field(default=None, max_length=4000)


class UpdateEntityRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=512)
    category: Optional[str] = Field(default=None, max_length=128)


class CreateRelationRequest(BaseModel):
    from_entity: str = Field(..., min_length=1, max_length=512)
    to_entity: str = Field(..., min_length=1, max_length=512)
    relation_type: str = Field(default="relates_to", max_length=128)


class EnrichStartRequest(BaseModel):
    limit: Optional[int] = None
    rebuild: bool = False
    delay: float = 0.0


def build_curation_router(db: Database, enricher: KnowledgeEnricher) -> APIRouter:
    router = APIRouter()

    @router.get("/api/entities")
    async def get_entities(category: Optional[str] = None):
        entities = db.get_entities(category=category)
        for e in entities:
            e["statements"] = db.get_statements(entity_id=e["id"])
        return entities

    @router.get("/api/categories")
    async def get_categories():
        return db.get_entity_categories()

    @router.get("/api/decisions")
    async def get_decisions(
        status: Optional[str] = None,
        q: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: int = 0,
    ):
        search_q = (query or q or "").strip() or None
        eff_limit = None if limit is not None and limit <= 0 else limit
        return db.get_decisions(status=status, query=search_q, limit=eff_limit, offset=offset)

    @router.get("/api/timeline")
    async def get_timeline(
        subject: Optional[str] = None,
        q: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: int = 0,
    ):
        search_q = (query or q or subject or "").strip() or None
        eff_limit = None if limit is not None and limit <= 0 else limit
        return db.get_timeline(subject=search_q, query=search_q, limit=eff_limit, offset=offset)

    @router.post("/api/seed", dependencies=[Depends(require_mutation_auth)])
    async def seed_data():
        count = seed_database(db)
        return {"status": "success", "seeded_count": count, "seeded_conversations": count, "stats": db.get_stats()}

    @router.post("/api/ingest", dependencies=[Depends(require_mutation_auth)])
    async def ingest_file(
        file: UploadFile = File(...),
        provider: str = Form("auto"),
    ):
        raw_filename = file.filename or "upload.json"
        raw_suffix = Path(raw_filename).suffix.lower()
        if raw_suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato file non supportato ('{raw_suffix}'). Formati ammessi: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
            )

        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        total_bytes = 0
        suffix = "".join(Path(raw_filename).suffixes) or ".json"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            try:
                while True:
                    chunk = await file.read(64 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Dimensione file oltre il limite consentito ({MAX_UPLOAD_MB} MB).",
                        )
                    tmp.write(chunk)
                tmp.flush()
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

        try:
            summary = await asyncio.to_thread(ingest_source, tmp_path, db, provider=provider)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "status": "success",
            "filename": file.filename,
            **summary,
            "stats": db.get_stats(),
        }

    @router.post("/api/statements/check-conflict")
    async def check_statement_conflict(payload: CheckConflictRequest):
        cat = payload.entity_category or guess_category(payload.subject)
        report = detect_contradiction(
            db=db,
            subject=payload.subject,
            predicate=payload.predicate,
            object_value=payload.object,
            statement_type=payload.statement_type.upper(),
            entity_category=cat,
        )
        return {"status": "success", "report": report}

    @router.post("/api/statements", dependencies=[Depends(require_mutation_auth)])
    async def create_statement(payload: CreateStatementRequest):
        cat = payload.entity_category or guess_category(payload.subject)
        ent_id = db.upsert_entity(payload.subject, cat)
        now_iso = utc_now_iso()

        # Contradiction Detection check
        conflict = detect_contradiction(
            db=db,
            subject=payload.subject,
            predicate=payload.predicate,
            object_value=payload.object,
            statement_type=payload.statement_type.upper(),
            entity_category=cat,
        )
        import json
        conflict_json = json.dumps(conflict, ensure_ascii=False) if conflict.get("has_conflict") else None

        stmt = StatementRecord(
            id=str(uuid.uuid4()),
            entity_id=ent_id,
            statement_type=payload.statement_type.upper(),
            subject=payload.subject.strip(),
            predicate=payload.predicate.strip(),
            object=payload.object.strip(),
            author="user",
            confidence=1.0,
            valid_from=now_iso,
            is_current=1,
            notes=payload.notes or "Inserito manualmente dall'utente",
            conflict_json=conflict_json,
        )
        stmt_id = db.insert_statement(stmt)
        return {
            "status": "success",
            "statement_id": stmt_id,
            "conflict": conflict,
            "stats": db.get_stats(),
        }

    @router.patch("/api/statements/{stmt_id}", dependencies=[Depends(require_mutation_auth)])
    async def update_statement(stmt_id: str, payload: UpdateStatementRequest):
        ok = db.update_statement(
            stmt_id=stmt_id,
            statement_type=payload.statement_type.upper() if payload.statement_type else None,
            is_current=payload.is_current,
            notes=payload.notes,
            conflict_json=payload.conflict_json,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Statement non trovato o nessuna modifica")
        return {"status": "success", "statement_id": stmt_id}

    @router.delete("/api/statements/{stmt_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_statement(stmt_id: str):
        ok = db.delete_statement(stmt_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Statement non trovato")
        return {"status": "success", "deleted_id": stmt_id, "stats": db.get_stats()}

    @router.post("/api/decisions", dependencies=[Depends(require_mutation_auth)])
    async def create_decision(payload: CreateDecisionRequest):
        status_val = payload.status.upper()
        if status_val not in ("EVALUATING", "ADOPTED", "ABANDONED", "SUPERSEDED"):
            status_val = "ADOPTED"
        dec = DecisionRecord(
            id=str(uuid.uuid4()),
            topic=payload.topic.strip(),
            decision_text=payload.decision_text.strip(),
            status=status_val,
            decided_at=payload.decided_at or utc_now_iso(),
        )
        dec_id = db.insert_decision(dec)
        return {"status": "success", "decision_id": dec_id, "stats": db.get_stats()}

    @router.patch("/api/decisions/{dec_id}", dependencies=[Depends(require_mutation_auth)])
    async def update_decision(dec_id: str, payload: UpdateDecisionRequest):
        status_val = payload.status.upper() if payload.status else None
        ok = db.update_decision(
            dec_id=dec_id,
            status=status_val,
            decision_text=payload.decision_text,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Decisione non trovata")
        return {"status": "success", "decision_id": dec_id}

    @router.delete("/api/decisions/{dec_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_decision(dec_id: str):
        ok = db.delete_decision(dec_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Decisione non trovata")
        return {"status": "success", "deleted_id": dec_id, "stats": db.get_stats()}

    @router.patch("/api/entities/{entity_id}", dependencies=[Depends(require_mutation_auth)])
    async def update_entity(entity_id: str, payload: UpdateEntityRequest):
        ok = db.update_entity(
            entity_id=entity_id,
            name=payload.name,
            category=payload.category,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Entità non trovata")
        return {"status": "success", "entity_id": entity_id}

    @router.delete("/api/entities/{entity_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_entity(entity_id: str):
        ok = db.delete_entity(entity_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Entità non trovata")
        return {"status": "success", "deleted_id": entity_id, "stats": db.get_stats()}

    # --- Knowledge Graph Relations Endpoints (Issue #30) ---

    @router.get("/api/relations")
    async def list_relations(
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        return db.get_relations(entity_id=entity_id, relation_type=relation_type, limit=limit, offset=offset)

    @router.post("/api/relations", dependencies=[Depends(require_mutation_auth)])
    async def create_relation(payload: CreateRelationRequest):
        from_ent = payload.from_entity.strip()
        to_ent = payload.to_entity.strip()
        if not from_ent or not to_ent or from_ent.lower() == to_ent.lower():
            raise HTTPException(status_code=400, detail="from_entity e to_entity devono essere diverse e non vuote")
        f_cat = guess_category(from_ent)
        t_cat = guess_category(to_ent)
        f_id = db.upsert_entity(from_ent, f_cat)
        t_id = db.upsert_entity(to_ent, t_cat)
        from engramvault.db.models import RelationRecord

        rel_rec = RelationRecord(
            id=str(uuid.uuid4()),
            from_entity_id=f_id,
            to_entity_id=t_id,
            relation_type=payload.relation_type.strip(),
        )
        rel_id = db.insert_relation(rel_rec)
        return {"status": "success", "relation_id": rel_id}

    @router.delete("/api/relations/{rel_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_relation(rel_id: str):
        ok = db.delete_relation(rel_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Relazione non trovata")
        return {"status": "success", "deleted_id": rel_id, "stats": db.get_stats()}

    # --- Batch Enrichment API Endpoints (Issue #7) ---

    @router.get("/api/enrich/status")
    async def get_enrich_status():
        return enricher.get_status()

    @router.post("/api/enrich/start", dependencies=[Depends(require_mutation_auth)])
    async def start_enrichment(payload: EnrichStartRequest):
        if enricher.is_running:
            return {"status": "already_running", "enrich_status": enricher.get_status()}
        enricher.start_background(limit=payload.limit, rebuild=payload.rebuild, delay=payload.delay)
        return {"status": "started", "enrich_status": enricher.get_status()}

    @router.post("/api/enrich/stop", dependencies=[Depends(require_mutation_auth)])
    async def stop_enrichment():
        enricher.stop()
        return {"status": "stopped", "enrich_status": enricher.get_status()}

    return router
