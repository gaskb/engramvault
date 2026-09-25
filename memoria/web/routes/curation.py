"""
Curation, Ingestion, Seed & Knowledge Enrichment Routes (Issue #26 / Q-01, S-05, R-05).
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from memoria.db.engine import Database, utc_now_iso
from memoria.db.models import DecisionRecord, StatementRecord
from memoria.ingest.pipeline import ingest_source
from memoria.knowledge.enricher import KnowledgeEnricher
from memoria.knowledge.extractor import guess_category
from memoria.seed.seed_data import seed_database
from memoria.web.routes.security import require_mutation_auth


class CreateStatementRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=512)
    predicate: str = Field(..., min_length=1, max_length=256)
    object: str = Field(..., min_length=1, max_length=4000)
    statement_type: str = Field(default="FACT", max_length=32)
    notes: Optional[str] = Field(default=None, max_length=2000)
    entity_category: Optional[str] = Field(default=None, max_length=128)


class UpdateStatementRequest(BaseModel):
    statement_type: Optional[str] = Field(default=None, max_length=32)
    is_current: Optional[int] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


class CreateDecisionRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=512)
    decision_text: str = Field(..., min_length=1, max_length=4000)
    status: str = Field(default="ADOPTED", max_length=32)
    decided_at: Optional[str] = Field(default=None, max_length=64)


class UpdateDecisionRequest(BaseModel):
    status: Optional[str] = Field(default=None, max_length=32)
    decision_text: Optional[str] = Field(default=None, max_length=4000)


class UpdateEntityRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=512)
    category: Optional[str] = Field(default=None, max_length=128)


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

    @router.get("/api/decisions")
    async def get_decisions(status: Optional[str] = None):
        return db.get_decisions(status=status)

    @router.get("/api/timeline")
    async def get_timeline(subject: Optional[str] = None):
        return db.get_timeline(subject=subject)

    @router.post("/api/seed", dependencies=[Depends(require_mutation_auth)])
    async def seed_data():
        count = seed_database(db)
        return {"status": "success", "seeded_conversations": count, "stats": db.get_stats()}

    @router.post("/api/ingest", dependencies=[Depends(require_mutation_auth)])
    async def ingest_file(
        file: UploadFile = File(...),
        provider: str = Form("auto"),
    ):
        suffix = "".join(Path(file.filename or "upload.json").suffixes) or ".json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            summary = ingest_source(tmp_path, db, provider=provider)
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "status": "success",
            "filename": file.filename,
            **summary,
            "stats": db.get_stats(),
        }

    @router.post("/api/statements", dependencies=[Depends(require_mutation_auth)])
    async def create_statement(payload: CreateStatementRequest):
        cat = payload.entity_category or guess_category(payload.subject)
        ent_id = db.upsert_entity(payload.subject, cat)
        now_iso = utc_now_iso()
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
        )
        stmt_id = db.insert_statement(stmt)
        return {"status": "success", "statement_id": stmt_id, "stats": db.get_stats()}

    @router.patch("/api/statements/{stmt_id}", dependencies=[Depends(require_mutation_auth)])
    async def update_statement(stmt_id: str, payload: UpdateStatementRequest):
        ok = db.update_statement(
            stmt_id=stmt_id,
            statement_type=payload.statement_type.upper() if payload.statement_type else None,
            is_current=payload.is_current,
            notes=payload.notes,
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
