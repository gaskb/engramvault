import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from memoria.config import DEFAULT_DB_PATH, DEFAULT_INCOMING_DIR, WATCH_INTERVAL, WATCH_ENABLED
from memoria.db.engine import Database
from memoria.db.models import DecisionRecord, StatementRecord
from memoria.knowledge.enricher import KnowledgeEnricher
from memoria.knowledge.extractor import guess_category
from memoria.knowledge.query_planner import QueryPlanner
from memoria.ingest.pipeline import ingest_source
from memoria.ingest.watcher import DropFolderWatcher
from memoria.seed.seed_data import seed_database

logger = logging.getLogger("memoria.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"


# Pydantic Schemas for Requests
class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"


class CreateStatementRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    statement_type: str = "FACT"
    notes: Optional[str] = None
    entity_category: Optional[str] = None


class UpdateStatementRequest(BaseModel):
    statement_type: Optional[str] = None
    is_current: Optional[int] = None
    notes: Optional[str] = None


class CreateDecisionRequest(BaseModel):
    topic: str
    decision_text: str
    status: str = "ADOPTED"
    decided_at: Optional[str] = None


class UpdateDecisionRequest(BaseModel):
    status: Optional[str] = None
    decision_text: Optional[str] = None


class UpdateEntityRequest(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None


class EnrichStartRequest(BaseModel):
    limit: Optional[int] = None
    rebuild: bool = False
    delay: float = 0.0


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> FastAPI:
    db = Database(db_path)
    planner = QueryPlanner(db)
    watcher = DropFolderWatcher(db, incoming_dir=DEFAULT_INCOMING_DIR, poll_interval=WATCH_INTERVAL)
    enricher = KnowledgeEnricher(db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop_event = asyncio.Event()

        async def _watch_loop():
            while not stop_event.is_set():
                try:
                    await asyncio.to_thread(watcher.scan_once)
                except Exception as e:
                    logger.error(f"Errore watcher drop folder: {e}")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=WATCH_INTERVAL)
                except asyncio.TimeoutError:
                    pass

        watch_task = None
        if WATCH_ENABLED:
            watch_task = asyncio.create_task(_watch_loop())
            logger.info(f"Watcher drop folder attivato su: {DEFAULT_INCOMING_DIR}")

        yield

        if watch_task:
            stop_event.set()
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
        enricher.stop()

    app = FastAPI(
        title="Memoria AI Anti",
        description="Personal AI Conversation Vault & Knowledge Layer",
        lifespan=lifespan,
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        stats = db.get_stats()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"stats": stats, "db_name": db.db_path.name},
        )

    @app.get("/api/stats")
    async def get_stats():
        return db.get_stats()

    @app.post("/api/query")
    async def run_query(payload: QueryRequest):
        return planner.route_query(payload.query, mode=payload.mode)

    @app.get("/api/conversations")
    async def list_conversations(provider: Optional[str] = None, limit: int = 50, offset: int = 0):
        return db.list_conversations(limit=limit, offset=offset, provider=provider)

    @app.get("/api/conversations/{conv_id}")
    async def get_conversation(conv_id: str):
        c = db.get_conversation(conv_id)
        if not c:
            raise HTTPException(status_code=404, detail="Conversazione non trovata")
        return c

    @app.get("/api/entities")
    async def get_entities(category: Optional[str] = None):
        entities = db.get_entities(category=category)
        for e in entities:
            e["statements"] = db.get_statements(entity_id=e["id"])
        return entities

    @app.get("/api/decisions")
    async def get_decisions(status: Optional[str] = None):
        return db.get_decisions(status=status)

    @app.get("/api/timeline")
    async def get_timeline(subject: Optional[str] = None):
        return db.get_timeline(subject=subject)

    @app.post("/api/seed")
    async def seed_data():
        count = seed_database(db)
        return {"status": "success", "seeded_count": count, "stats": db.get_stats()}

    @app.post("/api/ingest")
    async def upload_ingest(file: UploadFile = File(...), provider: str = Form("auto")):
        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            summary = ingest_source(tmp_path, db, provider=provider)
            summary["filename"] = file.filename
            summary["stats"] = db.get_stats()
            return summary
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @app.post("/api/ingest/scan")
    async def trigger_drop_folder_scan():
        results = await asyncio.to_thread(watcher.scan_once)
        return {
            "status": "success",
            "incoming_dir": str(watcher.incoming_dir),
            "processed": results,
            "stats": db.get_stats(),
        }

    # --- Curation Endpoints (Issue #5) ---

    @app.post("/api/statements")
    async def create_statement(payload: CreateStatementRequest):
        cat = payload.entity_category or guess_category(payload.subject)
        ent_id = db.upsert_entity(payload.subject, cat)
        now_iso = "now"
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

    @app.patch("/api/statements/{stmt_id}")
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

    @app.delete("/api/statements/{stmt_id}")
    async def delete_statement(stmt_id: str):
        ok = db.delete_statement(stmt_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Statement non trovato")
        return {"status": "success", "deleted_id": stmt_id, "stats": db.get_stats()}

    @app.post("/api/decisions")
    async def create_decision(payload: CreateDecisionRequest):
        status_val = payload.status.upper()
        if status_val not in ("EVALUATING", "ADOPTED", "ABANDONED", "SUPERSEDED"):
            status_val = "ADOPTED"
        dec = DecisionRecord(
            id=str(uuid.uuid4()),
            topic=payload.topic.strip(),
            decision_text=payload.decision_text.strip(),
            status=status_val,
            decided_at=payload.decided_at or "now",
        )
        dec_id = db.insert_decision(dec)
        return {"status": "success", "decision_id": dec_id, "stats": db.get_stats()}

    @app.patch("/api/decisions/{dec_id}")
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

    @app.delete("/api/decisions/{dec_id}")
    async def delete_decision(dec_id: str):
        ok = db.delete_decision(dec_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Decisione non trovata")
        return {"status": "success", "deleted_id": dec_id, "stats": db.get_stats()}

    @app.patch("/api/entities/{entity_id}")
    async def update_entity(entity_id: str, payload: UpdateEntityRequest):
        ok = db.update_entity(entity_id=entity_id, name=payload.name, category=payload.category)
        if not ok:
            raise HTTPException(status_code=404, detail="Entità non trovata")
        return {"status": "success", "entity_id": entity_id}

    @app.delete("/api/entities/{entity_id}")
    async def delete_entity(entity_id: str):
        ok = db.delete_entity(entity_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Entità non trovata")
        return {"status": "success", "deleted_id": entity_id, "stats": db.get_stats()}

    # --- Batch Enrich Endpoints (Issue #2) ---

    @app.post("/api/enrich/start")
    async def start_enrich(payload: EnrichStartRequest):
        if enricher.is_running:
            return {"status": "already_running", "enrich_status": enricher.get_status()}
        enricher.start_background(limit=payload.limit, rebuild=payload.rebuild, delay=payload.delay)
        return {"status": "started", "enrich_status": enricher.get_status()}

    @app.get("/api/enrich/status")
    async def get_enrich_status():
        return enricher.get_status()

    @app.post("/api/enrich/stop")
    async def stop_enrich():
        enricher.stop()
        return {"status": "stopped", "enrich_status": enricher.get_status()}

    return app
