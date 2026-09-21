import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from memoria.config import DEFAULT_DB_PATH, DEFAULT_INCOMING_DIR, WATCH_INTERVAL, WATCH_ENABLED
from memoria.db.engine import Database
from memoria.knowledge.query_planner import QueryPlanner
from memoria.ingest.pipeline import ingest_source
from memoria.ingest.watcher import DropFolderWatcher
from memoria.seed.seed_data import seed_database

logger = logging.getLogger("memoria.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"


class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> FastAPI:
    db = Database(db_path)
    planner = QueryPlanner(db)
    watcher = DropFolderWatcher(db, incoming_dir=DEFAULT_INCOMING_DIR, poll_interval=WATCH_INTERVAL)

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

    app = FastAPI(
        title="Memoria AI Anti",
        description="Personal AI Conversation Vault",
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
        res = planner.route_query(payload.query, mode=payload.mode)
        return res

    @app.get("/api/conversations")
    async def list_conversations(provider: Optional[str] = None, limit: int = 50, offset: int = 0):
        return db.list_conversations(limit=limit, offset=offset, provider=provider)

    @app.get("/api/conversations/{conv_id}")
    async def get_conversation(conv_id: str):
        c = db.get_conversation(conv_id)
        if not c:
            return JSONResponse({"error": "Conversazione non trovata"}, status_code=404)
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
        # Save uploaded file to temp directory
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
        """Immediately scans the incoming drop folder."""
        results = await asyncio.to_thread(watcher.scan_once)
        return {
            "status": "success",
            "incoming_dir": str(watcher.incoming_dir),
            "processed": results,
            "stats": db.get_stats(),
        }

    return app
