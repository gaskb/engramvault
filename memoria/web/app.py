import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from memoria.config import DEFAULT_DB_PATH
from memoria.db.engine import Database
from memoria.knowledge.query_planner import QueryPlanner
from memoria.adapters.registry import parse_source, detect_provider
from memoria.search.embeddings import EmbeddingEngine
from memoria.knowledge.extractor import KnowledgeExtractor
from memoria.seed.seed_data import seed_database, chunk_text

TEMPLATES_DIR = Path(__file__).parent / "templates"


class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"


def create_app(db_path: Path | str = DEFAULT_DB_PATH) -> FastAPI:
    app = FastAPI(title="Memoria AI Anti", description="Personal AI Conversation Vault")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    db = Database(db_path)
    planner = QueryPlanner(db)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        stats = db.get_stats()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"stats": stats, "db_name": db.db_path.name}
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
            conversations = parse_source(tmp_path, provider=provider)
            embedder = EmbeddingEngine()
            extractor = KnowledgeExtractor(db)
            imported = 0
            skipped = 0

            for conv in conversations:
                if db.conversation_exists_by_hash(conv.compute_hash()):
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
                "filename": file.filename,
                "imported": imported,
                "skipped": skipped,
                "stats": db.get_stats()
            }
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return app
