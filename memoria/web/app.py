import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from memoria.config import (
    API_VERSION,
    APP_TITLE,
    APP_VERSION,
    DEFAULT_DB_PATH,
    DEFAULT_INCOMING_DIR,
    DEFAULT_LANG,
    DESIGN_SYSTEM_URL,
    DOCS_LABEL,
    DOCS_URL,
    INSTANCE_NAME,
    MCP_PROTOCOL_VERSION,
    REPO_LABEL,
    REPO_URL,
    UI_VERSION,
    WATCH_ENABLED,
    WATCH_INTERVAL,
)
from memoria.db.engine import Database
from memoria.db.models import DecisionRecord, StatementRecord
from memoria.i18n import get_available_locales, get_locale_bundle, load_all_locales
from memoria.knowledge.enricher import KnowledgeEnricher
from memoria.knowledge.extractor import guess_category
from memoria.knowledge.query_planner import QueryPlanner
from memoria.ingest.pipeline import ingest_source
from memoria.ingest.watcher import DropFolderWatcher
from memoria.mcp_server import (
    MCP_SERVER_INFO,
    MCP_TOOLS_SCHEMA,
    build_agent_conversation_response,
    build_agent_search_response,
    handle_mcp_jsonrpc,
)
from memoria.search.embeddings import reindex_database_embeddings
from memoria.security.auth import (
    COOKIE_NAME,
    SecurityManager,
    verify_agent_request,
    verify_web_session,
)
from memoria.seed.seed_data import seed_database
from memoria.settings.manager import SettingsManager
from memoria.themes import ThemeManager

logger = logging.getLogger("memoria.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


# Pydantic Schemas for Requests
class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"
    lang: str = DEFAULT_LANG


class UpdateSettingsRequest(BaseModel):
    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    api_keys: Optional[Dict[str, str]] = None
    base_urls: Optional[Dict[str, str]] = None
    provider_active_models: Optional[Dict[str, str]] = None


class AddModelRequest(BaseModel):
    provider: str
    model_name: str


class RefreshModelsRequest(BaseModel):
    provider: Optional[str] = "all"


class TestProviderRequest(BaseModel):
    provider: str
    model: Optional[str] = None


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
    settings_mgr = SettingsManager(db)
    security_mgr = SecurityManager(db)
    theme_mgr = ThemeManager(db)
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
        version=API_VERSION,
        lifespan=lifespan,
    )
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "app_version": APP_VERSION,
                "api_version": API_VERSION,
                "mcp_version": MCP_PROTOCOL_VERSION,
            },
        )

    @app.post("/api/auth/login")
    async def auth_login(payload: Dict[str, Any]):
        password = str(payload.get("password", ""))
        if not security_mgr.verify_web_password(password):
            raise HTTPException(status_code=401, detail="Invalid password")
        cookie_val = security_mgr.create_session_cookie_value()
        resp = JSONResponse({"status": "success"})
        resp.set_cookie(
            key=COOKIE_NAME,
            value=cookie_val,
            httponly=True,
            samesite="lax",
            max_age=86400 * 30,
        )
        return resp

    @app.post("/api/auth/logout")
    async def auth_logout():
        resp = JSONResponse({"status": "success"})
        resp.delete_cookie(key=COOKIE_NAME)
        return resp

    @app.get("/api/security/status")
    async def get_security_status():
        return security_mgr.get_status()

    @app.post("/api/security/config")
    async def update_security_config(payload: Dict[str, Any]):
        return security_mgr.update_config(
            api_token_required=payload.get("api_token_required"),
            custom_api_token=payload.get("custom_api_token"),
            web_password=payload.get("web_password"),
            clear_web_password=bool(payload.get("clear_web_password", False)),
        )

    @app.post("/api/security/token/generate")
    async def generate_security_token():
        tok = security_mgr.generate_new_api_token()
        return {"status": "success", "api_token": tok, **security_mgr.get_status()}

    @app.get("/api/themes")
    async def get_themes():
        return theme_mgr.get_active_theme_info()

    @app.post("/api/themes/active")
    async def set_active_theme(payload: Dict[str, Any]):
        palette_id = str(payload.get("palette", "emerald"))
        mode = payload.get("mode")
        return theme_mgr.set_active_theme(palette_id, mode)

    @app.post("/api/themes")
    async def create_custom_theme(payload: Dict[str, Any]):
        return theme_mgr.save_custom_theme(
            name=str(payload.get("name", "Custom Theme")),
            primary=str(payload.get("primary", "#10b981")),
            secondary=str(payload.get("secondary", "#6366f1")),
            dark_bg=str(payload.get("dark_bg", "#0f1117")),
            dark_card=str(payload.get("dark_card", "#181d27")),
            light_bg=str(payload.get("light_bg", "#f8fafc")),
            light_card=str(payload.get("light_card", "#ffffff")),
            theme_id=payload.get("id"),
        )

    @app.delete("/api/themes/{theme_id}")
    async def delete_custom_theme(theme_id: str):
        return theme_mgr.delete_custom_theme(theme_id)

    @app.get("/api/onboarding/mcp-config")
    async def get_onboarding_mcp_config(request: Request):
        base_url = str(request.base_url).rstrip("/")
        sec = security_mgr.get_status()
        mcp_url = f"{base_url}/mcp"
        headers: Dict[str, str] = {}
        if sec.get("api_token_required") and sec.get("api_token"):
            headers["Authorization"] = f"Bearer {sec['api_token']}"
        server_entry: Dict[str, Any] = {"serverUrl": mcp_url}
        if headers:
            server_entry["headers"] = headers
        return {
            "mcp_url": mcp_url,
            "api_token_required": sec.get("api_token_required"),
            "config": {"mcpServers": {"memoria": server_entry}},
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, lang: Optional[str] = None):
        if not verify_web_session(request, db):
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "app_version": APP_VERSION,
                    "api_version": API_VERSION,
                    "mcp_version": MCP_PROTOCOL_VERSION,
                },
            )
        stats = db.get_stats()
        active_lang = (lang or db.get_setting("ui_language") or DEFAULT_LANG or "en").lower()[:2]
        bundle = get_locale_bundle(active_lang)
        theme_info = theme_mgr.get_active_theme_info()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "stats": stats,
                "db_name": db.db_path.name,
                "app_title": APP_TITLE,
                "app_version": APP_VERSION,
                "ui_version": UI_VERSION,
                "api_version": API_VERSION,
                "mcp_version": MCP_PROTOCOL_VERSION,
                "default_lang": active_lang,
                "available_locales": get_available_locales(),
                "i18n": bundle.get("ui", {}),
                "instance_name": INSTANCE_NAME,
                "docs_url": DOCS_URL,
                "docs_label": DOCS_LABEL,
                "repo_url": REPO_URL,
                "repo_label": REPO_LABEL,
                "design_system_url": DESIGN_SYSTEM_URL,
                "theme_info": theme_info,
            },
        )

    @app.get("/api/i18n")
    async def get_i18n_data(lang: Optional[str] = None):
        active_lang = (lang or db.get_setting("ui_language") or DEFAULT_LANG or "en").lower()[:2]
        return {
            "default_lang": active_lang,
            "available": get_available_locales(),
            "locales": load_all_locales(),
        }

    @app.post("/api/i18n/lang")
    async def set_default_language(payload: Dict[str, str]):
        new_lang = (payload.get("lang") or "en").lower()[:2]
        db.set_setting("ui_language", new_lang)
        return {"status": "success", "lang": new_lang, "bundle": get_locale_bundle(new_lang)}

    @app.get("/api/embeddings/status")
    async def get_embeddings_status():
        return planner.searcher.embedder.get_status()

    @app.post("/api/embeddings/reindex")
    async def reindex_embeddings():
        res = await asyncio.to_thread(reindex_database_embeddings, db, planner.searcher.embedder)
        planner.searcher._cached_matrix = None
        planner.searcher._cached_count = 0
        return res

    @app.get("/api/jev/status")
    async def get_jev_status():
        return planner.jev.get_status()

    @app.post("/api/jev/config")
    async def update_jev_config(payload: Dict[str, Any]):
        enabled = bool(payload.get("enabled", False))
        model = payload.get("model")
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")
        threshold = payload.get("threshold")
        return planner.jev.save_config(
            enabled=enabled,
            model=model,
            base_url=base_url,
            api_key=api_key,
            threshold=threshold,
        )

    @app.post("/api/jev/test")
    async def test_jev_decision(payload: Optional[Dict[str, str]] = None):
        sample_query = (payload or {}).get(
            "query",
            "Which audio interface in my inventory is best for recording electric guitar?",
        )
        decision = await asyncio.to_thread(planner.jev.route_query_decision, sample_query)
        return {
            "status": "ok",
            "sample_query": sample_query,
            "decision": decision,
            "jev_status": planner.jev.get_status(),
        }

    @app.get("/api/stats")
    async def get_stats():
        s = db.get_stats()
        s["app_version"] = APP_VERSION
        s["ui_version"] = UI_VERSION
        s["api_version"] = API_VERSION
        s["embedding_engine"] = planner.searcher.embedder.get_status()
        s["jev_system_one"] = planner.jev.get_status()
        return s

    @app.post("/api/query")
    async def run_query(payload: QueryRequest):
        return planner.route_query(payload.query, mode=payload.mode, lang=payload.lang)

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

    # --- Settings Endpoints (Issue #7) ---

    @app.get("/api/settings")
    async def get_settings():
        return settings_mgr.get_full_settings()

    @app.post("/api/settings")
    async def save_settings(payload: UpdateSettingsRequest):
        res = settings_mgr.update_settings(
            active_provider=payload.active_provider,
            active_model=payload.active_model,
            api_keys=payload.api_keys,
            base_urls=payload.base_urls,
            provider_active_models=payload.provider_active_models,
        )
        return {"status": "success", "settings": res}

    @app.get("/api/settings/models")
    async def get_provider_models(provider: str):
        models = settings_mgr.get_models_for_provider(provider)
        return {"provider": provider, "models": models}

    @app.post("/api/settings/models")
    async def add_custom_model(payload: AddModelRequest):
        models = settings_mgr.add_model_to_provider(payload.provider, payload.model_name)
        return {"status": "success", "provider": payload.provider, "models": models}

    @app.post("/api/settings/models/refresh")
    async def refresh_provider_models(payload: RefreshModelsRequest):
        if not payload.provider or payload.provider.lower() in ("all", "*"):
            res = await asyncio.to_thread(settings_mgr.fetch_all_live_models)
            return res
        res = await asyncio.to_thread(settings_mgr.fetch_live_models, payload.provider)
        return res

    @app.post("/api/settings/test")
    async def test_provider(payload: TestProviderRequest):
        client = settings_mgr.get_llm_client(provider_name=payload.provider, model=payload.model)
        start_t = asyncio.get_event_loop().time()
        try:
            res = await asyncio.to_thread(
                client.chat_completion,
                [{"role": "user", "content": "Respond strictly with the word 'OK'."}],
                max_tokens=200,
                timeout=20,
            )
            elapsed = int((asyncio.get_event_loop().time() - start_t) * 1000)
            if res:
                return {"status": "success", "response": res.strip(), "elapsed_ms": elapsed}
            err_msg = getattr(client.provider_instance, "last_error", None) or "Nessuna risposta dal provider. Verifica chiave o endpoint."
            return {"status": "error", "message": err_msg}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # --- Agent API v1 & MCP Server Endpoints (Issue #11) ---

    class AgentSearchRequest(BaseModel):
        query: str
        limit: int = 10
        category: Optional[str] = None
        provider: Optional[str] = None

    class AgentRememberRequest(BaseModel):
        record_type: str = "fact"  # "fact" | "decision"
        subject: Optional[str] = None
        predicate: str = "owned_by_user"
        object_value: Optional[str] = None
        category: Optional[str] = None
        notes: Optional[str] = None
        topic: Optional[str] = None
        decision_text: Optional[str] = None
        status: str = "ADOPTED"

    def _require_agent_auth(request: Request) -> None:
        if not verify_agent_request(request, db):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: valid Authorization: Bearer <MEMORIA_API_TOKEN> required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/api/v1/agent/search")
    async def agent_search_get(
        request: Request,
        q: str = "",
        query: Optional[str] = None,
        limit: int = 10,
        category: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        _require_agent_auth(request)
        search_q = (query or q or "").strip()
        base_url = str(request.base_url).rstrip("/")
        return build_agent_search_response(
            db=db,
            planner=planner,
            query=search_q,
            limit=limit,
            category=category,
            provider=provider,
            base_url=base_url,
        )

    @app.post("/api/v1/agent/search")
    async def agent_search_post(request: Request, payload: AgentSearchRequest):
        _require_agent_auth(request)
        base_url = str(request.base_url).rstrip("/")
        return build_agent_search_response(
            db=db,
            planner=planner,
            query=payload.query.strip(),
            limit=payload.limit,
            category=payload.category,
            provider=payload.provider,
            base_url=base_url,
        )

    @app.get("/api/v1/agent/conversation/{conv_id}")
    async def agent_get_conversation(request: Request, conv_id: str, max_messages: int = 40):
        _require_agent_auth(request)
        base_url = str(request.base_url).rstrip("/")
        res = build_agent_conversation_response(
            db=db,
            conv_id=conv_id,
            max_messages=max_messages,
            base_url=base_url,
        )
        if not res:
            raise HTTPException(status_code=404, detail=f"Conversazione {conv_id} non trovata")
        return res

    @app.post("/api/v1/agent/remember")
    async def agent_remember(request: Request, payload: AgentRememberRequest):
        _require_agent_auth(request)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rtype = payload.record_type.lower().strip()
        if rtype == "decision" or (payload.topic and payload.decision_text):
            if not payload.topic or not payload.decision_text:
                raise HTTPException(status_code=400, detail="topic e decision_text richiesti per una decisione")
            dec = DecisionRecord(
                id=str(uuid.uuid4()),
                topic=payload.topic.strip(),
                decision_text=payload.decision_text.strip(),
                status=payload.status.upper(),
                decided_at=now_iso,
            )
            dec_id = db.insert_decision(dec)
            return {
                "status": "success",
                "record_type": "decision",
                "id": dec_id,
                "topic": dec.topic,
                "decision_text": dec.decision_text,
            }
        else:
            if not payload.subject or not payload.object_value:
                raise HTTPException(status_code=400, detail="subject e object_value richiesti per un fatto")
            cat = payload.category or guess_category(payload.subject)
            ent_id = db.upsert_entity(payload.subject, cat)
            stmt = StatementRecord(
                id=str(uuid.uuid4()),
                entity_id=ent_id,
                statement_type="FACT",
                subject=payload.subject.strip(),
                predicate=(payload.predicate or "owned_by_user").strip(),
                object=payload.object_value.strip(),
                author="agent",
                confidence=1.0,
                valid_from=now_iso,
                is_current=1,
                notes=payload.notes or "Registrato da Agente AI",
            )
            stmt_id = db.insert_statement(stmt)
            return {
                "status": "success",
                "record_type": "fact",
                "id": stmt_id,
                "subject": stmt.subject,
                "predicate": stmt.predicate,
                "object_value": stmt.object,
                "category": cat,
            }

    @app.post("/api/v1/agent/ask")
    async def agent_ask(request: Request, payload: QueryRequest):
        _require_agent_auth(request)
        return planner.route_query(payload.query, mode=payload.mode)

    @app.get("/mcp")
    @app.get("/api/mcp")
    async def mcp_info():
        return {
            "status": "online",
            "server": MCP_SERVER_INFO,
            "tools_count": len(MCP_TOOLS_SCHEMA),
            "tools": MCP_TOOLS_SCHEMA,
            "endpoints": {
                "mcp_jsonrpc": "/mcp",
                "agent_search": "/api/v1/agent/search?q=...",
                "agent_conversation": "/api/v1/agent/conversation/{conv_id}",
                "agent_remember": "/api/v1/agent/remember",
                "agent_ask": "/api/v1/agent/ask",
                "openapi_docs": "/docs",
            },
        }

    @app.post("/mcp")
    @app.post("/api/mcp")
    async def mcp_endpoint(request: Request):
        if not verify_agent_request(request, db):
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": "Unauthorized: invalid or missing Bearer token"},
                },
            )
        base_url = str(request.base_url).rstrip("/")
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            )

        if isinstance(body, list):
            responses = []
            for item in body:
                r = handle_mcp_jsonrpc(item, db, planner, base_url=base_url)
                if r is not None:
                    responses.append(r)
            if not responses:
                return Response(status_code=202)
            resp_payload: Any = responses
        else:
            resp_payload = handle_mcp_jsonrpc(body, db, planner, base_url=base_url)
            if resp_payload is None:
                return Response(status_code=202)

        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept and "application/json" not in accept:
            sse_data = f"event: message\ndata: {json.dumps(resp_payload, ensure_ascii=False)}\n\n"
            return Response(content=sse_data, media_type="text/event-stream")

        return JSONResponse(content=resp_payload)

    return app

