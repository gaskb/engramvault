import asyncio
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from engramvault.config import (
    API_VERSION,
    APP_SUBTITLE,
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
from engramvault.db.engine import Database
from engramvault.i18n import get_available_locales, get_locale_bundle, load_all_locales
from engramvault.ingest.pipeline import ingest_source
from engramvault.ingest.watcher import DropFolderWatcher
from engramvault.knowledge.enricher import KnowledgeEnricher
from engramvault.knowledge.query_planner import QueryPlanner
from engramvault.search.embeddings import reindex_database_embeddings
from engramvault.security.auth import SecurityManager, verify_web_session
from engramvault.seed.seed_data import seed_database
from engramvault.settings.manager import SettingsManager
from engramvault.themes import ThemeManager
from engramvault.web.routes import (
    build_agent_router,
    build_curation_router,
    build_security_router,
    require_mutation_auth,
)

logger = logging.getLogger("engramvault.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


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
                    logger.error("Errore watcher drop folder: %s", e)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=WATCH_INTERVAL)
                except asyncio.TimeoutError:
                    pass

        watch_task = None
        if WATCH_ENABLED:
            watch_task = asyncio.create_task(_watch_loop())
            logger.info("Watcher drop folder attivato su: %s", DEFAULT_INCOMING_DIR)

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
        title=f"{APP_TITLE} — {APP_SUBTITLE}",
        description="Personal AI Conversation Vault & Epistemic Knowledge Layer",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.security_mgr = security_mgr
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.middleware("http")
    async def protect_web_and_api_security(request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        if path.startswith("/api/v1/agent") or path.startswith("/mcp"):
            return await call_next(request)
        if path.startswith("/api/auth/"):
            return await call_next(request)

        if security_mgr.is_web_password_enabled():
            is_public_api = (path == "/api/i18n") or (path == "/api/themes" and method == "GET")
            if path.startswith("/api/") and not is_public_api:
                if not security_mgr.is_request_authorized(request):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: valid Web Session or Bearer Token required."},
                    )
        else:
            if path.startswith("/api/security") or (
                path.startswith("/api/") and method in ("POST", "PUT", "PATCH", "DELETE")
            ):
                if not security_mgr.is_request_authorized(request):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: valid Web Session or Bearer Token required."},
                    )
        return await call_next(request)

    # Include modular APIRouters (Q-01)
    app.include_router(
        build_security_router(
            security_mgr=security_mgr,
            theme_mgr=theme_mgr,
            templates=templates,
        )
    )
    app.include_router(build_curation_router(db=db, enricher=enricher))
    app.include_router(build_agent_router(db=db, planner=planner))

    @app.get("/manifest.json")
    async def get_manifest():
        manifest_file = STATIC_DIR / "manifest.json"
        if not manifest_file.exists():
            raise HTTPException(status_code=404, detail="Manifest not found")
        return FileResponse(manifest_file, media_type="application/manifest+json")

    @app.get("/service-worker.js")
    async def get_service_worker():
        sw_file = STATIC_DIR / "service-worker.js"
        if not sw_file.exists():
            raise HTTPException(status_code=404, detail="Service worker not found")
        return FileResponse(
            sw_file,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )

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
                "app_subtitle": APP_SUBTITLE,
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

    @app.post("/api/i18n/lang", dependencies=[Depends(require_mutation_auth)])
    async def set_default_language(payload: Dict[str, str]):
        new_lang = (payload.get("lang") or "en").lower()[:2]
        db.set_setting("ui_language", new_lang)
        return {"status": "success", "lang": new_lang, "bundle": get_locale_bundle(new_lang)}

    @app.get("/api/embeddings/status")
    async def get_embeddings_status():
        return planner.searcher.embedder.get_status()

    @app.post("/api/embeddings/reindex", dependencies=[Depends(require_mutation_auth)])
    async def reindex_embeddings():
        res = await asyncio.to_thread(reindex_database_embeddings, db, planner.searcher.embedder)
        planner.searcher._cached_matrix = None
        planner.searcher._cached_count = 0
        return res

    @app.get("/api/jev/status")
    async def get_jev_status():
        return planner.jev.get_status()

    @app.post("/api/jev/config", dependencies=[Depends(require_mutation_auth)])
    async def update_jev_config(payload: Dict[str, Any]):
        enabled = bool(payload.get("enabled", False))
        model = payload.get("model")
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")
        threshold = payload.get("threshold")
        timeout = payload.get("timeout")
        return planner.jev.save_config(
            enabled=enabled,
            model=model,
            base_url=base_url,
            api_key=api_key,
            threshold=threshold,
            timeout=timeout,
        )

    @app.post("/api/jev/test", dependencies=[Depends(require_mutation_auth)])
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
        return await asyncio.to_thread(planner.route_query, payload.query, mode=payload.mode, lang=payload.lang)

    @app.get("/api/conversations")
    async def list_conversations(
        provider: Optional[str] = None,
        q: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        search_q = (query or q or "").strip() or None
        return db.list_conversations(limit=limit, offset=offset, provider=provider, query=search_q)

    @app.get("/api/conversations/{conv_id}")
    async def get_conversation(conv_id: str):
        c = db.get_conversation(conv_id)
        if not c:
            raise HTTPException(status_code=404, detail="Conversazione non trovata")
        return c

    @app.delete("/api/conversations/{conv_id}", dependencies=[Depends(require_mutation_auth)])
    async def delete_conversation(conv_id: str):
        ok = db.delete_conversation(conv_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Conversazione non trovata")
        return {"status": "success", "deleted_id": conv_id, "stats": db.get_stats()}

    @app.post("/api/ingest/scan", dependencies=[Depends(require_mutation_auth)])
    async def trigger_drop_folder_scan():
        results = await asyncio.to_thread(watcher.scan_once)
        return {
            "status": "success",
            "incoming_dir": str(watcher.incoming_dir),
            "processed": results,
            "stats": db.get_stats(),
        }

    # --- Settings Endpoints (Issue #7) ---

    @app.get("/api/settings")
    async def get_settings():
        return settings_mgr.get_full_settings()

    @app.post("/api/settings", dependencies=[Depends(require_mutation_auth)])
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

    @app.post("/api/settings/models", dependencies=[Depends(require_mutation_auth)])
    async def add_custom_model(payload: AddModelRequest):
        models = settings_mgr.add_model_to_provider(payload.provider, payload.model_name)
        return {"status": "success", "provider": payload.provider, "models": models}

    @app.post("/api/settings/models/refresh", dependencies=[Depends(require_mutation_auth)])
    async def refresh_provider_models(payload: RefreshModelsRequest):
        if not payload.provider or payload.provider.lower() in ("all", "*"):
            res = await asyncio.to_thread(settings_mgr.fetch_all_live_models)
            return res
        res = await asyncio.to_thread(settings_mgr.fetch_live_models, payload.provider)
        return res

    @app.post("/api/settings/test", dependencies=[Depends(require_mutation_auth)])
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
            err_msg = (
                getattr(client.provider_instance, "last_error", None)
                or "Nessuna risposta dal provider. Verifica chiave o endpoint."
            )
            return {"status": "error", "message": err_msg}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app

