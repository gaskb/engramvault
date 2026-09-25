"""
Agent API v1 & Model Context Protocol (MCP) Server Routes (Issue #26 / Q-01, S-03).
"""

import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from memoria.config import DEFAULT_LANG
from memoria.db.engine import Database, utc_now_iso
from memoria.db.models import DecisionRecord, StatementRecord
from memoria.knowledge.extractor import guess_category
from memoria.knowledge.query_planner import QueryPlanner
from memoria.mcp_server import (
    MCP_SERVER_INFO,
    MCP_TOOLS_SCHEMA,
    build_agent_conversation_response,
    build_agent_search_response,
    handle_mcp_jsonrpc,
)
from memoria.security.auth import verify_agent_request


class AgentSearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)
    category: Optional[str] = Field(default=None, max_length=128)
    provider: Optional[str] = Field(default=None, max_length=128)


class AgentRememberRequest(BaseModel):
    record_type: str = Field(default="fact", max_length=32)  # "fact" | "decision"
    subject: Optional[str] = Field(default=None, max_length=512)
    predicate: str = Field(default="owned_by_user", max_length=256)
    object_value: Optional[str] = Field(default=None, max_length=4000)
    category: Optional[str] = Field(default=None, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=2000)
    topic: Optional[str] = Field(default=None, max_length=512)
    decision_text: Optional[str] = Field(default=None, max_length=4000)
    status: str = Field(default="ADOPTED", max_length=32)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    mode: str = Field(default="auto", max_length=32)
    lang: str = Field(default=DEFAULT_LANG, max_length=8)


def build_agent_router(db: Database, planner: QueryPlanner) -> APIRouter:
    router = APIRouter()

    def require_agent_auth(request: Request) -> None:
        if not verify_agent_request(request, db):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: valid Authorization: Bearer <MEMORIA_API_TOKEN> required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @router.get("/api/v1/agent/search", dependencies=[Depends(require_agent_auth)])
    async def agent_search_get(
        request: Request,
        q: str = "",
        query: Optional[str] = None,
        limit: int = 10,
        category: Optional[str] = None,
        provider: Optional[str] = None,
    ):
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

    @router.post("/api/v1/agent/search", dependencies=[Depends(require_agent_auth)])
    async def agent_search_post(request: Request, payload: AgentSearchRequest):
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

    @router.get("/api/v1/agent/conversation/{conv_id}", dependencies=[Depends(require_agent_auth)])
    async def agent_get_conversation(request: Request, conv_id: str, max_messages: int = 40):
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

    @router.post("/api/v1/agent/remember", dependencies=[Depends(require_agent_auth)])
    async def agent_remember(payload: AgentRememberRequest):
        now_iso = utc_now_iso()
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

    @router.post("/api/v1/agent/ask", dependencies=[Depends(require_agent_auth)])
    async def agent_ask(payload: AgentQueryRequest):
        return planner.route_query(payload.query, mode=payload.mode, lang=payload.lang)

    @router.get("/mcp", dependencies=[Depends(require_agent_auth)])
    async def mcp_info_or_sse(request: Request):
        accept = request.headers.get("accept", "")
        base_url = str(request.base_url).rstrip("/")
        if "text/event-stream" in accept:
            sse_body = f"event: endpoint\ndata: {base_url}/mcp\n\n"
            return Response(content=sse_body, media_type="text/event-stream")
        return {
            "status": "online",
            "protocol": "MCP (Model Context Protocol)",
            "server": MCP_SERVER_INFO,
            "serverInfo": MCP_SERVER_INFO,
            "tools_count": len(MCP_TOOLS_SCHEMA),
            "transport": "Streamable HTTP (POST /mcp) & stdio (python -m memoria.mcp_server)",
            "endpoint": f"{base_url}/mcp",
            "agent_rest_api": {
                "search": f"{base_url}/api/v1/agent/search?q=<query>",
                "conversation": f"{base_url}/api/v1/agent/conversation/<id>",
                "remember": f"{base_url}/api/v1/agent/remember",
                "ask": f"{base_url}/api/v1/agent/ask",
            },
            "tools": MCP_TOOLS_SCHEMA,
        }

    @router.get("/mcp/sse", dependencies=[Depends(require_agent_auth)])
    async def mcp_sse_endpoint(request: Request):
        base_url = str(request.base_url).rstrip("/")
        sse_body = f"event: endpoint\ndata: {base_url}/mcp\n\n"
        return Response(content=sse_body, media_type="text/event-stream")

    @router.post("/mcp", dependencies=[Depends(require_agent_auth)])
    async def mcp_jsonrpc_endpoint(request: Request):
        base_url = str(request.base_url).rstrip("/")
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}},
            )

        if isinstance(body, list):
            responses = []
            for req_item in body:
                res = handle_mcp_jsonrpc(req_item, db=db, planner=planner, base_url=base_url)
                if res is not None:
                    responses.append(res)
            if not responses:
                return Response(status_code=202)
            resp_payload = responses
        else:
            resp_payload = handle_mcp_jsonrpc(body, db=db, planner=planner, base_url=base_url)
            if resp_payload is None:
                return Response(status_code=202)

        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept and "application/json" not in accept:
            sse_data = f"event: message\ndata: {json.dumps(resp_payload, ensure_ascii=False)}\n\n"
            return Response(content=sse_data, media_type="text/event-stream")

        return JSONResponse(content=resp_payload)

    return router
