"""
Agent API v1 & Model Context Protocol (MCP) Server Routes (Issue #26 / Q-01, S-03).
"""

import asyncio
import json
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from engramvault.config import DEFAULT_LANG
from engramvault.db.engine import Database, utc_now_iso
from engramvault.db.models import DecisionRecord, StatementRecord
from engramvault.knowledge.contradiction import detect_contradiction
from engramvault.knowledge.extractor import guess_category
from engramvault.knowledge.query_planner import QueryPlanner
from engramvault.mcp_server import (
    MCP_SERVER_INFO,
    MCP_TOOLS_SCHEMA,
    build_agent_conversation_response,
    build_agent_search_response,
    handle_mcp_jsonrpc,
)
from engramvault.security.auth import verify_agent_request


class AgentSearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)
    category: Optional[str] = Field(default=None, max_length=128)
    provider: Optional[str] = Field(default=None, max_length=128)


class AgentRememberRequest(BaseModel):
    record_type: Literal["fact", "claim", "decision"] = Field(default="fact")
    subject: Optional[str] = Field(default=None, max_length=512)
    predicate: str = Field(default="owned_by_user", max_length=256)
    object_value: Optional[str] = Field(default=None, max_length=4000)
    category: Optional[str] = Field(default=None, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=2000)
    topic: Optional[str] = Field(default=None, max_length=512)
    decision_text: Optional[str] = Field(default=None, max_length=4000)
    status: Literal["ADOPTED", "EVALUATING", "SUPERSEDED", "ABANDONED"] = Field(default="ADOPTED")


class AgentContextRequest(BaseModel):
    max_facts: int = Field(default=50, ge=1, le=500)
    max_decisions: int = Field(default=20, ge=1, le=200)
    format: str = Field(default="json", max_length=16)


class AgentCheckConflictRequest(BaseModel):
    subject: str = Field(..., max_length=512)
    predicate: str = Field(default="owned_by_user", max_length=256)
    object_value: str = Field(..., max_length=4000)
    record_type: Literal["fact", "claim", "decision"] = Field(default="fact")
    category: Optional[str] = Field(default=None, max_length=128)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., max_length=4000)
    mode: str = Field(default="auto", max_length=32)
    lang: str = Field(default=DEFAULT_LANG, max_length=8)


class AgentRelationRequest(BaseModel):
    from_entity: str = Field(..., max_length=512)
    relation_type: str = Field(default="relates_to", max_length=128)
    to_entity: str = Field(..., max_length=512)


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
        return await asyncio.to_thread(
            build_agent_search_response,
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
        return await asyncio.to_thread(
            build_agent_search_response,
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
        res = await asyncio.to_thread(
            build_agent_conversation_response,
            db=db,
            conv_id=conv_id,
            max_messages=max_messages,
            base_url=base_url,
        )
        if not res:
            raise HTTPException(status_code=404, detail=f"Conversazione {conv_id} non trovata")
        return res

    @router.get("/api/v1/agent/context", dependencies=[Depends(require_agent_auth)])
    async def agent_get_context(
        max_facts: int = 50,
        max_decisions: int = 20,
        format: str = "json",
    ):
        ctx = await asyncio.to_thread(db.build_agent_context, max_facts=max_facts, max_decisions=max_decisions)
        if format.lower() in ("markdown", "text", "md"):
            return Response(content=ctx["context_markdown"], media_type="text/markdown")
        return ctx

    @router.post("/api/v1/agent/context", dependencies=[Depends(require_agent_auth)])
    async def agent_post_context(payload: AgentContextRequest):
        ctx = await asyncio.to_thread(
            db.build_agent_context,
            max_facts=payload.max_facts,
            max_decisions=payload.max_decisions,
        )
        if payload.format.lower() in ("markdown", "text", "md"):
            return Response(content=ctx["context_markdown"], media_type="text/markdown")
        return ctx

    @router.post("/api/v1/agent/check_conflict", dependencies=[Depends(require_agent_auth)])
    async def agent_check_conflict(payload: AgentCheckConflictRequest):
        cat = payload.category or guess_category(payload.subject)
        report = detect_contradiction(
            db=db,
            subject=payload.subject,
            predicate=payload.predicate,
            object_value=payload.object_value,
            statement_type="FACT" if payload.record_type.lower() == "fact" else "CLAIM",
            entity_category=cat,
        )
        return {"status": "success", "report": report}

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

        # Run contradiction detection
        target_type = "CLAIM" if rtype == "claim" else "FACT"
        conflict = detect_contradiction(
            db=db,
            subject=payload.subject,
            predicate=(payload.predicate or "owned_by_user").strip(),
            object_value=payload.object_value.strip(),
            statement_type=target_type,
            entity_category=cat,
        )
        conflict_json = json.dumps(conflict, ensure_ascii=False) if conflict.get("has_conflict") else None
        supersedes_id = conflict.get("conflicting_id") if conflict.get("conflict_type") == "SUPERSEDES" else None
        if supersedes_id and target_type == "FACT":
            db.invalidate_statement(supersedes_id, valid_until=now_iso)

        stmt = StatementRecord(
            id=str(uuid.uuid4()),
            entity_id=ent_id,
            statement_type=target_type,
            subject=payload.subject.strip(),
            predicate=(payload.predicate or "owned_by_user").strip(),
            object=payload.object_value.strip(),
            author="agent",
            confidence=1.0,
            valid_from=now_iso,
            is_current=1,
            supersedes_id=supersedes_id,
            notes=payload.notes or "Registrato da Agente AI",
            conflict_json=conflict_json,
        )
        stmt_id = db.insert_statement(stmt)
        return {
            "status": "success",
            "record_type": target_type.lower(),
            "id": stmt_id,
            "subject": stmt.subject,
            "predicate": stmt.predicate,
            "object_value": stmt.object,
            "category": cat,
            "conflict": conflict,
        }

    @router.post("/api/v1/agent/ask", dependencies=[Depends(require_agent_auth)])
    async def agent_ask(payload: AgentQueryRequest):
        return await asyncio.to_thread(planner.route_query, payload.query, mode=payload.mode, lang=payload.lang)

    @router.get("/api/v1/agent/graph/{entity}", dependencies=[Depends(require_agent_auth)])
    async def agent_get_graph(entity: str, depth: int = 1):
        res = await asyncio.to_thread(db.get_entity_graph, entity, depth=depth)
        if not res.get("root"):
            raise HTTPException(status_code=404, detail=f"Entità '{entity}' non trovata nel Knowledge Graph.")
        return res

    @router.post("/api/v1/agent/relation", dependencies=[Depends(require_agent_auth)])
    async def agent_create_relation(payload: AgentRelationRequest):
        from_ent = payload.from_entity.strip()
        to_ent = payload.to_entity.strip()
        if not from_ent or not to_ent or from_ent.lower() == to_ent.lower():
            raise HTTPException(status_code=400, detail="from_entity e to_entity devono essere entità distinte e non vuote")
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
        return {
            "status": "success",
            "id": rel_id,
            "from_entity": from_ent,
            "relation_type": payload.relation_type,
            "to_entity": to_ent,
        }

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
            "transport": "Streamable HTTP (POST /mcp) & stdio (python -m engramvault.mcp_server)",
            "endpoint": f"{base_url}/mcp",
            "agent_rest_api": {
                "context": f"{base_url}/api/v1/agent/context",
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
                res = await asyncio.to_thread(handle_mcp_jsonrpc, req_item, db=db, planner=planner, base_url=base_url)
                if res is not None:
                    responses.append(res)
            if not responses:
                return Response(status_code=202)
            resp_payload = responses
        else:
            resp_payload = await asyncio.to_thread(handle_mcp_jsonrpc, body, db=db, planner=planner, base_url=base_url)
            if resp_payload is None:
                return Response(status_code=202)

        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept and "application/json" not in accept:
            sse_data = f"event: message\ndata: {json.dumps(resp_payload, ensure_ascii=False)}\n\n"
            return Response(content=sse_data, media_type="text/event-stream")

        return JSONResponse(content=resp_payload)

    return router
