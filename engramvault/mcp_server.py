"""
EngramVault — Agent API Engine & Model Context Protocol (MCP) Server (Issue #11).

Supports:
1. In-process execution for FastAPI endpoints (`/api/v1/agent/*` and `POST /mcp` Streamable HTTP / SSE)
2. Standalone `stdio` MCP Server (`python -m engramvault.mcp_server`) that proxies to `ENGRAMVAULT_API_URL`
   (`http://localhost:5006`) with automatic local database fallback.
"""

import json
import os
import sys
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from engramvault.config import API_VERSION, DEFAULT_DB_PATH
from engramvault.db.engine import Database
from engramvault.db.models import DecisionRecord, StatementRecord
from engramvault.knowledge.extractor import guess_category
from engramvault.knowledge.query_planner import QueryPlanner

DEFAULT_REMOTE_URL = (
    os.environ.get("ENGRAMVAULT_API_URL")
    or os.environ.get("MEMORIA_API_URL")
    or os.environ.get("MEMORIA_PUBLIC_URL")
    or "http://localhost:8080"
).rstrip("/")

MCP_SERVER_INFO = {
    "name": "engramvault-mcp",
    "version": API_VERSION,
}

MCP_INSTRUCTIONS = (
    "EngramVault è il Vault Storico delle conversazioni (ChatGPT, Claude, Gemini, Ollama, Antigravity), "
    "dei Fatti Verificati (inventario hardware/audio/software, configurazioni, preferenze) e delle Decisioni Architetturali di Gas.\n"
    "Usa `memoria_search` come primo strumento per cercare rapidamente (< 30ms) discussioni passate, fatti o decisioni su un argomento.\n"
    "Usa `memoria_get_conversation` per leggere i messaggi completi di una conversazione trovata.\n"
    "Usa `memoria_remember_fact` o `memoria_remember_decision` per salvare nuove decisioni o fatti emersi durante la sessione."
)

MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "memoria_search",
        "description": (
            "Ricerca ibrida ultra-veloce (< 30ms) nel Vault di EngramVault (Fatti verificati, Claims, Decisioni storiche "
            "e oltre 39.000 messaggi di conversazioni passate su ChatGPT, Claude, Gemini, Ollama e Antigravity). "
            "Restituisce sia un riassunto Markdown pronto per l'agente sia i dati strutturati con conversation_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Termini di ricerca o domanda (es. 'audiocassette rock', 'Zen Go interfaccia audio', 'proxmox backup', 'docker buildx').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di risultati per sezione (default: 10).",
                    "default": 10,
                },
                "category": {
                    "type": "string",
                    "description": "Filtro opzionale per categoria entità (es. 'audio', 'audiocassette', 'homelab', 'software', 'hardware').",
                },
                "provider": {
                    "type": "string",
                    "description": "Filtro opzionale per sorgente chat ('chatgpt', 'claude', 'claude-code', 'gemini', 'ollama', 'antigravity').",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memoria_get_conversation",
        "description": (
            "Recupera i dettagli completi, i fatti/decisioni estratti e la trascrizione dei messaggi di una specifica "
            "conversazione passata tramite il suo `conversation_id`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID della conversazione nel Vault di EngramVault.",
                },
                "max_messages": {
                    "type": "integer",
                    "description": "Numero massimo di messaggi da restituire (default: 40).",
                    "default": 40,
                },
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "memoria_ask",
        "description": (
            "Interroga il motore di sintesi LLM di EngramVault per ottenere una risposta ragionata multi-hop "
            "combinando l'inventario utente (FACT), i claims tecnici e le fonti storiche."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Domanda completa in linguaggio naturale.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "synthetic", "structured", "extractive"],
                    "description": "Modalità di esecuzione ('auto' consigliata).",
                    "default": "auto",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memoria_remember_fact",
        "description": (
            "Salva immediatamente un nuovo Fatto Verificato (FACT) o preferenza nel Knowledge Layer di EngramVault "
            "(es. possesso di un componente hardware/software, configurazione scelta, preferenza tecnica)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Soggetto o entità (es. 'Antelope Zen Go', 'TDK SA-X 90', 'Proxmox VM 103').",
                },
                "predicate": {
                    "type": "string",
                    "description": "Predicato/relazione (es. 'owned_by_user', 'configured_with', 'installed_on', 'prefers').",
                    "default": "owned_by_user",
                },
                "object_value": {
                    "type": "string",
                    "description": "Valore o stato dell'oggetto (es. 'true', 'localhost:5006', 'Interfaccia audio principale').",
                },
                "category": {
                    "type": "string",
                    "description": "Categoria dell'entità (es. 'audio', 'homelab', 'software', 'hardware').",
                },
                "notes": {
                    "type": "string",
                    "description": "Contesto o note aggiuntive sull'origine del fatto.",
                },
            },
            "required": ["subject", "predicate", "object_value"],
        },
    },
    {
        "name": "memoria_remember_decision",
        "description": (
            "Registra immediatamente una nuova Decisione Architetturale o Tecnica nel registro decisionale di EngramVault."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Ambito o tema della decisione (es. 'Standard UI Homelab', 'HTTPS Servizi Interni').",
                },
                "decision_text": {
                    "type": "string",
                    "description": "Descrizione della decisione presa, incluse le motivazioni e le alternative scartate.",
                },
                "status": {
                    "type": "string",
                    "enum": ["ADOPTED", "EVALUATING", "SUPERSEDED", "ABANDONED"],
                    "default": "ADOPTED",
                },
            },
            "required": ["topic", "decision_text"],
        },
    },
    {
        "name": "memoria_get_entity_graph",
        "description": (
            "Esplora il grafo di conoscenza (Knowledge Graph) intorno a un'entità fino a una profondità indicata (1 o 2 salti), "
            "restituendo le relazioni semantiche/fisiche connesse (es. 'connected_to', 'runs_on', 'part_of', 'depends_on', 'uses'), "
            "i nodi adiacenti e i fatti/claim associati a ciascun nodo."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Nome dell'entità o UUID da esplorare nel grafo (es. 'Yaesu FT-891', 'EngramVault', 'Docker', 'Proxmox').",
                },
                "depth": {
                    "type": "integer",
                    "description": "Profondità di attraversamento del grafo (default: 1, max: 2).",
                    "default": 1,
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "memoria_remember_relation",
        "description": (
            "Collega semanticamente due entità nel Knowledge Graph creando una relazione diretta "
            "(es. 'Yaesu FT-891' connected_to 'Antenna Dipolo', 'EngramVault' runs_on 'Docker')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_entity": {
                    "type": "string",
                    "description": "Nome dell'entità sorgente.",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Tipo di relazione (es. 'connected_to', 'runs_on', 'depends_on', 'part_of', 'compatible_with', 'subproject_of', 'uses', 'created_by').",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Nome dell'entità destinazione.",
                },
            },
            "required": ["from_entity", "relation_type", "to_entity"],
        },
    },
]


def build_agent_search_response(
    db: Database,
    planner: QueryPlanner,
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: str = DEFAULT_REMOTE_URL,
) -> Dict[str, Any]:
    """Executes a fast hybrid search across Knowledge Layer + FTS5/Vector Chunks and formats an LLM brief."""
    t0 = time.time()
    k_res = db.search_knowledge(query=query, limit=limit, category=category, provider=provider)
    chunks = planner.searcher.hybrid_search(query=query, limit=limit, provider=provider)

    # Enrich chunks with deep links
    for c in chunks:
        cid = c.get("conversation_id")
        c["web_url"] = f"{base_url}/?conv={cid}" if cid else base_url

    statements = k_res["statements"]
    decisions = k_res["decisions"]
    conversations = k_res["conversations"]

    # Knowledge Graph related entities
    entity_ids = list({s["entity_id"] for s in statements if s.get("entity_id")})
    with db.get_connection() as conn:
        q_clean = query.strip()
        matching_entities = conn.execute(
            "SELECT id FROM entities WHERE name LIKE ? OR ? LIKE '%' || normalized_name || '%' LIMIT 5",
            (f"%{q_clean}%", q_clean.lower())
        ).fetchall()
        for me in matching_entities:
            if me["id"] not in entity_ids:
                entity_ids.append(me["id"])

    relations = db.find_related_entities(entity_ids, limit=12) if entity_ids else []

    # Build Markdown Brief for Agents
    md_lines = [f"## 🧠 EngramVault — Risultati per: `{query}`\n"]

    if statements:
        md_lines.append(f"### 1. Fatti & Claims Rilevanti ({len(statements)})")
        for s in statements:
            st_type = s.get("statement_type", "FACT")
            subj = s.get("subject", "")
            pred = s.get("predicate", "")
            obj = s.get("object", "")
            cat = s.get("entity_category") or "generale"
            src_title = s.get("conversation_title") or "Manuale / Curation"
            src_prov = (s.get("conversation_provider") or "user").upper()
            dt = (s.get("valid_from") or "")[:10]
            conv_id = s.get("source_conversation_id")
            conv_ref = f" (`conv_id`: `{conv_id}`)" if conv_id else ""
            md_lines.append(
                f"- **[{st_type}] {subj}** (`{cat}`) $\\rightarrow$ `{pred}`: **{obj}** "
                f"— *Fonte: {src_prov} in \"{src_title}\" ({dt}){conv_ref}*"
            )
        md_lines.append("")

    if relations:
        md_lines.append(f"### 2. Relazioni Semantiche nel Knowledge Graph ({len(relations)})")
        for r in relations:
            fn = r.get("from_name", "")
            fc = r.get("from_category", "")
            rtype = r.get("relation_type", "")
            tn = r.get("to_name", "")
            tc = r.get("to_category", "")
            md_lines.append(f"- **{fn}** (`{fc}`) $\\xrightarrow{{\\text{{{rtype}}}}}$ **{tn}** (`{tc}`)")
        md_lines.append("")

    if decisions:
        md_lines.append(f"### 2. Decisioni Storiche ({len(decisions)})")
        for d in decisions:
            dt = (d.get("decided_at") or "")[:10]
            status = d.get("status", "ADOPTED")
            conv_id = d.get("source_conversation_id")
            conv_ref = f" (`conv_id`: `{conv_id}`)" if conv_id else ""
            md_lines.append(
                f"- **[{status}] {d['topic']}** ({dt}): {d['decision_text']}{conv_ref}"
            )
        md_lines.append("")

    if conversations:
        md_lines.append(f"### 3. Conversazioni con Titolo Corrispondente ({len(conversations)})")
        for c in conversations:
            dt = (c.get("created_at") or "")[:10]
            prov = (c.get("provider") or "chat").upper()
            md_lines.append(
                f"- **{c['title']}** ({prov}, {dt}) — `conversation_id`: `{c['id']}`"
            )
        md_lines.append("")

    if chunks:
        md_lines.append(f"### 4. Estratti Diretti dalle Conversazioni ({len(chunks)})")
        for idx, ch in enumerate(chunks, 1):
            dt = (ch.get("created_at") or "")[:10]
            prov = (ch.get("provider") or "chat").upper()
            title = ch.get("conv_title") or "Conversazione"
            cid = ch.get("conversation_id")
            snippet = (ch.get("chunk_text") or "").strip().replace("\n", " ")
            if len(snippet) > 420:
                snippet = snippet[:420] + "..."
            md_lines.append(
                f"{idx}. **[{prov} — {dt}] {title}** (`conversation_id`: `{cid}`)\n"
                f"   > {snippet}"
            )
        md_lines.append("")

    if not (statements or decisions or conversations or chunks):
        md_lines.append("_Nessun risultato specifico trovato nel Vault per questa query._")

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "query": query,
        "elapsed_ms": elapsed_ms,
        "counts": {
            "statements": len(statements),
            "relations": len(relations),
            "decisions": len(decisions),
            "matching_conversations": len(conversations),
            "message_chunks": len(chunks),
        },
        "statements": statements,
        "relations": relations,
        "decisions": decisions,
        "matching_conversations": conversations,
        "message_chunks": chunks,
        "markdown_brief": "\n".join(md_lines),
    }


def build_agent_conversation_response(
    db: Database,
    conv_id: str,
    max_messages: int = 40,
    base_url: str = DEFAULT_REMOTE_URL,
) -> Optional[Dict[str, Any]]:
    """Fetches a full conversation plus its extracted knowledge and formats an agent-readable transcript."""
    conv = db.get_conversation(conv_id)
    if not conv:
        return None

    knowledge = db.get_conversation_knowledge(conv_id)
    messages = conv.get("messages", [])
    total_messages = len(messages)
    sliced_messages = messages[:max_messages] if max_messages > 0 else messages

    md_lines = [
        f"## 💬 Conversazione: {conv.get('title')}",
        f"- **ID**: `{conv.get('id')}`",
        f"- **Provider**: `{(conv.get('provider') or '').upper()}`",
        f"- **Data**: `{(conv.get('created_at') or '')[:19]}`",
        f"- **Messaggi Totali**: {total_messages} (mostrati: {len(sliced_messages)})",
        f"- **Link Web**: {base_url}/?conv={conv.get('id')}\n",
    ]

    if knowledge["statements"]:
        md_lines.append("### Fatti e Claims Estratti da questa Chat:")
        for s in knowledge["statements"]:
            md_lines.append(f"- **[{s['statement_type']}] {s['subject']}** $\\rightarrow$ `{s['predicate']}`: {s['object']}")
        md_lines.append("")

    if knowledge["decisions"]:
        md_lines.append("### Decisioni Estratte da questa Chat:")
        for d in knowledge["decisions"]:
            md_lines.append(f"- **[{d['status']}] {d['topic']}**: {d['decision_text']}")
        md_lines.append("")

    md_lines.append("### Trascrizione Messaggi:")
    for m in sliced_messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        if len(content) > 1500:
            content = content[:1500] + "\n...[messaggio troncato]..."
        md_lines.append(f"#### [{role}]\n{content}\n")

    return {
        "conversation": {
            "id": conv.get("id"),
            "title": conv.get("title"),
            "provider": conv.get("provider"),
            "created_at": conv.get("created_at"),
            "total_messages": total_messages,
            "web_url": f"{base_url}/?conv={conv.get('id')}",
        },
        "knowledge": knowledge,
        "messages": sliced_messages,
        "markdown_transcript": "\n".join(md_lines),
    }


def handle_mcp_tool_call(
    name: str,
    arguments: Dict[str, Any],
    db: Database,
    planner: QueryPlanner,
    base_url: str = DEFAULT_REMOTE_URL,
) -> Dict[str, Any]:
    """Executes an MCP tool call in-process against Database & QueryPlanner."""
    try:
        if name == "memoria_search":
            query = str(arguments.get("query", "")).strip()
            limit = int(arguments.get("limit", 10))
            category = arguments.get("category")
            provider = arguments.get("provider")
            res = build_agent_search_response(
                db=db,
                planner=planner,
                query=query,
                limit=limit,
                category=category,
                provider=provider,
                base_url=base_url,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": res["markdown_brief"],
                    }
                ],
                "structuredContent": res,
            }

        elif name == "memoria_get_conversation":
            conv_id = str(arguments.get("conversation_id", "")).strip()
            max_messages = int(arguments.get("max_messages", 40))
            res = build_agent_conversation_response(
                db=db,
                conv_id=conv_id,
                max_messages=max_messages,
                base_url=base_url,
            )
            if not res:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Conversazione `{conv_id}` non trovata nel Vault."}],
                }
            return {
                "content": [{"type": "text", "text": res["markdown_transcript"]}],
                "structuredContent": res,
            }

        elif name == "memoria_ask":
            query = str(arguments.get("query", "")).strip()
            mode = str(arguments.get("mode", "auto"))
            res = planner.route_query(query, mode=mode)
            answer_md = res.get("answer") or json.dumps(res, ensure_ascii=False, indent=2)
            return {
                "content": [{"type": "text", "text": answer_md}],
                "structuredContent": res,
            }

        elif name == "memoria_remember_fact":
            subject = str(arguments.get("subject", "")).strip()
            predicate = str(arguments.get("predicate", "owned_by_user")).strip()
            object_value = str(arguments.get("object_value", "")).strip()
            category = arguments.get("category") or guess_category(subject)
            notes = arguments.get("notes") or "Registrato da Agente AI via MCP"

            ent_id = db.upsert_entity(subject, category)
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            stmt = StatementRecord(
                id=str(uuid.uuid4()),
                entity_id=ent_id,
                statement_type="FACT",
                subject=subject,
                predicate=predicate,
                object=object_value,
                author="agent",
                confidence=1.0,
                valid_from=now_iso,
                is_current=1,
                notes=notes,
            )
            stmt_id = db.insert_statement(stmt)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"✅ Fatto registrato in EngramVault (`id`: `{stmt_id}`): **{subject}** (`{category}`) -> `{predicate}`: **{object_value}**",
                    }
                ],
            }

        elif name == "memoria_remember_decision":
            topic = str(arguments.get("topic", "")).strip()
            decision_text = str(arguments.get("decision_text", "")).strip()
            status = str(arguments.get("status", "ADOPTED")).upper()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            dec = DecisionRecord(
                id=str(uuid.uuid4()),
                topic=topic,
                decision_text=decision_text,
                status=status,
                decided_at=now_iso,
            )
            dec_id = db.insert_decision(dec)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"✅ Decisione registrata in EngramVault (`id`: `{dec_id}`): **[{status}] {topic}** — {decision_text}",
                    }
                ],
            }

        elif name == "memoria_get_entity_graph":
            entity = str(arguments.get("entity", "")).strip()
            depth = int(arguments.get("depth", 1))
            graph_data = db.get_entity_graph(entity, depth=depth)
            root = graph_data.get("root")
            if not root:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Entità `{entity}` non trovata nel Knowledge Graph."}],
                }

            md_lines = [
                f"## 🕸️ Knowledge Graph: `{root['name']}` (`{root.get('category', 'general')}`)",
                f"- **Profondità esplorata**: {graph_data['depth']} salto/i",
                f"- **Nodi connessi**: {len(graph_data['nodes'])}",
                f"- **Archi/Relazioni**: {len(graph_data['edges'])}",
                "",
                "### Relazioni Semantiche & Connessioni:",
            ]
            if graph_data["edges"]:
                for e in graph_data["edges"]:
                    md_lines.append(
                        f"- **{e['from_name']}** (`{e['from_category']}`) $\\xrightarrow{{\\text{{{e['relation_type']}}}}}$ **{e['to_name']}** (`{e['to_category']}`)"
                    )
            else:
                md_lines.append("- _Nessuna relazione esplicita registrata per questa entità._")
            md_lines.append("")

            if graph_data["statements"]:
                md_lines.append(f"### Fatti & Claims sui nodi del grafo ({len(graph_data['statements'])}):")
                for s in graph_data["statements"][:15]:
                    md_lines.append(f"- **[{s['statement_type']}] {s['subject']}** $\\rightarrow$ `{s['predicate']}`: **{s['object']}**")
                md_lines.append("")

            return {
                "content": [{"type": "text", "text": "\n".join(md_lines)}],
                "structuredContent": graph_data,
            }

        elif name == "memoria_remember_relation":
            from_ent = str(arguments.get("from_entity", "")).strip()
            to_ent = str(arguments.get("to_entity", "")).strip()
            rel_type = str(arguments.get("relation_type", "relates_to")).strip()

            if not from_ent or not to_ent:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": "from_entity e to_entity sono obbligatori."}],
                }

            from engramvault.db.models import RelationRecord

            f_cat = guess_category(from_ent)
            t_cat = guess_category(to_ent)
            f_id = db.upsert_entity(from_ent, f_cat)
            t_id = db.upsert_entity(to_ent, t_cat)
            rel_rec = RelationRecord(
                id=str(uuid.uuid4()),
                from_entity_id=f_id,
                to_entity_id=t_id,
                relation_type=rel_type,
            )
            rel_id = db.insert_relation(rel_rec)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"✅ Relazione registrata in EngramVault (`id`: `{rel_id}`): **{from_ent}** (`{f_cat}`) $\\xrightarrow{{\\text{{{rel_type}}}}}$ **{to_ent}** (`{t_cat}`)",
                    }
                ],
            }

        else:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool MCP sconosciuto: `{name}`"}],
            }

    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Errore durante l'esecuzione di `{name}`: {e}"}],
        }


def handle_mcp_jsonrpc(
    payload: Dict[str, Any],
    db: Database,
    planner: QueryPlanner,
    base_url: str = DEFAULT_REMOTE_URL,
) -> Optional[Dict[str, Any]]:
    """Processes a single MCP JSON-RPC 2.0 request and returns the JSON-RPC 2.0 response dict."""
    method = payload.get("method", "")
    req_id = payload.get("id")
    params = payload.get("params") or {}

    # Notifications have no id and expect no response body
    if req_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        client_proto = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_proto,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": MCP_SERVER_INFO,
                "instructions": MCP_INSTRUCTIONS,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": MCP_TOOLS_SCHEMA,
            },
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        call_result = handle_mcp_tool_call(
            name=tool_name,
            arguments=tool_args,
            db=db,
            planner=planner,
            base_url=base_url,
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": call_result,
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


def _forward_to_remote_mcp(
    payload: Dict[str, Any],
    remote_url: str,
    api_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Forwards an MCP JSON-RPC request to the remote container (`http://localhost:5006/mcp`)."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    token = api_token or os.environ.get("MEMORIA_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        f"{remote_url}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read().decode("utf-8").strip()
        if not raw:
            return None
        return json.loads(raw)


def run_stdio_server():
    """Runs the MCP Server over stdio, forwarding to `MEMORIA_API_URL` (localhost:5006) with local SQLite fallback."""
    local_db: Optional[Database] = None
    local_planner: Optional[QueryPlanner] = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        resp = None
        try:
            resp = _forward_to_remote_mcp(payload, DEFAULT_REMOTE_URL)
        except Exception:
            if local_db is None:
                local_db = Database(DEFAULT_DB_PATH)
                local_planner = QueryPlanner(local_db)
            resp = handle_mcp_jsonrpc(payload, local_db, local_planner, base_url=DEFAULT_REMOTE_URL)

        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
