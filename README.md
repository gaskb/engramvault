# Memoria AI (`memoria_ai_anti`) — v1.5.1

> **Self-hosted, local-first AI Conversation Vault, Epistemic Knowledge Graph (`FACT` / `CLAIM` / `DECISION`), Hybrid Search Engine, and Native MCP Server for AI Agents.**
>
> *Un punto centralizzato, leggero e intelligente per memorizzare, indicizzare e distillare conoscenza da migliaia di conversazioni con diverse AI (ChatGPT, Claude, Claude Code, Google Antigravity, Gemini, OpenWebUI/Ollama, Markdown).*

[![Version](https://img.shields.io/badge/Memoria_AI-v1.5.1-emerald.svg)](#)
[![REST API](https://img.shields.io/badge/REST_API-v1-teal.svg)](#agent-api--mcp-server)
[![MCP Ready](https://img.shields.io/badge/MCP-2.0-indigo.svg)](#agent-api--mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

---

## 💡 Philosophy & Architecture

When working across multiple AI assistants and coding agents (**ChatGPT**, **Claude**, **Claude Code**, **Google Antigravity**, **Gemini**, **Ollama**), personal context and technical decisions get fragmented into isolated silos.

`Memoria AI` unifies your entire AI history into a single **SQLite (`vault.db`)** instance organized in 4 layers:

1. **Conversation Vault**: Normalized, immutable, SHA-256 deduplicated archive of all raw dialogues and agent transcripts.
2. **Multi-Backend Hybrid Search (`FTS5 BM25` + `384-dim Semantic Vectors` + `RRF`)**:
   - Supports **External Embedding APIs** (Google Gemini `text-embedding-004`, Ollama `nomic-embed-text`, OpenAI `/v1/embeddings`), **FastEmbed ONNX**, and a zero-latency **Multilingual Concept-Cluster Vectorizer (IT/EN)** (`multilingual-concept-hash-384`), plus one-click global Vault Re-Indexing (`POST /api/embeddings/reindex`).
3. **Epistemic Knowledge Layer (`FACT` vs `CLAIM` vs `DECISION` + `TIMELINE`)**:
   - **`FACT`**: Verified user inventory, owned hardware/software, and permanent configurations (`owned_by_user`, `specs`, `configured_with`).
   - **`CLAIM`**: Technical statements and advice extracted from AI responses.
   - **`DECISION`**: Explicit architectural, technical, or purchasing choices (`ADOPTED`, `EVALUATING`, `SUPERSEDED`).
   - **`TIMELINE`**: Temporal state tracking (`valid_from`, `valid_until`, `is_current`).
4. **System 1 (Optional Jev) + System 2 (LLM) Synthesis**:
   - **System 1 Fast Decision Engine (Optional — TypeSafe Jev / OpenRouter `typesafe/jev-1.13`)**: Evaluates `Choice`, `Noul`, and `Score` primitives in ~100ms for query routing, category filtering, batch enrichment noise gatekeeping, and chunk reranking—with automatic zero-latency local fallback when disabled or offline.
   - **System 2 Synthesis**: Synthesizes multi-hop answers using your configured LLM provider (Gemini, Anthropic, OpenAI, OpenRouter, Groq, or local Ollama).

---

## 🤖 Agent API & Native MCP Server

Memoria AI exposes both a **Model Context Protocol (MCP JSON-RPC 2.0)** endpoint (`POST /mcp`) and a **REST Agent API** (`/api/v1/agent/*`) so autonomous agents (Antigravity, Claude Code, Cursor, OpenWebUI) can query your historical memory and record new facts/decisions in real time:

| MCP Tool / REST Endpoint | Description |
|---|---|
| `memoria_search` (`GET /api/v1/agent/search?q=...`) | Hybrid search across `FACT`, `CLAIM`, `DECISION`, and 80,000+ conversation chunks in `< 30ms`. |
| `memoria_get_conversation` (`GET /api/v1/agent/conversation/{id}`) | Retrieves the full chronological transcript of any conversation by `conversation_id`. |
| `memoria_ask` (`POST /api/query`) | Executes multi-hop synthesis across verified inventory (`FACT`) and past discussions. |
| `memoria_remember_fact` (`POST /api/knowledge/statement`) | Persists a verified `FACT` or `CLAIM` directly into the Knowledge Layer. |
| `memoria_remember_decision` (`POST /api/knowledge/decision`) | Records a technical or architectural `DECISION` in the permanent registry. |

---

## 🌐 Internationalization (`i18n`) & White-Labeling

- **Modular `i18n` (`memoria/i18n/locales/*.json`)**: Ships with **English (`en`)** as default and full **Italian (`it`)** support, switchable live in the UI header. Adding a new language requires only dropping a `<lang>.json` file in `memoria/i18n/locales/` (auto-discovered with automatic fallback to `en.json`).
- **White-Label Environment Variables**:
  ```env
  MEMORIA_DEFAULT_LANG=en
  MEMORIA_EMBEDDING_BACKEND=auto          # auto | api | gemini | ollama | fastembed | native
  MEMORIA_JEV_ENABLED=false               # Optional System 1 Jev engine (true | false)
  MEMORIA_JEV_MODEL=typesafe/jev-1.13
  MEMORIA_INSTANCE_NAME=Local Node
  MEMORIA_DOCS_URL=
  MEMORIA_REPO_URL=https://github.com/gaskb/memoria-ai
  ```

---

## 🔄 Automatic Sync Daemons (Claude Code & Google Antigravity)

Includes incremental sync scripts and macOS LaunchAgents (`deploy/launchd/`) to automatically ship new local sessions to your Memoria AI server:
- `memoria sync-antigravity --brain-dir ~/.gemini/antigravity/brain --remote-url http://<host>:5006`
- `memoria sync-claude --claude-dir ~/.claude/projects --remote-url http://<host>:5006`

---

## 🐳 Quick Start (Docker Compose)

```bash
docker compose up -d
```
Open `http://localhost:5006` (or `http://localhost:8080` locally via `memoria serve`).

---

## 🧪 Running Tests

```bash
./venv/bin/pytest
```

---

## 📄 License & Support

Released under the **MIT License**.

*Memoria AI creates a private, self-hosted memory for your AI workflow and is distributed for free. If you find it useful, feel free to [buy me a coffee on Ko-fi ☕](https://ko-fi.com/gaskb).*

*Memoria AI è gratuito. Se ti torna utile, sentiti libero di [offrirmi un caffè su Ko-fi ☕](https://ko-fi.com/gaskb).*
