# Engram AI — AI Memory Vault (`engramvault.dev` · `engramvault.net`) — v2.1.0

> **Self-hosted, local-first AI Conversation Vault, Epistemic Knowledge Graph (`FACT` / `CLAIM` / `DECISION`), Hybrid Search Engine, and Native MCP Server for AI Agents.**
>
> *Un punto centralizzato, leggero e intelligente per memorizzare, indicizzare e distillare conoscenza da migliaia di conversazioni con diverse AI (ChatGPT, Claude, Claude Code, Google Antigravity, Gemini, OpenWebUI/Ollama, Markdown).*

[![Version](https://img.shields.io/badge/Engram_AI-v2.1.0-emerald.svg)](https://engramvault.dev)
[![REST API](https://img.shields.io/badge/REST_API-v1-teal.svg)](#agent-api--mcp-server)
[![MCP Ready](https://img.shields.io/badge/MCP-2.0-indigo.svg)](#agent-api--mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

---

## 💡 Philosophy & Architecture

When working across multiple AI assistants and coding agents (**ChatGPT**, **Claude**, **Claude Code**, **Google Antigravity**, **Gemini**, **Ollama**), personal context and technical decisions get fragmented into isolated silos.

`Memoria AI` unifies your entire AI history into a single **SQLite (`vault.db`)** instance organized in 4 layers:

```mermaid
flowchart LR
    subgraph Sources["AI Exports & Live Sync"]
        A1["ChatGPT / Claude / Gemini JSON"]
        A2["Claude Code & Antigravity JSONL"]
        A3["Drop-Folder Auto Watcher"]
    end
    subgraph Vault["Memoria AI (SQLite + FTS5 + 384-dim Vectors)"]
        B1["Immutable Conversation Vault (SHA-256)"]
        B2["Epistemic Graph (FACT · CLAIM · DECISION · TIMELINE)"]
        B3["Hybrid Search (BM25 + Semantic Vectors + RRF)"]
    end
    subgraph Agents["System 1/2 & MCP Clients"]
        C1["Optional Jev System 1 Router (~100ms)"]
        C2["Native MCP 2.0 Server (/mcp)"]
        C3["REST Agent API (/api/v1/agent/*)"]
    end
    Sources --> Vault --> Agents
```

1. **Conversation Vault**: Normalized, immutable, SHA-256 deduplicated archive of all raw dialogues and agent transcripts.
2. **Multi-Backend Hybrid Search (`FTS5 BM25` + `384-dim Semantic Vectors` + `RRF`)**:
   - Supports **External Embedding APIs** (Google Gemini `text-embedding-004`, Ollama `nomic-embed-text`, OpenAI `/v1/embeddings`), **FastEmbed ONNX**, and a zero-latency **Multilingual Concept-Cluster Vectorizer (IT/EN)** (`multilingual-concept-hash-384`), plus one-click global Vault Re-Indexing (`POST /api/embeddings/reindex`).
3. **Epistemic Knowledge Layer (`FACT` vs `CLAIM` vs `DECISION` + `TIMELINE`)**:
   - **`FACT`**: Verified user inventory, owned hardware/software, and permanent configurations (`owned_by_user`, `specs`, `configured_with`).
   - **`CLAIM`**: Technical statements and advice extracted from AI responses.
   - **`DECISION`**: Explicit architectural, technical, or purchasing choices (`ADOPTED`, `EVALUATING`, `ABANDONED`).
   - **`TIMELINE`**: Temporal state tracking (`valid_from`, `valid_until`, `is_current`).
4. **System 1 (Optional Jev) + System 2 (LLM) Synthesis**:
   - **System 1 Fast Decision Engine (Optional — TypeSafe Jev / OpenRouter `typesafe/jev-1.13`)**: Evaluates `Choice`, `Noul`, and `Score` primitives in ~100ms for query routing, category filtering, batch enrichment noise gatekeeping, and chunk reranking—with automatic zero-latency local fallback when disabled or offline.
   - **System 2 Synthesis**: Synthesizes multi-hop answers using your configured LLM provider (Gemini, Anthropic, OpenAI, OpenRouter, Groq, or local Ollama).

---

## 🚀 60-Second Quickstart & First-Run Onboarding Wizard

```bash
docker run -d --name memoria-ai -p 5006:8080 -v memoria_data:/app/data gaskb/memoria-ai-anti:latest
```

Open **`http://localhost:5006`**:
1. **First-Run Wizard (`🚀 Quickstart & MCP`)**: Automatically guides you on a fresh instance (or anytime via the header button):
   - **Step 1**: Connect local **Ollama** or a cloud API key (multilingual 384-dim semantic search works immediately with zero config).
   - **Step 2**: Drag-and-drop your `conversations.json` export OR click **`✨ Load Demo Dataset`** to immediately populate 6 sample conversations, verified `FACT` inventory, and `DECISION` records.
   - **Step 3**: Click **`📋 Copy MCP Config JSON`** to paste your live server configuration (including Bearer Token if enabled) into **Google Antigravity**, **Claude Code**, **Claude Desktop**, or **Cursor**.

---

## 🎨 Multi-Theme Engine & Custom Theme Studio (`gaskb-style` M3)

- **5 Built-in M3 Palettes (`Dark` & `Light` modes)**: Switch instantaneously in the header between **🟢 Emerald Cyber**, **🟠 Amber Gold**, **🟣 Indigo Deep**, **🔴 Ruby Crimson**, and **⚪ Slate Minimal**.
- **Custom Theme Studio (`+ Create Custom Theme`)**: Pick custom Primary, Secondary, Background, and Surface colors with live preview, save to SQLite (`POST /api/themes`), and export/import theme JSON.

---

## 🔐 Security, Bearer Token & Web UI Password

- **API / MCP Bearer Token (`MEMORIA_API_TOKEN`)**: Protect `/api/v1/agent/*` and `/mcp` with `Authorization: Bearer <token>`. Generate, view, copy, or toggle enforcement live in **⚙️ Settings → Security**.
- **Optional Web UI Password (`MEMORIA_WEB_PASSWORD`)**: Enable an `HttpOnly` signed session cookie and login screen (`/login`) when exposing the Web UI outside a trusted LAN.

---

## 🤖 Agent API & Native MCP Server

| MCP Tool / REST Endpoint | Description |
|---|---|
| `memoria_search` (`GET /api/v1/agent/search?q=...`) | Hybrid search across `FACT`, `CLAIM`, `DECISION`, and 80,000+ conversation chunks in `< 30ms`. |
| `memoria_get_conversation` (`GET /api/v1/agent/conversation/{id}`) | Retrieves the full chronological transcript of any conversation by `conversation_id`. |
| `memoria_ask` (`POST /api/v1/agent/ask`) | Executes multi-hop synthesis across verified inventory (`FACT`) and past discussions. |
| `memoria_remember_fact` (`POST /api/v1/agent/remember`) | Persists a verified `FACT` or `CLAIM` directly into the Knowledge Layer. |
| `memoria_remember_decision` (`POST /api/v1/agent/remember`) | Records a technical or architectural `DECISION` in the permanent registry. |

---

## 🌐 Internationalization (`i18n`) & Environment Variables

- **Modular `i18n` (`memoria/i18n/locales/*.json`)**: Ships with **English (`en`)** as default and full **Italian (`it`)** support across all 6 tabs, modals, and dynamic cards. Adding a new language requires only dropping `<lang>.json` in `memoria/i18n/locales/`.
- **Environment Variables**:
  ```env
  MEMORIA_DEFAULT_LANG=en
  MEMORIA_DEFAULT_THEME=emerald           # emerald | amber | indigo | ruby | slate
  MEMORIA_API_TOKEN=                      # Optional Bearer token for /api/v1/agent/* and /mcp
  MEMORIA_WEB_PASSWORD=                   # Optional password for Web UI login screen
  MEMORIA_EMBEDDING_BACKEND=auto          # auto | api | gemini | ollama | fastembed | native
  MEMORIA_JEV_ENABLED=false               # Optional System 1 Jev engine (true | false)
  MEMORIA_JEV_MODEL=typesafe/jev-1.13
  ```

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
