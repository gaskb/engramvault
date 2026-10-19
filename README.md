# EngramVault — AI Memory Vault (`engramvault.net` · `engramvault.dev`) — v2.13.0

> **Self-hosted, local-first AI Conversation Vault, Epistemic Knowledge Graph (`FACT` / `CLAIM` / `DECISION`), Hybrid Search Engine, and Native MCP Server for AI Agents.**
>
> *Un punto centralizzato, leggero e intelligente per memorizzare, indicizzare e distillare conoscenza da migliaia di conversazioni con diverse AI (ChatGPT, Claude, Claude Code, Google Antigravity, Gemini, OpenWebUI/Ollama, Markdown).*

[![Version](https://img.shields.io/badge/EngramVault-v2.13.0-emerald.svg)](https://engramvault.net)
[![REST API](https://img.shields.io/badge/REST_API-v1-teal.svg)](#agent-api--mcp-server)
[![MCP Ready](https://img.shields.io/badge/MCP-2.0-indigo.svg)](#agent-api--mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/gaskb/engramvault/actions/workflows/ci.yml/badge.svg)](https://github.com/gaskb/engramvault/actions)

**English** · [🇮🇹 Italiano](README.it.md)

---

## 💡 Philosophy & Architecture

When working across multiple AI assistants and coding agents (**ChatGPT**, **Claude**, **Claude Code**, **Google Antigravity**, **Gemini**, **Ollama**), personal context and technical decisions get fragmented into isolated silos.

`EngramVault` unifies your entire AI history into a single **SQLite (`vault.db`)** instance organized in 4 layers:

```mermaid
flowchart LR
    subgraph Sources["AI Exports & Live Sync"]
        A1["ChatGPT / Claude / Gemini JSON"]
        A2["Claude Code & Antigravity JSONL"]
        A3["Drop-Folder Auto Watcher"]
    end
    subgraph Vault["EngramVault (SQLite + FTS5 + 384-dim Vectors)"]
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
   - **System 1 Fast Decision Engine (Optional — TypeSafe Jev / OpenRouter `typesafe/jev-1.13` or Local `jev-system1-local` on CPU)**: Evaluates `Choice`, `Noul`, and `Score` primitives in ~100ms (cloud) or ~1.5s (local CPU via quantized Qwen 2.5 1.5B on `llama.cpp:server`) for query routing, category filtering, batch enrichment noise gatekeeping, and chunk reranking—with automatic zero-latency local heuristic fallback when disabled or offline.
   - **System 2 Synthesis**: Synthesizes multi-hop answers using your configured LLM provider (Gemini, Anthropic, OpenAI, OpenRouter, Groq, or local Ollama).

---

## 🧠 Epistemic Hierarchy & AI Extraction (Human-in-the-Loop)

Unlike traditional RAG systems or unstructured note tools that blindly trust LLM summaries, **EngramVault enforces a strict epistemic hierarchy**:

* **`EVIDENCE` (Immutable Truth)**: The exact raw transcripts from ChatGPT, Claude, Antigravity, or Gemini. Nothing is ever altered, hallucinated, or discarded.
* **`CLAIM` / Extracted Relations (AI-Assisted)**: Suggested statements, concepts, and relationships distilled from conversations via LLMs or local heuristics. Every extracted statement is explicitly tagged as an unverified `CLAIM` until validated.
* **`FACT` (Human-Verified Knowledge)**: Statements confirmed, promoted, or directly authored by the user (hardware specs, system topology, verified configurations). Preserved with SCD Type 2 temporal validity (`valid_from`, `valid_until`, `is_current`, `supersedes_id`).
* **`DECISION` (Explicit Choices)**: Architectural or technical trade-offs recorded with rationale, status (`ADOPTED`, `EVALUATING`, `ABANDONED`), and source references.

### Privacy & Extraction Modes:
* **100% Offline / Local-First (Default)**: EngramVault operates entirely within your network. Heuristic extraction and concept clustering run locally without sending any data to external cloud APIs.
* **LLM-Powered Distillation (Optional)**: If you provide an API key (Ollama, Gemini, Anthropic, OpenAI, DeepSeek, Grok), the batch enricher uses the LLM to structure complex entity graphs. You always retain complete control via the **Curation Studio** to review, promote, or discard suggested claims.

---

## 🚀 60-Second Quickstart

```bash
docker run -d --name engramvault -p 5006:8080 -v engramvault_data:/app/data gaskb/engramvault:latest
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

## 🔐 Security, Bearer Token, Web Lockdown & Ingest Protection (v2.9.1)

- **Full Web API Lockdown (`MEMORIA_WEB_PASSWORD`)**: When web password protection is active, all `/api/*` read and write endpoints (`/api/conversations`, `/api/entities`, `/api/decisions`, `/api/relations`, `/api/stats`, `/api/settings`) strictly require an authenticated web session cookie or Bearer Token, rejecting unauthorized requests with `401 Unauthorized`.
- **API / MCP Bearer Token (`MEMORIA_API_TOKEN`)**: Protect `/api/v1/agent/*` and `/mcp` with `Authorization: Bearer <token>`. Supported transparently in `ClaudeCodeSyncer`, `AntigravitySyncer`, and stdio MCP proxy. Generate, view, copy, or toggle enforcement live in **⚙️ Settings → Security**.
- **Safe Streaming Ingest & Resource Limits (`MEMORIA_MAX_UPLOAD_MB`)**:
  - Enforces strict extension whitelist (`.json`, `.jsonl`, `.zip`, `.md`, `.txt`), rejecting unauthorized formats with `400 Bad Request`.
  - Streams uploads in chunks with configurable ceiling (default: `100MB`), cleanly returning `413 Payload Too Large` if breached.
  - Offloads parsing, chunking, and embedding generation to `asyncio.to_thread` to maintain zero-latency event loop responsiveness.
- **Strict Schema Validation (Pydantic Literals)**: Agent and curation endpoints validate status and statement types via Pydantic Literals, returning clean `422 Unprocessable Entity` responses rather than unhandled database 500 errors.
- **Stored XSS Defense**: All Markdown rendering and card interpolations are strictly sanitized with **DOMPurify** and HTML entity escaping.
- **Provider API Key Privacy**: Google Gemini API keys are sent via official `x-goog-api-key` HTTP headers rather than URL query parameters to prevent exposure in proxy logs.

---

## 📲 Progressive Web App (PWA) & Mobile Integration (v2.4.0)

Engram AI / EngramVault can be installed directly onto any smartphone (Android and iOS) as a **Progressive Web App**:

- **Native Standalone Experience**: Launches in full-screen standalone mode without browser navigation bars, featuring custom high-resolution icons (192x192, 512x512, maskable, and 180x180 Apple Touch Icon).
- **Web Share Target (Android)**: Integrates directly with the native OS share sheet. Share any text, article link, YouTube video, or note from Chrome, WhatsApp, or other apps directly to EngramVault to trigger instant hybrid search or save as a verified `FACT` or `DECISION`.
- **Service Worker & Offline Shell**: Static application assets (CSS design system, themes, icons, core shell) are cached locally via `service-worker.js` with a stale-while-revalidate strategy for instant load times and offline shell resilience.
- **Mobile App Backlog Roadmap**: Tracks future native/hybrid extensions (Issue #27) including Android home screen widgets, Whisper-powered quick audio memos, and local read-only SQLite replicas.

### How to Install on Mobile:
- **Android (Chrome / Brave / Edge)**: Open your instance URL (e.g. `http://localhost:5006` or `https://vault.example.com`) $\rightarrow$ tap the browser menu (⋮) $\rightarrow$ select **"Install app"** or **"Add to Home screen"**.
- **iOS (Safari)**: Open your instance URL $\rightarrow$ tap the Share button (⎋) $\rightarrow$ select **"Add to Home Screen"**.

## 🕸️ Knowledge Graph & Multi-Hop Reasoning (v2.9.0)

Engram AI models interconnected homelab topology, software stacks, hardware setups, and project dependencies using an epistemic **Knowledge Graph**:

- **Typed Directed Edges**: First-class relations linking entities with semantic predicates:
  - `connected_to` (e.g. `Focusrite Scarlett 2i2` $\rightarrow$ `MacBook Pro M1`)
  - `runs_on` / `installed_on` (e.g. `EngramVault` $\rightarrow$ `Docker Server 10.0.0.5`)
  - `depends_on` / `uses` (e.g. `Home Assistant` $\rightarrow$ `Zigbee2MQTT`)
  - `part_of` / `subproject_of` (e.g. `Agent Curation UI` $\rightarrow$ `Engram AI`)
  - `compatible_with`, `created_by`, `configured_with`.
- **Multi-Hop Traversal (BFS)**:
  - 1-hop and 2-hop graph expansion integrated natively into the DB engine (`Database.get_entity_graph`).
  - Automatically queries outward connections during agent search (`memoria_search`) and synthesizes relational context in `memoria_ask` / `planner.execute_synthetic`.
- **Interactive Web UI Relation Explorer**:
  - Live filterable grid of entity relationships directly within the **Entities & Facts** tab.
  - Interactive creation modal (`+ Add Relation`) and instant deletion/curation tools.
  - Live relationship counter (`enrich-stat-relations`) in the Ingest & Enrich dashboard.
- **Automated Two-Stage Relation Extraction CLI (`scripts/extract_relations.py`)**:
  - *Stage 1*: Zero-latency heuristic mining from existing statement predicates (`installed_on`, `runs_on`, `connected_to`, `part_of`, `depends_on`).
  - *Stage 2*: Token-conservative batch LLM extraction across historical conversations.

---

## 🏷️ Open-World Taxonomy & Dynamic Category Auto-Discovery (v2.8.0)

- **Dynamic Open-World Category Auto-Discovery**: The Level 3 Epistemic Knowledge Enricher now feeds known categories dynamically into extraction prompts. When encountering entities in novel specialized domains (e.g. `ham_radio`, `homelab_infra`, `audio_pro_studio`, `hardware_pc`, `cooking`, `personal_life`), it automatically creates concise snake_case categories rather than dumping them into a generic catch-all bucket.
- **Category Promotion, Sanitization & Canonical Aliasing (v2.9.2)**: `Database.upsert_entity` sanitizes categories into snake_case and transparently canonicalizes fragmented variants into unified domains (e.g. `homelab`, `infrastruttura`, and `homelab & infra` $\rightarrow$ `homelab_infra`; `hardware` $\rightarrow$ `hardware_pc`; `feature` $\rightarrow$ `audio_interface`).
- **Batch Re-clustering CLI (`scripts/recluster_categories.py`)**: Migrates existing legacy entities from `general` to specific domain categories using a two-stage pipeline:
  1. *Stage 1 (Zero-Latency Heuristics)*: Matches against domain keywords instantly (e.g. transceivers, sound cards, hypervisors, decks).
  2. *Stage 2 (Batch LLM Index-Mapping)*: Groups entities in efficient 50-item indexed payloads for sub-second, token-conservative classification via `deepseek-chat` or local models.
  ```bash
  # Dry-run preview
  python scripts/recluster_categories.py --dry-run
  # Full live execution on vault.db
  python scripts/recluster_categories.py --db-path vault.db --batch-size 50
  ```

---

## ⚡ Concurrency, CLI & UX Enhancements (v2.3.0)

- **Non-Blocking Async Event Loop**: All synchronous SQLite, BM25, embedding matrix operations, and LLM queries are offloaded via `asyncio.to_thread` to ensure sub-millisecond API responsiveness under load.
- **Vector BLOB Reshape Protection**: Validates 384-dimensional float buffer integrity (`1536` bytes) in `_ensure_vector_cache`, gracefully skipping malformed entries without runtime crashes.
- **Real-Time Unenriched Conversation Tracker**: The "Ingest & Enrich" view displays real-time counters of unanalyzed conversations (`X in attesa / Y totali`), providing clear feedback on batch extraction status.
- **Viewport Layout Optimization**: Zero-margin base styling ensures the footer is cleanly visible at the bottom of the viewport on initial load without default vertical scrollbars.
- **CLI Robustness**: Full unit-test suite for `engramvault` CLI commands (`init`, `seed`, `stats`, `search`, `ask`, `entities`, `decisions`, `timeline`) with support for global `--db` positioning (legacy `memoria` alias preserved).

---

## 🔒 Sensitive Workspace Exclusion & Sentinel Files (v2.12.6)

Prevent confidential projects, NDA repositories, or personal workspaces from syncing to EngramVault across both **Google Antigravity** and **Claude Code**:

- **Sentinel Files**: Simply drop a file named `.no-vault`, `.no-engramvault`, or `.engramvaultignore` into the root of any confidential project folder. Both syncers recursively detect these sentinels and exclude the session.
- **Pattern Exclusions**: Pass `--exclude "*pattern1*,*pattern2*"` via CLI, set the `ENGRAMVAULT_SYNC_EXCLUDE` environment variable, or add patterns to `~/.engramvault/sync_ignore.txt`.
- **Zero Overhead**: Excluded sessions are remembered with `ignored: true` in sync state so they are skipped without rescanning transcripts.

---

## 🤖 Agent API & Native MCP Server

| MCP Tool / REST Endpoint | Description |
|---|---|
| `memoria_search` (`GET /api/v1/agent/search?q=...`) | Hybrid search across `FACT`, `CLAIM`, `DECISION`, Knowledge Graph relations, and 80,000+ conversation chunks in `< 30ms`. |
| `memoria_get_conversation` (`GET /api/v1/agent/conversation/{id}`) | Retrieves the full chronological transcript of any conversation by `conversation_id`. |
| `memoria_get_entity_graph` (`GET /api/v1/agent/graph/{entity}`) | Traverses outgoing & incoming Knowledge Graph connections (1-hop / 2-hop) for any entity. |
| `memoria_ask` (`POST /api/v1/agent/ask`) | Executes multi-hop synthesis across verified inventory (`FACT`), graph topology, and past discussions. |
| `memoria_remember_fact` (`POST /api/v1/agent/remember`) | Persists a verified `FACT` or `CLAIM` directly into the Knowledge Layer. |
| `memoria_remember_decision` (`POST /api/v1/agent/remember`) | Records a technical or architectural `DECISION` in the permanent registry. |
| `memoria_remember_relation` (`POST /api/v1/agent/relation`) | Connects two entities with a typed directed edge (`from_entity`, `relation_type`, `to_entity`). |

---

## 🌐 Internationalization (`i18n`) & Environment Variables

- **Modular `i18n` (`engramvault/i18n/locales/*.json`)**: Ships with **English (`en`)** as default and full **Italian (`it`)** support across all 6 tabs, modals, and dynamic cards. Adding a new language requires only dropping `<lang>.json` in `engramvault/i18n/locales/`.
- **Environment Variables**:
  All variables support the `ENGRAMVAULT_` prefix (with transparent fallback to legacy `MEMORIA_` prefixes):
  ```env
  ENGRAMVAULT_DEFAULT_LANG=en
  ENGRAMVAULT_DEFAULT_THEME=emerald       # emerald | amber | indigo | ruby | slate
  ENGRAMVAULT_API_TOKEN=                  # Optional Bearer token for /api/v1/agent/* and /mcp
  ENGRAMVAULT_WEB_PASSWORD=               # Optional password for Web UI login screen
  ENGRAMVAULT_EMBEDDING_BACKEND=auto      # auto | api | gemini | ollama | fastembed | native
  ENGRAMVAULT_LLM_PROVIDER=               # Optional LLM provider: openai | anthropic | gemini | grok | deepseek | qwen
  ENGRAMVAULT_JEV_ENABLED=false           # Optional System 1 Jev engine (true | false)
  ENGRAMVAULT_JEV_MODEL=typesafe/jev-1.13 # typesafe/jev-1.13 or local qwen2.5-1.5b
  ENGRAMVAULT_JEV_BASE_URL=https://openrouter.ai/api/v1
  ENGRAMVAULT_JEV_TIMEOUT=5.0             # Timeout in seconds before falling back to local heuristic
  ```

---

## 🎨 UI & Design System (`gaskb_style`)

- **Pure Vector SVG Iconography**: Zero emoji di sistema nei menu, card e azioni; set completo di icone vettoriali SVG scalabili ad alta definizione con ereditarietà di colore `currentColor` per una visualizzazione nitida su qualsiasi densità di pixel.
- **Multi-Theme Engine (M3 Dark & Light)**: 5 palette integrate (`emerald`, `amber`, `indigo`, `ruby`, `slate`) con Custom Theme Studio per creare, esportare e importare temi personalizzati.

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📄 License & Support

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

*EngramVault creates a private, self-hosted memory for your AI workflow and is distributed for free. If you find it useful, feel free to [buy me a coffee on Ko-fi ☕](https://ko-fi.com/gaskb).*

*EngramVault è gratuito. Se ti torna utile, sentiti libero di [offrirmi un caffè su Ko-fi ☕](https://ko-fi.com/gaskb).*
