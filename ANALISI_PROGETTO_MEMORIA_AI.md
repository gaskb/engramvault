# Report di Analisi e Revisione Tecnica: Memoria AI (`memoria_ai_anti`)

> **Data Analisi:** 26 Settembre 2026  
> **Versione Analizzata:** v2.2.0 (UI: 2.2.0 / API: v1 / MCP: 2.0)  
> **Repository:** `memoria_ai_anti` (Homelab Node `.220`)  
> **Oggetto:** Audit di validità, qualità del codice, solidità, sicurezza, copertura test, architettura e UX/UI.

---

## 1. Giudizio Complessivo & Sintesi Esecutiva

Il progetto **Memoria AI** (`memoria_ai_anti`) è un progetto **estremamente valido, concettualmente avanzato e ben posizionato per un ambiente Homelab**. 

Risolve un problema reale e critico: la frammentazione e la dispersione delle conversazioni con diversi assistenti AI (ChatGPT, Claude, Claude Code, Gemini, Ollama, Antigravity) e la perdita di memoria a lungo termine degli agenti di sviluppo.

### I grandi punti di forza del progetto:
1. **Architettura Zero-Overhead per Homelab:** L'idea di non usare un database vettoriale esterno pesante (come Qdrant, Milvus o Pinecone), affidandosi a un unico file SQLite con FTS5 per il full-text search, BLOB vettoriali a 384 dimensioni e **Reciprocal Rank Fusion (RRF)** per la ricerca ibrida, è brillante per efficienza, consumi di memoria e facilità di backup.
2. **Epistemic Cleanliness (FACT vs CLAIM & SCD Type 2):** La separazione epistemica tra dichiarazioni dell'utente (fatti verificati sull'inventario, setup o preferenze) e pareri/risposte generate dai modelli AI (claim), unita alla storicizzazione temporale (`is_current`, `valid_from`, `valid_until`, `supersedes_id`), è una delle intuizioni concettuali migliori viste in progetti di memoria personale.
3. **Integrazione Nativa con l'Ecosistema Agentico (MCP & REST API):** Il supporto completo al Model Context Protocol (MCP) sia via HTTP Streamable/SSE sia via stdio, insieme alle REST API per agenti (`/api/v1/agent/*`), rende il sistema immediatamente fruibile da Antigravity, Claude Code, Cursor e script di automazione.
4. **Design System & Aderenza agli Standard Homelab:** L'adozione rigorosa dello standard `gaskb_style`, i doppi badge di versione (`UI_VERSION` / `API_VERSION`), il multi-theme engine (5 palette e supporto Light/Dark) e l'interfaccia curata ne fanno un'applicazione gradevole e coerente con il resto dell'homelab.

Nonostante l'alta qualità generale e una suite di **69 test automatici tutti passanti**, l'analisi ha evidenziato **alcune criticità di sicurezza da sanare con priorità alta**, diversi **anti-pattern di codice** (duplicazione rotte, codice sincrono nell'event loop async) e **margini di irrobustimento** sul fronte della robustezza e della manutenibilità frontend.

Di seguito vengono dettagliate le osservazioni suddivise nelle categorie richieste.

---

## 2. Sicurezza (Security)

### 🔴 Critico: Token Leak su Endpoint Non Autenticato (`/api/onboarding/mcp-config`)
- **Posizione:** `memoria/web/routes/security.py`, righe 158–173.
- **Problema:** L'endpoint `GET /api/onboarding/mcp-config` è esposto pubblicamente **senza alcuna dipendenza di autenticazione** (`require_mutation_auth` o Bearer check). Il metodo legge `sec = security_mgr.get_status()` e, se il token è richiesto, include nei dati restituiti l'header:
  ```python
  headers["Authorization"] = f"Bearer {sec['api_token']}"
  ```
  Chiunque sulla rete locale (o su Internet, se esposto tramite reverse proxy senza basic auth esterna) può interrogare `GET http://<host>:5006/api/onboarding/mcp-config` e **ricevere il token API in chiaro**, vanificando completamente la protezione Bearer Token del server MCP e delle API agenti.
- **Rimedio Raccomandato:** Proteggere l'endpoint con `dependencies=[Depends(require_mutation_auth)]` oppure richiedere una sessione web autenticata prima di mostrare la configurazione con il token in chiaro.

---

### 🔴 Alto: Rischio Stored XSS nel Frontend Web (`templates/index.html`)
- **Posizione:** `memoria/web/templates/index.html` (funzioni di rendering `renderMarkdown`, card statements, decisioni, cronologia, messaggi).
- **Problema:** Nel file JavaScript della dashboard:
  1. I campi del database (titoli conversazioni, subject e object degli statement, topic e testo delle decisioni, contenuti dei messaggi) vengono iniettati direttamente in blocchi `innerHTML` tramite template literals JavaScript (es. `card.innerHTML = `... <span>${stmt.subject}</span> ...``) **senza alcuna funzione di escaping HTML** (`escapeHtml` non è implementata nel template).
  2. La funzione `renderMarkdown()` utilizza `marked.parse(processed)` iniettandolo direttamente in `document.getElementById('query-answer-content').innerHTML = ...`. Nelle versioni recenti di `marked.js`, i tag HTML e gli schemi URI (es. `[test](javascript:alert(1))`) non vengono filtrati di default.
  Se una conversazione passata importata contiene frammenti HTML/JS dannosi o un payload proveniente da un modello AI / prompt injection, questo viene eseguito nel browser dell'utente amministratore.
- **Rimedio Raccomandato:**
  - Introdurre una funzione helper `escapeHtml(str)` per tutte le interpolazioni in template string.
  - Includere la libreria `DOMPurify` prima di invocare `marked.parse` (`DOMPurify.sanitize(marked.parse(...))`).

---

### 🟡 Medio: Esposizione API Key Google Gemini nella Query String
- **Posizione:** `memoria/llm/providers/gemini.py`, riga 44:
  ```python
  url = f"{self.base_url}/v1beta/models/{clean_model}:generateContent?key={self.api_key}"
  ```
- **Problema:** Passare la chiave API in query string espone la chiave ai log di accesso di eventuali proxy HTTP intermedi, ai log del server web e alle tracce di debug in caso di crash di rete. In `memoria/search/embeddings.py` (riga 144) era stata correttamente implementata la regola `S-01` usando l'header HTTP `x-goog-api-key: key`.
- **Rimedio Raccomandato:** Allineare `GeminiProvider` a usare l'header `headers={"x-goog-api-key": self.api_key}` rimuovendo `?key=` dall'URL.

---

### 🟡 Medio: Mancanza di Rate Limiting / Protezione Brute-Force su Login Web
- **Posizione:** `memoria/web/routes/security.py`, endpoint `POST /api/auth/login`.
- **Problema:** L'endpoint di login verifica la password tramite PBKDF2 ma non ha alcun meccanismo di rate limiting, tarpit o blocco temporaneo su tentativi falliti ripetuti.
- **Rimedio Raccomandato:** Aggiungere un contatore di tentativi falliti in-memory con ban temporaneo (es. 5 tentativi errati = blocco di 60 secondi) o un middleware di rate limiting (es. `slowapi`).

---

### 🔵 Basso: API Key Salvate in Chiaro su Database SQLite
- **Posizione:** `memoria/db/schema.sql` (tabella `settings`) e `memoria/settings/manager.py`.
- **Problema:** Le chiavi dei vari provider (OpenAI, Anthropic, Gemini, Grok, DeepSeek) vengono memorizzate in chiaro nella tabella `settings` (`key = "api_key:openai"`). Se il file `vault.db` viene estratto o salvato come backup non cifrato, le credenziali sono leggibili.
- **Rimedio Raccomandato:** Cifrare i valori dei secret a riposo usando `AES-GCM` con chiave derivata da `security_signing_secret` o una `MEMORIA_MASTER_KEY`.

---

## 3. Solidità & Affidabilità (Robustness & Concurrency)

### 🟡 Blocco Sincrono dell'Event Loop FastAPI (Async vs Sync Anti-Pattern)
- **Posizione:** `memoria/web/routes/agent.py` e `memoria/web/app.py`.
- **Problema:** Diversi endpoint FastAPI sono dichiarati come `async def` (es. `agent_search_get`, `agent_search_post`, `agent_ask`, `run_query`), ma all'interno invocano metodi sincroni CPU-bound o I/O-bound pesanti:
  - `planner.searcher.hybrid_search()` (calcola FTS5, embeddings vettoriali, ranking RRF).
  - `planner.route_query()` (può interrogare via HTTP sincrono l'LLM tramite `urllib.request.urlopen` con timeout fino a 30s).
  In FastAPI/Starlette, se una funzione è `async def`, gira direttamente sull'event loop principale asyncio. Le chiamate bloccanti fermano l'intero server web, impedendo a qualsiasi altra richiesta o healthcheck Docker di essere evasa fino al completamento.
- **Rimedio Raccomandato:** Per operazioni sincrone lunghe in FastAPI ci sono due opzioni pulite:
  1. Definirle come normali `def` (FastAPI le invia automaticamente al `ThreadPoolExecutor` di default).
  2. Oppure eseguire la chiamata con `await asyncio.to_thread(planner.route_query, ...)`.

---

### 🟡 Rischio Crash su Numpy Reshape con Dimensioni Vettoriali Disallineate
- **Posizione:** `memoria/search/hybrid.py`, righe 59–61:
  ```python
  raw_blobs = b"".join(r[1] for r in rows)
  self._cached_matrix = np.frombuffer(raw_blobs, dtype=np.float32).reshape((len(rows), -1))
  ```
- **Problema:** Il codice concatena tutti i BLOB vettoriali della tabella `chunks` e tenta di fare un `reshape((len(rows), -1))`. Se per qualsiasi ragione nel database è presente anche un solo chunk con un BLOB di lunghezza differente (ad esempio creato durante un test, o derivante da un cambio di modello di embedding da 384 a 768 o 1536 float senza reindex totale, o un BLOB corrotto), `np.frombuffer().reshape()` solleva un'eccezione irreversibile `ValueError: cannot reshape array of size X into shape (Y, -1)`. Questo manda in crash l'intero motore di ricerca vettoriale.
- **Rimedio Raccomandato:** Verificare prima che tutti i blob abbiano dimensione fissa pari a `EMBEDDING_DIM * 4` byte (1536 byte per 384 float32) oppure filtrare i record difformi con logging di warning.

---

### 🟡 Sync Remoti e MCP Standalone Falliscono se il Server ha l'Autenticazione Attiva
- **Posizione:** `memoria/sync/claude_code.py` (`post_multipart_file`), `memoria/sync/antigravity.py` e `memoria/mcp_server.py` (`_forward_to_remote_mcp`).
- **Problema:** 
  1. La funzione `post_multipart_file` usata per inviare archivi al server remoto `http://localhost:5006/api/ingest` non accetta né invia l'header `Authorization: Bearer <TOKEN>`. Se sul server è attivata la sicurezza (`MEMORIA_API_TOKEN`), qualsiasi operazione di sync remoto restituisce HTTP 401.
  2. In `memoria/mcp_server.py`, la funzione `_forward_to_remote_mcp` invia richieste JSON-RPC al container remoto senza passare il token di autorizzazione. Se la sicurezza è abilitata sul container, la chiamata remota fallisce con 401 e il processo fa un fallback trasparente e silenzioso sul database SQLite locale (`~/.memoria/vault.db`), illudendo l'utente di interrogare il server homelab centralizzato.
- **Rimedio Raccomandato:** Leggere `MEMORIA_API_TOKEN` dalle variabili d'ambiente in `_forward_to_remote_mcp` e in `post_multipart_file`, passando l'header `Authorization: Bearer <token>`.

---

### 🔵 Assenza di un Meccanismo di Migrazione Schema Versionato
- **Posizione:** `memoria/db/engine.py` (`_init_db()`) e `memoria/db/schema.sql`.
- **Problema:** Lo schema viene eseguito con `conn.executescript(schema_sql)`. I cambi di schema successivi sono gestiti con logica imperativa ad-hoc (es. righe 67–85 con `try... SELECT COUNT(*) FROM statements_fts`). Quando il database cresce o se cambiano colonne/vincoli su tabelle esistenti, questo approccio diventa fragile e rischia incoerenze.
- **Rimedio Raccomandato:** Introdurre una tabella `schema_migrations(version INT, applied_at TEXT)` con script di migrazione numerati o usare uno strumento leggero di migrazione.

---

## 4. Qualità del Codice & Code Smells

### 🟠 Rotte Duplicate e Shadowing tra `web/app.py` e `web/routes/curation.py`
- **Posizione:** `memoria/web/app.py` (righe 291–325) e `memoria/web/routes/curation.py` (righe 65–105).
- **Problema:** In `app.py` viene montato il router modulare:
  ```python
  app.include_router(build_curation_router(db=db, enricher=enricher))
  ```
  Tuttavia, più avanti nello stesso file `app.py`, vengono ridefiniti nuovamente sullo stesso path i seguenti endpoint:
  - `GET /api/entities`
  - `GET /api/decisions`
  - `GET /api/timeline`
  - `POST /api/seed`
  - `POST /api/ingest`
  In FastAPI, quando due rotte hanno lo stesso percorso e metodo, la prima registrata intercetta la richiesta e la seconda diventa **codice morto/shadowed**. Inoltre, le due implementazioni differiscono leggermente nei dettagli (es. `/api/seed` in `curation.py` restituisce `"seeded_conversations"`, mentre in `app.py` restituisce `"seeded_count"`).
- **Rimedio Raccomandato:** Rimuovere le route duplicate da `memoria/web/app.py` lasciandole esclusivamente nel rispettivo router `web/routes/curation.py`.

---

### 🟠 Monolite Frontend Inline di 2626 Righe in `templates/index.html`
- **Posizione:** `memoria/web/templates/index.html`.
- **Problema:** L'intero frontend Single Page Application (centinaia di funzioni JS, gestione eventi, stato reattivo, manipolazione DOM, rendering tabelle, modali, gestione traduzioni e stili CSS) si trova interamente all'interno di un unico file Jinja2 da oltre 160 KB.
- **Rimedio Raccomandato:**
  - Suddividere il JavaScript in moduli statici in `memoria/web/static/js/` (es. `app.js`, `api.js`, `themes.js`, `markdown.js`, `modals.js`).
  - Separare il template Jinja2 in macro/partial (es. `partials/header.html`, `partials/tabs/query.html`, `partials/tabs/conversations.html`, ecc.).

---

### 🟡 Accoppiamento Tematico Hardcoded nell'Estrattore Heuristico
- **Posizione:** `memoria/knowledge/extractor.py`.
- **Problema:** L'estrattore euristico locale contiene dizionari e regex hardcoded specifici per il dominio delle audiocassette e schede audio (es. `CATEGORY_KEYWORDS` per TDK, Maxell, Audient, Zen Go). Sebbene utilissimo per il caso d'uso primario dell'utente, conversazioni su altri domini (programmazione Python, Docker, ricette, medicina, finanza) non vengono categorizzate correttamente dall'euristica base.
- **Rimedio Raccomandato:** Spostare i pattern e le keyword di categorizzazione in una configurazione estensibile (es. un file JSON/YAML o tabella DB) per consentire l'aggiunta di nuove categorie senza modificare il codice sorgente.

---

### 🔵 Pin della Versione di NumPy
- **Posizione:** `pyproject.toml` e `requirements.txt`:
  ```toml
  numpy>=1.24.0,<2.0.0
  ```
- **Problema:** Il vincolo `<2.0.0` crea incompatibilità di build sui sistemi con Python 3.13 e 3.14 (dove NumPy 2.x è fortemente consigliato/richiesto per la C-API). Nel container Docker (Python 3.12-slim) non crea problemi, ma chi installa il pacchetto localmente su Python recente incontra conflitti di installazione.
- **Rimedio Raccomandato:** Verificare la compatibilità del codice con NumPy 2.x e allentare il vincolo a `numpy>=1.24.0`.

---

## 5. Copertura Test & Gap di Testing

Attualmente la suite di test comprende **19 file di test con 69 test automatici**, tutti con esito positivo in ~12 secondi. 

Tuttavia, l'analisi evidenzia aree scoperte da test:

| Modulo / Funzionalità | Stato Copertura | Rischio / Gap Riscontrato |
|---|---|---|
| **CLI (`memoria/cli/main.py`)** | ❌ **0% (Nessun test)** | I 14 comandi CLI (`init`, `seed`, `ingest`, `watch`, `sync`, `enrich`, `ask`, `search`, `entities`, ecc.) non sono coperti da test unitari o funzionali. |
| **MCP Server Stdio (`memoria/mcp_server.py`)** | ⚠️ **Parziale** | Viene testato `handle_mcp_jsonrpc` in-process su HTTP, ma non il loop `run_stdio_server()` né l'inoltro verso il server remoto `_forward_to_remote_mcp()`. |
| **Edge case Ingestion & Zip-Slip** | ⚠️ **Parziale** | Nessun test verifica la gestione di archivi ZIP con file corrotti, percorsi malevoli (`../../`) o file vuoti/non UTF-8. |
| **Concorrenza SQLite sotto carico** | ❌ **Non testato** | Nessun test di concorrenza stress-testa scritture contemporanee tra watcher in background, reindex embeddings, batch enricher e richieste REST. |
| **Network Failure & Retry LLM Providers** | ⚠️ **Parziale** | I test sui provider LLM mockano risposte positive, ma non testano casi di timeout HTTP, 429 rate limit persistente o payload JSON malformati dal provider. |

---

## 6. Errori Vari & Bug Riscontrati

1. **Discrepanza Nomenclatura App (`APP_TITLE` vs `memoria_ai_anti`):**
   In `memoria/config.py`:
   ```python
   APP_TITLE = os.getenv("MEMORIA_APP_TITLE", "Engram AI")
   APP_SUBTITLE = os.getenv("MEMORIA_APP_SUBTITLE", "AI Memory Vault")
   ```
   Il titolo predefinito è "Engram AI", mentre tutta la documentazione, i comandi CLI, il package e le istruzioni homelab si riferiscono a "Memoria AI". Questo genera ambiguità se le variabili d'ambiente non sono esplicitamente impostate nel container.
2. **Incoerenza Chiavi Risposta tra Endpoint Seed:**
   - In `memoria/web/routes/curation.py`: `{"status": "success", "seeded_conversations": count, ...}`
   - In `memoria/web/app.py`: `{"status": "success", "seeded_count": count, ...}`
   Un client API che si aspetta una chiave specifica riceve un risultato imprevedibile a seconda di quale delle due definizioni prevale.
3. **Avviso di Deprecazione Starlette TestClient nei Test:**
   ```
   StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
   ```
   Risolvibile aggiornando la libreria di test client o le dipendenze dev.

---

## 7. Scelte Architetturali e Stilistiche

### Valutazione dell'Architettura
- **Modello Dati a 3 Livelli:**
  - *Livello 1 (Archivio Immutabile):* `conversations` e `messages` conservano la verità storica intatta senza alterazioni. **Scelta eccellente.**
  - *Livello 2 (Retrieval Ibrido):* `chunks` con trigger automatici su `chunks_fts` (FTS5) e BLOB di embedding. **Scelta eccellente.**
  - *Livello 3 (Epistemic Knowledge Graph):* `entities`, `statements` (FACT/CLAIM), `decisions`, `relations`. **Scelta pionieristica e solida.**
- **RRF (Reciprocal Rank Fusion):**
  La combinazione BM25 + Similarità Coseno con $k = 60$ permette di superare i limiti tipici dei vettori puri su codici e sigle alfanumeriche (es. `TDK SA-X`, `Audient iD22`), mantenendo la capacità di cogliere affinità concettuali.
- **Doppio Sistema Cognitivo (System 1 Jev + System 2 LLM):**
  L'adozione del router veloce (Jev) per smistare le query prima di invocare modelli di sintesi più lenti o costosi è un pattern architetturale all'avanguardia per l'ottimizzazione delle risorse in homelab.

---

## 8. UX e UI

### Valutazione dell'Interfaccia Utente
- **Stile `gaskb_style`:** L'interfaccia rispetta in modo eccellente lo standard UI dell'homelab: dark theme nativo (#0c0f17 / #161b26), finiture in glassmorphism, tipografia pulita, contrasti elevati e palette tematiche intercambiabili (Emerald, Indigo, Slate, Amber, Ruby).
- **Dual Badges:** Presenti sia nel footer che nei metadata (`UI_VERSION: 2.2.0` / `API_VERSION: v1`), pienamente conformi alle linee guida.
- **Multi-lingua (i18n):** Supporto bilingue integrato (IT ed EN) con selettore rapido e bundle dedicati per i prompt di sistema.
- **Curation Studio:** Ottima la presenza di un pannello interattivo per validare, modificare o invalidare manualmente i fatti estratti e le decisioni architetturali.

### Punti Deboli UX da Migliorare
1. **Paginazione Mancante nella Vista Conversazioni:** Con oltre 1.700 conversazioni e 37.000 messaggi nel database, caricare troppe conversazioni nella UI senza virtualizzazione o paginazione numerata appesantisce il rendering del browser.
2. **Feedback di Stato Asincrono:** Durante il reindex vettoriale (`POST /api/embeddings/reindex`) o il batch enrichment, la UI mostra uno spinner ma non una barra di avanzamento percentuale in tempo reale (SSE o polling continuo).

---

## 9. Tabella di Riepilogo & Piano di Azione Raccomandato

| Priorità | Ambito | Azione Raccomandata | Difficoltà |
|---|---|---|---|
| **P1** | **Sicurezza** | Proteggere `/api/onboarding/mcp-config` con `require_mutation_auth` per evitare il leak del Bearer Token. | Bassa (2 righe) |
| **P1** | **Sicurezza** | Aggiungere `DOMPurify` su `marked.parse` e funzione `escapeHtml()` sulle interpolazioni in `index.html`. | Media |
| **P1** | **Sicurezza** | Usare header `x-goog-api-key` in `GeminiProvider` invece di passare la chiave in query string (`?key=`). | Bassa |
| **P2** | **Solidità** | Passare il Bearer token in `_forward_to_remote_mcp` e `sync/claude_code.py` per consentire sync con server protetti. | Bassa |
| **P2** | **Qualità** | Rimuovere le route duplicate e ridondanti in `memoria/web/app.py` consolidandole in `curation.py`. | Bassa |
| **P2** | **Solidità** | Rimuovere le chiamate bloccanti sincrone dall'event loop di FastAPI usando `asyncio.to_thread` o route `def`. | Media |
| **P3** | **Solidità** | Proteggere `np.reshape` in `hybrid.py` per gestire eventuali chunk con BLOB vettoriali difformi o corrotti. | Bassa |
| **P3** | **Testing** | Aggiungere una suite `tests/test_cli.py` per coprire i comandi del file `cli/main.py`. | Media |
| **P3** | **Manutenibilità** | Spostare il JavaScript monolitico di `index.html` in file statici modulari in `memoria/web/static/js/`. | Media/Alta |

---

## 10. Conclusione

`memoria_ai_anti` è un progetto di **notevole valore tecnologico ed ergonomico**, concepito con una visione chiara e con scelte di design molto intelligenti per un homelab self-hosted. 

Risolvendo le criticità di sicurezza identificate (in primis la protezione dell'endpoint di onboarding token e la sanitizzazione XSS del frontend) ed eliminando le duplicazioni di route, il progetto raggiunge un livello di **solidità, affidabilità e maturità di livello enterprise-grade**.
