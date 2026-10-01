# Architettura di EngramVault

Questo documento descrive le scelte architetturali, il modello dati e il funzionamento interno di **EngramVault**.

## 1. Principi Fondamentali

1. **Source of Truth Immutabile:**
   Le conversazioni originali con le AI non vengono mai modificate o rimpiazzate da riassunti. Rimangono memorizzate integra nel database come fonti primarie verificabili.
2. **Knowledge Layer Derivato:**
   La conoscenza distillata (fatti, claim, decisioni) è un livello *derivato* sopra l'archivio. Ogni elemento di conoscenza mantiene il puntatore (`source_message_id`, `source_conversation_id`) alla chat da cui proviene.
3. **Epistemic Cleanliness (Separazione Fatti vs Opinioni):**
   - `FACT`: Dichiarazioni effettuate dall'utente (es. oggetti posseduti, preferenze, parametri del proprio setup).
   - `CLAIM`: Suggerimenti, analisi o pareri forniti dal modello AI (es. "i nastri Type II sono ideali per il rock").
4. **Dimensione Temporale (SCD Type 2):**
   La memoria dell'utente cambia nel tempo (acquisto, vendita, cambio di decisioni). I fatti mantengono uno storico con `valid_from`, `valid_until`, `is_current` e `supersedes_id`.
5. **Zero-Overhead su Homelab (CPU-Only):**
   Nessun database vettoriale esterno residente in memoria. L'intero sistema risiede in un unico file SQLite (`vault.db`), con ricerca testuale FTS5 e vettori conservati come BLOB.

---

## 2. Modello Dati (Schema Relazionale)

```
[ conversations ] 1 ──< [ messages ] 1 ──< [ chunks ] (FTS5 indexed)
       │                      │
       │                      │ (source_message_id)
       │                      ▼
       └──────────────> [ statements ] (FACT vs CLAIM)
                              │
                              ▼
                         [ entities ]
                              │
                              ▼
                         [ relations ]
```

### Schema delle Tabelle

- `conversations`: Metadati della conversazione (`id`, `provider`, `title`, `created_at`, `content_hash`).
- `messages`: Messaggi individuali (`id`, `role`, `content`, `ts`, `sequence_order`).
- `chunks`: Segmenti di testo per l'indicizzazione semantica e lessicale.
- `chunks_fts`: Tabella virtuale FTS5 di SQLite sincronizzata tramite trigger automatici `AFTER INSERT`, `AFTER DELETE`, `AFTER UPDATE`.
- `entities`: Nomi canonici degli oggetti/argomenti identificati (`name`, `category`, `normalized_name`).
- `statements`: Triple o predicati estratti (`subject`, `predicate`, `object`, `statement_type`, `author`, `is_current`, `valid_from`, `valid_until`).
- `decisions`: Registro delle scelte intraprese (`topic`, `decision_text`, `status`, `decided_at`).
- `relations`: Connessioni tra entità nel grafo relazionale.

---

## 3. Motore di Ricerca Ibrida (Hybrid Retrieval con RRF)

I motori di ricerca puramente semantici (vettori) falliscono regolarmente su codici prodotto e sigle alfanumeriche (es. `TDK SA-X`, `Maxell XLII-S`, `Audient iD22`), confondendole o generalizzandole.
Al contempo, la ricerca puramente lessicale (BM25) fallisce sui sinonimi e concetti affini.

**EngramVault** unisce i due mondi tramite **Reciprocal Rank Fusion (RRF)**:

$$RRF(d) = \frac{w_{bm25}}{k + rank_{bm25}(d)} + \frac{w_{vec}}{k + rank_{vec}(d)}$$

- $rank_{bm25}$: posizione del documento nei risultati FTS5 (BM25 ranking).
- $rank_{vec}$: posizione del documento secondo la similarità coseno del vettore di embedding.
- Costante di smoothing $k = 60$.

---

## 4. Integrazione con AI Esterne e Locali

Il modulo `engramvault.llm.client` supporta qualsiasi endpoint compatibile OpenAI (incluso Ollama in locale o gateway cloud per Gemini, Claude, OpenAI):

1. **Elaborazione Query Sintetiche:** Invia al modello solo i top chunk e i fatti pertinenti estratti localmente, risparmiando token e preservando la privacy.
2. **Estrazione Strutturata:** Funzione `extract_knowledge_with_llm()` per generare entità e relazioni complesse in batch.
