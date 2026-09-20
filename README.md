# Memoria AI Anti (memoria_ai_anti)

> **Un punto centralizzato, leggero e intelligente per memorizzare, indicizzare e distillare conoscenza da centinaia di conversazioni con diverse AI (ChatGPT, Claude, Claude Code, Gemini, Markdown).**

Progettato specificamente per girare su un **homelab senza GPU** (basso consumo di risorse, zero demoni pesanti), ma con la possibilità di connettersi ad **AI esterne (o locali)** per compiti di sintesi e distillazione avanzata.

---

## 💡 Il Problema e la Filosofia

Usando molteplici assistenti AI per scopi diversi (programmazione, hobby, musica, cucina), si finisce per disperdere la propria conoscenza in diversi "silos":
1. **Problema di Retrieval:** Non ci si ricorda dove si è avuta una certa conversazione e si perde tempo a cercarla.
2. **Problema di Conoscenza / Memoria:** Una domanda complessa (es. *"Quali sono le migliori audiocassette per registrare un brano rock tra le audiocassette che ho?"*) non si trova in una singola chat. L'inventario delle cassette potrebbe essere in una chat con ChatGPT, il confronto tecnico tra nastri Type I e Type II in una con Claude, e la calibrazione del registratore in una sessione con Gemini.

`memoria_ai_anti` organizza il sistema su tre livelli:
1. **Conversation Vault:** Archivio normalizzato, immutabile e verificabile di tutte le conversazioni originali.
2. **Search Layer (Ibrido):** Combinazione di ricerca lessicale per parole esatte (**FTS5 / BM25**) e semantica (**Vettori L2 normalizzati**) fuse con **RRF (Reciprocal Rank Fusion)**. Trova sia i modelli esatti (es. *TDK SA-X*, *Audient iD22*) sia i concetti generici.
3. **Knowledge Layer (Fatti vs Claim & Timeline):**
   - **`FACT`:** Fatti dichiarati o verificati dall'utente (es. *"Possiedo una TDK SA-X"*, *"Ho venduto l'Audient iD22"*).
   - **`CLAIM`:** Informazioni e consigli tecnici dati dalle AI (es. *"I nastri Type II offrono maggiore saturazione per il rock"*).
   - **`DECISION`:** Decisioni esplicite dell'utente (es. *"Ho scelto la Zen Go come scheda audio principale"*).
   - **`TIMELINE`:** Tracciamento della validità temporale (`valid_from`, `valid_until`, `is_current`, `supersedes`).

---

## 🚀 Caratteristiche Principali

* **Database unico SQLite (`vault.db`):** Niente database vettoriali complessi o servizi esterni. Un solo file ACID. Backup istantaneo con `cp vault.db backup.db`.
* **Zero requisiti GPU:** Calcolo degli embeddings e della ricerca ibrida su CPU in pochi millisecondi.
* **Supporto AI Esterne e Locali:** Disaccoppiato dal modello di generazione. Puoi interrogare il sistema senza LLM (estrazione pura e veloce), con **Ollama** locale (`qwen2.5:3b`, `llama3.2:3b`) o con **API cloud** (Google Gemini Flash, Anthropic Claude, OpenAI, Groq) inviando all'esterno solo i pochi estratti rilevanti.
* **Adapter multi-provider inclusi:**
  * **ChatGPT:** parsing completo dell'albero di nodi (`mapping`) e dell'export ZIP/JSON.
  * **Claude Web:** parsing degli export ufficiali JSON.
  * **Claude Code (CLI):** ingest automatico delle sessioni in `~/.claude/projects/` (evitando la cancellazione automatica a 30 giorni).
  * **Gemini:** supporto ad archivi Takeout e JSON.
  * **Markdown:** parsing di appunti e chat con frontmatter YAML.
* **Deduplicazione basata su Hash:** Ingestione idempotente con hash SHA-256 dei contenuti.
* **Dashboard Web & CLI:** Interfaccia web scattante in dark mode e tool da terminale completo.

---

## 🛠️ Installazione Rapida

### 1. Requisiti
* Python 3.10+
* Nessuna scheda video richiesta

### 2. Setup Locale
```bash
git clone ssh://git@github.com/root/memoria_ai_anti.git
cd memoria_ai_anti

# Crea virtualenv ed installa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 📖 Utilizzo della CLI

### 1. Inizializzazione e Dati Demo
```bash
# Inizializza il vault
memoria init

# Popola con il dataset di esempio (incluso lo scenario audiocassette e audio gear)
memoria seed

# Visualizza statistiche del database
memoria stats
```

### 2. Importazione delle tue conversazioni
```bash
# Import automatico (rileva il formato)
memoria ingest /percorso/a/conversations.json

# Importa le sessioni di Claude Code
memoria ingest ~/.claude/projects --provider claude_code

# Importa uno ZIP di ChatGPT
memoria ingest export_chatgpt.zip --provider chatgpt
```

### 3. Interrogare la memoria
```bash
# Domanda complessa in modalità automatica o sintetica
memoria ask "quali sono le migliori audiocassette per registrare un brano rock tra le audiocassette che ho ?"

# Query strutturata sull'inventario
memoria ask "Quali cassette possiedo?" --mode structured

# Query sulle decisioni prese
memoria ask "Che cosa ho deciso riguardo all'interfaccia audio?" --mode structured

# Ricerca estrattiva pura (trova le citazioni nelle chat)
memoria search "TDK SA-X"
```

### 4. Esplorazione Conoscenza e Timeline
```bash
# Elenco entità e relativi FACT / CLAIM
memoria entities

# Solo per una categoria (es. audiocassette o audio_interface)
memoria entities --category audiocassette

# Registro delle decisioni
memoria decisions

# Cronologia e cambiamenti di stato di un oggetto
memoria timeline "Audient iD22"
```

### 5. Avviare la Dashboard Web
```bash
memoria serve --host 0.0.0.0 --port 8080
```
Apri il browser su `http://localhost:8080` per esplorare la tua memoria, cercare tra le chat, consultare i grafi di entità e importare file via drag-and-drop.

---

## 🐳 Docker & Homelab Deploy

### Con Docker Compose:
```bash
docker compose up -d
```
Il database verrà salvato nella cartella locale `./data/vault.db`.

### Con Systemd (Servizio e Sync Notturno):
Nella cartella `deploy/systemd/` trovi i file di configurazione pronti:
* `memoria-ai.service`: avvia il server Web come servizio di sistema.
* `memoria-ai-sync.service` & `memoria-ai-sync.timer`: esegue ogni notte alle 03:30 l'ingest automatico dei nuovi log di Claude Code e genera un backup compresso di SQLite.

---

## 🧪 Esecuzione dei Test

Per eseguire l'intera suite di test:
```bash
python3 run_tests.py
```
Oppure con `pytest`:
```bash
pytest tests/
```

---

## 📄 Licenza
MIT License - Realizzato per la gestione della conoscenza personale.
