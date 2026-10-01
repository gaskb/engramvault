# Come Contribuire a EngramVault

[🇬🇧 English](CONTRIBUTING.md) · **Italiano**

---

Grazie per il tuo interesse nel contribuire a **EngramVault**! Accogliamo con entusiasmo contributi dalla community: segnalazioni di bug, correzioni, adapter per nuovi provider AI e miglioramenti alla UI.

## Codice di Condotta

Tratta chiunque con rispetto ed empatia. Ci impegniamo a mantenere un ambiente amichevole, accogliente e privo di molestie per tutti i partecipanti.

---

## Per Iniziare

### 1. Fork e Clone

```bash
git clone https://github.com/gaskb/engramvault.git
cd engramvault
```

### 2. Configura l'Ambiente Virtuale

Usando `uv` (consigliato):
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install pytest httpx
```

Oppure con `venv` standard:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### 3. Esegui la Suite di Test

Prima di apportare modifiche, assicurati che tutti i test passino:
```bash
pytest
```
EngramVault adotta una politica di zero regressioni: tutti i 100+ test devono avere esito positivo.

---

## Linee Guida per lo Sviluppo

### Architettura & Gerarchia Epistemica
- Mantieni la separazione tra dati grezzi (`conversations`, `messages`, `chunks`), elementi dedotti (`claims`, `relations`) e conoscenza verificata (`facts`, `decisions`).
- Rispetta la filosofia **Zero Dipendenze Esterne Pesanti**: EngramVault si basa su SQLite (FTS5 + vettori) e puro Python/NumPy per le operazioni base. Evita di aggiungere database esterni pesanti come requisiti obbligatori.

### Standard di Codice
- **Python:** compatibile con Python 3.10+. Usa type hints ove pratico.
- **Frontend:** HTML/CSS/Vanilla JS puro (renderizzato tramite template Jinja2) senza build chain complesse. Mantienilo leggero e veloce.
- **Formattazione & Linting:** esegui `ruff check .` prima di inviare modifiche.

---

## Invio di una Pull Request

1. Crea un branch descrittivo:
   ```bash
   git checkout -b feat/add-new-adapter
   ```
2. Scrivi test unitari o di integrazione in `tests/`.
3. Verifica che `pytest` passi al 100%.
4. Esegui il commit con messaggi semantici chiari (es. `feat: add mistral adapter`, `fix: handle edge case in zip import`).
5. Apri una Pull Request verso il branch `main` su GitHub.

Grazie per aiutarci a rendere la memoria per AI accessibile, privata e verificabile!
