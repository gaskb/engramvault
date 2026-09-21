import re
from typing import Dict, Any, List, Optional
from memoria.db.engine import Database
from memoria.search.hybrid import HybridSearcher
from memoria.llm.client import LLMClient


class QueryPlanner:
    def __init__(
        self,
        db: Database,
        searcher: Optional[HybridSearcher] = None,
        llm: Optional[LLMClient] = None
    ):
        self.db = db
        self.searcher = searcher or HybridSearcher(db)
        self.llm = llm or LLMClient()

    def route_query(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """Routes user query into Extractive, Structured, or Synthetic execution."""
        q_lower = query.lower()

        if mode == "auto":
            if any(k in q_lower for k in ("dove ho parlato", "quando ho detto", "citami", "trova la chat", "cerca")):
                mode = "extractive"
            elif any(k in q_lower for k in ("cosa possiedo", "quali ho", "elenco delle", "quali cassette ho", "che setup", "decisioni prese")):
                mode = "structured"
            else:
                mode = "synthetic"

        if mode == "structured":
            return self.execute_structured(query)
        elif mode == "synthetic":
            return self.execute_synthetic(query)
        else:
            return self.execute_extractive(query)

    def execute_extractive(self, query: str, limit: int = 8) -> Dict[str, Any]:
        hits = self.searcher.hybrid_search(query, limit=limit)
        return {
            "query": query,
            "mode": "extractive",
            "count": len(hits),
            "results": hits,
            "answer": None
        }

    def execute_structured(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        facts = []
        decisions = []
        timeline = []

        if "decision" in q_lower or "deciso" in q_lower:
            decisions = self.db.get_decisions()
            summary_lines = ["### Registro delle Decisioni Personali:"]
            for d in decisions:
                summary_lines.append(f"- **{d['topic']}** ({d['decided_at'][:10]}): {d['decision_text']} `[Stato: {d['status']}]`")
            answer = "\n".join(summary_lines)

        elif "timeline" in q_lower or "cambiato" in q_lower or "setup" in q_lower:
            timeline = self.db.get_timeline()
            summary_lines = ["### Cronologia e Cambiamenti di Stato:"]
            for ev in timeline:
                dt = (ev.get("event_date") or "")[:10] or "N/D"
                curr = " (Attuale)" if ev.get("is_current") else " (Passato/Dismesso)"
                summary_lines.append(f"- `[{dt}]` **{ev['subject']}** -> {ev['predicate']}: *{ev['object']}*{curr}")
            answer = "\n".join(summary_lines)

        else:
            # Default: Inventory / Owned facts
            facts = self.db.get_statements(predicate="owned_by_user", is_current=1)
            summary_lines = ["### Oggetti e Componenti Attualmente Posseduti (FACT):"]
            for f in facts:
                cat = f.get("entity_category", "generale")
                src = f.get("conversation_title", "Archivio")
                prov = f.get("conversation_provider", "chat")
                dt = (f.get("valid_from") or "")[:10]
                summary_lines.append(f"- **{f['subject']}** ({cat}) — *Fonte: {prov} in '{src}' ({dt})*")
            answer = "\n".join(summary_lines)

        return {
            "query": query,
            "mode": "structured",
            "facts": facts,
            "decisions": decisions,
            "timeline": timeline,
            "answer": answer
        }

    def execute_synthetic(self, query: str) -> Dict[str, Any]:
        """
        Synthesizes multi-hop answer across:
        1. User inventory (FACT)
        2. Technical discussions / models advice (CLAIM & Hybrid Retrieval)
        Uses external or local LLM if configured; otherwise uses deterministic synthesizer.
        """
        q_lower = query.lower()

        # Step 1: Check user-owned items in inventory
        owned_stmts = self.db.get_statements(predicate="owned_by_user", is_current=1)

        # Step 2: Hybrid search for context chunks
        hits = self.searcher.hybrid_search(query, limit=8)

        # Step 3: Check if external / local LLM can synthesize
        if self.llm.is_configured():
            llm_response = self.llm.synthesize_answer(query, owned_stmts, hits)
            if llm_response:
                return {
                    "query": query,
                    "mode": f"synthetic ({self.llm.provider_name.capitalize()}-powered)",
                    "answer": llm_response,
                    "sources": hits,
                    "facts": owned_stmts,
                }

        # Fallback to local deterministic distillation engine
        lines = [f"### Risposta Sintetica a: *\"{query}\"*\n"]

        # If audiocassette / rock question
        if any(w in q_lower for w in ("cassett", "nastr", "tape", "rock")):
            cassettes_owned = [
                s for s in owned_stmts 
                if (s.get("entity_category") == "audiocassette" or any(k in s["subject"].lower() for k in ("tdk", "maxell", "sony", "cassetta", "nastro")))
            ]
            
            lines.append("#### 1. Il tuo inventario verificato (`FACT`):")
            if cassettes_owned:
                for c in cassettes_owned:
                    prov = (c.get('conversation_provider') or '').upper()
                    dt = (c.get('valid_from') or '')[:10]
                    lines.append(f"- **{c['subject']}** — Dichiarato da te in *'{c.get('conversation_title')}'* ({prov}, {dt})")
            else:
                lines.append("- Nessuna cassetta specifica registrata come attiva nell'inventario.")

            lines.append("\n#### 2. Dati tecnici e confronti estratti dalle discussioni (`CLAIM`):")
            relevant_hits = [h for h in hits if any(w in h["chunk_text"].lower() for w in ("rock", "saturazione", "type", "cromo", "tdk", "maxell", "bias", "hf"))]
            for h in (relevant_hits or hits)[:3]:
                snippet = h["chunk_text"][:220].replace("\n", " ").strip()
                lines.append(f"- **{h['conv_title']}** ({h['provider'].upper()}, {h['created_at'][:10]}): *\"{snippet}...\"*")

            lines.append("\n#### 3. Sintesi e Raccomandazione Distillata:")
            has_tdk_sax = any("sa-x" in s["subject"].lower() or "sa" in s["subject"].lower() for s in cassettes_owned)
            has_maxell = any("xlii" in s["subject"].lower() for s in cassettes_owned)
            has_sony_hf = any("hf" in s["subject"].lower() for s in cassettes_owned)

            if has_tdk_sax:
                lines.append("• **TDK SA-X (Scelta consigliata per Rock):** Essendo un nastro Type II (High Bias) al cromo/Avilyn a doppio strato, offre la massima headroom sui medi e alti con una saturazione calda ideale per chitarre elettriche e batteria.")
            if has_maxell:
                lines.append("• **Maxell XLII (Ottima alternativa):** Altro nastro Type II bilanciato con eccellente rapporto segnale/rumore per mix rock.")
            if has_sony_hf:
                lines.append("• **Sony HF (Non consigliata per Rock saturo):** È un nastro Type I al ferro; ottimo per voce o acustico, ma tende a comprimere/distorcere eccessivamente con carichi energetici elevati.")
        else:
            # General synthesis
            lines.append("#### 1. Fatti e Informazioni Utente Correlate (`FACT`):")
            if owned_stmts:
                for s in owned_stmts[:4]:
                    lines.append(f"- **{s['subject']}**: {s['predicate']} = {s['object']} *(da {s.get('conversation_title')})*")

            lines.append("\n#### 2. Passaggi Rilevanti Trovati (Ricerca Ibrida BM25 + Vector):")
            for h in hits[:4]:
                snippet = h["chunk_text"][:200].replace("\n", " ")
                lines.append(f"- *[{h['provider'].upper()} - {h['conv_title']}]*: {snippet}...")

        answer = "\n".join(lines)
        return {
            "query": query,
            "mode": "synthetic (Local Rule-based Engine)",
            "answer": answer,
            "sources": hits,
            "facts": owned_stmts,
            "claims": []
        }
