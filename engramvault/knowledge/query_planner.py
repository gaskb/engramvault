import re
from typing import Any, Dict, List, Optional

from engramvault.config import DEFAULT_LANG
from engramvault.db.engine import Database
from engramvault.knowledge.query_router import QueryRouter, detect_query_lang as _detect_query_lang
from engramvault.llm.client import LLMClient
from engramvault.llm.jev import JevSystemOneEngine
from engramvault.search.hybrid import HybridSearcher


class QueryPlanner:
    def __init__(
        self,
        db: Database,
        searcher: Optional[HybridSearcher] = None,
        llm: Optional[LLMClient] = None,
        jev: Optional[JevSystemOneEngine] = None,
        router: Optional[QueryRouter] = None,
    ):
        self.db = db
        self.searcher = searcher or HybridSearcher(db)
        self._llm = llm
        self.jev = jev or JevSystemOneEngine(db=db)
        self.router = router or QueryRouter(jev=self.jev)

    @property
    def llm(self) -> LLMClient:
        if self._llm is not None:
            return self._llm
        from engramvault.settings.manager import SettingsManager

        return SettingsManager(self.db).get_llm_client()

    def route_query(self, query: str, mode: str = "auto", lang: str = DEFAULT_LANG) -> Dict[str, Any]:
        """Routes user query via QueryRouter and executes Extractive, Structured, or Synthetic synthesis."""
        routing = self.router.classify(query, requested_mode=mode, fallback_lang=lang)
        eff_mode = routing["effective_mode"]
        eff_lang = routing["effective_lang"]
        route_decision = routing["route_decision"]

        if eff_mode == "structured":
            res = self.execute_structured(query, lang=eff_lang)
        elif eff_mode == "synthetic":
            res = self.execute_synthetic(query, lang=eff_lang, route_decision=route_decision)
        else:
            res = self.execute_extractive(query, lang=eff_lang)

        res["system_one_routing"] = route_decision
        return res

    def execute_extractive(self, query: str, limit: int = 8, lang: str = DEFAULT_LANG) -> Dict[str, Any]:
        hits = self.searcher.hybrid_search(query, limit=limit)
        return {
            "query": query,
            "mode": "extractive",
            "lang": lang,
            "count": len(hits),
            "results": hits,
            "answer": None,
        }

    def execute_structured(self, query: str, lang: str = DEFAULT_LANG) -> Dict[str, Any]:
        q_lower = query.lower()
        is_it = lang == "it"
        facts = []
        decisions = []
        timeline = []

        if any(w in q_lower for w in ("decision", "deciso", "scelt", "choice")):
            decisions = self.db.get_decisions()
            header = "### Registro delle Decisioni Personali:" if is_it else "### Personal Decision Registry:"
            summary_lines = [header]
            for d in decisions:
                summary_lines.append(
                    f"- **{d['topic']}** ({d['decided_at'][:10]}): {d['decision_text']} `[Status: {d['status']}]`"
                )
            answer = "\n".join(summary_lines)

        elif any(w in q_lower for w in ("timeline", "cambiato", "setup", "evolution", "cronologia")):
            timeline = self.db.get_timeline()
            header = "### Cronologia e Cambiamenti di Stato:" if is_it else "### Chronological Timeline & State Changes:"
            summary_lines = [header]
            for ev in timeline:
                dt = (ev.get("event_date") or "")[:10] or "N/A"
                curr = (
                    (" (Attuale)" if ev.get("is_current") else " (Passato/Dismesso)")
                    if is_it
                    else (" (Current)" if ev.get("is_current") else " (Past/Superseded)")
                )
                summary_lines.append(f"- `[{dt}]` **{ev['subject']}** -> {ev['predicate']}: *{ev['object']}*{curr}")
            answer = "\n".join(summary_lines)

        else:
            facts = self.db.get_statements(predicate="owned_by_user", is_current=1)
            header = (
                "### Oggetti e Componenti Attualmente Posseduti (FACT):"
                if is_it
                else "### Currently Owned Items & Verified Facts (FACT):"
            )
            summary_lines = [header]
            for f in facts:
                cat = f.get("entity_category", "general")
                src = f.get("conversation_title", "Archive")
                prov = f.get("conversation_provider", "chat")
                dt = (f.get("valid_from") or "")[:10]
                src_label = "Fonte" if is_it else "Source"
                summary_lines.append(f"- **{f['subject']}** ({cat}) — *{src_label}: {prov} in '{src}' ({dt})*")
            answer = "\n".join(summary_lines)

        return {
            "query": query,
            "mode": "structured",
            "lang": lang,
            "facts": facts,
            "decisions": decisions,
            "timeline": timeline,
            "answer": answer,
        }

    def execute_synthetic(
        self,
        query: str,
        lang: str = DEFAULT_LANG,
        route_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Domain-agnostic multi-hop synthesis across:
        1. User inventory (`FACT`) filtered dynamically by query categories/entities
        2. Technical claims (`CLAIM`) and hybrid search conversation chunks (`hits`)
        Uses configured LLM provider if active; otherwise uses the domain-agnostic rule engine.
        """
        q_lower = query.lower()
        is_it = lang == "it"
        decision = route_decision or self.jev.route_query_decision(query)

        owned_stmts = self.db.get_statements(predicate="owned_by_user", is_current=1)
        target_cat = decision.get("category", "all")
        if target_cat and target_cat != "all":
            cat_filtered = [s for s in owned_stmts if s.get("entity_category") == target_cat]
            if cat_filtered:
                owned_stmts = cat_filtered

        all_claims = self.db.get_statements(statement_type="CLAIM")
        raw_hits = self.searcher.hybrid_search(query, limit=12)
        hits = self.jev.rerank_chunks(query, raw_hits, top_k=8)

        if self.llm.is_configured():
            llm_response = self.llm.synthesize_answer(query, owned_stmts, hits, lang=lang)
            if llm_response:
                return {
                    "query": query,
                    "mode": f"synthetic ({self.llm.provider_name.capitalize()}-powered)",
                    "lang": lang,
                    "answer": llm_response,
                    "sources": hits,
                    "facts": owned_stmts,
                }

        # Domain-agnostic deterministic distillation engine (Issue #13)
        q_tokens = set(re.findall(r"\b[\w\-\.]{3,}\b", q_lower))

        # Dynamically match owned items whose category or subject intersects the query or retrieved chunks
        matched_owned = []
        for s in owned_stmts:
            cat = (s.get("entity_category") or "").lower()
            subj = (s.get("subject") or "").lower()
            if (
                (cat and (cat in q_lower or any(t in cat or cat in t for t in q_tokens)))
                or any(t in subj for t in q_tokens)
                or any(subj in (h.get("chunk_text") or "").lower() for h in hits[:4])
            ):
                matched_owned.append(s)

        target_owned = matched_owned if matched_owned else owned_stmts[:10]

        title_prefix = "Risposta Sintetica a" if is_it else "Synthetic Analysis for"
        lines = [f"### {title_prefix}: *\"{query}\"*\n"]

        sec1 = "1. Il tuo inventario verificato (`FACT`):" if is_it else "1. Your Verified Inventory (`FACT`):"
        lines.append(f"#### {sec1}")
        if target_owned:
            for item in target_owned:
                prov = (item.get("conversation_provider") or "vault").upper()
                dt = (item.get("valid_from") or "")[:10]
                conv_id = item.get("source_conversation_id") or ""
                conv_title = item.get("conversation_title") or "Archive"
                link = f"[{conv_title}](#conv-{conv_id})" if conv_id else f"*{conv_title}*"
                lines.append(f"- **{item['subject']}** (`{item.get('entity_category', 'general')}`) — {link} ({prov}, {dt})")
        else:
            lines.append(
                "- Nessun elemento specifico registrato nell'inventario."
                if is_it
                else "- No specific items registered in the verified inventory."
            )

        # Collect graph relations for matched entities
        target_entity_ids = [item.get("entity_id") for item in target_owned if item.get("entity_id")]
        relations = self.db.find_related_entities(target_entity_ids, limit=8) if target_entity_ids else []
        if relations:
            sec_rel = "Relazioni Semantiche & Connessioni (`GRAPH`):" if is_it else "Knowledge Graph Connections (`GRAPH`):"
            lines.append(f"\n#### {sec_rel}")
            for r in relations:
                lines.append(
                    f"- **{r['from_name']}** (`{r['from_category']}`) $\\rightarrow$ `{r['relation_type']}`: **{r['to_name']}** (`{r['to_category']}`)"
                )

        sec2 = (
            "2. Dati tecnici e confronti estratti dalle discussioni (`CLAIM`):"
            if is_it
            else "2. Technical Claims & Context from Past Conversations (`CLAIM`):"
        )
        lines.append(f"\n#### {sec2}")
        for h in hits[:4]:
            snippet = (h.get("chunk_text") or "")[:240].replace("\n", " ").strip()
            cid = h.get("conversation_id") or ""
            ctitle = h.get("conv_title") or "Conversation"
            cprov = (h.get("provider") or "chat").upper()
            cdt = (h.get("created_at") or "")[:10]
            lines.append(f"- **[{ctitle}](#conv-{cid})** ({cprov}, {cdt}): *\"{snippet}...\"*")

        sec3 = (
            "3. Sintesi e Raccomandazione su Misura per il Tuo Inventario:"
            if is_it
            else "3. Tailored Recommendation Based on Your Inventory:"
        )
        lines.append(f"\n#### {sec3}")

        if target_owned:
            for item in target_owned:
                subj = item["subject"]
                subj_lower = subj.lower()
                # Gather claims and chunk lines specifically mentioning this owned item
                item_claims = [
                    f"{c.get('predicate')}: {c.get('object')}"
                    for c in all_claims
                    if subj_lower in (c.get("subject") or "").lower() or (c.get("subject") or "").lower() in subj_lower
                ]
                item_snippets = []
                for h in hits:
                    for sentence in re.split(r"(?<=[.!?])\s+|\n+", h.get("chunk_text") or ""):
                        if subj_lower in sentence.lower() or any(part in sentence.lower() for part in subj_lower.split() if len(part) >= 3):
                            clean_s = sentence.strip()
                            if clean_s and clean_s not in item_snippets:
                                item_snippets.append(clean_s)

                details = "; ".join(item_claims[:2] + item_snippets[:2])
                if not details:
                    details = item.get("notes") or item.get("object") or "Confermato nell'inventario utente"
                lines.append(f"- **{subj}**: {details}")
        else:
            lines.append(
                "Nessun incrocio diretto con l'inventario attivo; consulta gli estratti sopra."
                if is_it
                else "No direct inventory match found; see conversation excerpts above."
            )

        return {
            "query": query,
            "mode": "synthetic (Local Rule-based Engine)",
            "lang": lang,
            "answer": "\n".join(lines),
            "sources": hits,
            "facts": target_owned or owned_stmts,
        }
