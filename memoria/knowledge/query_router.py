"""
QueryRouter (Issue #26 / A-01):
Dedicated routing and language classification component separated from QueryPlanner execution.
"""

from typing import Any, Dict, Optional

from memoria.config import DEFAULT_LANG
from memoria.llm.jev import JevSystemOneEngine


def detect_query_lang(query: str, fallback: str = DEFAULT_LANG) -> str:
    """Detects whether a user query is in Italian ('it') or English ('en')."""
    q = (query or "").lower()
    it_markers = (
        "quali ", "quale ", "cosa ", "come ", "dove ", "quando ",
        "possiedo", "abbiamo", "nelle ", "delle ", "che ho", "deciso",
    )
    en_markers = (
        "which ", "what ", "how ", "where ", "when ",
        "do i own", "have i ", "between ", "my inventory", "decisions",
    )
    if any(m in q for m in it_markers):
        return "it"
    if any(m in q for m in en_markers):
        return "en"
    return fallback or "en"


class QueryRouter:
    """Routes user queries across Extractive, Structured, and Synthetic modes via System 1 (Jev) or local rules."""

    def __init__(self, jev: Any):
        if isinstance(jev, JevSystemOneEngine):
            self.jev = jev
        else:
            self.jev = JevSystemOneEngine(jev)

    def detect_query_lang(self, query: str, fallback: str = DEFAULT_LANG) -> str:
        return detect_query_lang(query, fallback=fallback)

    def select_route(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        route_decision = self.jev.route_query_decision(query)
        eff_mode = route_decision.get("mode", "synthetic") if mode == "auto" else mode
        return {**route_decision, "mode": eff_mode}

    def classify(self, query: str, requested_mode: str = "auto", fallback_lang: str = DEFAULT_LANG) -> Dict[str, Any]:
        eff_lang = detect_query_lang(query, fallback=fallback_lang)
        route_decision = self.jev.route_query_decision(query)
        mode = route_decision.get("mode", "synthetic") if requested_mode == "auto" else requested_mode
        return {
            "effective_mode": mode,
            "effective_lang": eff_lang,
            "route_decision": route_decision,
        }

