"""
Optional System 1 Fast Decision Engine (TypeSafe Jev / OpenRouter `typesafe/jev-1.13`).

Provides typed, non-generative System 1 primitives (`Choice`, `Noul`, `Score`) for:
1. Query Routing & Category Classification (`route_query_decision`)
2. Batch Enrichment Gatekeeping (`evaluate_enrichment_gate`)
3. Post-Retrieval Chunk Reranking (`rerank_chunks`)

100% Optional with automatic Circuit-Breaker fallback to zero-latency local heuristics
when disabled (`jev_enabled=False`) or when offline / rate-limited.
"""
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from memoria.config import (
    JEV_API_KEY,
    JEV_BASE_URL,
    JEV_ENABLED,
    JEV_GATEKEEPER_THRESHOLD,
    JEV_MODEL,
)
from memoria.db.engine import Database

VALID_CATEGORIES = [
    "all",
    "audio_interface",
    "audiocassette",
    "tape_deck",
    "microfoni",
    "monitor_cuffie",
    "homelab",
    "software",
    "hardware",
    "general",
]


class JevSystemOneEngine:
    """
    System 1 Decision Engine (`Choice`, `Noul`, `Score`) with circuit-breaker local fallback.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        timeout_seconds: float = 0.75,
    ):
        self.db = db
        self._explicit_enabled = enabled
        self._explicit_api_key = api_key
        self._explicit_base_url = base_url
        self._explicit_model = model
        self._explicit_threshold = threshold
        self.timeout_seconds = timeout_seconds

        # Runtime telemetry
        self.stats = {
            "calls_total": 0,
            "calls_succeeded": 0,
            "local_fallbacks": 0,
            "gatekeeper_accepted": 0,
            "gatekeeper_filtered": 0,
            "last_latency_ms": 0.0,
            "last_error": None,
        }

    def get_config(self) -> Dict[str, Any]:
        """Reads dynamic configuration from Database settings or environment fallbacks."""
        enabled = self._explicit_enabled
        if enabled is None:
            if self.db:
                raw_en = self.db.get_setting("jev_enabled", "")
                if raw_en != "":
                    enabled = raw_en.lower() in ("true", "1", "yes")
            if enabled is None:
                enabled = JEV_ENABLED

        model = self._explicit_model or (self.db.get_setting("jev_model") if self.db else None) or JEV_MODEL
        base_url = (
            self._explicit_base_url
            or (self.db.get_setting("jev_base_url") if self.db else None)
            or JEV_BASE_URL
        ).rstrip("/")

        raw_thresh = self._explicit_threshold
        if raw_thresh is None and self.db:
            db_t = self.db.get_setting("jev_threshold")
            if db_t:
                try:
                    raw_thresh = float(db_t)
                except ValueError:
                    pass
        threshold = raw_thresh if raw_thresh is not None else JEV_GATEKEEPER_THRESHOLD

        # Resolve API key: explicit -> db jev_api_key -> db openrouter key -> env
        api_key = self._explicit_api_key or (self.db.get_setting("jev_api_key") if self.db else "") or JEV_API_KEY
        if not api_key and self.db:
            try:
                from memoria.settings.manager import SettingsManager

                sm = SettingsManager(self.db)
                or_cfg = sm.get_provider_config("openrouter")
                if or_cfg and or_cfg.get("api_key"):
                    api_key = or_cfg["api_key"]
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY", "")

        return {
            "enabled": bool(enabled),
            "model": model,
            "base_url": base_url,
            "api_key_configured": bool(api_key and api_key != "dummy"),
            "api_key": api_key,
            "threshold": float(threshold),
        }

    def save_config(
        self,
        enabled: bool,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Persists Jev System 1 configuration in SQLite settings."""
        if self.db:
            self.db.set_setting("jev_enabled", "true" if enabled else "false")
            if model is not None:
                self.db.set_setting("jev_model", model.strip())
            if base_url is not None:
                self.db.set_setting("jev_base_url", base_url.strip())
            if api_key is not None and api_key.strip() and not api_key.startswith("***"):
                self.db.set_setting("jev_api_key", api_key.strip())
            if threshold is not None:
                self.db.set_setting("jev_threshold", str(float(threshold)))
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        cfg = self.get_config()
        return {
            "enabled": cfg["enabled"],
            "active": bool(cfg["enabled"] and cfg["api_key_configured"]),
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key_configured": cfg["api_key_configured"],
            "threshold": cfg["threshold"],
            "stats": dict(self.stats),
        }

    def _call_jev_primitives(self, state_text: str, questions_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executes a single-pass System 1 decision call via OpenRouter (`typesafe/jev-1.13`)
        or TypeSafe SystemOne endpoint, returning structured `Choice`, `Noul`, and `Score` values.
        """
        cfg = self.get_config()
        if not cfg["enabled"] or not cfg["api_key_configured"]:
            return None

        self.stats["calls_total"] += 1
        t0 = time.perf_counter()

        try:
            endpoint = f"{cfg['base_url']}/chat/completions"
            if cfg["base_url"].endswith("/system-one"):
                endpoint = cfg["base_url"]

            prompt_payload = {
                "system_one_state": state_text[:3500],
                "questions": questions_schema,
            }

            body = {
                "model": cfg["model"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Jev System 1 Decision Engine. Evaluate the provided state against the "
                            "typed questions (Choice, Noul [0..1], Score [0..1]) and return ONLY a compact JSON object "
                            "with the exact question keys."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
                "temperature": 0.0,
                "max_tokens": 180,
                "response_format": {"type": "json_object"},
            }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg['api_key']}",
                    "HTTP-Referer": "https://github.com/gaskb/memoria-ai",
                    "X-Title": "Memoria AI (System 1 Router)",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                content = raw.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                parsed = json.loads(content)
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.stats["calls_succeeded"] += 1
                self.stats["last_latency_ms"] = round(elapsed, 1)
                return parsed
        except Exception as e:
            self.stats["local_fallbacks"] += 1
            self.stats["last_error"] = str(e)
            return None

    # -------------------------------------------------------------------------
    # 1. Smart Query Router & Category Classifier
    # -------------------------------------------------------------------------
    def route_query_decision(self, query: str) -> Dict[str, Any]:
        """
        Determines query execution mode (`Choice`), category filter (`Choice`),
        `strict_owned_only` (`Noul`), and `is_decision_lookup` (`Noul`).
        """
        schema = {
            "mode": {
                "type": "Choice",
                "options": ["extractive", "structured", "synthetic"],
                "instructions": "extractive=locate exact chat quotes; structured=list inventory/decisions; synthetic=recommend or compare across inventory and knowledge.",
            },
            "category": {
                "type": "Choice",
                "options": VALID_CATEGORIES,
                "instructions": "Target domain category if the query focuses on a specific domain, otherwise 'all'.",
            },
            "strict_owned_only": {
                "type": "Noul",
                "question": "Is the user asking specifically about equipment/software they personally own in their inventory?",
            },
            "is_decision_lookup": {
                "type": "Noul",
                "question": "Is the user asking about past architectural, hardware, or technical decisions?",
            },
        }

        remote = self._call_jev_primitives(query, schema)
        if remote and remote.get("mode") in ("extractive", "structured", "synthetic"):
            cat = remote.get("category", "all")
            if cat not in VALID_CATEGORIES:
                cat = "all"
            return {
                "mode": remote["mode"],
                "category": cat,
                "strict_owned_only": float(remote.get("strict_owned_only", 0.5)),
                "is_decision_lookup": float(remote.get("is_decision_lookup", 0.2)),
                "engine": "jev_system_1",
            }

        # Zero-latency Local Heuristic Fallback
        q_lower = (query or "").lower()
        if any(
            k in q_lower
            for k in (
                "dove ho parlato",
                "quando ho detto",
                "citami",
                "trova la chat",
                "cerca",
                "where did i mention",
                "when did i say",
                "find the chat",
            )
        ):
            mode = "extractive"
        elif any(
            k in q_lower
            for k in (
                "cosa possiedo",
                "quali ho",
                "elenco delle",
                "quali cassette possiedo",
                "che setup",
                "decisioni prese",
                "cosa ho deciso",
                "what do i own",
                "list my",
                "what equipment",
                "what decisions",
            )
        ):
            mode = "structured"
        else:
            mode = "synthetic"

        from memoria.knowledge.extractor import guess_category

        detected_cat = guess_category(query)
        if detected_cat == "general":
            detected_cat = "all"

        strict_owned = 0.9 if any(w in q_lower for w in ("che ho", "possiedo", "mio", "miei", "i own", "my ")) else 0.3
        is_dec = 0.9 if any(w in q_lower for w in ("deciso", "decision", "scelto", "perché abbiamo", "why did we")) else 0.1

        return {
            "mode": mode,
            "category": detected_cat,
            "strict_owned_only": strict_owned,
            "is_decision_lookup": is_dec,
            "engine": "local_heuristic",
        }

    # -------------------------------------------------------------------------
    # 2. Batch Enrichment Gatekeeper (Noise Filter before System 2 LLM)
    # -------------------------------------------------------------------------
    def evaluate_enrichment_gate(self, conversation_snippet: str) -> Dict[str, Any]:
        """
        Evaluates whether a conversation contains durable facts or technical decisions
        worth extracting with the System 2 generative LLM.
        """
        cfg = self.get_config()
        threshold = cfg["threshold"]

        schema = {
            "has_permanent_fact": {
                "type": "Noul",
                "question": "Does the user state ownership, purchase, or permanent configuration of hardware, audio gear, server, or software?",
            },
            "has_architectural_decision": {
                "type": "Noul",
                "question": "Is a lasting technical, architectural, or purchasing decision made between alternatives?",
            },
            "memory_score": {
                "type": "Score",
                "scale": ["ephemeral_debug_noise", "casual_chat", "useful_reference", "core_permanent_memory"],
                "instructions": "Rate the long-term personal knowledge value of this conversation.",
            },
        }

        remote = self._call_jev_primitives(conversation_snippet, schema)
        if remote and ("has_permanent_fact" in remote or "has_architectural_decision" in remote):
            p_fact = float(remote.get("has_permanent_fact", 0.0))
            p_dec = float(remote.get("has_architectural_decision", 0.0))
            m_score = float(remote.get("memory_score", max(p_fact, p_dec)))
            combined = max(p_fact, p_dec, m_score)
            should_enrich = combined >= threshold
            if should_enrich:
                self.stats["gatekeeper_accepted"] += 1
            else:
                self.stats["gatekeeper_filtered"] += 1
            return {
                "should_enrich": should_enrich,
                "has_permanent_fact": round(p_fact, 3),
                "has_architectural_decision": round(p_dec, 3),
                "confidence": round(combined, 3),
                "engine": "jev_system_1",
            }

        # Local heuristic fallback: always allow unless purely empty/system log
        text_low = (conversation_snippet or "").lower()
        signal_words = (
            "possiedo",
            "comprato",
            "ho preso",
            "uso ",
            "deciso",
            "scelto",
            "server",
            "scheda",
            "cassetta",
            "docker",
            "proxmox",
            "i own",
            "bought",
            "decided",
            "configured",
        )
        has_signal = any(w in text_low for w in signal_words) or len(text_low) > 250
        return {
            "should_enrich": bool(has_signal),
            "has_permanent_fact": 0.75 if has_signal else 0.2,
            "has_architectural_decision": 0.65 if has_signal else 0.1,
            "confidence": 0.75 if has_signal else 0.2,
            "engine": "local_heuristic",
        }

    # -------------------------------------------------------------------------
    # 3. Post-Retrieval Chunk Reranker (`Score`)
    # -------------------------------------------------------------------------
    def rerank_chunks(self, query: str, hits: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Optionally re-scores top retrieved chunks using Jev `Score` primitive when enabled.
        """
        if not hits or len(hits) <= 2:
            return hits[:top_k]

        cfg = self.get_config()
        if not cfg["enabled"] or not cfg["api_key_configured"]:
            return hits[:top_k]

        candidates = hits[: min(len(hits), 12)]
        state_lines = [f"QUERY: {query}\n"]
        schema: Dict[str, Any] = {}
        for idx, h in enumerate(candidates):
            key = f"chunk_{idx}"
            snippet = (h.get("content") or "")[:300].replace("\n", " ")
            state_lines.append(f"[{key}] ({h.get('title', '')}): {snippet}")
            schema[key] = {
                "type": "Score",
                "scale": ["irrelevant", "tangential", "helpful", "directly_answers"],
            }

        remote = self._call_jev_primitives("\n".join(state_lines), schema)
        if not remote:
            return hits[:top_k]

        scored_hits = []
        for idx, h in enumerate(candidates):
            key = f"chunk_{idx}"
            jev_val = remote.get(key)
            try:
                bonus = float(jev_val) if isinstance(jev_val, (int, float)) else 0.5
            except Exception:
                bonus = 0.5
            h_copy = dict(h)
            base_score = float(h_copy.get("rrf_score", h_copy.get("score", 0.5)))
            h_copy["jev_score"] = round(bonus, 3)
            h_copy["combined_score"] = base_score + (0.35 * bonus)
            scored_hits.append(h_copy)

        scored_hits.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_hits[:top_k]
