import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
from engramvault.db.engine import Database
from engramvault.db.models import DecisionRecord, StatementRecord
from engramvault.knowledge.extractor import KnowledgeExtractor, guess_category
from engramvault.llm.client import LLMClient, parse_llm_json
from engramvault.llm.jev import JevSystemOneEngine

logger = logging.getLogger("engramvault.enricher")


def _clean_str(val: Any, default: str = "") -> str:
    """Sanitizes text by removing null bytes and unpaired Unicode surrogates."""
    if val is None:
        return default
    s = str(val).replace("\x00", "")
    # Remove surrogate code points that crash utf-8 encoders in SQLite
    return s.encode("utf-8", "ignore").decode("utf-8").strip()


class KnowledgeEnricher:
    """
    Batch processor to systematically extract and structure knowledge (entities, facts,
    claims, decisions, timeline) across conversations using external/local LLM or heuristic rules,
    with optional Jev System 1 Gatekeeper pre-filtering.
    """

    def __init__(
        self,
        db: Database,
        llm: Optional[LLMClient] = None,
        delay_between_calls: float = 0.1,
        jev: Optional[JevSystemOneEngine] = None,
    ):
        self.db = db
        self._custom_llm = llm
        self.delay_between_calls = delay_between_calls
        self.extractor = KnowledgeExtractor(db)
        self.jev = jev or JevSystemOneEngine(db=db)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

        m_name = self.llm.model if self.llm.is_configured() else "rule-based-heuristics"
        self.status = {
            "is_running": False,
            "total_to_process": 0,
            "processed": 0,
            "new_facts": 0,
            "new_decisions": 0,
            "new_relations": 0,
            "jev_filtered": 0,
            "current_conversation": "",
            "extractor_model": m_name,
            "error": None,
        }

    @property
    def llm(self) -> LLMClient:
        if self._custom_llm is not None:
            return self._custom_llm
        from engramvault.settings.manager import SettingsManager
        return SettingsManager(self.db).get_llm_client()

    def get_status(self) -> Dict[str, Any]:
        st = dict(self.status)
        st["jev_gatekeeper"] = self.jev.get_status()
        try:
            total_convs = self.db.get_stats().get("conversations", 0)
            unenriched = self.db.count_unenriched_conversations()
            st["total_conversations"] = total_convs
            st["unenriched_count"] = unenriched
            st["enriched_count"] = max(0, total_convs - unenriched)
        except Exception as exc:
            logger.debug("Failed to calculate enrich stats: %s", exc)
            st["total_conversations"] = 0
            st["unenriched_count"] = 0
            st["enriched_count"] = 0
        return st

    def extract_with_llm(self, conv_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Invokes configured LLM with prompt for structured entity/fact/claim/decision extraction."""
        messages = conv_record.get("messages", [])
        if not messages:
            return None

        # Format conversation snippets up to 6000 chars for token budget
        lines = []
        for m in messages:
            role = m.get("role", "user").upper()
            text = (m.get("content") or "").strip()
            lines.append(f"{role}: {text}")
        dialogue_text = "\n\n".join(lines)[:6000]

        # Optional System 1 Jev Gatekeeper pre-filter before calling expensive System 2 LLM
        if self.jev.get_config().get("enabled"):
            gate = self.jev.evaluate_enrichment_gate(f"TITLE: {conv_record.get('title', '')}\n{dialogue_text[:2500]}")
            if not gate.get("should_enrich", True):
                self.status["jev_filtered"] = self.status.get("jev_filtered", 0) + 1
                return {"entities": [], "facts": [], "claims": [], "decisions": [], "jev_filtered": True}

        known_cats = self.db.get_entity_categories()
        fallback_cats = [
            "audiocassette", "audio_interface", "hardware_deck", "audio_pro_studio",
            "ham_radio", "homelab_infra", "hardware_pc", "software", "project",
            "cooking", "personal_life", "general"
        ]
        merged_cats = list(dict.fromkeys(known_cats + fallback_cats))
        cats_hint = ", ".join(merged_cats[:25])

        system_prompt = (
            "Sei un estrattore di conoscenza per la memoria personale di uno sviluppatore e appassionato tech.\n"
            "Analizza il dialogo ed estrai SOLO informazioni certe e concrete in JSON con questo schema:\n"
            "{\n"
            "  \"entities\": [{\"name\": \"...\", \"category\": \"<categoria>\"}],\n"
            "  \"facts\": [{\"subject\": \"...\", \"predicate\": \"owned_by_user|uses|preference\", \"object\": \"...\", \"notes\": \"...\"}],\n"
            "  \"claims\": [{\"subject\": \"...\", \"predicate\": \"suitable_for|hardware_quality|feature\", \"object\": \"...\"}],\n"
            "  \"decisions\": [{\"topic\": \"...\", \"decision\": \"...\", \"status\": \"ADOPTED|EVALUATING|ABANDONED\"}],\n"
            "  \"relations\": [{\"from_entity\": \"...\", \"relation\": \"connected_to|runs_on|part_of|depends_on|compatible_with|subproject_of|uses|created_by\", \"to_entity\": \"...\"}]\n"
            "}\n"
            "Regole di categorizzazione entità (Open-World Dynamic Taxonomy):\n"
            f"- Categorie esistenti consigliate: {cats_hint}.\n"
            "- Se un'entità appartiene chiaramente a una di queste categorie, riusala.\n"
            "- Se appartiene a un nuovo dominio specialistico (es. 'fotografia', 'giardinaggio', 'domotica'), crea una nuova categoria sintetica in snake_case (tutto minuscolo, senza spazi).\n"
            "- Usa 'general' solo come ultima risorsa per concetti generici o astratti.\n"
            "Regole tassative sui contenuti:\n"
            "1. 'facts': SOLO ciò che l'utente dichiara esplicitamente di possedere, usare o fare.\n"
            "2. 'claims': affermazioni o pareri tecnici formulati dall'AI.\n"
            "3. 'relations': SOLO relazioni logiche o fisiche dirette tra due entità concrete (es. apparato -> antenna, app -> container/db).\n"
            "4. Rispondi ESCLUSIVAMENTE con il blocco JSON valido, senza testo introduttivo."
        )

        user_content = f"TITOLO: {conv_record.get('title', '')}\nDATA: {conv_record.get('created_at', '')[:10]}\n\nDIALOGO:\n{dialogue_text}"

        res = self.llm.chat_completion(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1500,
        )
        return parse_llm_json(res)

    def run_batch(
        self,
        limit: Optional[int] = None,
        rebuild: bool = False,
        delay: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes batch enrichment across conversations.
        Supports resume or complete rebuild.
        """
        if delay is not None:
            self.delay_between_calls = delay

        if rebuild:
            logger.info("Modalità --rebuild: azzeramento Level 3 knowledge layer...")
            self.db.reset_knowledge_layer()

        unenriched = self.db.get_unenriched_conversations(limit=limit)
        total = len(unenriched)

        self.is_running = True
        model_name = self.llm.model if self.llm.is_configured() else "heuristics-engine"
        self.status = {
            "is_running": True,
            "total_to_process": total,
            "processed": 0,
            "new_facts": 0,
            "new_decisions": 0,
            "new_relations": 0,
            "current_conversation": "",
            "extractor_model": model_name,
            "error": None,
        }

        logger.info(f"Avvio batch enrichment su {total} conversazioni con modello '{model_name}'...")

        try:
            for item in unenriched:
                if not self.is_running:
                    logger.info("Batch enrichment interrotto manualmente.")
                    break

                conv_id = item["id"]
                title = item.get("title", "")
                self.status["current_conversation"] = title

                try:
                    conv_record = self.db.get_conversation(conv_id)
                    if not conv_record or not conv_record.get("messages"):
                        self.db.mark_conversation_enriched(conv_id, model_name, 0)
                        self.status["processed"] += 1
                        continue

                    extracted_count = 0
                    parsed = None
                    if self.llm.is_configured():
                        try:
                            parsed = self.extract_with_llm(conv_record)
                        except Exception as parse_exc:
                            logger.warning(
                                "Errore durante estrazione LLM per conversazione '%s' (%s): %s",
                                title, conv_id, parse_exc
                            )
                            parsed = None

                    if parsed and isinstance(parsed, dict):
                        # Ingest LLM structured data
                        ts = conv_record.get("created_at")
                        first_msg_id = conv_record["messages"][0]["id"] if conv_record["messages"] else None

                        # 1. Entities
                        extracted_entity_cats = {}
                        for ent in parsed.get("entities", []):
                            name = _clean_str(ent.get("name") if isinstance(ent, dict) else "")
                            raw_cat = _clean_str(ent.get("category") if isinstance(ent, dict) else "")
                            cat = raw_cat or guess_category(name)
                            if name:
                                self.db.upsert_entity(name, cat)
                                extracted_entity_cats[name.strip().lower()] = cat

                        # 2. Facts
                        for f in parsed.get("facts", []):
                            if not isinstance(f, dict):
                                continue
                            subj = _clean_str(f.get("subject"))
                            pred = _clean_str(f.get("predicate"), "owned_by_user")
                            obj = _clean_str(f.get("object"), "true")
                            if subj:
                                norm_subj = subj.strip().lower()
                                cat = extracted_entity_cats.get(norm_subj) or guess_category(subj)
                                ent_id = self.db.upsert_entity(subj, cat)
                                stmt = StatementRecord(
                                    id=str(uuid.uuid4()),
                                    entity_id=ent_id,
                                    statement_type="FACT",
                                    subject=subj,
                                    predicate=pred,
                                    object=str(obj),
                                    author="user",
                                    confidence=1.0,
                                    source_message_id=first_msg_id,
                                    source_conversation_id=conv_id,
                                    valid_from=ts,
                                    is_current=1,
                                    notes=_clean_str(f.get("notes")) or f"Estratto da {model_name}",
                                )
                                self.db.insert_statement(stmt)
                                self.status["new_facts"] += 1
                                extracted_count += 1

                        # 3. Claims
                        for c in parsed.get("claims", []):
                            if not isinstance(c, dict):
                                continue
                            subj = _clean_str(c.get("subject"))
                            pred = _clean_str(c.get("predicate"), "technical_claim")
                            obj = _clean_str(c.get("object"))
                            if subj and obj:
                                norm_subj = subj.strip().lower()
                                cat = extracted_entity_cats.get(norm_subj) or guess_category(subj)
                                ent_id = self.db.upsert_entity(subj, cat)
                                stmt = StatementRecord(
                                    id=str(uuid.uuid4()),
                                    entity_id=ent_id,
                                    statement_type="CLAIM",
                                    subject=subj,
                                    predicate=pred,
                                    object=str(obj),
                                    author="assistant",
                                    confidence=0.85,
                                    source_message_id=first_msg_id,
                                    source_conversation_id=conv_id,
                                    valid_from=ts,
                                    is_current=1,
                                    notes=f"Claim tecnico estratto da {model_name}",
                                )
                                self.db.insert_statement(stmt)
                                extracted_count += 1

                        # 4. Decisions
                        for d in parsed.get("decisions", []):
                            if not isinstance(d, dict):
                                continue
                            topic = _clean_str(d.get("topic"), "Decisione Personale")
                            dec_text = _clean_str(d.get("decision"))
                            status_val = _clean_str(d.get("status"), "ADOPTED").upper()
                            if status_val not in ("EVALUATING", "ADOPTED", "ABANDONED", "SUPERSEDED"):
                                status_val = "ADOPTED"
                            if dec_text:
                                dec = DecisionRecord(
                                    id=str(uuid.uuid4()),
                                    topic=topic,
                                    decision_text=dec_text,
                                    status=status_val,
                                    decided_at=ts or "N/D",
                                    source_message_id=first_msg_id,
                                    source_conversation_id=conv_id,
                                )
                                self.db.insert_decision(dec)
                                self.status["new_decisions"] += 1
                                extracted_count += 1

                        # 5. Relations
                        for r in parsed.get("relations", []):
                            if not isinstance(r, dict):
                                continue
                            f_ent = _clean_str(r.get("from_entity"))
                            t_ent = _clean_str(r.get("to_entity"))
                            rel_type = _clean_str(r.get("relation") or r.get("relation_type"), "relates_to")
                            if f_ent and t_ent and f_ent.lower() != t_ent.lower():
                                f_cat = extracted_entity_cats.get(f_ent.lower()) or guess_category(f_ent)
                                t_cat = extracted_entity_cats.get(t_ent.lower()) or guess_category(t_ent)
                                f_id = self.db.upsert_entity(f_ent, f_cat)
                                t_id = self.db.upsert_entity(t_ent, t_cat)
                                from engramvault.db.models import RelationRecord
                                rel_rec = RelationRecord(
                                    id=str(uuid.uuid4()),
                                    from_entity_id=f_id,
                                    to_entity_id=t_id,
                                    relation_type=rel_type,
                                    source_message_id=first_msg_id,
                                )
                                self.db.insert_relation(rel_rec)
                                self.status["new_relations"] = self.status.get("new_relations", 0) + 1
                                extracted_count += 1
                    else:
                        # Fallback to deterministic heuristic extractor
                        from engramvault.adapters.base import NormalizedConversation, NormalizedMessage
                        norm_msgs = [
                            NormalizedMessage(role=m["role"], text=m["content"], ts=m["ts"])
                            for m in conv_record["messages"]
                        ]
                        norm_conv = NormalizedConversation(
                            assistant=conv_record["provider"],
                            title=conv_record["title"],
                            date=conv_record["created_at"],
                            external_ref=conv_record["external_ref"],
                            messages=norm_msgs,
                        )
                        self.extractor.extract_from_conversation(norm_conv, conv_id, conv_record["messages"])
                        extracted_count = 1

                    self.db.mark_conversation_enriched(conv_id, model_name, extracted_count)
                    self.status["processed"] += 1
                except Exception as conv_exc:
                    logger.error(
                        "Errore durante l'elaborazione della conversazione '%s' (%s): %s",
                        title, conv_id, conv_exc, exc_info=True
                    )
                    # Mark enriched with 0 extracted so the batch doesn't get stuck indefinitely
                    try:
                        self.db.mark_conversation_enriched(conv_id, model_name, 0)
                        self.status["processed"] += 1
                    except Exception:
                        pass

                if self.delay_between_calls > 0:
                    time.sleep(self.delay_between_calls)

        except Exception as e:
            logger.error(f"Errore durante batch enrich: {e}", exc_info=True)
            self.status["error"] = str(e)
        finally:
            self.is_running = False
            self.status["is_running"] = False

        return self.get_status()

    def start_background(
        self,
        limit: Optional[int] = None,
        rebuild: bool = False,
        delay: Optional[float] = None,
    ):
        """Launches batch enrich in a background daemon thread."""
        if self.is_running:
            return
        self._thread = threading.Thread(
            target=self.run_batch,
            kwargs={"limit": limit, "rebuild": rebuild, "delay": delay},
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self.is_running = False
