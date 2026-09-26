import re
import uuid
from typing import List, Tuple, Dict, Any, Optional
from memoria.adapters.base import NormalizedConversation, NormalizedMessage
from memoria.db.models import StatementRecord, DecisionRecord, EntityRecord, RelationRecord
from memoria.db.engine import Database

# Audio gear, cassette, homelab, dev and personal entity keywords for categorizing
CATEGORY_KEYWORDS = {
    "audiocassette": [
        "tdk", "maxell", "sony hf", "sony ufx", "sony ux", "basf", "cassetta", "audiocassetta",
        "nastro magnetico", "type ii", "type i", "type iv", "metal tape", "avilyn", "ferrocromo", "c-60", "c-90"
    ],
    "audio_interface": [
        "audient", "zen go", "focusrite", "motu", "scarlett", "rme", "interfaccia audio", "scheda audio", "audio interface"
    ],
    "hardware_deck": [
        "nakamichi", "walkman", "teac", "tape deck", "piastra a cassette", "registratore a cassette",
        "capstan", "pinch roller", "cinghia", "cinghie"
    ],
    "audio_pro_studio": [
        "mixer", "preamp", "preamplificatore", "shure sm7b", "shure sm58", "monitor da studio",
        "cuffie monitor", "microfono", "compressore", "outboard", "cavo bilanciato", "xlr", "akg", "sennheiser", "beyerdynamic"
    ],
    "ham_radio": [
        "yaesu", "icom", "kenwood", "radioamatore", "antenne", "balun", "dipolo", "accordatore",
        "hf", "vhf", "uhf", "nominativo", "qrz", "cw", "ssb", "ricetrasmettitore", "baofeng", "qso", "swr"
    ],
    "homelab_infra": [
        "proxmox", "docker", "reverse proxy", "nginx", "npm", "vlan", "tailscale", "zfs", "raid",
        "dns", "pi-hole", "lxc", "router", "switch", "pve", "synology", "nas", "truenas", "opnsense", "wireguard"
    ],
    "hardware_pc": [
        "cpu", "gpu", "rtx", "amd", "intel", "ram", "nvme", "ssd", "alimentatore", "motherboard",
        "scheda madre", "vram", "pc fisso", "workstation", "thinkpad", "ryzen"
    ],
    "software": [
        "fastapi", "sqlite", "python", "docker compose", "react", "vue", "gitlab", "typescript",
        "javascript", "bash", "linux", "git", "css", "html", "pwa", "api", "rest", "sdk", "postgres", "redis"
    ],
    "project": [
        "livellatore", "quattrotracce", "engramvault", "memoria", "digestly", "openstorybox",
        "cozytabs", "eggonomics", "pingsentry", "jar", "checkdns", "hyphencheck", "trucksavekit"
    ],
    "cooking": [
        "ricetta", "impasto", "pizza", "lievito", "farina", "idratazione", "maturazione", "forno"
    ],
    "personal_life": [
        "viaggio", "vacanza", "salute", "casa", "auto", "spesa", "medico", "palestra"
    ],
}


def guess_category(name: str) -> str:
    lower = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return cat
    return "general"


def split_items(item_list_str: str) -> List[str]:
    """Splits strings like 'TDK SA-X, Maxell XLII e Sony HF' into individual items."""
    # Replace conjunctions ' e ' or ' ed ' with comma
    normalized = re.sub(r"\s+(?:e|ed)\s+", ", ", item_list_str)
    raw_tokens = [t.strip() for t in normalized.split(",") if t.strip()]
    cleaned = []
    for token in raw_tokens:
        clean = re.sub(r"^(?:delle|dei|degli|un|uno|una|il|lo|la|l')\s+", "", token, flags=re.IGNORECASE).strip()
        clean = clean.strip(".:; ")
        if len(clean) >= 2 and clean.lower() not in ("cose", "tutto", "niente", "qualcosa", "tempo", "queste", "questi"):
            cleaned.append(clean)
    return cleaned


class KnowledgeExtractor:
    def __init__(self, db: Database):
        self.db = db

    def extract_from_conversation(
        self,
        conv: NormalizedConversation,
        conv_id: str,
        message_records: List[Dict[str, Any]]
    ):
        """Extracts facts, claims, decisions and updates entities and timeline."""
        for msg in message_records:
            msg_id = msg["id"]
            role = msg["role"]
            text = msg["content"]
            ts = msg["ts"] or conv.date

            if role == "user":
                self._extract_user_facts_and_decisions(text, ts, msg_id, conv_id)
            elif role == "assistant":
                self._extract_assistant_claims(text, ts, msg_id, conv_id)

    def _register_ownership_fact(
        self,
        item: str,
        owned: bool,
        ts: str,
        msg_id: str,
        conv_id: str,
        notes: str = "Dichiarato dall'utente"
    ):
        category = guess_category(item)
        entity_id = self.db.upsert_entity(item, category)

        # Invalidate previous ownership states if selling or changing
        prev_stmts = self.db.get_statements(entity_id=entity_id, predicate="owned_by_user", is_current=1)
        supersedes_id = None
        if prev_stmts:
            for prev in prev_stmts:
                self.db.invalidate_statement(prev["id"], valid_until=ts)
                supersedes_id = prev["id"]

        stmt = StatementRecord(
            id=str(uuid.uuid4()),
            entity_id=entity_id,
            statement_type="FACT",
            subject=item,
            predicate="owned_by_user",
            object="true" if owned else "false",
            author="user",
            confidence=1.0,
            source_message_id=msg_id,
            source_conversation_id=conv_id,
            valid_from=ts,
            valid_until=None,
            is_current=1 if owned else 0,
            supersedes_id=supersedes_id,
            notes=notes
        )
        self.db.insert_statement(stmt)

    def _extract_user_facts_and_decisions(self, text: str, ts: str, msg_id: str, conv_id: str):
        # 1. Ownership: positive (lists and single items)
        # E.g. "Ho queste audiocassette: TDK SA-X, Maxell XLII e Sony HF."
        match_lists = re.search(
            r"\b(?:ho queste audiocassette|possiedo le seguenti audiocassette|ho questi nastri)[:\s]+([^.]+)",
            text, re.IGNORECASE
        )
        if match_lists:
            items_str = match_lists.group(1)
            for item in split_items(items_str):
                self._register_ownership_fact(item, True, ts, msg_id, conv_id)

        # E.g. "Nel mio setup ho la Audient iD22" / "Ho comprato la TDK SA-X"
        pos_singles = re.finditer(
            r"\b(?:nel mio setup ho|possiedo|ho comprato|ho preso|uso)\s+(?:il|lo|la|l'|un|una|delle|dei)?\s*([A-Za-z0-9\-\s\+]{2,30}?)(?=[,.\n]| e |\bma\b|$)",
            text, re.IGNORECASE
        )
        for m in pos_singles:
            raw_item = m.group(1).strip()
            for item in split_items(raw_item):
                self._register_ownership_fact(item, True, ts, msg_id, conv_id)

        # 2. Ownership: negative (selling / discarding)
        # E.g. "Ho venduto la Audient iD22 e ho deciso..."
        neg_matches = re.finditer(
            r"\b(?:ho venduto|venduta|dismesso|non uso più|non ho più)\s+(?:il|lo|la|l'|un|una)?\s*([A-Za-z0-9\-\s\+]{2,30}?)(?=[,.\n]| e |\bma\b|$)",
            text, re.IGNORECASE
        )
        for m in neg_matches:
            raw_item = m.group(1).strip()
            for item in split_items(raw_item):
                self._register_ownership_fact(item, False, ts, msg_id, conv_id, notes="Oggetto venduto/dismesso")

        # 3. Decisions
        decision_patterns = [
            re.compile(r"\b(?:ho deciso di|alla fine ho scelto di|alla fine scelgo|ho preferito)\s+([^.\n]+)", re.IGNORECASE),
            re.compile(r"\b(?:per il progetto|riguardo a)\s+([^,\.\n]{3,40})\s+(?:ho optato per|faccio|ho deciso di usare)\s+([^.\n]+)", re.IGNORECASE),
            re.compile(r"\b(?:alla fine non compro|ho scartato)\s+([^.\n]+)", re.IGNORECASE),
        ]

        for pattern in decision_patterns:
            for match in pattern.finditer(text):
                groups = [g.strip() for g in match.groups() if g]
                if not groups:
                    continue

                topic = "Decisione Personale"
                dec_text = groups[0]
                if len(groups) >= 2:
                    topic = groups[0]
                    dec_text = groups[1]

                dec = DecisionRecord(
                    id=str(uuid.uuid4()),
                    topic=topic,
                    decision_text=dec_text,
                    status="ADOPTED",
                    decided_at=ts,
                    source_message_id=msg_id,
                    source_conversation_id=conv_id
                )
                self.db.insert_decision(dec)

                # If decision was "usare X come interfaccia principale", also register X as active gear
                opt_match = re.search(r"usare\s+(?:la|il|l')?\s*([A-Za-z0-9\-\s]{2,20})\s+come", dec_text, re.IGNORECASE)
                if opt_match:
                    item_name = opt_match.group(1).strip()
                    self._register_ownership_fact(item_name, True, ts, msg_id, conv_id, notes="Scelto come principale")

    def _extract_assistant_claims(self, text: str, ts: str, msg_id: str, conv_id: str):
        # Scan for technical claims about cassettes / rock / gear
        claim_patterns = [
            (r"\b(TDK\s+[A-Za-z0-9\-]+|Maxell\s+[A-Za-z0-9\-]+|Sony\s+[A-Za-z0-9\-]+)\b.*?(\b(?:adatta|ottima|ideale|migliore|consigliata|saturazione|headroom|distorsione|normale|cromo)\b[^\.\n]+)", "suitable_for"),
            (r"\b(Zen\s+Go|Audient\s+iD\d+|Nakamichi\s+[A-Za-z0-9\-]+)\b.*?(\b(?:convertitori|preamplificatori|affidabilità|driver|testine|bias)\b[^\.\n]+)", "hardware_quality")
        ]

        for pat_str, pred in claim_patterns:
            matches = re.finditer(pat_str, text, re.IGNORECASE)
            for m in matches:
                subject = m.group(1).strip()
                claim_obj = m.group(2).strip()
                cat = guess_category(subject)
                entity_id = self.db.upsert_entity(subject, cat)

                stmt = StatementRecord(
                    id=str(uuid.uuid4()),
                    entity_id=entity_id,
                    statement_type="CLAIM",
                    subject=subject,
                    predicate=pred,
                    object=claim_obj,
                    author="assistant",
                    confidence=0.85,
                    source_message_id=msg_id,
                    source_conversation_id=conv_id,
                    valid_from=ts,
                    is_current=1,
                    notes="Affermazione tecnica dell'assistente"
                )
                self.db.insert_statement(stmt)
