from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ConversationRecord:
    id: str
    provider: str
    external_ref: str
    title: str
    created_at: str
    content_hash: str
    metadata_json: str = "{}"


@dataclass
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    ts: str
    sequence_order: int
    metadata_json: str = "{}"


@dataclass
class ChunkRecord:
    id: str
    message_id: str
    conversation_id: str
    chunk_index: int
    chunk_text: str
    embedding: Optional[bytes] = None


@dataclass
class EntityRecord:
    id: str
    name: str
    category: str
    normalized_name: str
    created_at: str


@dataclass
class StatementRecord:
    id: str
    entity_id: Optional[str]
    statement_type: str  # 'FACT' | 'CLAIM'
    subject: str
    predicate: str
    object: str
    author: str  # 'user' | 'assistant'
    confidence: float = 1.0
    source_message_id: Optional[str] = None
    source_conversation_id: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    is_current: int = 1
    supersedes_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class DecisionRecord:
    id: str
    topic: str
    decision_text: str
    status: str = "ADOPTED"
    decided_at: str = ""
    source_message_id: Optional[str] = None
    source_conversation_id: Optional[str] = None


@dataclass
class RelationRecord:
    id: str
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    source_message_id: Optional[str] = None
