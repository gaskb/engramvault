from typing import List
from engramvault.adapters.base import NormalizedConversation, NormalizedMessage
from engramvault.db.engine import Database
from engramvault.search.embeddings import EmbeddingEngine
from engramvault.knowledge.extractor import KnowledgeExtractor
from engramvault.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_sample_conversations() -> List[NormalizedConversation]:
    return [
        # 1. ChatGPT: Inventario Audiocassette
        NormalizedConversation(
            assistant="chatgpt",
            title="Inventario studio e audiocassette vintage",
            date="2024-02-14T15:30:00Z",
            external_ref="chatgpt-conv-001",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Sto riordinando il mio studio analogico. Ho queste audiocassette: TDK SA-X, Maxell XLII e Sony HF. Vorrei iniziare a registrarci qualcosa con il registratore a cassette Nakamichi CR-3 che ho fatto revisionare.",
                    ts="2024-02-14T15:30:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="Ottimo setup! Il Nakamichi CR-3 è una macchina a tre testine straordinaria per la calibrazione fine del bias. Tra le cassette che hai citato, hai sia nastri al ferro Type I (come la classica Sony HF) sia nastri ad alto bias al cromo Type II (come la TDK SA-X e la Maxell XLII).",
                    ts="2024-02-14T15:31:20Z"
                )
            ]
        ),

        # 2. Claude: Confronto nastri e resa sul Rock
        NormalizedConversation(
            assistant="claude",
            title="Migliori nastri magnetici per registrazioni Rock",
            date="2024-05-20T10:15:00Z",
            external_ref="claude-conv-002",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Quali sono le caratteristiche dei diversi tipi di audiocassette quando si registra un brano rock con chitarre distorte e batteria potente?",
                    ts="2024-05-20T10:15:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="Per il genere Rock, la dinamica e la saturazione sulle alte e medie frequenze sono fondamentali.\n\n"
                         "1. **TDK SA-X (Type II - High Bias):** È considerata una delle migliori cassette per registrare rock. Il nastro Avilyn a doppio strato permette di spingere i picchi a +3dB / +4dB con una saturazione molto calda sulle chitarre senza sbavare sui transienti dei piatti.\n\n"
                         "2. **Maxell XLII (Type II):** Molto pulita, basso rumore di fondo, ottima linearità, validissima alternativa per un sound rock compatto.\n\n"
                         "3. **Sony HF (Type I - Normal Bias):** Ottima per musica acustica o registrazioni vocali, ma nel rock tende a saturare prematuramente e generare distorsione fangosa sui riff veloci e compressi.",
                    ts="2024-05-20T10:17:45Z"
                )
            ]
        ),

        # 3. ChatGPT: Setup Interfaccia Audio (2024)
        NormalizedConversation(
            assistant="chatgpt",
            title="Valutazione scheda audio per home studio",
            date="2024-08-10T18:00:00Z",
            external_ref="chatgpt-conv-003",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Nel mio setup ho la Audient iD22 collegata via ADAT, ma sto valutando una scheda più compatta per registrazioni portatili. Come si comporta la Zen Go?",
                    ts="2024-08-10T18:00:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="L'Audient iD22 ha preamplificatori discreti eccellenti in classe A, ma è ingombrante e richiede alimentatore esterno. L'Antelope Zen Go Synergy Core è alimentata via bus USB-C, ha convertitori ad altissima gamma dinamica (127 dB) e DSP integrato con emulazioni vintage.",
                    ts="2024-08-10T18:01:30Z"
                )
            ]
        ),

        # 4. Claude Code: Sessione CLI e Decisione Backend
        NormalizedConversation(
            assistant="claude_code",
            title="[dev] Refactoring API audio e scelta architettura",
            date="2024-11-05T09:40:00Z",
            external_ref="claude-code-session-004",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Dobbiamo decidere l'architettura per il microservizio di elaborazione tracce. Per il progetto audio ho deciso di usare FastAPI per il backend invece di Flask.",
                    ts="2024-11-05T09:40:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="Ottima decisione. FastAPI offre supporto nativo asincrono con `asyncio`, validazione automatica Pydantic e documentazione OpenAPI automatica, ideale per streaming audio a bassa latenza.",
                    ts="2024-11-05T09:41:10Z"
                )
            ]
        ),

        # 5. Claude: Cambio Setup e Vendita iD22 (Timeline 2026)
        NormalizedConversation(
            assistant="claude",
            title="Aggiornamento hardware studio e mastering",
            date="2026-03-02T16:20:00Z",
            external_ref="claude-conv-005",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Ho venduto la Audient iD22 e alla fine ho deciso di usare la Zen Go come interfaccia principale per tutte le produzioni e registrazioni da banco.",
                    ts="2026-03-02T16:20:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="Scelta molto pratica. Concentrarsi su un'unica interfaccia snella come la Zen Go semplifica il routing e ti garantisce portabilità senza sacrificare la qualità di conversione.",
                    ts="2026-03-02T16:21:40Z"
                )
            ]
        ),

        # 6. Gemini: Ricetta Pizza Napoletana
        NormalizedConversation(
            assistant="gemini",
            title="Idratazione impasto pizza contemporanea",
            date="2026-06-18T11:00:00Z",
            external_ref="gemini-conv-006",
            messages=[
                NormalizedMessage(
                    role="user",
                    text="Voglio fare una pizza napoletana con 70% di idratazione usando farina W300. Quante ore di lievitazione consigli a temperatura ambiente?",
                    ts="2026-06-18T11:00:00Z"
                ),
                NormalizedMessage(
                    role="assistant",
                    text="Con una farina di forza W300 e idratazione al 70%, ti consiglio una maturazione totale di 24 ore: 18 ore di puntata in frigo a 4°C per rilassare la maglia glutinica, seguite da 6 ore di appretto a temperatura ambiente (20-22°C).",
                    ts="2026-06-18T11:01:15Z"
                )
            ]
        )
    ]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits long text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def seed_database(db: Database) -> int:
    """Populates database with sample conversations, embeddings, and knowledge items."""
    embedder = EmbeddingEngine()
    extractor = KnowledgeExtractor(db)
    conversations = get_sample_conversations()
    count = 0

    for conv in conversations:
        if db.conversation_exists_by_hash(conv.compute_hash()):
            continue

        chunk_payloads = []
        for idx, msg in enumerate(conv.messages):
            c_list = chunk_text(msg.text)
            for c_str in c_list:
                blob = embedder.embed_to_blob(c_str)
                chunk_payloads.append((idx, c_str, blob))

        conv_id = db.insert_conversation_with_chunks(conv, chunk_payloads)

        # Knowledge Extraction
        conv_record = db.get_conversation(conv_id)
        if conv_record:
            extractor.extract_from_conversation(conv, conv_id, conv_record["messages"])

        count += 1

    return count
