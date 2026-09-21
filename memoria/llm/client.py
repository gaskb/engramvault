import json
from typing import Any, Dict, List, Optional
from memoria.llm.base import BaseLLMProvider, LLMResponse
from memoria.llm.factory import get_llm_provider


class LLMClient:
    """
    Unified high-level LLM client supporting Anthropic, OpenAI, Gemini, Grok, DeepSeek, and Qwen.
    Maintains 100% backward compatibility while delegating to specialized provider adapters.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider_instance: BaseLLMProvider = get_llm_provider(
            provider_name=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.provider_name = self.provider_instance.provider_name
        self.base_url = self.provider_instance.base_url
        self.api_key = self.provider_instance.api_key
        self.model = self.provider_instance.model

    def is_configured(self) -> bool:
        """Returns True if a real endpoint or valid API key is configured."""
        return self.provider_instance.is_configured()

    def generate_full(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> Optional[LLMResponse]:
        """Returns the full LLMResponse including reasoning and usage statistics."""
        if not self.is_configured():
            return None
        return self.provider_instance.generate(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> Optional[str]:
        """
        Sends chat completion request to the configured provider.
        Returns generated text content or None if request failed.
        """
        resp = self.generate_full(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return resp.content if resp else None

    def synthesize_answer(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Synthesizes an answer using the configured AI provider given retrieved facts and chunks.
        """
        system_prompt = (
            "Sei l'assistente personale intelligente di 'Memoria AI'.\n"
            "Il tuo compito è rispondere in modo pratico, diretto ed estremamente utile alla domanda dell'utente.\n"
            "Incrocia i dati personali dell'utente (oggetti che possiede nel suo inventario FACT, decisioni passate) "
            "con i frammenti delle sue conversazioni precedenti (CLAIM / discussioni) e le tue competenze tecniche.\n\n"
            "Linee guida di risposta:\n"
            "1. Rispondi subito alla domanda con una raccomandazione chiara, concreta e personalizzata su misura per ciò che possiede.\n"
            "   Fornisci un elenco strutturato e ragionato, ad esempio:\n"
            "   'Tra le opzioni che possiedi nel tuo inventario, le scelte migliori per [scopo] sono:\n"
            "   1) [Modello/Elemento] - Ideale se cerchi [caratteristica/risultato]...\n"
            "   2) [Modello/Elemento] - Consigliato se vuoi un risultato più [caratteristica]...\n"
            "   3) [Modello/Elemento] - Consigliato solo se...'\n"
            "2. Se l'utente chiede cosa scegliere tra ciò che ha, considera SOLO gli elementi confermati nel suo inventario (FACT), "
            "menzionando eventuali altri solo come confronto o nota a margine.\n"
            "3. Motiva ogni scelta con dettagli tecnici specifici e collega i consigli alle discussioni passate citando le fonti se rilevanti.\n"
            "4. Cita le fonti delle conversazioni passate riutilizzando i link markdown [Titolo Conversazione](#conv-ID) presenti negli estratti.\n"
            "5. Mantieni uno stile diretto, da esperto di settore, senza preamboli burocratici."
        )

        facts_text = "\n".join([
            f"- [{f.get('statement_type', 'FACT')}] {f.get('subject')} -> {f.get('predicate')}: {f.get('object')} (Fonte: [{f.get('conversation_title', 'Inventario')}](#conv-{f.get('source_conversation_id', '')}))"
            for f in facts
        ]) if facts else "Nessun fatto specifico trovato."

        chunks_text = "\n\n".join([
            f"--- FONTE: [{c.get('conv_title', 'Conversazione')}](#conv-{c.get('conversation_id', '')}) | {c.get('provider')} ({c.get('created_at', '')[:10]}) ---\n{c.get('chunk_text', '')}"
            for c in chunks
        ]) if chunks else "Nessun frammento rilevante trovato."

        user_content = (
            f"DOMANDA DELL'UTENTE:\n{query}\n\n"
            f"FATTI ACCERTATI NELL'INVENTARIO/MEMORIA DELL'UTENTE:\n{facts_text}\n\n"
            f"ESTRATTI PERTINENTI DALLE CONVERSAZIONI PASSATE:\n{chunks_text}\n\n"
            "Fornisci una sintesi accurata, motivata e con citazioni."
        )

        messages = [
            {"role": "user", "content": user_content}
        ]

        resp = self.generate_full(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        if not resp:
            return None

        # If reasoning was extracted (DeepSeek R1, Qwen QwQ), we can append or expose it
        if resp.reasoning:
            return f"> **Ragionamento AI ({resp.provider}):**\n> {resp.reasoning}\n\n{resp.content}"

        return resp.content

    def extract_knowledge_with_llm(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """
        Uses configured AI provider to extract entities, facts, and decisions in structured JSON.
        """
        system_prompt = (
            "Estrai entità, fatti personali dichiarati dall'utente e decisioni da questa conversazione.\n"
            "Rispondi SOLO in formato JSON valido con questa struttura:\n"
            "{\n"
            "  \"entities\": [{\"name\": \"...\", \"category\": \"audiocassette|audio_interface|software|...\"}],\n"
            "  \"facts\": [{\"subject\": \"...\", \"predicate\": \"owned_by_user|preference|...\", \"object\": \"...\"}],\n"
            "  \"decisions\": [{\"topic\": \"...\", \"decision\": \"...\", \"status\": \"ADOPTED\"}]\n"
            "}"
        )
        messages = [
            {"role": "user", "content": conversation_text[:4000]}
        ]
        res = self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=1000,
        )
        if res:
            try:
                clean = res.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                return json.loads(clean.strip())
            except Exception:
                return None
        return None
