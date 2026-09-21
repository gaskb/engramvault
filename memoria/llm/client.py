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
            "Sei un assistente personale di 'Memoria AI'. Il tuo compito è rispondere alla domanda "
            "dell'utente basandoti ESCLUSIVAMENTE sui fatti registrati dall'utente (FACT) e sui contenuti "
            "delle sue precedenti conversazioni con varie AI (CLAIM / CHUNKS).\n"
            "Regole importanti:\n"
            "1. Distingui sempre chiaramente ciò che l'utente ha dichiarato (fatti/inventario/decisioni) "
            "da ciò che l'AI ha suggerito o consigliato (opinioni/claim tecnici).\n"
            "2. Cita sempre le fonti (es. [Fonte: Claude - Titolo Chat, 2024-05-20]).\n"
            "3. Se l'informazione non è presente nei dati forniti, dillo chiaramente senza inventare."
        )

        facts_text = "\n".join([
            f"- [{f.get('statement_type', 'FACT')}] {f.get('subject')} -> {f.get('predicate')}: {f.get('object')} (Fonte: {f.get('conversation_title', 'N/D')})"
            for f in facts
        ]) if facts else "Nessun fatto specifico trovato."

        chunks_text = "\n\n".join([
            f"--- FONTE: {c.get('provider')} | {c.get('conv_title')} ({c.get('created_at', '')[:10]}) ---\n{c.get('chunk_text', '')}"
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
