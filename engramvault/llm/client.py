import json
from typing import Any, Dict, List, Optional
from engramvault.llm.base import BaseLLMProvider, LLMResponse
from engramvault.llm.factory import get_llm_provider


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
        lang: str = "en",
    ) -> Optional[str]:
        """
        Synthesizes an answer using the configured AI provider given retrieved facts and chunks.
        """
        from engramvault.i18n import get_prompt

        system_prompt = get_prompt("synthesis_system", lang) or (
            "You are the intelligent personal assistant of 'EngramVault'.\n"
            "Cross-reference the user's verified inventory (FACT) and decisions with past conversation excerpts (CLAIM).\n"
            "Always cite conversation sources using markdown links [Title](#conv-ID) and reply in the language of the user's question."
        )

        facts_text = "\n".join([
            f"- [{f.get('statement_type', 'FACT')}] {f.get('subject')} -> {f.get('predicate')}: {f.get('object')} (Source: [{f.get('conversation_title', 'Inventory')}](#conv-{f.get('source_conversation_id', '')}))"
            for f in facts
        ]) if facts else "No specific verified facts found."

        chunks_text = "\n\n".join([
            f"--- SOURCE: [{c.get('conv_title', 'Conversation')}](#conv-{c.get('conversation_id', '')}) | {c.get('provider')} ({c.get('created_at', '')[:10]}) ---\n{c.get('chunk_text', '')}"
            for c in chunks
        ]) if chunks else "No relevant excerpts found."

        user_content = (
            f"USER QUESTION:\n{query}\n\n"
            f"VERIFIED USER INVENTORY & FACTS:\n{facts_text}\n\n"
            f"RELEVANT EXCERPTS FROM PAST CONVERSATIONS:\n{chunks_text}\n\n"
            "Provide an accurate, well-reasoned synthesis with clickable citations."
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
        return parse_llm_json(res)


def parse_llm_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parses JSON returned by an LLM, stripping optional Markdown code fences (Q-02)."""
    if not raw:
        return None
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        return json.loads(clean.strip())
    except Exception as exc:
        logger.debug("Failed to parse LLM JSON response: %s", exc)
        return None
