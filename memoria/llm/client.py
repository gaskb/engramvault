import os
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any


class LLMClient:
    """
    Universal lightweight client for local (Ollama) or external (Gemini, OpenAI, Claude, Groq, OpenRouter)
    LLMs using standard OpenAI-compatible REST API endpoints without external SDK dependencies.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        # Configure endpoint with priority: explicit arg -> env var -> defaults
        self.base_url = (
            base_url
            or os.getenv("MEMORIA_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:11434/v1"
        ).rstrip("/")

        self.api_key = (
            api_key
            or os.getenv("MEMORIA_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or "dummy"
        )

        self.model = (
            model
            or os.getenv("MEMORIA_LLM_MODEL")
            or "qwen2.5:3b"
        )

    def is_configured(self) -> bool:
        """Returns True if a real endpoint or key appears configured."""
        if not self.base_url:
            return False
        # If localhost, it might be running Ollama
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return True
        # If external, check if API key is provided and not default dummy
        return bool(self.api_key and self.api_key != "dummy")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30
    ) -> Optional[str]:
        """
        Sends chat completion request to configured OpenAI-compatible endpoint.
        Returns generated text or None if failed.
        """
        if not self.is_configured():
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
        except Exception:
            return None
        return None

    def synthesize_answer(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Synthesizes an answer using external/local AI given retrieved facts and chunks."""
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
            f"- [{f.get('statement_type')}] {f.get('subject')} -> {f.get('predicate')}: {f.get('object')} (Fonte: {f.get('conversation_title', 'N/D')})"
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        return self.chat_completion(messages, temperature=0.3, max_tokens=1200)

    def extract_knowledge_with_llm(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """
        Uses an external AI to extract entities, facts, and decisions in structured JSON.
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text[:4000]}
        ]
        res = self.chat_completion(messages, temperature=0.1, max_tokens=1000)
        if res:
            try:
                # remove codeblocks if present
                clean = res.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                return json.loads(clean.strip())
            except Exception:
                return None
        return None
