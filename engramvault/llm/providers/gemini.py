import os
from typing import Any, Dict, List, Optional
from engramvault.llm.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """
    Adapter for Google Gemini via Google AI Studio native generateContent API.
    Provides direct, high-speed access to Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("MEMORIA_LLM_API_KEY"),
            base_url=(
                base_url
                or os.getenv("GEMINI_BASE_URL")
                or "https://generativelanguage.googleapis.com"
            ),
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "gemini-3-flash-preview",
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> Optional[LLMResponse]:
        if not self.is_configured():
            return None

        clean_model = self.model.replace("models/", "").strip()
        url = f"{self.base_url}/v1beta/models/{clean_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        # Format chat messages into Gemini contents structure
        contents: List[Dict[str, Any]] = []
        sys_instructions = [system_prompt] if system_prompt else []

        for msg in messages:
            role = msg.get("role", "user")
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            if role == "system":
                sys_instructions.append(text)
            else:
                gemini_role = "model" if role in ("assistant", "model") else "user"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": text}],
                })

        # Ensure at least one user content exists
        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": max(0.0, min(temperature, 2.0)),
                "maxOutputTokens": max_tokens,
            },
        }

        if sys_instructions:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(sys_instructions)}],
            }

        data = self._send_http_request(url, headers, payload, timeout=timeout)
        if not data:
            return None

        candidates = data.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = []
        thought_parts = []
        for p in parts:
            if p.get("thought") is True:
                thought_parts.append(p.get("text", ""))
            else:
                text_parts.append(p.get("text", ""))
        text = "".join(text_parts).strip()
        reasoning = "\n".join(thought_parts).strip() if thought_parts else None
        if not text and thought_parts:
            text = reasoning or ""

        usage_raw = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_raw.get("promptTokenCount", 0),
            "completion_tokens": usage_raw.get("candidatesTokenCount", 0),
            "cached_tokens": usage_raw.get("cachedContentTokenCount", 0),
            "thought_tokens": usage_raw.get("thoughtsTokenCount", 0),
        }

        return LLMResponse(
            content=text,
            reasoning=reasoning,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            raw=data,
        )
