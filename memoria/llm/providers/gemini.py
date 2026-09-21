import os
from typing import Any, Dict, List, Optional
from memoria.llm.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """
    Adapter for Google Gemini (Gemini 2.0 Flash / Pro) via OpenAI-compatible endpoint
    or Google AI Studio.
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
                or "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "gemini-2.0-flash",
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

        formatted_messages: List[Dict[str, str]] = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = self._send_http_request(url, headers, payload, timeout=timeout)
        if not data or "choices" not in data or not data["choices"]:
            return None

        choice = data["choices"][0]
        text = choice.get("message", {}).get("content", "") or ""

        usage_raw = data.get("usage", {})
        cached_tokens = (
            usage_raw.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or usage_raw.get("cached_content_token_count", 0)
        )

        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "cached_tokens": cached_tokens,
        }

        return LLMResponse(
            content=text.strip(),
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            raw=data,
        )
