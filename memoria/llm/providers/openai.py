import os
from typing import Any, Dict, List, Optional
from memoria.llm.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """
    Adapter for OpenAI (GPT-4o, o1, o3-mini).
    Supports automatic prompt caching and reasoning models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MEMORIA_LLM_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "gpt-4o",
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_reasoning_model(self) -> bool:
        """Determines if the model belongs to the o1 or o3 reasoning series."""
        m = self.model.lower()
        return m.startswith("o1") or m.startswith("o3")

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

        is_reasoning = self.is_reasoning_model()
        system_role = "developer" if is_reasoning else "system"

        formatted_messages: List[Dict[str, str]] = []
        if system_prompt:
            formatted_messages.append({"role": system_role, "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and is_reasoning:
                role = "developer"
            formatted_messages.append({"role": role, "content": content})

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }

        if is_reasoning:
            # Reasoning models require max_completion_tokens and disallow custom temperature
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature

        data = self._send_http_request(url, headers, payload, timeout=timeout)
        if not data or "choices" not in data or not data["choices"]:
            return None

        choice = data["choices"][0]
        text = choice.get("message", {}).get("content", "") or ""

        usage_raw = data.get("usage", {})
        prompt_details = usage_raw.get("prompt_tokens_details", {})
        cached_tokens = prompt_details.get("cached_tokens", 0)

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
