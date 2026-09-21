import os
from typing import Any, Dict, List, Optional
from memoria.llm.base import BaseLLMProvider, LLMResponse


class DeepSeekProvider(BaseLLMProvider):
    """
    Adapter for DeepSeek (DeepSeek-V3, DeepSeek-R1).
    Extracts native reasoning_content for R1 and tracks disk-cached tokens.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("MEMORIA_LLM_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "deepseek-chat",
        )

    @property
    def provider_name(self) -> str:
        return "deepseek"

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
        msg = choice.get("message", {})
        content = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning_content")

        usage_raw = data.get("usage", {})
        cache_hits = usage_raw.get("prompt_cache_hit_tokens", 0)
        cache_miss = usage_raw.get("prompt_cache_miss_tokens", 0)

        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "cached_tokens": cache_hits,
            "prompt_cache_hit_tokens": cache_hits,
            "prompt_cache_miss_tokens": cache_miss,
        }

        return LLMResponse(
            content=content,
            reasoning=reasoning.strip() if reasoning else None,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            raw=data,
        )
