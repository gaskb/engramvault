import os
import re
from typing import Any, Dict, List, Optional
from memoria.llm.base import BaseLLMProvider, LLMResponse


class QwenProvider(BaseLLMProvider):
    """
    Adapter for Qwen (Ollama local / DashScope Cloud).
    Sanitizes and extracts reasoning from <think>...</think> tags.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("MEMORIA_LLM_API_KEY") or "dummy",
            base_url=(
                base_url
                or os.getenv("QWEN_BASE_URL")
                or os.getenv("DASHSCOPE_BASE_URL")
                or os.getenv("MEMORIA_LLM_BASE_URL")
                or "http://localhost:11434/v1"
            ),
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "qwen2.5:3b",
        )

    @property
    def provider_name(self) -> str:
        return "qwen"

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
        raw_content = choice.get("message", {}).get("content", "") or ""

        # Extract <think>...</think> tags if present
        think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        else:
            reasoning = None
            content = raw_content.strip()

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "cached_tokens": usage_raw.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        }

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            raw=data,
        )
