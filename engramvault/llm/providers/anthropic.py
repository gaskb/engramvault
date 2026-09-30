import os
from typing import Any, Dict, List, Optional
from engramvault.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """
    Adapter for Anthropic Claude (Messages API).
    Supports prompt caching with ephemeral checkpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("MEMORIA_LLM_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1",
            model=model or os.getenv("MEMORIA_LLM_MODEL") or "claude-sonnet-4-5-20250929",
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

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

        # Extract system prompt if present inside messages
        chat_messages: List[Dict[str, str]] = []
        extracted_sys: List[str] = []
        if system_prompt:
            extracted_sys.append(system_prompt)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                extracted_sys.append(content)
            else:
                chat_messages.append({"role": role, "content": content})

        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        # Fable models deprecate temperature parameter
        if "fable" not in self.model.lower():
            payload["temperature"] = min(max(temperature, 0.0), 1.0)

        # Format system prompt with cache_control for prompt caching
        if extracted_sys:
            combined_sys = "\n\n".join(extracted_sys)
            payload["system"] = [
                {
                    "type": "text",
                    "text": combined_sys,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        data = self._send_http_request(url, headers, payload, timeout=timeout)
        if not data:
            return None

        # Parse response content
        content_blocks = data.get("content", [])
        text_parts = [
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        ]
        text = "".join(text_parts).strip()

        usage_raw = data.get("usage", {})
        usage = {
            "input_tokens": usage_raw.get("input_tokens", 0),
            "output_tokens": usage_raw.get("output_tokens", 0),
            "cache_creation_input_tokens": usage_raw.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage_raw.get("cache_read_input_tokens", 0),
        }

        return LLMResponse(
            content=text,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            raw=data,
        )
