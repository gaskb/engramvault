import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    reasoning: Optional[str] = None
    provider: str = ""
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Enforces unified interfaces, input guardrails, and resilient HTTP retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or ""
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the provider (e.g. anthropic, openai)."""
        pass

    def is_configured(self) -> bool:
        """Returns True if required credentials or endpoint are provided."""
        if not self.base_url and not self.api_key:
            return False
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return True
        return bool(self.api_key and self.api_key != "dummy")

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> Optional[LLMResponse]:
        """Executes a completion request against the provider."""
        pass

    def _send_http_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: int = 30,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Sends HTTP POST request with defensive retry on transient 429/5xx errors.
        """
        data_bytes = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **headers}

        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=data_bytes,
                    headers=req_headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if 200 <= response.status < 300:
                        raw_body = response.read().decode("utf-8")
                        return json.loads(raw_body)
            except urllib.error.HTTPError as e:
                # Retry on rate limits (429) or server errors (500, 502, 503, 504)
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    sleep_time = (2 ** attempt) * 0.5
                    time.sleep(sleep_time)
                    continue
                return None
            except Exception:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                return None

        return None
