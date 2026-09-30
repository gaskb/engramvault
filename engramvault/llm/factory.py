import os
from typing import Dict, Optional, Type
from engramvault.llm.base import BaseLLMProvider
from engramvault.llm.providers.anthropic import AnthropicProvider
from engramvault.llm.providers.openai import OpenAIProvider
from engramvault.llm.providers.gemini import GeminiProvider
from engramvault.llm.providers.grok import GrokProvider
from engramvault.llm.providers.deepseek import DeepSeekProvider
from engramvault.llm.providers.qwen import QwenProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "grok": GrokProvider,
    "xai": GrokProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "ollama": QwenProvider,
}

CANONICAL_PROVIDERS: Dict[str, str] = {
    "claude": "anthropic",
    "google": "gemini",
    "xai": "grok",
    "ollama": "qwen",
}


def detect_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Infers the most appropriate canonical provider based on configuration."""
    if provider_name and provider_name.lower() in PROVIDER_REGISTRY:
        p = provider_name.lower()
        return CANONICAL_PROVIDERS.get(p, p)

    env_provider = os.getenv("MEMORIA_LLM_PROVIDER", "").lower()
    if env_provider in PROVIDER_REGISTRY:
        return CANONICAL_PROVIDERS.get(env_provider, env_provider)

    m = (model or os.getenv("MEMORIA_LLM_MODEL", "")).lower()
    if "claude" in m:
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "chatgpt")):
        return "openai"
    if "gemini" in m:
        return "gemini"
    if "grok" in m:
        return "grok"
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m:
        return "qwen"

    u = (base_url or os.getenv("MEMORIA_LLM_BASE_URL", "")).lower()
    if "anthropic.com" in u:
        return "anthropic"
    if "openai.com" in u:
        return "openai"
    if "googleapis.com" in u:
        return "gemini"
    if "x.ai" in u:
        return "grok"
    if "deepseek.com" in u:
        return "deepseek"
    if "dashscope.aliyuncs.com" in u:
        return "qwen"

    # Check available API keys in environment
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
        return "grok"
    if os.getenv("DASHSCOPE_API_KEY"):
        return "qwen"

    # Default fallback: local Ollama / Qwen compatible
    return "qwen"


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMProvider:
    """Factory function to instantiate the selected LLM provider."""
    selected_name = detect_provider(provider_name, model=model, base_url=base_url)
    cls = PROVIDER_REGISTRY.get(selected_name, QwenProvider)
    return cls(api_key=api_key, base_url=base_url, model=model)
