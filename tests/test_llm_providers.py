import json
from unittest.mock import MagicMock, patch
import pytest
from memoria.llm.base import LLMResponse
from memoria.llm.factory import detect_provider, get_llm_provider
from memoria.llm.providers.anthropic import AnthropicProvider
from memoria.llm.providers.openai import OpenAIProvider
from memoria.llm.providers.gemini import GeminiProvider
from memoria.llm.providers.grok import GrokProvider
from memoria.llm.providers.deepseek import DeepSeekProvider
from memoria.llm.providers.qwen import QwenProvider
from memoria.llm.client import LLMClient


def test_detect_provider():
    # Direct provider name
    assert detect_provider("anthropic") == "anthropic"
    assert detect_provider("openai") == "openai"
    assert detect_provider("claude") == "anthropic"

    # Inferred from model
    assert detect_provider(model="claude-3-5-sonnet-20241022") == "anthropic"
    assert detect_provider(model="gpt-4o") == "openai"
    assert detect_provider(model="o3-mini") == "openai"
    assert detect_provider(model="gemini-2.0-flash") == "gemini"
    assert detect_provider(model="grok-2") == "grok"
    assert detect_provider(model="deepseek-reasoner") == "deepseek"
    assert detect_provider(model="qwen2.5:7b") == "qwen"

    # Inferred from URL
    assert detect_provider(base_url="https://api.anthropic.com/v1") == "anthropic"
    assert detect_provider(base_url="https://api.openai.com/v1") == "openai"
    assert detect_provider(base_url="https://generativelanguage.googleapis.com/v1beta/openai") == "gemini"
    assert detect_provider(base_url="https://api.x.ai/v1") == "grok"
    assert detect_provider(base_url="https://api.deepseek.com") == "deepseek"


def test_anthropic_provider():
    provider = AnthropicProvider(api_key="sk-ant-test1234", model="claude-3-5-sonnet-20241022")
    assert provider.is_configured() is True
    assert provider.provider_name == "anthropic"

    mock_response = {
        "content": [{"type": "text", "text": "Risposta Claude"}],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 80,
            "cache_read_input_tokens": 20,
        },
    }

    with patch.object(provider, "_send_http_request", return_value=mock_response) as mock_send:
        res = provider.generate(
            messages=[{"role": "user", "content": "Ciao"}],
            system_prompt="Sei un assistente",
            temperature=0.3,
            max_tokens=500,
        )

        assert res is not None
        assert res.content == "Risposta Claude"
        assert res.provider == "anthropic"
        assert res.usage["cache_read_input_tokens"] == 20
        assert mock_send.called
        call_url, call_headers, call_payload = mock_send.call_args[0][:3]
        assert "messages" in call_url
        assert call_headers["x-api-key"] == "sk-ant-test1234"
        assert call_headers["anthropic-beta"] == "prompt-caching-2024-07-31"
        assert call_payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_standard_and_reasoning_models():
    # Standard GPT-4o
    p_standard = OpenAIProvider(api_key="sk-proj-test1234", model="gpt-4o")
    assert p_standard.is_reasoning_model() is False

    mock_gpt4o = {
        "choices": [{"message": {"role": "assistant", "content": "Risposta GPT-4o"}}],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 15},
        },
    }

    with patch.object(p_standard, "_send_http_request", return_value=mock_gpt4o) as mock_send:
        res = p_standard.generate([{"role": "user", "content": "Test"}], system_prompt="Sys")
        assert res is not None
        assert res.content == "Risposta GPT-4o"
        assert res.usage["cached_tokens"] == 15
        _, _, payload = mock_send.call_args[0][:3]
        assert payload["messages"][0]["role"] == "system"
        assert "max_tokens" in payload
        assert "temperature" in payload

    # Reasoning o3-mini
    p_reasoning = OpenAIProvider(api_key="sk-proj-test1234", model="o3-mini")
    assert p_reasoning.is_reasoning_model() is True

    with patch.object(p_reasoning, "_send_http_request", return_value=mock_gpt4o) as mock_send:
        p_reasoning.generate([{"role": "user", "content": "Test"}], system_prompt="Sys")
        _, _, payload = mock_send.call_args[0][:3]
        # o3-mini uses developer role and max_completion_tokens, no custom temperature
        assert payload["messages"][0]["role"] == "developer"
        assert "max_completion_tokens" in payload
        assert "temperature" not in payload


def test_gemini_provider():
    provider = GeminiProvider(api_key="AIzaSyTest1234", model="gemini-2.0-flash")
    assert provider.is_configured() is True
    assert provider.provider_name == "gemini"

    mock_gemini = {
        "choices": [{"message": {"role": "assistant", "content": "Risposta Gemini"}}],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "cached_content_token_count": 30,
        },
    }

    with patch.object(provider, "_send_http_request", return_value=mock_gemini):
        res = provider.generate([{"role": "user", "content": "Ciao"}])
        assert res is not None
        assert res.content == "Risposta Gemini"
        assert res.usage["cached_tokens"] == 30


def test_grok_provider():
    provider = GrokProvider(api_key="xai-test1234", model="grok-2-1212")
    assert provider.is_configured() is True
    assert provider.provider_name == "grok"

    mock_grok = {
        "choices": [{"message": {"role": "assistant", "content": "Risposta Grok"}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10},
    }

    with patch.object(provider, "_send_http_request", return_value=mock_grok):
        res = provider.generate([{"role": "user", "content": "Ciao"}])
        assert res is not None
        assert res.content == "Risposta Grok"


def test_deepseek_reasoning_extraction():
    provider = DeepSeekProvider(api_key="sk-ds-test1234", model="deepseek-reasoner")
    assert provider.is_configured() is True
    assert provider.provider_name == "deepseek"

    mock_r1 = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Risposta finale DeepSeek-R1",
                "reasoning_content": "Passaggio logico 1: verifica premesse...",
            }
        }],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "prompt_cache_hit_tokens": 90,
            "prompt_cache_miss_tokens": 30,
        },
    }

    with patch.object(provider, "_send_http_request", return_value=mock_r1):
        res = provider.generate([{"role": "user", "content": "Spiega"}])
        assert res is not None
        assert res.content == "Risposta finale DeepSeek-R1"
        assert res.reasoning == "Passaggio logico 1: verifica premesse..."
        assert res.usage["prompt_cache_hit_tokens"] == 90
        assert res.usage["cached_tokens"] == 90


def test_qwen_think_tag_parsing():
    provider = QwenProvider(base_url="http://localhost:11434/v1", model="qwq:32b")
    assert provider.is_configured() is True
    assert provider.provider_name == "qwen"

    raw_text = "<think>\nConsidero i nastri TDK SA-X e Maxell XLII.\n</think>\nEcco la raccomandazione rock: TDK SA-X."

    mock_qwen = {
        "choices": [{"message": {"role": "assistant", "content": raw_text}}],
        "usage": {"prompt_tokens": 60, "completion_tokens": 35},
    }

    with patch.object(provider, "_send_http_request", return_value=mock_qwen):
        res = provider.generate([{"role": "user", "content": "Quale cassetta?"}])
        assert res is not None
        assert res.reasoning == "Considero i nastri TDK SA-X e Maxell XLII."
        assert res.content == "Ecco la raccomandazione rock: TDK SA-X."
        assert "<think>" not in res.content


def test_llm_client_facade():
    client = LLMClient(provider="deepseek", api_key="sk-test", model="deepseek-chat")
    assert client.is_configured() is True
    assert client.provider_name == "deepseek"

    # Test synthesize_answer with reasoning inclusion
    mock_resp = LLMResponse(
        content="Sintesi per rock: usa TDK SA-X",
        reasoning="Valutato il bias Type II",
        provider="deepseek",
        model="deepseek-reasoner",
    )

    with patch.object(client.provider_instance, "generate", return_value=mock_resp):
        ans = client.synthesize_answer("Quale cassetta rock?", [], [])
        assert ans is not None
        assert "Ragionamento AI (deepseek)" in ans
        assert "Sintesi per rock: usa TDK SA-X" in ans

    # Test extract_knowledge_with_llm JSON parsing
    mock_json_resp = LLMResponse(
        content='```json\n{"entities": [{"name": "TDK SA-X", "category": "audiocassette"}]}\n```',
        provider="deepseek",
        model="deepseek-chat",
    )
    with patch.object(client.provider_instance, "generate", return_value=mock_json_resp):
        res_json = client.extract_knowledge_with_llm("Ho registrato con una TDK SA-X")
        assert res_json is not None
        assert res_json["entities"][0]["name"] == "TDK SA-X"


def test_provider_resilience_and_retry():
    provider = AnthropicProvider(api_key="sk-test", base_url="https://api.anthropic.com/v1")
    
    # Test network failure / 500 returns None gracefully without exception
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        res = provider.generate([{"role": "user", "content": "Test"}])
        assert res is None


def test_unconfigured_provider():
    # If no key and external endpoint
    provider = AnthropicProvider(api_key="", base_url="https://api.anthropic.com/v1")
    assert provider.is_configured() is False
    res = provider.generate([{"role": "user", "content": "Test"}])
    assert res is None

