import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from memoria.db.engine import Database

# Provider metadata and known out-of-the-box models
PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "name": "Qwen / Local (Ollama, vLLM)",
        "default_url": "http://localhost:11434/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "env_url": "MEMORIA_LLM_BASE_URL",
        "is_local": True,
        "default_models": [
            "qwen2.5:3b",
            "qwen2.5:7b",
            "qwen2.5-coder:7b",
            "llama3.2:3b",
            "mistral",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "default_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "env_url": "OPENAI_BASE_URL",
        "is_local": False,
        "default_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o3-mini",
        ],
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "default_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "env_url": "ANTHROPIC_BASE_URL",
        "is_local": False,
        "default_models": [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-opus-4-5-20251101",
        ],
    },
    "gemini": {
        "name": "Google Gemini",
        "default_url": "https://generativelanguage.googleapis.com",
        "env_key": "GEMINI_API_KEY",
        "env_url": "GEMINI_BASE_URL",
        "is_local": False,
        "default_models": [
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-3.5-flash",
        ],
    },
    "grok": {
        "name": "xAI Grok",
        "default_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "env_url": "XAI_BASE_URL",
        "is_local": False,
        "default_models": [
            "grok-2-latest",
            "grok-beta",
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "default_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "env_url": "DEEPSEEK_BASE_URL",
        "is_local": False,
        "default_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
}


def mask_key(key: str) -> str:
    """Masks secret API key for safe UI display (e.g. sk-••••••••1234)."""
    if not key or key == "dummy":
        return ""
    if len(key) <= 8:
        return "••••••••"
    prefix = key[:3] if len(key) > 12 else ""
    suffix = key[-4:]
    return f"{prefix}••••••••{suffix}"


class SettingsManager:
    """
    Manages persistent system configuration: API keys, base URLs,
    active provider/model, custom model catalogs, and live model fetching.
    """

    def __init__(self, db: Database):
        self.db = db

    def get_api_key(self, provider: str) -> str:
        prov = provider.lower()
        key = self.db.get_setting(f"api_key:{prov}")
        if key:
            return key
        env_var = PROVIDER_DEFAULTS.get(prov, {}).get("env_key", "")
        if env_var and os.getenv(env_var):
            return os.getenv(env_var, "")
        if prov == "grok" and os.getenv("GROK_API_KEY"):
            return os.getenv("GROK_API_KEY", "")
        return ""

    def get_base_url(self, provider: str) -> str:
        prov = provider.lower()
        url = self.db.get_setting(f"base_url:{prov}")
        if url:
            return url
        env_var = PROVIDER_DEFAULTS.get(prov, {}).get("env_url", "")
        if env_var and os.getenv(env_var):
            return os.getenv(env_var, "")
        return PROVIDER_DEFAULTS.get(prov, {}).get("default_url", "")

    def is_provider_configured(self, provider: str) -> bool:
        prov = provider.lower()
        meta = PROVIDER_DEFAULTS.get(prov)
        if not meta:
            return False
        if meta.get("is_local"):
            return True
        key = self.get_api_key(prov)
        return bool(key and key != "dummy")

    def get_active_provider(self) -> str:
        stored = self.db.get_setting("active_provider")
        if stored and stored.lower() in PROVIDER_DEFAULTS:
            return stored.lower()
        env_p = os.getenv("MEMORIA_LLM_PROVIDER", "").lower()
        if env_p and env_p in PROVIDER_DEFAULTS:
            return env_p
        for p in ("qwen", "openai", "gemini", "anthropic", "deepseek", "grok"):
            if self.is_provider_configured(p):
                return p
        return "qwen"

    def get_active_model(self, provider: Optional[str] = None) -> str:
        prov = (provider or self.get_active_provider()).lower()
        models = self.get_models_for_provider(prov)
        stored = self.db.get_setting(f"active_model:{prov}")
        if stored and (stored in models or not models):
            return stored
        if prov == self.get_active_provider():
            global_model = self.db.get_setting("active_model")
            if global_model and (global_model in models or not models):
                return global_model
        env_m = os.getenv("MEMORIA_LLM_MODEL")
        if env_m and not provider and (env_m in models or not models):
            return env_m
        return models[0] if models else "qwen2.5:3b"

    def get_models_for_provider(self, provider: str) -> List[str]:
        prov = provider.lower()
        stored = self.db.get_setting(f"models:{prov}")
        if stored:
            try:
                models = json.loads(stored)
                if isinstance(models, list) and models:
                    return models
            except Exception:
                pass
        return list(PROVIDER_DEFAULTS.get(prov, {}).get("default_models", ["default"]))

    def set_models_for_provider(self, provider: str, models: List[str]):
        prov = provider.lower()
        cleaned = [m.strip() for m in models if m and m.strip()]
        # Preserve order while removing duplicates
        seen = set()
        deduped = [m for m in cleaned if not (m in seen or seen.add(m))]
        self.db.set_setting(f"models:{prov}", json.dumps(deduped, ensure_ascii=False))

    def add_model_to_provider(self, provider: str, model_name: str) -> List[str]:
        m = model_name.strip()
        if not m:
            return self.get_models_for_provider(provider)
        models = self.get_models_for_provider(provider)
        if m not in models:
            models.append(m)
            self.set_models_for_provider(provider, models)
        return models

    def get_full_settings(self) -> Dict[str, Any]:
        """Returns the full settings schema for the UI."""
        active_prov = self.get_active_provider()
        active_model = self.get_active_model(active_prov)

        providers_dict = {}
        for p, meta in PROVIDER_DEFAULTS.items():
            key = self.get_api_key(p)
            base_url = self.get_base_url(p)
            is_cfg = self.is_provider_configured(p)
            models = self.get_models_for_provider(p)
            prov_active_model = self.get_active_model(p)

            providers_dict[p] = {
                "name": meta["name"],
                "is_local": meta["is_local"],
                "base_url": base_url,
                "api_key_masked": mask_key(key),
                "has_key": bool(key),
                "is_configured": is_cfg,
                "models": models,
                "active_model": prov_active_model,
            }

        return {
            "active_provider": active_prov,
            "active_model": active_model,
            "providers": providers_dict,
        }

    def update_settings(
        self,
        active_provider: Optional[str] = None,
        active_model: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        base_urls: Optional[Dict[str, str]] = None,
        provider_active_models: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if active_provider and active_provider.lower() in PROVIDER_DEFAULTS:
            self.db.set_setting("active_provider", active_provider.lower())

        if active_model:
            self.db.set_setting("active_model", active_model.strip())
            target_prov = active_provider.lower() if active_provider else self.get_active_provider()
            self.db.set_setting(f"active_model:{target_prov}", active_model.strip())

        if provider_active_models:
            for p, m in provider_active_models.items():
                if p.lower() in PROVIDER_DEFAULTS and m:
                    self.db.set_setting(f"active_model:{p.lower()}", m.strip())

        if base_urls:
            for p, u in base_urls.items():
                if p.lower() in PROVIDER_DEFAULTS and u is not None:
                    self.db.set_setting(f"base_url:{p.lower()}", u.strip())

        if api_keys:
            for p, k in api_keys.items():
                prov = p.lower()
                if prov not in PROVIDER_DEFAULTS:
                    continue
                k_val = k.strip() if k else ""
                # Skip if value is masked or empty
                if not k_val or "••••" in k_val:
                    continue
                self.db.set_setting(f"api_key:{prov}", k_val)
                # Auto-sync live models for newly configured provider
                try:
                    sync_res = self.fetch_live_models(prov)
                    if sync_res.get("status") == "success" and sync_res.get("models"):
                        first_m = sync_res["models"][0]
                        cur_m = self.get_active_model(prov)
                        if cur_m not in sync_res["models"]:
                            self.db.set_setting(f"active_model:{prov}", first_m)
                            if self.get_active_provider() == prov:
                                self.db.set_setting("active_model", first_m)
                except Exception:
                    pass

        return self.get_full_settings()

    def fetch_live_models(self, provider: str) -> Dict[str, Any]:
        """Interrogates the provider API to refresh available models."""
        prov = provider.lower()
        if prov not in PROVIDER_DEFAULTS:
            return {"status": "error", "message": f"Provider {provider} non supportato", "models": []}

        api_key = self.get_api_key(prov)
        base_url = self.get_base_url(prov).rstrip("/")

        fetched: List[str] = []
        try:
            if prov == "gemini":
                if not api_key:
                    return {"status": "error", "message": "API key mancante per Gemini", "models": self.get_models_for_provider(prov)}
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_models = data.get("models", [])
                    skip_patterns = (
                        "-tuning", "-image", "-tts", "-transcribe",
                        "robotics", "computer-use", "customtools", "gemini-2.5-flash",
                    )
                    for rm in raw_models:
                        name = rm.get("name", "").replace("models/", "")
                        methods = rm.get("supportedGenerationMethods", [])
                        if "generateContent" in methods and "gemini" in name.lower() and not any(p in name.lower() for p in skip_patterns):
                            fetched.append(name)
                    priority = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-pro-latest"]
                    fetched.sort(key=lambda x: priority.index(x) if x in priority else 99)

            elif prov == "anthropic":
                if not api_key:
                    return {"status": "error", "message": "API key mancante per Anthropic", "models": self.get_models_for_provider(prov)}
                url = f"{base_url}/models"
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    for m in data.get("data", []):
                        mid = m.get("id")
                        if mid:
                            fetched.append(mid)

            elif prov in ("openai", "deepseek", "grok"):
                if not api_key and prov != "qwen":
                    return {"status": "error", "message": f"API key mancante per {prov}", "models": self.get_models_for_provider(prov)}
                url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    for m in data.get("data", []):
                        mid = m.get("id")
                        if not mid:
                            continue
                        if prov == "openai":
                            if mid.startswith(("gpt-4", "o1", "o3", "chatgpt")):
                                fetched.append(mid)
                        else:
                            fetched.append(mid)

            elif prov == "qwen":
                # Check OpenAI-compatible /v1/models or Ollama /api/tags
                models_url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                req = urllib.request.Request(models_url, headers=headers, method="GET")
                try:
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        for m in data.get("data", []):
                            if m.get("id"):
                                fetched.append(m["id"])
                except Exception:
                    # Fallback to Ollama native /api/tags
                    root_url = base_url.replace("/v1", "")
                    tags_url = f"{root_url}/api/tags"
                    req2 = urllib.request.Request(tags_url, method="GET")
                    with urllib.request.urlopen(req2, timeout=6) as resp2:
                        data2 = json.loads(resp2.read().decode("utf-8"))
                        for m in data2.get("models", []):
                            if m.get("name"):
                                fetched.append(m["name"])

            if fetched:
                # Merge existing defaults/custom models to never lose user models
                current = self.get_models_for_provider(prov)
                combined = fetched + [m for m in current if m not in fetched]
                self.set_models_for_provider(prov, combined)
                return {
                    "status": "success",
                    "count": len(fetched),
                    "models": self.get_models_for_provider(prov),
                }
            else:
                return {
                    "status": "warning",
                    "message": "Nessun modello trovato dall'API",
                    "models": self.get_models_for_provider(prov),
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Errore durante il recupero dei modelli: {str(e)}",
                "models": self.get_models_for_provider(prov),
            }

    def get_llm_client(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Creates an initialized LLMClient using current settings."""
        from memoria.llm.client import LLMClient

        p = (provider_name or self.get_active_provider()).lower()
        m = model or self.get_active_model(p)
        key = self.get_api_key(p)
        url = self.get_base_url(p)
        return LLMClient(provider=p, model=m, api_key=key, base_url=url)
