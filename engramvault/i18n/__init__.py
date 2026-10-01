import json
from pathlib import Path
from typing import Any, Dict, List

from engramvault.config import DEFAULT_LANG

LOCALES_DIR = Path(__file__).parent / "locales"
_LOCALE_CACHE: Dict[str, Dict[str, Any]] = {}


def load_all_locales(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Auto-discovers and loads all `<lang>.json` files inside `engramvault/i18n/locales/`."""
    global _LOCALE_CACHE
    if _LOCALE_CACHE and not force_reload:
        return _LOCALE_CACHE

    loaded: Dict[str, Dict[str, Any]] = {}
    if LOCALES_DIR.exists():
        for fpath in sorted(LOCALES_DIR.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                code = (data.get("code") or fpath.stem).lower()
                loaded[code] = data
            except Exception:
                continue

    # Ensure fallback merge against 'en' so partial locale files never have missing keys
    en_base = loaded.get("en", {"ui": {}, "prompts": {}})
    for code, data in loaded.items():
        if code != "en":
            merged_ui = dict(en_base.get("ui", {}))
            merged_ui.update(data.get("ui", {}))
            data["ui"] = merged_ui

            merged_prompts = dict(en_base.get("prompts", {}))
            merged_prompts.update(data.get("prompts", {}))
            data["prompts"] = merged_prompts

    _LOCALE_CACHE = loaded
    return _LOCALE_CACHE


def get_available_locales() -> List[Dict[str, str]]:
    """Returns metadata (`code`, `name`, `flag`) for all registered locales."""
    locales = load_all_locales()
    return [
        {
            "code": code,
            "name": data.get("name", code.upper()),
            "flag": data.get("flag", "🌐"),
        }
        for code, data in locales.items()
    ]


def get_locale_bundle(lang: str = DEFAULT_LANG) -> Dict[str, Any]:
    """Returns the complete bundle (`ui`, `prompts`, `code`, `name`, `flag`) for `lang`, defaulting to `en`."""
    locales = load_all_locales()
    norm = (lang or DEFAULT_LANG or "en").lower()[:2]
    return locales.get(norm) or locales.get("en") or {"code": "en", "name": "English", "flag": "🇬🇧", "ui": {}, "prompts": {}}


def get_prompt(prompt_key: str, lang: str = DEFAULT_LANG) -> str:
    """Returns the localized LLM prompt string for `prompt_key`."""
    bundle = get_locale_bundle(lang)
    return bundle.get("prompts", {}).get(prompt_key, "")


def get_available_languages() -> List[Dict[str, str]]:
    """Alias for get_available_locales()."""
    return get_available_locales()


def get_translations(lang: str = DEFAULT_LANG) -> Dict[str, str]:
    """Returns the flat dictionary of UI translation strings for `lang`."""
    bundle = get_locale_bundle(lang)
    return bundle.get("ui", {})


def t(key: str, lang: str = DEFAULT_LANG, **kwargs: Any) -> str:
    """Translates a UI or prompt key with optional string formatting."""
    bundle = get_locale_bundle(lang)
    val = bundle.get("ui", {}).get(key) or bundle.get("prompts", {}).get(key) or key
    if kwargs and isinstance(val, str):
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val

