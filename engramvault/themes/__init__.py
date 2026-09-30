"""
Multi-Theme Engine for Memoria AI (Issue #20).
Supports 5 built-in M3 presets (emerald, amber, indigo, ruby, slate) in Dark & Light modes,
plus user-created Custom Themes persisted in SQLite and exportable/importable as JSON.
"""

import json
import re
from typing import Any, Dict, List, Optional

from engramvault.config import DEFAULT_THEME_PALETTE
from engramvault.db.engine import Database


def _hex_to_rgb_str(hex_color: str) -> str:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return "16, 185, 129"
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r}, {g}, {b}"
    except ValueError:
        return "16, 185, 129"


PRESET_THEMES: Dict[str, Dict[str, Any]] = {
    "emerald": {
        "id": "emerald",
        "name": "Emerald Cyber",
        "icon": "🟢",
        "builtin": True,
        "primary": "#10b981",
        "primary_rgb": "16, 185, 129",
        "primary_hover": "#059669",
        "secondary": "#6366f1",
        "secondary_rgb": "99, 102, 241",
        "dark_bg": "#0f1117",
        "dark_surface": "#10141d",
        "dark_card": "#181d27",
        "dark_border": "#262f40",
        "light_bg": "#f8fafc",
        "light_surface": "#f1f5f9",
        "light_card": "#ffffff",
        "light_border": "#cbd5e1",
    },
    "amber": {
        "id": "amber",
        "name": "Amber Gold",
        "icon": "🟠",
        "builtin": True,
        "primary": "#f59e0b",
        "primary_rgb": "245, 158, 11",
        "primary_hover": "#d97706",
        "secondary": "#ea580c",
        "secondary_rgb": "234, 88, 12",
        "dark_bg": "#12100e",
        "dark_surface": "#181410",
        "dark_card": "#211c16",
        "dark_border": "#362e24",
        "light_bg": "#fffbeb",
        "light_surface": "#fef3c7",
        "light_card": "#ffffff",
        "light_border": "#fde68a",
    },
    "indigo": {
        "id": "indigo",
        "name": "Indigo Deep",
        "icon": "🟣",
        "builtin": True,
        "primary": "#6366f1",
        "primary_rgb": "99, 102, 241",
        "primary_hover": "#4f46e5",
        "secondary": "#06b6d4",
        "secondary_rgb": "6, 182, 212",
        "dark_bg": "#0d0f1a",
        "dark_surface": "#111526",
        "dark_card": "#171c32",
        "dark_border": "#283054",
        "light_bg": "#f5f7ff",
        "light_surface": "#eef2ff",
        "light_card": "#ffffff",
        "light_border": "#c7d2fe",
    },
    "ruby": {
        "id": "ruby",
        "name": "Ruby Crimson",
        "icon": "🔴",
        "builtin": True,
        "primary": "#f43f5e",
        "primary_rgb": "244, 63, 94",
        "primary_hover": "#e11d48",
        "secondary": "#a855f7",
        "secondary_rgb": "168, 85, 247",
        "dark_bg": "#140d11",
        "dark_surface": "#1c1117",
        "dark_card": "#26171f",
        "dark_border": "#3d2431",
        "light_bg": "#fff1f2",
        "light_surface": "#ffe4e6",
        "light_card": "#ffffff",
        "light_border": "#fecdd3",
    },
    "slate": {
        "id": "slate",
        "name": "Slate Minimal",
        "icon": "⚪",
        "builtin": True,
        "primary": "#38bdf8",
        "primary_rgb": "56, 189, 248",
        "primary_hover": "#0284c7",
        "secondary": "#94a3b8",
        "secondary_rgb": "148, 163, 184",
        "dark_bg": "#090d14",
        "dark_surface": "#0f172a",
        "dark_card": "#1e293b",
        "dark_border": "#334155",
        "light_bg": "#f8fafc",
        "light_surface": "#f1f5f9",
        "light_card": "#ffffff",
        "light_border": "#cbd5e1",
    },
}


class ThemeManager:
    """Manages preset palettes, active theme state, and user-created custom themes."""

    def __init__(self, db: Database):
        self.db = db

    def get_custom_themes(self) -> Dict[str, Dict[str, Any]]:
        raw = self.db.get_setting("custom_themes_json")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get_all_themes(self) -> Dict[str, Dict[str, Any]]:
        merged = dict(PRESET_THEMES)
        custom = self.get_custom_themes()
        for k, v in custom.items():
            merged[k] = v
        return merged

    def get_active_theme_info(self) -> Dict[str, Any]:
        all_themes = self.get_all_themes()
        active_id = self.db.get_setting("active_theme_palette") or DEFAULT_THEME_PALETTE
        if active_id not in all_themes:
            active_id = "emerald"
        active_mode = self.db.get_setting("active_theme_mode") or "dark"
        return {
            "active_palette": active_id,
            "active_mode": active_mode,
            "theme": all_themes[active_id],
            "themes": list(all_themes.values()),
        }

    def set_active_theme(self, palette_id: str, mode: Optional[str] = None) -> Dict[str, Any]:
        all_themes = self.get_all_themes()
        if palette_id in all_themes:
            self.db.set_setting("active_theme_palette", palette_id)
        if mode in ("dark", "light"):
            self.db.set_setting("active_theme_mode", mode)
        return self.get_active_theme_info()

    def save_custom_theme(
        self,
        name: str,
        primary: str,
        secondary: str = "#6366f1",
        dark_bg: str = "#0f1117",
        dark_card: str = "#181d27",
        light_bg: str = "#f8fafc",
        light_card: str = "#ffffff",
        theme_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        slug = theme_id or re.sub(r"[^a-z0-9_-]", "-", name.lower().strip())[:32].strip("-")
        if not slug or slug in PRESET_THEMES:
            slug = f"custom-{slug or 'theme'}"

        theme_obj = {
            "id": slug,
            "name": name.strip() or "Custom Theme",
            "icon": "🎨",
            "builtin": False,
            "primary": primary,
            "primary_rgb": _hex_to_rgb_str(primary),
            "primary_hover": primary,
            "secondary": secondary,
            "secondary_rgb": _hex_to_rgb_str(secondary),
            "dark_bg": dark_bg,
            "dark_surface": dark_bg,
            "dark_card": dark_card,
            "dark_border": "#2d3748",
            "light_bg": light_bg,
            "light_surface": light_bg,
            "light_card": light_card,
            "light_border": "#cbd5e1",
        }
        custom = self.get_custom_themes()
        custom[slug] = theme_obj
        self.db.set_setting("custom_themes_json", json.dumps(custom))
        self.db.set_setting("active_theme_palette", slug)
        return self.get_active_theme_info()

    def delete_custom_theme(self, theme_id: str) -> Dict[str, Any]:
        custom = self.get_custom_themes()
        if theme_id in custom:
            del custom[theme_id]
            self.db.set_setting("custom_themes_json", json.dumps(custom))
        if self.db.get_setting("active_theme_palette") == theme_id:
            self.db.set_setting("active_theme_palette", "emerald")
        return self.get_active_theme_info()
