"""
save/settings_save.py
Persists user preferences (mouse sensitivity, render distance, FPS display,
FPS limit) to a single JSON file outside any specific world's save folder,
since these are global app settings rather than per-world data. Without
this, settings silently reset to defaults every time the game restarted.
"""

import json
import os

import config
import paths


def _settings_path():
    """Not a module-level constant any more: paths.data_path creates the parent
    directory as a side effect, and doing that at import time means the folder
    appears merely because something imported this module. See save/world_save.py
    for why the path cannot just be config.SAVES_DIR."""
    return paths.data_path(config.SAVES_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "mouse_sensitivity": 1.0,
    "render_distance": config.DEFAULT_RENDER_DISTANCE,
    "show_fps": True,
    "fps_limit": config.DEFAULT_FPS_LIMIT,
    # Multiplayer. Global rather than per-world for the same reason as
    # everything else here: a name and a friend's address belong to the person
    # sitting at the keyboard, not to any one save folder.
    #
    # The address starts EMPTY. It was prefilled with a plausible-looking
    # 26.31.44.10:25565 once, which was worse than blank in both directions: the
    # IP was invented, and the port is now assigned fresh by the kernel every
    # time a world is opened (see GameServer.start), so a remembered one is
    # wrong the next day. The field fills itself in from the last address that
    # actually worked, which is the only prefill worth having.
    "username": "Player",
    "last_server_address": "",
}


def load_settings() -> dict:
    """Returns saved settings merged over the defaults (so new settings
    added in future updates always have a sane value even for old saves)."""
    settings = dict(DEFAULT_SETTINGS)
    settings_path = _settings_path()
    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            settings.update(saved)
        except (json.JSONDecodeError, OSError):
            pass  # corrupted settings file - fall back to defaults rather than crashing
    return settings


def save_settings(settings: dict):
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
