"""Env-хелперы PG-слоя: булевы и числовые ручки читаются в момент вызова.

Вынесено из pg_sources.py волной 5 техдолга без изменения поведения.
"""

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default
