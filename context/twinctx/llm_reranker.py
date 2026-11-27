"""LLM-as-reranker: фильтрует BM25 top-k через один вызов LiteLLM.

Зачем: cross-encoder на CPU даёт 2–4с задержку и плохо фильтрует русский («борщ»
проходит). LLM через LiteLLM (уже работает на сервере, уже оплачен) понимает
контекст и русский лучше любой маленькой модели — и стоит ~500 токенов ≈ 0.15₽.

Один вызов: запрос + заголовки top-k документов → LLM возвращает JSON со
score 0–10. Нерелевантные отфильтровываются по порогу.

Скорость — СВЕРХКРИТИЧНА (владелец). Поэтому:
  • таймаут вызова — TWINCTX_LLM_RERANKER_TIMEOUT (дефолт 5с, не 30)
  • circuit breaker: 2 медленных/ошибочных вызова подряд → отключение
    на TWINCTX_LLM_RERANKER_COOLDOWN секунд (дефолт 300 = 5 мин)
  • TWINCTX_LLM_RERANKER=0 — мгновенное полное выключение
  • при отключении поиск работает как чистый BM25 (без реранкера)

Env:
  TWINCTX_LLM_RERANKER=1          # вкл | 0/пусто = выкл
  TWINCTX_LLM_RERANKER_TIMEOUT=5  # сек на вызов (дефолт 5)
  TWINCTX_LLM_RERANKER_COOLDOWN=300  # сек отключения после сбоев (дефолт 300)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

# ── Circuit breaker (модульный: живёт между запросами) ─────────────────────
_BREAKER = {"failures": 0, "disabled_until": 0.0, "last_ms": 0}


def _breaker_timeout() -> float:
    try:
        return float(os.environ.get("TWINCTX_LLM_RERANKER_TIMEOUT", "5"))
    except ValueError:
        return 5.0


def _breaker_cooldown() -> float:
    try:
        return float(os.environ.get("TWINCTX_LLM_RERANKER_COOLDOWN", "300"))
    except ValueError:
        return 300.0


def _breaker_available() -> bool:
    """True если реранкер доступен (не в cooldown)."""
    return time.monotonic() >= _BREAKER["disabled_until"]


def _breaker_record(success: bool, elapsed_ms: int) -> None:
    """Записать результат вызова: успех сбрасывает счётчик, неудача — инкремент."""
    _BREAKER["last_ms"] = elapsed_ms
    if success and elapsed_ms <= _breaker_timeout() * 1000:
        _BREAKER["failures"] = 0
        return
    _BREAKER["failures"] += 1
    if _BREAKER["failures"] >= 2:
        _BREAKER["disabled_until"] = time.monotonic() + _breaker_cooldown()
        _BREAKER["failures"] = 0


def _litellm_base() -> str:
    return os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1").rstrip("/")


def _litellm_key() -> str:
    return os.environ.get("LITELLM_API_KEY", "")


def _chat(model: str, messages: list[dict], max_tokens: int, timeout: float) -> str:
    req = urllib.request.Request(
        f"{_litellm_base()}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_litellm_key()}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


def llm_rerank(
    query: str,
    documents: list[str],
    model: str = "",
    *,
    timeout: float = 0,
) -> Optional[list[float]]:
    """Реранкинг через LLM: scores 0–10 для каждого документа или None.

    None = выключен/cooldown/ошибка/таймаут → движок продолжает без реранкера.
    Circuit breaker: 2 подряд медленных/ошибочных вызова → отключение на cooldown.
    """
    if not documents:
        return []
    if not _litellm_key():
        return None
    if not _breaker_available():
        return None  # circuit breaker открыт — работаем как чистый BM25

    timeout = timeout or _breaker_timeout()
    model = model or os.environ.get("TWINCTX_LLM_RERANKER_MODEL", "chat")

    doc_parts = []
    for i, text in enumerate(documents):
        snippet = text[:500].replace("\n", " ").strip()
        doc_parts.append(f"[{i}] {snippet}")
    docs_block = "\n".join(doc_parts)

    prompt = (
        f"Запрос: {query}\n\n"
        f"Документы:\n{docs_block}\n\n"
        "Оцени релевантность каждого документа запросу от 0 до 10.\n"
        "0 = не относится. 10 = прямо отвечает.\n"
        'Ответь СТРОГО JSON: {"scores": [s0, s1, ...]}'
    )

    t0 = time.monotonic()
    try:
        response = _chat(model, [{"role": "user", "content": prompt}], max_tokens=200, timeout=timeout)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        scores = parsed.get("scores", [])
        if len(scores) != len(documents):
            _breaker_record(False, elapsed_ms)
            return None

        _breaker_record(True, elapsed_ms)
        return [float(s) for s in scores]
    except Exception:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _breaker_record(False, elapsed_ms)
        return None


def from_env() -> bool:
    """Включён ли LLM-реранкер (env + circuit breaker)."""
    if os.environ.get("TWINCTX_LLM_RERANKER", "").strip() not in ("1", "true", "on"):
        return False
    return _breaker_available()


def breaker_status() -> dict:
    """Состояние circuit breaker — для диагностики и health."""
    return {
        "available": _breaker_available(),
        "failures": _BREAKER["failures"],
        "cooldown_remaining_s": max(0, int(_BREAKER["disabled_until"] - time.monotonic())),
        "last_call_ms": _BREAKER["last_ms"],
        "timeout_s": _breaker_timeout(),
        "cooldown_s": _breaker_cooldown(),
    }


def llm_reranker_status() -> dict:
    """Диагностика для страницы подключений / health."""
    return {
        "enabled": from_env(),
        "model": os.environ.get("TWINCTX_LLM_RERANKER_CHAT_MODEL", "chat"),
        "threshold": os.environ.get("TWINCTX_LLM_RERANKER_MIN", "5.0"),
        "base_url": _litellm_base(),
        "breaker": breaker_status(),
    }


def reset_breaker() -> None:
    """Сбросить circuit breaker (для диагностики/ручного включения)."""
    _BREAKER["failures"] = 0
    _BREAKER["disabled_until"] = 0.0
