"""Реранкер: cross-encoder фильтр качества топ-k после BM25/vector fusion.

Наивный RAG не даёт буста — top-k из BM25 содержит мусор. Реранкер
обрабатывает каждую пару (query, document) cross-encoder'ом и отбрасывает
нерелевантные. Модель — BAAI/bge-reranker-v2-m3 (мультиязычный, русский,
ONNX CPU ~100 мс на 20 пар). Решение владельца 22.11: «реранкер нам нужен 100%.

Env: TWINCTX_RERANKER — пусто (off) | "bge-reranker-v2-m3" (вкл).
"""

from __future__ import annotations

import os
from typing import Optional, Protocol


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def _from_fastembed(model_name: str) -> Optional[Reranker]:
    """Через fastembed TextCrossEncoder (ONNX, CPU, без PyTorch)."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        encoder = TextCrossEncoder(model_name)
        return _FastembedReranker(encoder)
    except ImportError:
        return None
    except Exception:
        return None


class _FastembedReranker:
    """Обёртка над fastembed TextCrossEncoder — возвращает сырые scores."""

    def __init__(self, encoder):
        self._encoder = encoder

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = list(self._encoder.rerank(query, documents))
        return scores


def from_env() -> Optional[Reranker]:
    """Читает TWINCTX_RERANKER: пусто/none → None (выключен); имя модели → реранкер.

    Дефолтная модель: BAAI/bge-reranker-v2-m3 (мультиязычный, ~568M params,
    ONNX CPU, покрытие русского подтверждено arXiv:2504.12879).
    """
    model = os.environ.get("TWINCTX_RERANKER", "").strip()
    if not model or model.lower() in ("none", "off", "0", "disabled"):
        return None
    if model == "bge-reranker-v2-m3":
        model = "BAAI/bge-reranker-v2-m3"
    return _from_fastembed(model)


def filter_reranked(
    items: list,
    scores: list[float],
    threshold: float = 0.0,
    k: int = 8,
) -> tuple[list, list[float]]:
    """Отсекает нерелевантные по reranker-score, возвращает top-k sorted by score.

    Score может быть отрицательным (bge-reranker logit) — порог обычно 0.0.
    items и scores должны быть одной длины (zip-семантика).
    """
    if len(items) != len(scores):
        return items[:k], scores[:k] if len(scores) >= len(items) else scores
    pairs = list(zip(items, scores))
    pairs.sort(key=lambda p: -p[1])
    passed = [(item, score) for item, score in pairs if score >= threshold]
    items_out = [item for item, _ in passed[:k]]
    scores_out = [score for _, score in passed[:k]]
    return items_out, scores_out
