"""Единая фабрика memory-атома (волна 3 / W3.3): один путь записи памяти вместо трёх расползшихся.

## Contract: memory
### Inputs
Черновик факта и хранилище. Для записи вектора вызывающий код может передать
отложенное получение эмбеддера и инвалидацию кэша.
### Outputs
`build_memory_atom` возвращает Atom, `remember` возвращает его id. Ошибки
хранилища и эмбеддера передаются вызывающему коду.
### State
Факт записывается через Store до получения эмбеддера. Кэш принадлежит вызывающему коду.
### Multi-call invariants
Для идентификатора от содержимого используется идемпотентная запись, если
хранилище её поддерживает. CLI и MCP сохраняют строгую запись с id от времени.
### Edge cases
Без явной темы используется general. Без valid_from берётся время наблюдения.
Без эмбеддера или вектора факт доступен для BM25. Ошибка настройки эмбеддера
оставляет сохранённый факт. Для хранилища без идемпотентной записи остаётся add_memory.
### Dependencies
Store и часы из contracts, идентификаторы из ingest, эмбеддер и кэш вызывающей стороны.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import contracts
from .ingest import atom_id


@dataclass(frozen=True)
class MemoryDraft:
    """Что запоминаем; всё различие между точками входа — параметры, не код."""

    text: str
    about: str = ""  # подтема: source = memory/<about or 'general'>
    trust: float = 0.7
    supersedes: str = ""
    valid_from: str = ""  # пусто = сейчас
    identity: str = ""  # непусто — id, вычисленный из содержимого (Hermes)
    title: str = ""  # пусто = about or 'memory'


def build_memory_atom(draft: MemoryDraft) -> contracts.Atom:
    """Atom из черновика: идентичность, время и умолчания вычисляются в одном месте."""
    ts = contracts.now()
    src = f"memory/{draft.about or 'general'}"
    return contracts.Atom(
        id=draft.identity or atom_id(src, int(time.time() * 1000)),
        source=src,
        corpus="memory",
        kind="memory",
        doc_type="fact",
        title=draft.title or draft.about or "memory",
        text=draft.text,
        observed_at=ts,
        valid_from=draft.valid_from or ts,
        trust=draft.trust,
        supersedes=draft.supersedes,
    )


def remember(
    store: contracts.Store,
    draft: MemoryDraft,
    *,
    get_embedder: Callable[[], contracts.Embedder | None] | None = None,
    refresh_cache: Callable[[], None] | None = None,
) -> str:
    """Записать факт памяти единым путём; вернуть id атома.

    Эмбеддер подготавливается после записи факта. При отсутствии вектора
    факт остаётся доступным для BM25, кэш векторов сохраняется.
    """
    atom = build_memory_atom(draft)
    write = store.add_memory
    if draft.identity:
        add_idempotent = getattr(store, "add_memory_idempotent", None)
        if callable(add_idempotent):
            write = add_idempotent
    write(atom)
    embedder = get_embedder() if get_embedder is not None else None
    if embedder:
        vector = embedder.embed_passages([atom.text])[0]
        if vector is not None:
            store.put_vectors([(atom.id, vector)])
            if refresh_cache is not None:
                refresh_cache()
    return atom.id
