"""Проверяемый поиск проектных документов Vault.

## Contract: ProjectSearch
Inputs: явный идентификатор проекта, вопрос и ограниченный онлайн-поиск.
Outputs: цитата действующей редакции, явная неопределённость или код отказа.
State: до двух локальных ходов, без записи в Vault и без вызова модели.
Multi-call: отдельный дедлайн на ход; QMD один раз после пустой выдачи.
Edge cases: пустой проект, сбой twinctx, занятость QMD, изменившийся Markdown,
отсутствующая, неоднозначная или циклическая связь редакций, дедлайн чтения.
Dependencies: twinctx, Vault и локальный HTTP QMD без прокси или переадресации.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
from pathlib import Path
from threading import BoundedSemaphore
from time import monotonic

from twinctx.engine import AssembleRequest
from twinctx.ingest import sha1, split_md
from twinctx.text import query_terms, words
from twinctx.vault_projects import active_revision, normalize_project, read_document, scan_projects

logger = logging.getLogger(__name__)
_DEADLINE_SECONDS = 8.0
_PRIMARY_LIMIT = 60
_SLOTS = BoundedSemaphore(2)

_UNCERTAIN = "Не удалось подтвердить действующую редакцию."
_EMPTY = "В проектных документах Vault не нашёл совпадений."
_TIMEOUT = "Истёк лимит времени на поиск по проекту."
_PRIMARY = "Основной поиск twinctx не завершился. Резервный поиск не запускался."
_QMD = "Резервный поиск QMD сейчас недоступен."
_BUSY = "Поиск по проекту занят. Попробуй ещё раз чуть позже."


def _qmd_search(project: str, query: str, timeout: float) -> list[dict]:
    from .qmd_client import hits
    return hits("/vsearch", {"project": project, "q": query}, timeout)


def _overlap(terms: list[str], text: str) -> int:
    tokens = words(text)
    return sum(any(token.startswith(term) for token in tokens) for term in terms)


def _quote(query: str, raw: bytes, path: str) -> tuple[str, str] | None:
    terms = query_terms(query)
    if not terms:
        return None
    chunks = split_md(raw.decode())
    spans = []
    for heading, start, end, text in chunks:
        if not text:
            continue
        # A quoted excerpt is always a verbatim substring of the live Markdown.
        # Rank individual sentences rather than reporting a whole file as an answer.
        for excerpt in re.split(r"(?<=[.!?])\s+|\n+", text):
            excerpt = excerpt.strip()
            if not 18 <= len(excerpt) <= 500 or excerpt.startswith("#"):
                continue
            direct = _overlap(terms, excerpt)
            if direct >= min(2, len(terms)):
                spans.append((direct, _overlap(terms, heading), excerpt, start, end))
    if not spans:
        return None
    _, _, excerpt, start, end = max(spans, key=lambda row: row[:2])
    return excerpt, f"vault/{path}:{start}-{end}"


def answer_project(
    project: str,
    query: str,
    *,
    search,
    vault_root: Path,
    qmd_search=None,
    admitted_at: float | None = None,
) -> str:
    start = admitted_at if admitted_at is not None else monotonic()
    route = "twinctx"
    status = "primary_failed"

    def remaining() -> float:
        seconds = start + _DEADLINE_SECONDS - monotonic()
        if seconds <= 0:
            raise TimeoutError
        return seconds

    if not _SLOTS.acquire(blocking=False):
        logger.info("project_search route=none outcome=busy")
        return _BUSY
    try:
        if normalize_project(project) != project or not query.strip() or len(query) > 500:
            status = "invalid_query"
            return _EMPTY
        remaining()
        status = "revision_uncertain"
        documents = scan_projects(vault_root, checkpoint=remaining)
        documents = {path: doc for path, doc in documents.items() if doc.project == project}
        status = "primary_failed"
        atoms, info = search(
            AssembleRequest(query=query, corpora=["vault"], include_memory=False, graph=False, k=_PRIMARY_LIMIT,
                            allowed_sources=tuple(f"vault/{path}" for path in documents))
        )
        remaining()
        if set(info["degraded"]) - {"vector_ranker_unavailable", "embedding_unavailable", "bm25_only", "vec_mmap", "no_vectors_indexed"}:
            return _PRIMARY
        status = "revision_uncertain"
        candidates = {}
        for atom in atoms:
            if atom.source.startswith("vault/") and atom.source[6:] in documents:
                candidates.setdefault(atom.source[6:], info.get("source_sha1", {}).get(atom.source))
        if not candidates:
            # A saturated ranked list means we cannot establish an empty search
            # for this project. Do not use the fallback for that partial result.
            if atoms:
                status = "primary_truncated"
                return _PRIMARY
            if qmd_search is None:
                if os.environ.get("TWINCTX_QMD_ENABLED") != "1":
                    return _EMPTY
                qmd_search = _qmd_search
            route = "qmd"
            status = "qmd_failed"
            hits = qmd_search(project, query, remaining())
            remaining()
            if not isinstance(hits, list) or len(hits) > 5:
                raise ValueError("qmd_invalid_reply")
            for hit in hits:
                path = hit["path"]
                if path not in documents or hit["sha256"] != documents[path].sha256:
                    status = "revision_uncertain"
                    return _UNCERTAIN
                candidates[path] = None

        status = "empty"
        if not candidates:
            return _EMPTY
        status = "revision_uncertain"
        for path, indexed_sha in candidates.items():
            remaining()
            original = read_document(vault_root, path)
            if hashlib.sha256(original).hexdigest() != documents[path].sha256:
                continue
            if route == "twinctx" and indexed_sha != sha1(original):
                continue
            active = active_revision(documents, path)
            if active is None:
                continue
            raw = original if active == path else read_document(vault_root, active)
            if hashlib.sha256(raw).hexdigest() != documents[active].sha256:
                continue
            result = _quote(query, raw, active)
            if result is None and route != "qmd":
                continue
            # Re-read the project's source map after selection. Renames, new
            # successors and edits during lookup invalidate the decision.
            refreshed = scan_projects(vault_root, checkpoint=remaining)
            if documents != {name: doc for name, doc in refreshed.items() if doc.project == project}:
                return _UNCERTAIN
            remaining()
            status = "verified"
            if result is None:
                return f"Возможный источник по смыслу: vault/{active}. Цитата для ответа не найдена."
            excerpt, provenance = result
            return f"«{excerpt}» [{provenance}]"
        return _UNCERTAIN
    except TimeoutError:
        status = "deadline"
        return _TIMEOUT
    except urllib.error.HTTPError as error:
        if error.code == 504:
            status = "deadline"
            return _TIMEOUT
        status = "qmd_busy" if error.code == 429 else "qmd_failed"
        return _BUSY if error.code == 429 else _QMD
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        if status == "primary_failed":
            return _PRIMARY
        return _QMD if status == "qmd_failed" else _UNCERTAIN
    finally:
        _SLOTS.release()
        logger.info("project_search route=%s outcome=%s elapsed_ms=%d",route,status,(monotonic()-start)*1000)
