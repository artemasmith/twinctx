"""## Contract: PgOnlineSearch

### Inputs
`search` и `assemble` принимают `AssembleRequest`. Фабрика чтения принимает
функцию остатков дедлайна в секундах и открывает контекст read-only сеанса.
`search_task_sources` принимает тот же запрос с k от 1 до 60 и использует
область tasks. Для идентификатора карточки t_<8hex> отбираются только
начала YAML, содержащие его лексемы. Потребитель сверяет поле идентичности.

### Outputs
Списки и манифесты имеют форму `Engine`. После сбоя дополнительной стадии
возвращаются уже отобранные фрагменты с пометкой `degraded`. Если поиск
прерван до отбора, манифест пуст, а `nothing_relevant` равно false.
Поиск задач дополняет отобранные фрагменты недостающими началами найденных
YAML-документов из того же снимка. При неполноте и превышении лимита
возвращается пустой результат с причиной в `degraded`.

### State
Дедлайн, результат и сеанс чтения принадлежат одному вызову. Слот общий
на процесс. Запись и конфигурация эмбеддера остаются в исходном канале.

### Multi-call invariants
Поиск только читает. Один занятый поток на процесс без очереди. Слот
освобождается после выхода из сеанса, результат другого вызова не используется.

### Edge cases
Пустой запрос и запрос без совпадений дают завершившийся пустой поиск.
Занятый слот возвращает `online_search_busy` немедленно. При истечении
дедлайна возвращается `online_search_deadline`, а занятый поток закрывает
сеанс. Обрыв базы или отказ подключения возвращает `online_read_failed`
без реквизитов. Повтор после закрытия соединения открывает новый сеанс.

### Dependencies
Лексический `Engine` читает атомы, источники и граф через внедрённый адаптер.
SQL, реквизиты и TLS принадлежат `stores`. Сетевые вызовы эмбеддера возможны
только у отдельной операции записи памяти, которая запрашивает эмбеддер.
"""

import re
from collections import Counter
from dataclasses import replace
from functools import cached_property
from threading import BoundedSemaphore, Event, Thread
from time import monotonic

from .contracts import now
from .engine import Engine, resolve_graph

_SEARCH_SECONDS = 7.0
_SEARCH_SLOT = BoundedSemaphore(1)
_TASK_SOURCE_LIMIT = 16
_CARD_ID = re.compile(r"t_[0-9a-f]{8}", re.IGNORECASE | re.ASCII)


class PgOnlineSearch:
    def __init__(self, store, reader, build_embedder, *, rerank=True, search_seconds=None):
        self.store = store
        self._reader = reader
        self._build_embedder = build_embedder
        self._rerank = rerank
        self._search_seconds = search_seconds

    @cached_property
    def _write_embedder(self):
        return self._build_embedder()

    def memory_embedder(self):
        return self._write_embedder

    def _run(self, operation, *, rerank=True):
        deadline = monotonic() + (self._search_seconds if self._search_seconds is not None else _SEARCH_SECONDS)
        if not _SEARCH_SLOT.acquire(blocking=False):
            return None, "online_search_busy"

        done = Event()
        result, partial, reason = None, None, "online_read_failed"

        def checkpoint(value):
            nonlocal partial
            partial = value

        def remaining():
            seconds = deadline - monotonic()
            if seconds <= 0:
                raise TimeoutError
            return seconds

        def run():
            nonlocal result, reason
            try:
                with self._reader(remaining) as reader:
                    engine = Engine(reader, lexical_only=True, check_deadline=remaining, rerank=self._rerank and rerank)
                    found = operation(engine, checkpoint)
                remaining()
                result, reason = found, ""
            except TimeoutError:
                reason = "online_search_deadline"
            except Exception:
                if monotonic() >= deadline:
                    reason = "online_search_deadline"
            finally:
                _SEARCH_SLOT.release()
                done.set()

        try:
            Thread(target=run, daemon=True, name="twinctx-online-read").start()
        except (RuntimeError, OSError):
            _SEARCH_SLOT.release()
            return None, "online_read_failed"

        if not done.wait(max(0.0, deadline - monotonic())):
            reason = "online_search_deadline"
        if reason and partial is not None:
            return partial, "online_optional_stage_incomplete"
        return result, reason

    def search(self, request):
        result, reason = self._run(lambda engine, checkpoint: engine.search(request, checkpoint=checkpoint))
        if result is not None:
            atoms, info = result
            if reason:
                info["degraded"].append(reason)
            return atoms, info
        if reason:
            return [], {"counts": Counter(), "degraded": [reason], "terms": []}
        return result

    def search_task_sources(self, request):
        """Complete ranked task evidence with current headers in one read session."""
        if type(request.k) is not int or not 1 <= request.k <= 60:
            raise ValueError("task search k must be between 1 and 60")
        scoped = replace(request, corpora=["tasks"], as_of=request.as_of or now())

        def search_by_card(engine, _checkpoint):
            card = scoped.query.strip().lower()
            limit = min(scoped.k, _TASK_SOURCE_LIMIT)
            headers = engine.store.task_headers_by_card(card, scoped.as_of, scoped.min_trust, limit + 1)
            sources = {atom.source for atom in headers}
            info = {
                "counts": Counter(
                    task_search_hits=len(headers), task_yaml_sources=len(sources), task_yaml_headers=len(headers),
                ),
                "degraded": [],
                "terms": [card],
            }
            if len(headers) > limit or len(sources) != len(headers):
                info["degraded"].append("online_task_sources_overflow")
                return [], info
            return sorted(headers, key=lambda atom: atom.source), info

        def search_with_headers(engine, _checkpoint):
            ranked, info = engine.search(scoped)
            atoms = [
                atom for atom in ranked
                if atom.corpus == "tasks" and atom.source.startswith("tasks/")
                and (not atom.valid_from or atom.valid_from <= scoped.as_of)
                and (not atom.valid_to or atom.valid_to > scoped.as_of)
            ]
            sources = list(dict.fromkeys(
                atom.source for atom in atoms
                if atom.kind == "knowledge" and atom.doc_type == "task" and atom.source.endswith((".yml", ".yaml"))
            ))
            info["counts"].update(task_search_hits=len(ranked), task_yaml_sources=len(sources))
            if len(ranked) >= scoped.k or len(sources) > _TASK_SOURCE_LIMIT:
                info["degraded"].append("online_task_sources_overflow")
                return [], info
            headers = engine.store.task_source_headers(sources, scoped.as_of, scoped.min_trust)
            by_source = {atom.source: atom for atom in headers}
            info["counts"]["task_yaml_headers"] = len(headers)
            if len(by_source) != len(headers):
                info["degraded"].append("online_task_sources_overflow")
                return [], info
            if by_source.keys() != set(sources):
                info["degraded"].append("online_task_sources_incomplete")
                return [], info
            seen = {atom.id for atom in atoms}
            atoms.extend(by_source[source] for source in sources if by_source[source].id not in seen)
            return atoms, info

        operation = search_by_card if _CARD_ID.fullmatch(scoped.query.strip()) else search_with_headers
        result, reason = self._run(operation, rerank=False)
        if reason:
            return [], {"counts": Counter(), "degraded": [reason], "terms": []}
        return result

    def current_memory_facts(self, source, as_of, *, limit=128):
        """Return a complete bounded snapshot for one source, or an explicit failure."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("memory fact limit must be between 1 and 128")
        atoms, reason = self._run(
            lambda engine, _checkpoint: engine.store.current_memory_facts(source, as_of, limit + 1)
        )
        if reason:
            return [], reason
        if len(atoms) > limit:
            return [], "online_memory_facts_overflow"
        return atoms, ""

    def assemble(self, request):
        result, reason = self._run(lambda engine, checkpoint: engine.assemble(request, checkpoint=checkpoint))
        if result is not None:
            if reason:
                result["manifest"]["degraded"].append(reason)
            return result
        if reason:
            return {
                "packed": "",
                "manifest": {
                    "query": request.query,
                    "terms": [],
                    "tokens": 0,
                    "budget": request.budget_tokens,
                    "items": [],
                    "nothing_relevant": False,
                    "graph_enabled": resolve_graph(request),
                    "graph_expanded_count": 0,
                    "graph_hops_used": 0,
                    "graph_edges_considered": 0,
                    "degraded": [reason],
                    "source_ages": {},
                },
            }
        return result
