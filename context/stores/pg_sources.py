"""Isolated PostgreSQL source-index slice; NOT a complete twinctx Store.

This module is deliberately not registered in open_store(). It provides atomic
source replacement, reads and bounded keyword search for a dedicated customer
database, plus graph and vector operations. Memory writes, journal recovery and
data migration are NOT implemented here: they live in the PgMemoryCanon wrapper
that open_store(postgres) puts around this slice; do not select the slice directly.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import struct
import time
from contextlib import contextmanager, nullcontext, suppress
from ipaddress import ip_address
from itertools import chain
from threading import RLock
from urllib.parse import unquote, urlparse

from twinctx.connectors import TELEGRAM_ARCHIVE_MESSAGE_PREFIX, TELEGRAM_BUSINESS_MESSAGE_PREFIX
from twinctx.contracts import COLS, Atom, now
from twinctx.text import quoted_subject_terms

from .memory_history import family_query
from .pg_bm25 import (  # noqa: F401 — фасад волны 5: тесты зовут pg_sources.<имя>
    _BM25_B,
    _BM25_K1,
    _BM25_POOL,
    _BM25_WEIGHTS,
    _FTS_STATS_TTL,
    PgBm25Mixin,
)
from .pg_env import _env_flag, _env_number  # noqa: F401
from .pg_graph import PgGraphMixin
from .pg_schema import _SCHEMA, _SCHEMA_ALL_VERSION, PgSchemaMixin  # noqa: F401
from .pg_text import project_text, restore_text
from .pg_vectors import _VECTOR_BATCH_SIZE, _VECTOR_PAGE_MAX, PgVectorsMixin  # noqa: F401


class SourceOperationConflictError(ValueError):
    """Operation ID was reused with a different source request."""


class _PgSearchReads(PgBm25Mixin):
    """Queries shared by the main index and its isolated read-only sessions."""

    def _run_read(self, sql, **params):
        return self.con.run(sql, **params)

    def _check_search_deadline(self):
        """Offline and maintenance calls have no online deadline."""

    def neighbors(self, ids: list[str], per_seed: int, project: str = "", as_of: str = "") -> list[dict]:
        """Return bounded edges, checking validity of the actual neighboring atom."""
        if type(per_seed) is not int or per_seed < 1:
            return []
        as_of = as_of or now()
        out = []
        with self._lock:
            for seed in ids:
                rows = self._run_read(
                    """select case when e.src=:seed then e.dst else e.src end as nb,
                              e.src,e.dst,e.kind,e.weight,e.project,e.provenance,e.basis,e.owner
                       from twinctx_v1.edges e
                       join twinctx_v1.atoms a on a.id=case when e.src=:seed then e.dst else e.src end
                       where (e.src=:seed or e.dst=:seed)
                         and (e.valid_from is null or e.valid_from='' or e.valid_from<=:at)
                         and (e.valid_to is null or e.valid_to='' or e.valid_to>:at)
                         and (a.valid_from is null or a.valid_from='' or a.valid_from<=:at)
                         and (a.valid_to is null or a.valid_to='' or a.valid_to>:at)
                         and not starts_with(a.source, :telegram_archive)
                         and not starts_with(a.source, :telegram_business)
                         and (:project='' or e.project=:project)
                       order by e.weight desc,nb limit :lim""",
                    seed=seed,
                    at=as_of,
                    project=project,
                    lim=per_seed,
                    telegram_archive=TELEGRAM_ARCHIVE_MESSAGE_PREFIX,
                    telegram_business=TELEGRAM_BUSINESS_MESSAGE_PREFIX,
                )
                keys = ("nb", "src", "dst", "kind", "weight", "project", "provenance", "basis", "owner")
                out.extend(dict(zip(keys, row)) for row in rows)
        return out

    @staticmethod
    def _atom(row):
        if len(row) != len(COLS) + 1:
            raise ValueError("atom query must include raw text")
        values = {c: (v if v is not None else Atom.__dataclass_fields__[c].default) for c, v in zip(COLS, row[:-1])}
        values["text"] = restore_text(values["text"], row[-1])
        return Atom(**values)

    def memory_family(self, seed: str) -> list[Atom]:
        with self._lock:
            rows = self._run_read(family_query("twinctx_v1.atoms", raw_text=True), seed=seed)
            return [self._atom(row) for row in rows]

    def get(self, ids: list[str]) -> dict[str, Atom]:
        if not ids:
            return {}
        with self._lock:
            rows = self._run_read(
                f"select {','.join(COLS)},text_raw from twinctx_v1.atoms where id=any(cast(:ids as text[]))", ids=ids
            )
            atoms = [self._atom(row) for row in rows]
            return {a.id: a for a in atoms}

    def source_ages(self) -> dict[str, str]:
        """Latest indexed timestamp per corpus, with SQLite-compatible NULLs."""
        with self._lock:
            rows = self._run_read('select corpus, max(indexed_at collate "C") from twinctx_v1.sources group by corpus')
            return {corpus: indexed for corpus, indexed in rows}


class _PgSearchSession(_PgSearchReads):
    """A single worker owns this session and uses only the shared search queries."""

    def __init__(self, connection, remaining, stats):
        self._connection = connection
        self._remaining = remaining
        self._fts_stats = stats
        self._fts_writes = 0  # the reader does no writes and expires stats by TTL
        self._lock = nullcontext()

    def _check_search_deadline(self):
        self._remaining()

    def current_memory_facts(self, source, as_of, limit):
        rows = self._run_read(
            f"""select {",".join(COLS)}, text_raw from twinctx_v1.atoms
                where source=:source and kind='memory' and doc_type='fact' and trust>=0.9
                  and (valid_from is null or valid_from='' or valid_from<=:as_of)
                  and (valid_to is null or valid_to='' or valid_to>:as_of)
                limit :limit""",
            source=source,
            as_of=as_of,
            limit=limit,
        )
        return [self._atom(row) for row in rows]

    def task_source_headers(self, sources, as_of, min_trust):
        if not sources:
            return []
        return self._task_headers(
            "source=any(cast(:sources as text[]))",
            as_of,
            min_trust,
            len(sources) + 1,
            sources=sources,
        )

    def task_headers_by_card(self, card, as_of, min_trust, limit):
        return self._task_headers(
            "fts_text @@ plainto_tsquery('simple'::regconfig, :card)",
            as_of,
            min_trust,
            limit,
            card=card,
        )

    def _task_headers(self, predicate, as_of, min_trust, limit, **params):
        rows = self._run_read(
            f"""select {",".join(COLS)}, text_raw from twinctx_v1.atoms
                where {predicate}
                  and corpus='tasks' and kind='knowledge' and doc_type='task'
                  and starts_with(source,'tasks/') and (right(source, 4)='.yml' or right(source, 5)='.yaml')
                  and start_line=1 and trust>=:min_trust
                  and (valid_from is null or valid_from='' or valid_from<=:as_of)
                  and (valid_to is null or valid_to='' or valid_to>:as_of)
                limit :limit""",
            as_of=as_of,
            min_trust=min_trust,
            limit=limit,
            **params,
        )
        return [self._atom(row) for row in rows]

    def indexed_source_hashes(self, sources):
        if not sources:
            return {}
        rows = self._run_read(
            "select source,sha from twinctx_v1.sources where source=any(cast(:sources as text[]))",
            sources=sorted(sources),
        )
        return dict(rows)

    def online_candidates(self, terms, request, as_of):
        """Apply visibility before each bounded lookup and aggregate only its small result."""
        from .pg_bm25 import _tsquery

        words = [_tsquery(f'"{term}"*') for term in terms]
        words = [word for word in words if word is not None]
        pairs = [f"({left} & {right})" for left, right in zip(words, words[1:])]
        subject = [_tsquery(f'"{term}"*') for term in quoted_subject_terms(request.query)]
        fact_probe = ["(" + " <-> ".join(value for value in subject if value) + ")"] if subject else []
        patterns = fact_probe + pairs + words
        if not patterns or request.allowed_sources == ():
            return [], 0
        scope_is_explicit = request.allowed_sources is not None
        per_probe = min(120, 480 // len(patterns))
        rows = self._run_read(
            """with sampled as (
                 select hits.id, p.priority, count(*) over (partition by p.priority) as frequency
               from unnest(cast(:patterns as text[])) with ordinality as p(pattern, priority)
               cross join lateral (select to_tsquery('simple'::regconfig, p.pattern) as query) q
               cross join lateral (
                 select a.id from twinctx_v1.atoms a
                 where (a.fts_title @@ q.query or a.fts_heading @@ q.query or a.fts_text @@ q.query)
                   and not starts_with(a.source, :telegram_archive)
                   and not starts_with(a.source, :telegram_business)
                   and (cardinality(cast(:corpora as text[]))=0 or exists (
                     select 1 from unnest(cast(:corpora as text[])) as scopes(prefix)
                     where starts_with(a.corpus, scopes.prefix)
                   ))
                   and (not :scope_is_explicit or a.source=any(cast(:allowed_sources as text[])))
                   and a.trust >= :min_trust
                   and (p.priority > :fact_probes or (a.kind='memory' and a.doc_type='fact' and a.trust>=0.9))
                   and (a.kind <> 'memory' or (
                     :include_memory
                     and (a.valid_from is null or a.valid_from='' or a.valid_from<=:as_of)
                     and (a.valid_to is null or a.valid_to='' or a.valid_to>:as_of)
                   ))
                 limit :per_probe
               ) hits
               )
               select id, sum(case
                 when priority <= :fact_probes then 8.0
                 when priority <= :pair_count then 2.0
                 else 1.0 end / sqrt(frequency)) as strength
               ,max(frequency) as sampled_frequency
               from sampled
               group by id
               order by strength desc, min(priority), id
               limit :max_candidates""",
            patterns=patterns,
            corpora=request.corpora,
            min_trust=request.min_trust,
            include_memory=request.include_memory,
            as_of=as_of,
            per_probe=per_probe + int(scope_is_explicit),
            max_candidates=480 + int(scope_is_explicit),
            scope_is_explicit=scope_is_explicit,
            allowed_sources=list(request.allowed_sources or ()),
            fact_probes=len(fact_probe),
            pair_count=len(fact_probe) + len(pairs),
            telegram_archive=TELEGRAM_ARCHIVE_MESSAGE_PREFIX,
            telegram_business=TELEGRAM_BUSINESS_MESSAGE_PREFIX,
        )
        if scope_is_explicit and (len(rows) > 480 or any(row[2] > per_probe for row in rows)):
            raise RuntimeError("project search candidate limit reached")
        return [(row[0], row[1]) for row in rows], len(rows)

    @staticmethod
    def statement_timeout(remaining):
        # Leave time for PostgreSQL to report cancellation before the socket expires.
        return max(1, int(remaining * 1000) - 75)

    def _run_read(self, sql, **params):
        remaining = self._remaining()
        # pg8000 has no public per-query timeout. Use its active socket, which is
        # the TLS socket on remote connections, without changing the main store.
        sock = self._connection._usock
        try:
            sock.settimeout(remaining)
            self._connection.run(
                "select set_config('statement_timeout', :timeout, true)",
                timeout=str(self.statement_timeout(remaining)),
            )
            sock.settimeout(self._remaining())
            return self._connection.run(sql, **params)
        except Exception as exc:
            details = exc.args[0] if exc.args else None
            if isinstance(details, dict) and details.get("C") == "57014":
                raise TimeoutError from exc
            raise


def _pg_connection_args(dsn):
    url = urlparse(dsn)
    if (
        url.scheme not in ("postgres", "postgresql")
        or not url.hostname
        or not url.path.strip("/")
        or url.query
        or url.fragment
    ):
        raise ValueError("explicit PostgreSQL DSN with database is required")
    try:
        local = ip_address(url.hostname).is_loopback
    except ValueError:
        local = url.hostname == "localhost"
    # pg8000's default is opportunistic TLS without peer verification. Use
    # a verified TLS context for non-loopback hosts; never silently downgrade.
    tls = False if local else ssl.create_default_context()
    return dict(
        user=unquote(url.username or "postgres"),
        password=unquote(url.password) if url.password else None,
        host=url.hostname,
        port=url.port or 5432,
        database=unquote(url.path.lstrip("/")),
        ssl_context=tls,
        # The same socket handles SQL reads as well as connection setup.
        # Cold indexed scans and first-time index creation may exceed 5 s.
        timeout=120,
    )


def _online_reader(connection_args):
    """Build a read-only session factory without reopening or migrating the store."""
    import pg8000.native
    from pg8000.exceptions import InterfaceError

    stats = {"ts": 0.0, "n": 0, "avgdl": {}, "df": {}}

    @contextmanager
    def open_session(remaining):
        timeout = remaining()
        connection = pg8000.native.Connection(
            **(connection_args | {"timeout": timeout}),
            application_name="aitwin-twinctx-online",
            startup_params={"statement_timeout": str(_PgSearchSession.statement_timeout(timeout))},
        )
        try:
            if time.monotonic() - stats["ts"] >= _FTS_STATS_TTL:
                stats["df"].clear()
            connection._usock.settimeout(remaining())
            connection.run("begin isolation level repeatable read read only")
            yield _PgSearchSession(connection, remaining, stats)
        finally:
            # A broken connection can already be closed by pg8000. Cleanup
            # must preserve the original SQL timeout or connection failure.
            with suppress(InterfaceError):
                connection.close()

    return open_session


def open_online_reader(dsn):
    """Build a read-only session factory without initializing a write store."""
    return _online_reader(_pg_connection_args(dsn))


class PgSourceIndex(_PgSearchReads, PgSchemaMixin, PgGraphMixin, PgVectorsMixin):
    """Transactional source operations; one instance owns one DB connection.

    Only standalone tests/migration development may instantiate this slice. Every
    tenant requires a dedicated DB until explicit server-enforced isolation exists.
    """

    def __init__(self, dsn: str, *, experimental: bool = False):
        if experimental is not True:
            raise RuntimeError("incomplete source index requires explicit experimental opt-in")
        import pg8000.native

        self._connection_args = _pg_connection_args(dsn)
        self.con = pg8000.native.Connection(**self._connection_args)
        self._lock = RLock()
        self._vector_connections = set()
        self._closed = False
        self._fts_stats = {"ts": 0.0, "n": 0, "avgdl": {}, "df": {}}
        self._fts_writes = 0  # атомы, изменённые с последнего полного пересчёта FTS-статистики
        try:
            compatible = self.con.run(
                "select current_setting('server_encoding')='UTF8', lower('Я')='я', 'Ж' ~ '^[[:alnum:]]+$'"
            )
            if compatible != [[True, True, True]]:
                raise RuntimeError("PostgreSQL Unicode text-search locale is incompatible")
            # Warm-start: составной маркер schema-all позволяет пропустить миграционную
            # транзакцию (advisory lock + 5 зондов). Полный путь остаётся источником истины:
            # при отсутствии/несовпадении маркера он прогоняется и детекторы drift работают.
            if not self._schema_all_current():
                with self._transaction():
                    self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-schema-v1'))")
                    self.con.run(_SCHEMA)
                    self._ensure_fts_schema()
                    self._ensure_graph_schema()
                    self._ensure_atom_precision()
                    self._ensure_atom_text_raw()
                    self._ensure_source_operation_schema()
                self.con.run(
                    "insert into twinctx_v1.schema_meta(key,value) values('schema-all',:v) "
                    "on conflict(key) do update set value=excluded.value",
                    v=json.dumps(
                        {"v": _SCHEMA_ALL_VERSION, "snapshot": self._schema_probe_snapshot()},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
        except BaseException:
            self.con.close()
            raise

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.con.run("begin")
            try:
                yield
                self.con.run("commit")
            except BaseException:
                try:
                    self.con.run("rollback")
                except Exception:
                    pass
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for connection in tuple(self._vector_connections):
                try:
                    connection.close()
                except Exception:
                    pass
                finally:
                    self._vector_connections.discard(connection)
            self.con.close()

    def online_reader(self):
        return _online_reader(self._connection_args)

    def _lock_source(self, source):
        # Graph rebuilds must see a stable source snapshot without acquiring one
        # PostgreSQL advisory lock for every indexed document.
        self.con.run("select pg_advisory_xact_lock_shared(hashtext('aitwin-twinctx-graph-v1'))")
        self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-source-v1'), hashtext(:s))", s=source)

    def source_revision(self, source):
        """Snapshot source SHA and tuple version, or None when absent."""
        with self._lock:
            return self._source_revision(source)

    def _source_revision(self, source):
        rows = self.con.run("select sha,xmin::text from twinctx_v1.sources where source=:source", source=source)
        return tuple(rows[0]) if rows else None

    @staticmethod
    def _source_request_hash(action, source, expected, *, corpus=None, sha=None, atoms=(), updated=""):
        digest = hashlib.sha256(b"aitwin-source-operation-v1\0")
        header = {
            "action": action,
            "source": source,
            "expected": expected,
            "corpus": corpus,
            "sha": sha,
            "updated": updated,
        }
        for value in chain((header,), ({c: getattr(atom, c) for c in COLS} for atom in atoms)):
            encoded = json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
            digest.update(struct.pack("!Q", len(encoded)))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _validate_source_operation(source, expected, operation_id):
        if not isinstance(source, str) or not source or "\x00" in source:
            raise ValueError("valid source required")
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or "\x00" in operation_id
            or len(operation_id.encode()) > 200
        ):
            raise ValueError("valid operation id required")
        if expected is not None and (
            not isinstance(expected, tuple)
            or len(expected) != 2
            or (expected[0] is not None and not isinstance(expected[0], str))
            or not isinstance(expected[1], str)
            or not expected[1].isdecimal()
        ):
            raise ValueError("invalid expected source revision")

    def _source_operation(self, source, expected, operation_id, request_hash, apply):
        with self._transaction():
            self._lock_source(source)
            previous = self.con.run(
                "select request_hash from twinctx_v1.source_operations where operation_id=:id", id=operation_id
            )
            if previous:
                if previous != [[request_hash]]:
                    raise SourceOperationConflictError("source operation id reused with different request")
                return "already_applied"
            if self._source_revision(source) != expected:
                return "conflict"
            apply()
            self.con.run(
                "insert into twinctx_v1.source_operations(operation_id,request_hash,source) values(:id,:hash,:source)",
                id=operation_id,
                hash=request_hash,
                source=source,
            )
            return "applied"

    def _delete_knowledge(self, source):
        ids = "select id from twinctx_v1.atoms where source=:s and kind='knowledge'"
        removed = int(self.con.run(f"select count(*) from ({ids}) t", s=source)[0][0])
        self.con.run(f"delete from twinctx_v1.edges where src in ({ids}) or dst in ({ids})", s=source)
        self.con.run(f"delete from twinctx_v1.vectors where atom_id in ({ids})", s=source)
        self.con.run("delete from twinctx_v1.atoms where source=:s and kind='knowledge'", s=source)
        self._invalidate_fts_stats(changed=removed or 1)

    def replace_source(self, source: str, corpus: str, sha: str, atoms: list[Atom], updated: str = "") -> None:
        if not isinstance(source, str) or not source or not isinstance(corpus, str):
            raise ValueError("valid source and corpus required")
        if any(not isinstance(a, Atom) or a.source != source or a.kind != "knowledge" for a in atoms):
            raise ValueError("source atoms must be knowledge belonging to the same source")
        with self._transaction():
            self._lock_source(source)
            self._write_source(source, corpus, sha, atoms, updated)

    def replace_sources(self, rows: list[tuple[str, str, str, list[Atom], str]]) -> None:
        """Replace the entire connector batch in one transaction, in input order."""
        if not isinstance(rows, (list, tuple)):
            raise ValueError("batch must be a finite sequence")
        for source, corpus, _sha, atoms, _updated in rows:
            if not isinstance(source, str) or not source or not isinstance(corpus, str):
                raise ValueError("valid source and corpus required")
            if not isinstance(atoms, (list, tuple)):
                raise ValueError("source atoms must be a finite sequence")
            if any(not isinstance(a, Atom) or a.source != source or a.kind != "knowledge" for a in atoms):
                raise ValueError("source atoms must be knowledge belonging to the same source")
        if not rows:
            return
        with self._transaction():
            # Every individual source writer takes this same lock in shared
            # mode before acquiring a source lock. Taking it exclusively once
            # preserves atomic batches without one PG lock per source.
            self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-graph-v1'))")
            for source, corpus, sha, atoms, updated in rows:
                self._write_source(source, corpus, sha, atoms, updated)

    def replace_sources_if_current(self, rows, *, expected, operation_id):
        """Atomically replace a guarded batch and record its replay identity."""
        if not isinstance(rows, (list, tuple)) or not rows or len(rows) > 250 or not isinstance(expected, dict):
            raise ValueError("nonempty source batch and expected revisions required")
        prepared = []
        names = set()
        for source, corpus, sha, atoms, updated in rows:
            if source in names or source not in expected:
                raise ValueError("batch sources must be unique and have expected revisions")
            self._validate_source_operation(source, expected[source], operation_id)
            if not isinstance(corpus, str) or not isinstance(sha, str) or not isinstance(atoms, (list, tuple)):
                raise ValueError("valid corpus, sha, and source atoms required")
            if any(not isinstance(atom, Atom) or atom.source != source or atom.kind != "knowledge" for atom in atoms):
                raise ValueError("source atoms must be knowledge belonging to the same source")
            names.add(source)
            copied = [Atom(**{column: getattr(atom, column) for column in COLS}) for atom in atoms]
            prepared.append((source, corpus, sha, copied, updated))
        if set(expected) != names:
            raise ValueError("expected revisions must match batch sources")
        captured = {name: expected[name] for name in names}
        digest = hashlib.sha256(b"aitwin-guarded-source-batch-v1\0")
        for source, corpus, sha, atoms, updated in prepared:
            digest.update(
                bytes.fromhex(
                    self._source_request_hash(
                        "replace", source, captured[source], corpus=corpus, sha=sha, atoms=atoms, updated=updated
                    )
                )
            )
        request_hash = digest.hexdigest()
        with self._transaction():
            self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-graph-v1'))")
            previous = self.con.run(
                "select request_hash from twinctx_v1.source_operations where operation_id=:id", id=operation_id
            )
            if previous:
                if previous != [[request_hash]]:
                    raise SourceOperationConflictError("source operation id reused with different request")
                return "already_applied"
            found = self.con.run(
                "select source,sha,xmin::text from twinctx_v1.sources where source=any(cast(:sources as text[]))",
                sources=list(names),
            )
            current = {source: (sha, version) for source, sha, version in found}
            if any(current.get(source) != captured[source] for source in names):
                return "conflict"
            for source, corpus, sha, atoms, updated in prepared:
                self._write_source(source, corpus, sha, atoms, updated)
            self.con.run(
                "insert into twinctx_v1.source_operations(operation_id,request_hash,source) "
                "values(:id,:hash,'__batch__')",
                id=operation_id,
                hash=request_hash,
            )
            return "applied"

    def _write_source(self, source, corpus, sha, atoms, updated):
        cols = ",".join(COLS) + ",text_raw"
        binds = ",".join(":" + c for c in COLS) + ",:text_raw"
        update = ",".join(f"{c}=excluded.{c}" for c in COLS if c != "id") + ",text_raw=excluded.text_raw"
        self._delete_knowledge(source)
        bulk_min = int(_env_number("TWINCTX_BULK_MIN_ATOMS", 50))
        chunk_size = int(_env_number("TWINCTX_BULK_CHUNK", 200))
        if len(atoms) >= bulk_min:
            # Волна 2: партия едет чанками по одному INSERT на чанк — вместо round-trip на атом.
            # Identity-guard прежний: on conflict .. where пропускает чужие id, returning их не
            # возвращает, недобор строк = ValueError (транзакция откатывается целиком).
            for start in range(0, len(atoms), chunk_size):
                chunk = atoms[start : start + chunk_size]
                values = []
                params: dict = {}
                for i, item in enumerate(chunk):
                    row = {c: getattr(item, c) for c in COLS}
                    row["text"], row["text_raw"] = project_text(item.text)
                    values.append("(" + ",".join(f":{c}_{i}" for c in COLS) + ",:text_raw_" + str(i) + ")")
                    for c in COLS:
                        params[f"{c}_{i}"] = row[c]
                    params[f"text_raw_{i}"] = row["text_raw"]
                result = self.con.run(
                    f"""insert into twinctx_v1.atoms({cols}) values {",".join(values)}
                    on conflict(id) do update set {update}
                    where twinctx_v1.atoms.source=excluded.source and twinctx_v1.atoms.kind='knowledge'
                    returning id""",
                    **params,
                )
                if len(result) != len(chunk):
                    raise ValueError("atom identity belongs to another source or memory")
        else:
            for item in atoms:
                params = {c: getattr(item, c) for c in COLS}
                params["text"], params["text_raw"] = project_text(item.text)
                result = self.con.run(
                    f"""insert into twinctx_v1.atoms({cols}) values({binds})
                    on conflict(id) do update set {update}
                    where twinctx_v1.atoms.source=excluded.source and twinctx_v1.atoms.kind='knowledge'
                    returning id""",
                    **params,
                )
                if not result:
                    raise ValueError("atom identity belongs to another source or memory")
        self.con.run(
            """insert into twinctx_v1.sources(source,corpus,sha,indexed_at,updated)
            values(:s,:c,:h,:i,:u) on conflict(source) do update set corpus=excluded.corpus,
            sha=excluded.sha,indexed_at=excluded.indexed_at,updated=excluded.updated""",
            s=source,
            c=corpus,
            h=sha,
            i=now(),
            u=updated,
        )
        self._invalidate_fts_stats(changed=len(atoms))

    def replace_source_if_current(self, source, corpus, sha, atoms, updated="", *, expected, operation_id):
        self._validate_source_operation(source, expected, operation_id)
        if not isinstance(corpus, str) or not isinstance(sha, str):
            raise ValueError("valid corpus and sha required")
        if any(not isinstance(a, Atom) or a.source != source or a.kind != "knowledge" for a in atoms):
            raise ValueError("source atoms must be knowledge belonging to the same source")
        request_hash = self._source_request_hash(
            "replace", source, expected, corpus=corpus, sha=sha, atoms=atoms, updated=updated
        )
        return self._source_operation(
            source,
            expected,
            operation_id,
            request_hash,
            lambda: self._write_source(source, corpus, sha, atoms, updated),
        )

    def drop_source_if_current(self, source, *, expected, operation_id):
        self._validate_source_operation(source, expected, operation_id)
        request_hash = self._source_request_hash("drop", source, expected)

        def drop():
            self._delete_knowledge(source)
            self.con.run("delete from twinctx_v1.sources where source=:source", source=source)

        return self._source_operation(source, expected, operation_id, request_hash, drop)

    def drop_source(self, source: str) -> None:
        with self._transaction():
            self._lock_source(source)
            self._delete_knowledge(source)
            self.con.run("delete from twinctx_v1.sources where source=:s", s=source)

    def drop_sources(self, sources: list[str]) -> None:
        """Remove knowledge and its derived data for all sources atomically."""
        if not isinstance(sources, (list, tuple)):
            raise ValueError("source batch must be a finite sequence")
        if not sources:
            return
        if any(not isinstance(source, str) or "\x00" in source for source in sources):
            raise ValueError("valid source list required")
        sources = list(dict.fromkeys(sources))
        with self._transaction():
            self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-graph-v1'))")
            ids = "select id from twinctx_v1.atoms where source=any(cast(:sources as text[])) and kind='knowledge'"
            removed = int(self.con.run(f"select count(*) from ({ids}) t", sources=sources)[0][0])
            self.con.run(f"delete from twinctx_v1.edges where src in ({ids}) or dst in ({ids})", sources=sources)
            self.con.run(f"delete from twinctx_v1.vectors where atom_id in ({ids})", sources=sources)
            self.con.run(
                "delete from twinctx_v1.atoms where source=any(cast(:sources as text[])) and kind='knowledge'",
                sources=sources,
            )
            self.con.run("delete from twinctx_v1.sources where source=any(cast(:sources as text[]))", sources=sources)
            self._invalidate_fts_stats(changed=removed or 1)

    def source_rows(self, source: str) -> list[tuple[str, str]]:
        with self._lock:
            rows = self.con.run("select id,text,text_raw from twinctx_v1.atoms where source=:s order by ord", s=source)
            return [(row[0], restore_text(row[1], row[2])) for row in rows]

    def first_atom_id(self, source: str) -> str | None:
        with self._lock:
            rows = self.con.run("select id from twinctx_v1.atoms where source=:s order by ord limit 1", s=source)
            return rows[0][0] if rows else None

    def source_sha(self, source: str) -> str | None:
        with self._lock:
            rows = self.con.run("select sha from twinctx_v1.sources where source=:s", s=source)
            return rows[0][0] if rows else None

    def sources(self, corpus: str | None = None) -> list[str]:
        with self._lock:
            if corpus:
                rows = self.con.run("select source from twinctx_v1.sources where corpus=:c", c=corpus)
            else:
                rows = self.con.run("select source from twinctx_v1.sources")
            return [r[0] for r in rows]

    def by_source(self, source: str) -> list[Atom]:
        with self._lock:
            rows = self.con.run(
                f"select {','.join(COLS)},text_raw from twinctx_v1.atoms where source=:s order by ord", s=source
            )
            return [self._atom(row) for row in rows]

    def memories(self) -> list[Atom]:
        """Return the memory index, preserving nullable fields and validity history.

        This only reads the index; it does not provide journal/canon writes.
        """
        with self._lock:
            rows = self.con.run(
                f"select {','.join(COLS)},text_raw from twinctx_v1.atoms where kind='memory' "
                'order by valid_from collate "C" asc nulls first, id collate "C"'
            )
            atoms = []
            for row in rows:
                values = dict(zip(COLS, row[:-1]))
                values["text"] = restore_text(values["text"], row[-1])
                atoms.append(Atom(**values))
            return atoms

    def stats(self) -> dict:
        """Obtain all counts in one PostgreSQL statement and one snapshot."""
        with self._lock:
            row = self.con.run(
                """select
                    (select count(*) from twinctx_v1.atoms),
                    (select count(*) from twinctx_v1.atoms where kind='memory'),
                    (select count(*) from twinctx_v1.sources),
                    (select count(*) from twinctx_v1.edges),
                    (select count(*) from twinctx_v1.vectors),
                    coalesce((select jsonb_object_agg(kind,n)::text from
                        (select kind,count(*) as n from twinctx_v1.edges group by kind) x), '{}'),
                    coalesce((select jsonb_object_agg(corpus,n)::text from
                        (select corpus,count(*) as n from twinctx_v1.atoms group by corpus) x), '{}')"""
            )[0]
            return {
                "atoms": row[0],
                "memory": row[1],
                "sources": row[2],
                "edges": row[3],
                "edges_by_kind": json.loads(row[5]),
                "vectors": row[4],
                "by_corpus": json.loads(row[6]),
            }
