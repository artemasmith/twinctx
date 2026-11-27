"""Векторный слой PG-индекса: партии вставки, курсорная итерация, коннект-реестр.

Вынесено из pg_sources.py волной 5 техдолга без изменения поведения.
"""

from twinctx.contracts import COLS, Atom

from .pg_text import project_text

_VECTOR_BATCH_SIZE = 256


_VECTOR_PAGE_MAX = 2048


class PgVectorsMixin:
    def missing_vector_atoms(self, ids):
        """Return current knowledge atoms with no vector for retry/backfill."""
        if not ids:
            return []
        with self._lock:
            rows = self.con.run(
                f"select {','.join('a.' + c for c in COLS)}, a.text_raw from twinctx_v1.atoms a "
                "where a.id=any(cast(:ids as text[])) and a.kind='knowledge' "
                "and not exists (select 1 from twinctx_v1.vectors v where v.atom_id=a.id)",
                ids=ids,
            )
            return [self._atom(row) for row in rows]

    def _vector_atom_versions(self, ids: list[str]) -> dict[str, tuple[str, str]]:
        """Read source and tuple version before and after acquiring source locks."""
        found = {}
        for offset in range(0, len(ids), _VECTOR_BATCH_SIZE):
            part = ids[offset : offset + _VECTOR_BATCH_SIZE]
            rows = self.con.run(
                "select id, source, xmin::text from twinctx_v1.atoms where id=any(cast(:ids as text[]))",
                ids=part,
            )
            found.update((row[0], (row[1], row[2])) for row in rows)
        return found

    def put_vectors(self, rows: list[tuple[str, bytes]]) -> None:
        """Upsert an entire batch atomically, serialized with source replacement.

        SQLite permits vectors with no atom, so this slice preserves that behavior.
        Existing atom versions must not change while this write waits for its locks.
        The API cannot validate embeddings computed before the call without a
        caller-supplied source/atom revision; runtime integration remains blocked.
        """
        if not rows:
            return
        with self._transaction():
            ids = []
            for row in rows:
                if len(row) != 2 or not isinstance(row[0], str):
                    raise ValueError("invalid vector row")
                if row[1] is not None and not isinstance(row[1], (bytes, bytearray, memoryview)):
                    raise ValueError("invalid vector blob")
                ids.append(row[0])
            initial = self._vector_atom_versions(ids)
            sources = list({value[0] for value in initial.values()})
            if sources:
                ordered = self.con.run(
                    "select source,hashtext(source) from unnest(cast(:sources as text[])) source "
                    "order by hashtext(source),source",
                    sources=sources,
                )
                for source, _ in ordered:
                    self._lock_source(source)
            if self._vector_atom_versions(ids) != initial:
                raise RuntimeError("atom changed while writing vectors")
            for offset in range(0, len(rows), _VECTOR_BATCH_SIZE):
                # PostgreSQL cannot update the same key twice within one INSERT;
                # retain SQLite's last-write-wins semantics within each batch.
                part = {
                    key: (bytes(value) if value is not None else None)
                    for key, value in rows[offset : offset + _VECTOR_BATCH_SIZE]
                }
                values = ",".join(f"(:id{i},:vec{i})" for i in range(len(part)))
                params = {}
                for i, (key, blob) in enumerate(part.items()):
                    params[f"id{i}"] = key
                    params[f"vec{i}"] = blob
                self.con.run(
                    f"insert into twinctx_v1.vectors(atom_id,vec) values {values} "
                    "on conflict(atom_id) do update set vec=excluded.vec",
                    **params,
                )

    def put_vectors_if_current(self, rows: list[tuple[Atom, bytes]]) -> int:
        """Write vectors only for passages unchanged since embedding began."""
        if not rows:
            return 0
        with self._transaction():
            sources = list({atom.source for atom, _ in rows})
            ordered = self.con.run(
                "select source,hashtext(source) from unnest(cast(:sources as text[])) source "
                "order by hashtext(source),source",
                sources=sources,
            )
            for source, _ in ordered:
                self._lock_source(source)
            written = 0
            for atom, vector in rows:
                projected, raw = project_text(atom.text)
                matches = self.con.run(
                    """select id from twinctx_v1.atoms where id=:id and source=:source and kind=:kind
                         and title is not distinct from :title and heading is not distinct from :heading
                         and text is not distinct from :text and text_raw is not distinct from :raw and valid_to is not distinct from :valid_to
                       for update""",
                    id=atom.id,
                    source=atom.source,
                    kind=atom.kind,
                    title=atom.title,
                    heading=atom.heading,
                    text=projected,
                    raw=raw,
                    valid_to=atom.valid_to,
                )
                if matches:
                    self.con.run(
                        "insert into twinctx_v1.vectors(atom_id,vec) values(:id,:vec) "
                        "on conflict(atom_id) do update set vec=excluded.vec",
                        id=atom.id,
                        vec=vector,
                    )
                    written += 1
            return written

    def atoms_without_vectors(self, limit: int) -> list[Atom]:
        if not isinstance(limit, int):
            raise ValueError("integer limit required")
        # SQLite LIMIT -1 means no limit.
        clause = "" if limit < 0 else " limit :lim"
        params = {} if limit < 0 else {"lim": int(limit)}
        with self._lock:
            rows = self.con.run(
                f"select {','.join(COLS)},text_raw from twinctx_v1.atoms a "
                "where (a.valid_to is null or a.valid_to='') "
                "and not exists (select 1 from twinctx_v1.vectors v where v.atom_id=a.id) "
                "order by a.source,a.ord,a.id" + clause,
                **params,
            )
            return [self._atom(row) for row in rows]

    def _vector_connection(self):
        import pg8000.native

        return pg8000.native.Connection(**self._connection_args)

    def iter_vectors(self, page_size: int = _VECTOR_BATCH_SIZE):
        """Yield bounded pages from one repeatable-read snapshot on a separate connection.

        Exhaust or explicitly close the iterator to promptly release its cursor,
        transaction and connection. PostgreSQL handles cleanup if the client dies.
        """
        if type(page_size) is not int or not 1 <= page_size <= _VECTOR_PAGE_MAX:
            raise ValueError("invalid vector page size")
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            conn = self._vector_connection()
            self._vector_connections.add(conn)
        try:
            conn.run("begin isolation level repeatable read read only")
            conn.run(
                "declare twinctx_vector_scan no scroll cursor for "
                "select atom_id,vec from twinctx_v1.vectors order by atom_id"
            )
            while True:
                page = conn.run(f"fetch forward {page_size} from twinctx_vector_scan")
                if not page:
                    break
                for key, blob in page:
                    yield key, bytes(blob) if blob is not None else None
        finally:
            try:
                conn.run("rollback")
            except Exception:
                pass  # A disconnected/closed session already released its transaction.
            try:
                conn.close()
            except Exception:
                if not self._closed:
                    raise
            finally:
                with self._lock:
                    self._vector_connections.discard(conn)

    def vectors(self) -> list[tuple[str, bytes]]:
        """SQLite-compatible materialized snapshot; use iter_vectors for bounded reads."""
        return list(self.iter_vectors())
