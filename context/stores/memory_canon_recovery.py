"""Offline recovery of legacy JSONL memory facts; no runtime integration.

The caller must provide an immutable, verified snapshot and fence every writer.
This tool cannot invent historical standalone closes absent from the canon.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from twinctx.contracts import COLS, Atom

_MAX_FILE = 128 * 1024 * 1024
_MAX_LINE = 1024 * 1024
_MAX_RECORDS = 100_000


class CanonConflictError(RuntimeError):
    """Canon or target is ambiguous; preserve both and reconcile explicitly."""


def open_recovery_store(path: Path):
    """Open an index for recovery without running migrations on an existing DB."""
    from .sqlite import SCHEMA, SqliteStore

    path = Path(path)
    if not path.exists():
        return SqliteStore(path)
    db = sqlite3.connect(path.absolute().as_uri() + "?mode=rw", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with sqlite3.connect(":memory:") as template:
            template.executescript(SCHEMA)
            expected = dict(template.execute(
                "select name,sql from sqlite_master where type='table' and name in ('atoms','atoms_fts')"))
        actual = dict(db.execute(
            "select name,sql from sqlite_master where type='table' and name in ('atoms','atoms_fts')"))
        if actual != expected or db.execute("select 1 from sqlite_master where type='trigger' limit 1").fetchone():
            raise CanonConflictError("unsupported recovery index schema or trigger")
        store = object.__new__(SqliteStore)
        store.db = db
        store._memory_thread_lock = threading.RLock()
        return store
    except BaseException:
        db.close()
        raise


def _columns(atom):
    return {key: getattr(atom, key) for key in COLS}


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise CanonConflictError("duplicate JSON field")
        out[key] = value
    return out


def reconstruct_canon(path: Path, *, allow_dangling: bool = False) -> dict[str, Atom]:
    """Validate legacy atom lines and reconstruct supersede closures without writing.

    Repeated IDs are allowed only when all fields except valid_to are equal and
    the validity progresses from open to closed or remains unchanged. An absent
    predecessor, inconsistent closure, split, cycle, unknown closure time,
    truncated tail, or conflicting duplicate fails without choosing a winner.
    Caller must ensure the file is an immutable consistent snapshot. An
    explicitly enabled legacy mode preserves unresolved predecessor IDs rather
    than inventing the missing facts; all existing relationships remain checked.
    """
    path = Path(path)
    if path.stat().st_size > _MAX_FILE:
        raise CanonConflictError("canon exceeds recovery size limit")
    states = {}
    total = records = 0
    with path.open("rb") as stream:
        while raw := stream.readline(_MAX_LINE + 1):
            total += len(raw)
            if total > _MAX_FILE or len(raw) > _MAX_LINE:
                raise CanonConflictError("canon exceeds recovery size limit")
            if not raw.endswith(b"\n"):
                raise CanonConflictError("incomplete canon tail")
            if not raw.strip():
                continue
            records += 1
            if records > _MAX_RECORDS:
                raise CanonConflictError("too many canon records")
            try:
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
                if not isinstance(value, dict) or set(value) - set(COLS):
                    raise CanonConflictError("invalid canon fields")
                atom = Atom(**value)
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                raise CanonConflictError("invalid canon record") from exc
            required = (atom.id, atom.source, atom.corpus, atom.kind, atom.text)
            optional_text = (atom.doc_type, atom.title, atom.heading, atom.updated, atom.observed_at,
                             atom.valid_from, atom.valid_to, atom.supersedes)
            integers = (atom.start_line, atom.end_line, atom.ord)
            if (atom.kind != "memory" or not atom.id
                    or any(not isinstance(v, str) or "\x00" in v for v in required)
                    or any(v is not None and (not isinstance(v, str) or "\x00" in v) for v in optional_text)
                    or any(v is not None and (type(v) is not int or not -(1 << 63) <= v < (1 << 63)) for v in integers)
                    or atom.trust is not None and (type(atom.trust) not in (int, float) or not math.isfinite(atom.trust))):
                raise CanonConflictError("invalid memory identity, types or validity")
            previous = states.get(atom.id)
            if previous is not None:
                a, b = _columns(previous), _columns(atom)
                old_end, new_end = a.pop("valid_to"), b.pop("valid_to")
                if a != b or (old_end and old_end != new_end):
                    raise CanonConflictError("conflicting memory versions")
            states[atom.id] = atom
    children = {}
    for atom in states.values():
        parent = atom.supersedes
        if not parent:
            continue
        if parent not in states and allow_dangling:
            continue
        if parent not in states or not atom.valid_from:
            raise CanonConflictError("missing predecessor or closure time")
        if parent in children and children[parent] != atom.id:
            raise CanonConflictError("multiple successors require reconciliation")
        children[parent] = atom.id
    finished = set()
    for start in states:
        path_seen = set()
        current = start
        while current and current in states and current not in finished:
            if current in path_seen:
                raise CanonConflictError("cyclic supersede history")
            path_seen.add(current)
            current = states[current].supersedes
        finished.update(path_seen)
    for parent_id, child_id in children.items():
        parent, child = states[parent_id], states[child_id]
        if parent.valid_to and parent.valid_to != child.valid_from:
            raise CanonConflictError("predecessor closure conflicts with successor")
        if not parent.valid_to:
            states[parent_id] = replace(parent, valid_to=child.valid_from)
    return states


def compare_index(store, state: dict[str, Atom]) -> dict[str, list[str]]:
    """Compare all persistent fields with a fenced SQLite snapshot; never modify it."""
    index = {atom.id: atom for atom in store.memories()}
    canon_ids, index_ids = set(state), set(index)
    return {
        "missing_in_canon": sorted(index_ids - canon_ids),
        "missing_in_index": sorted(canon_ids - index_ids),
        "different": sorted(i for i in canon_ids & index_ids if _columns(state[i]) != _columns(index[i])),
    }


def restore_memory(store, state: dict[str, Atom], *, offline: bool = False) -> int:
    """Restore a reconstructed snapshot atomically into a fenced SQLite index.

    Existing unrelated memory or any unequal/colliding atom ID aborts the entire
    transaction. The caller must ensure tenant identity, backup, and writer fence.
    It does not write embeddings, canon, or PostgreSQL and is not a Store API.
    """
    if offline is not True:
        raise RuntimeError("offline recovery opt-in required")
    from .sqlite import SqliteStore

    if type(store) is not SqliteStore or store.db.in_transaction:
        raise RuntimeError("dedicated idle SQLite store required")
    if not isinstance(state, dict) or any(
        not isinstance(v, Atom) or key != v.id or v.kind != "memory" for key, v in state.items()
    ):
        raise CanonConflictError("invalid reconstructed state")
    try:
        store.db.execute("begin immediate")
        existing_memory = {a.id: a for a in store.memories()}
        if set(existing_memory) - set(state):
            raise CanonConflictError("target has unrelated memory")
        ids = list(state)
        for offset in range(0, len(ids), 500):
            part = ids[offset : offset + 500]
            if not part:
                continue
            placeholders = ",".join("?" for _ in part)
            for row in store.db.execute(f"select {','.join(COLS)} from atoms where id in ({placeholders})", part):
                current = Atom(**dict(zip(COLS, row)))
                if _columns(current) != _columns(state[current.id]):
                    raise CanonConflictError("target has conflicting atom")
        for key, atom in state.items():
            if key not in existing_memory:
                store._insert(atom)
        for key, atom in state.items():
            row = store.db.execute(f"select {','.join(COLS)} from atoms where id=?", (key,)).fetchone()
            if row is None or _columns(Atom(**dict(zip(COLS, row)))) != _columns(atom):
                raise CanonConflictError("recovered memory differs from canon")
            fts = store.db.execute("select title,heading,text from atoms_fts where atom_id=?", (key,)).fetchall()
            if [tuple(row) for row in fts] != [(atom.title, atom.heading, atom.text)]:
                raise CanonConflictError("recovered memory search index differs from canon")
        store.db.commit()
        return len(state)
    except BaseException:
        store.db.rollback()
        raise
