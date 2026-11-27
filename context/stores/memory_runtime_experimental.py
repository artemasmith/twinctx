"""Opt-in memory coordination prototype; external writers must be fenced.

This does not establish fencing for Obsidian, git, old app processes or other
hosts. It requires an existing trusted canonical file and local filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from twinctx.contracts import COLS, Atom

from .memory_canon_recovery import CanonConflictError, reconstruct_canon
from .memory_closure_protocol import _fields, _lock, _sync_published


@contextmanager
def _guard_locked(store, canon):
    """Match standalone close: SQLite first, then the stable sibling lock."""
    if store.db.in_transaction:
        raise RuntimeError('memory operation requires an idle connection')
    lock_fd = None
    try:
        store.db.execute('pragma synchronous=FULL')
        store.db.execute('begin immediate')
        lock_fd = os.open(str(canon) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        info = os.fstat(lock_fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077):
            raise CanonConflictError('invalid journal lock')
        _lock(lock_fd)
        info = canon.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
            raise CanonConflictError('untrusted journal file')
        yield
        store.db.commit()
    except BaseException:
        store.db.rollback()
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


@contextmanager
def _guard(store, canon):
    # sqlite3 connections are shared in some callers; serialize their memory
    # transactions within this process as well as across processes.
    with store._memory_thread_lock:
        with _guard_locked(store, canon):
            yield


def _row(store, identity):
    row = store.db.execute(f"select {','.join(COLS)} from atoms where id=?", (identity,)).fetchone()
    return Atom(**dict(zip(COLS, row))) if row else None


def _publish(canon, atom, checkpoint):
    info = canon.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
        raise CanonConflictError('untrusted journal file')
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix='.memory-add-', dir=canon.parent)
        with os.fdopen(fd, 'wb') as out, canon.open('rb') as source:
            current = os.fstat(source.fileno())
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise CanonConflictError('journal changed during preparation')
            os.fchown(out.fileno(), info.st_uid, info.st_gid)
            os.fchmod(out.fileno(), stat.S_IMODE(info.st_mode))
            shutil.copyfileobj(source, out, 1024 * 1024)
            out.write((json.dumps(_fields(atom), ensure_ascii=False, sort_keys=True) + '\n').encode())
            out.flush()
            os.fsync(out.fileno())
        # Validate the complete candidate before publication, including size,
        # field types and supersede graph; a bad atom must not poison the canon.
        candidate = reconstruct_canon(Path(tmp))
        if atom.id not in candidate or _fields(candidate[atom.id]) != _fields(atom):
            raise CanonConflictError('candidate memory is not reproducible')
        if checkpoint:
            checkpoint('after_file_sync')
        now = canon.lstat()
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        ):
            raise CanonConflictError('journal changed during preparation')
        os.replace(tmp, canon)
        tmp = None
        if checkpoint:
            checkpoint('after_rename')
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def add_memory(store, canon, atom, *, _checkpoint=None):
    """Publish or reconcile one exact memory fact, refusing divergent history."""
    canon = Path(canon)
    if not isinstance(atom, Atom) or atom.kind != 'memory' or not atom.id or atom.valid_to:
        raise ValueError('an open memory atom with id is required')
    if atom.supersedes and (atom.supersedes == atom.id or not atom.valid_from):
        raise ValueError('supersede requires a distinct predecessor and validity start')
    with _guard(store, canon):
        state = reconstruct_canon(canon)
        indexed = {entry.id: entry for entry in store.memories()}
        # Only the operation being retried may be present in the journal but
        # absent from the index; never let an unrelated write hide an earlier
        # unknown outcome. A pending supersede may close its own predecessor.
        if set(indexed) - set(state) or (set(state) - set(indexed) - {atom.id}):
            raise CanonConflictError('unreconciled memory operation exists')
        for identity in set(state) & set(indexed) - {atom.id}:
            left, right = _fields(state[identity]), _fields(indexed[identity])
            if identity == atom.supersedes and atom.id in state:
                canon_end, index_end = left.pop('valid_to'), right.pop('valid_to')
                if canon_end != atom.valid_from or index_end not in ('', None, atom.valid_from):
                    raise CanonConflictError('unreconciled predecessor closure')
            if left != right:
                raise CanonConflictError('unreconciled memory content')
        existing_canon = state.get(atom.id)
        existing_index = _row(store, atom.id)
        if existing_canon and _fields(existing_canon) != _fields(atom):
            raise CanonConflictError('memory identity differs from canon')
        if existing_index and _fields(existing_index) != _fields(atom):
            raise CanonConflictError('memory identity differs from index')
        if existing_index:
            fts = store.db.execute('select title,heading,text from atoms_fts where atom_id=?', (atom.id,)).fetchall()
            if [tuple(row) for row in fts] != [(atom.title, atom.heading, atom.text)]:
                raise CanonConflictError('index search projection differs')
        if existing_index and not existing_canon:
            raise CanonConflictError('memory exists only in index')
        if not existing_index and store.db.execute(
            'select 1 from atoms_fts where atom_id=? limit 1', (atom.id,)
        ).fetchone():
            raise CanonConflictError('orphan search entry requires reconciliation')
        if atom.supersedes:
            previous_canon = state.get(atom.supersedes)
            previous_index = _row(store, atom.supersedes)
            if not previous_canon or not previous_index or previous_index.kind != 'memory':
                raise CanonConflictError('predecessor is missing')
            left, right = _fields(previous_canon), _fields(previous_index)
            canon_end, index_end = left.pop('valid_to'), right.pop('valid_to')
            if left != right or index_end not in ('', None, atom.valid_from):
                raise CanonConflictError('predecessor differs')
            expected_canon_end = atom.valid_from if existing_canon else ''
            if (canon_end or '') != expected_canon_end:
                raise CanonConflictError('predecessor is closed or conflicts')
        if not existing_canon:
            _publish(canon, atom, _checkpoint)
        _sync_published(canon)
        if _checkpoint:
            _checkpoint('after_canon_publish')
        if atom.supersedes and not previous_index.valid_to:
            store.db.execute('update atoms set valid_to=? where id=?', (atom.valid_from, atom.supersedes))
        if not existing_index:
            store._insert(atom)
    if _checkpoint:
        _checkpoint('after_db_commit')


def import_canon(store, canon):
    """Import one locked canonical snapshot; never overwrite differing rows."""
    canon = Path(canon)
    with _guard(store, canon):
        state = reconstruct_canon(canon)
        existing = {a.id: a for a in store.memories()}
        if set(existing) - set(state):
            raise CanonConflictError('index contains memory absent from canon')
        for identity, atom in state.items():
            current = _row(store, identity)
            if current and _fields(current) != _fields(atom):
                raise CanonConflictError('index has a conflicting atom')
            if not current and store.db.execute(
                'select 1 from atoms_fts where atom_id=? limit 1', (identity,)
            ).fetchone():
                raise CanonConflictError('orphan search entry requires reconciliation')
        for identity, atom in state.items():
            if identity not in existing:
                store._insert(atom)
        for identity, atom in state.items():
            fts = store.db.execute('select title,heading,text from atoms_fts where atom_id=?', (identity,)).fetchall()
            if [tuple(row) for row in fts] != [(atom.title, atom.heading, atom.text)]:
                raise CanonConflictError('index search projection differs')
        _sync_published(canon)
        return len(state)


def export_canon(store, canon, destination):
    """Check agreement before publishing a new file; never truncate an old file."""
    canon, destination = Path(canon), Path(destination)
    with _guard(store, canon):
        state = reconstruct_canon(canon)
        current = {a.id: a for a in store.memories()}
        if set(state) != set(current) or any(_fields(state[k]) != _fields(current[k]) for k in state):
            raise CanonConflictError('canon and index differ; cannot export')
        _sync_published(canon)
        if destination.absolute() == canon.absolute():
            return len(state)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, 'wb') as output, canon.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            # An incomplete export remains visible for operator reconciliation.
            raise
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return len(state)
