"""Experimental offline standalone-close protocol; not connected to runtime.

All writers, including external clients, MUST be fenced for the entire call.
The caller provides a trusted local filesystem, paths and a verified backup.
This module does not fence legacy writers or claim distributed atomicity.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from twinctx.contracts import COLS, Atom

from .memory_canon_recovery import CanonConflictError, open_recovery_store, reconstruct_canon


def _fields(atom):
    return {key: getattr(atom, key) for key in COLS}


def _sync_published(path):
    """Retry uncertain publication durability before confirming SQLite."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _lock(fd, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('memory journal lock timed out')
            time.sleep(min(remaining, 0.05))


def close_memory_offline(canon_path, db_path, atom_id, valid_to, *, offline=False, _checkpoint=None):
    """Close one matching fact with legacy-compatible JSONL and crash reconciliation.

    Only a single-host, fenced offline operation is supported. The journal is
    published first; if the process dies before SQLite commit, retrying the
    identical close reconciles the index without appending a duplicate. A
    different close or divergent content is rejected. Existing missing or
    corrupted histories require manual reconciliation, not automatic guessing.
    """
    if offline is not True:
        raise RuntimeError('offline writer fencing required')
    if any(not isinstance(x, str) or not x or '\x00' in x for x in (atom_id, valid_to)):
        raise ValueError('memory id and validity end must be nonempty text')
    path, db_path = Path(canon_path), Path(db_path)
    for item in (path, db_path):
        info = item.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CanonConflictError('expected unique regular file')
    store = open_recovery_store(db_path)
    lock_fd = None
    pending = None
    try:
        store.db.execute('pragma synchronous=FULL')
        store.db.execute('begin immediate')
        lock_fd = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock_stat = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1
                or lock_stat.st_uid != os.geteuid() or stat.S_IMODE(lock_stat.st_mode) & 0o077):
            raise CanonConflictError('invalid journal lock')
        _lock(lock_fd)
        canonical = reconstruct_canon(path)
        indexed = {entry.id: entry for entry in store.memories()}
        if set(canonical) != set(indexed):
            raise CanonConflictError('unreconciled memory operation exists')
        if any(_fields(canonical[key]) != _fields(indexed[key]) for key in canonical if key != atom_id):
            raise CanonConflictError('unreconciled memory content')
        old = canonical.get(atom_id)
        row = store.db.execute(f"select {','.join(COLS)} from atoms where id=?", (atom_id,)).fetchone()
        if old is None or row is None:
            raise CanonConflictError('memory is missing from canon or index')
        current = Atom(**dict(zip(COLS, row)))
        a, b = _fields(old), _fields(current)
        canon_end, index_end = a.pop('valid_to'), b.pop('valid_to')
        if old.kind != 'memory' or a != b:
            raise CanonConflictError('memory content differs between canon and index')
        if canon_end and canon_end != valid_to or index_end and index_end != valid_to:
            raise CanonConflictError('memory has a different closure')
        if not canon_end and index_end:
            raise CanonConflictError('index closure absent from canon requires reconciliation')
        if canon_end == valid_to and index_end == valid_to:
            _sync_published(path)
            store.db.commit()
            return 'already_closed'
        if not canon_end:
            original = path.lstat()
            if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
                raise CanonConflictError('journal changed type or links')
            fd, pending = tempfile.mkstemp(prefix='.memory-close-', dir=path.parent)
            with os.fdopen(fd, 'wb') as out, open(path, 'rb') as source:
                opened = os.fstat(source.fileno())
                if (opened.st_dev, opened.st_ino) != (original.st_dev, original.st_ino):
                    raise CanonConflictError('journal changed during preparation')
                if opened.st_uid != os.geteuid():
                    raise CanonConflictError('journal owner differs from writer')
                os.fchown(out.fileno(), opened.st_uid, opened.st_gid)
                os.fchmod(out.fileno(), stat.S_IMODE(opened.st_mode))
                shutil.copyfileobj(source, out, 1024 * 1024)
                closed = replace(old, valid_to=valid_to)
                out.write((json.dumps(_fields(closed), ensure_ascii=False, sort_keys=True) + '\n').encode())
                out.flush()
                os.fsync(out.fileno())
            if _checkpoint:
                _checkpoint('after_file_sync')
            observed = path.lstat()
            if (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns) != (
                original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns
            ):
                raise CanonConflictError('journal changed during preparation')
            os.replace(pending, path)
            pending = None
            if _checkpoint:
                _checkpoint('after_rename')
        _sync_published(path)
        if _checkpoint:
            _checkpoint('after_canon_publish')
        result = store.db.execute(
            "update atoms set valid_to=? where id=? and kind='memory' and (valid_to='' or valid_to is null)",
            (valid_to, atom_id),
        )
        if result.rowcount != 1:
            raise CanonConflictError('memory changed during close')
        store.db.commit()
        if _checkpoint:
            _checkpoint('after_db_commit')
        return 'closed'
    except BaseException:
        store.db.rollback()
        raise
    finally:
        if pending:
            try:
                os.unlink(pending)
            except FileNotFoundError:
                pass
        if lock_fd is not None:
            os.close(lock_fd)
        store.db.close()
