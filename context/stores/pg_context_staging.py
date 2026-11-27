"""Offline staging of a context snapshot and its memory canon into PostgreSQL.

The caller must fence all writers and use a dedicated empty database. This
operation never selects the runtime store or starts any service.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

from .memory_canon_recovery import reconstruct_canon
from .pg_bulk import _connection, _transaction, _verify_identity
from .pg_memory_projection import PgMemoryProjection
from .pg_migration_watchdog import migrate_offline_guarded
from .pg_sources import PgSourceIndex


def _canon_digest(path):
    parent = Path(path).parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) & 0o077):
        raise ValueError('canonical snapshot directory must be private and owned by the operator')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 128 * 1024 * 1024):
            raise ValueError('canonical snapshot must be a private regular file owned by the operator')
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
        return digest.hexdigest()


def stage_context(source, snapshot, canon_snapshot, migration_dsn, runtime_dsn, *, identity,
                  postgres_storage_paths, max_snapshot_bytes, min_free_bytes,
                  snapshot_timeout_seconds=3600, offline=False):
    """Copy, independently verify, and register matching canonical memories.

    An error may leave a partial target; the caller must inspect it rather than
    blindly retry. The source and canonical snapshot are never changed.
    """
    if offline is not True:
        raise ValueError('offline writer fencing is required')
    if type(max_snapshot_bytes) is not int or not 1 <= max_snapshot_bytes <= 1024**4:
        raise ValueError('explicit bounded snapshot size is required')
    if type(min_free_bytes) is not int or not 1 <= min_free_bytes <= 1024**4:
        raise ValueError('positive bounded disk reserve is required')
    if type(snapshot_timeout_seconds) is not int or not 1 <= snapshot_timeout_seconds <= 3600:
        raise ValueError('invalid snapshot timeout')
    source_path = Path(source).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError('source must be a regular file')
    snapshot_path = Path(snapshot)
    parent = snapshot_path.parent.resolve(strict=True)
    parent_info = parent.stat()
    if (not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_uid != os.geteuid()
            or parent_info.st_mode & 0o077):
        raise ValueError('snapshot directory must be private and owned by the operator')
    if snapshot_path.exists() or snapshot_path.is_symlink() or parent / snapshot_path.name == source_path:
        raise ValueError('snapshot destination must be new and distinct')
    if not postgres_storage_paths:
        raise ValueError('PostgreSQL storage paths are required')
    storage = [Path(path).resolve(strict=True) for path in postgres_storage_paths]
    if any(not path.is_dir() for path in storage):
        raise ValueError('PostgreSQL storage paths must be directories')
    if shutil.disk_usage(parent).free < max_snapshot_bytes + min_free_bytes:
        raise RuntimeError('insufficient disk space for snapshot and reserve')
    if any(shutil.disk_usage(path).free <= min_free_bytes for path in storage):
        raise RuntimeError('insufficient disk reserve for PostgreSQL')
    canon_path = Path(canon_snapshot)
    before = _canon_digest(canon_path)
    canonical = reconstruct_canon(canon_path, allow_dangling=True)
    if _canon_digest(canon_path) != before:
        raise RuntimeError('canonical snapshot changed during validation')
    # Both credentials must point to exactly the independently identified DB.
    with _connection(migration_dsn, 30) as con:
        with _transaction(con):
            _verify_identity(con, identity)
    with _connection(runtime_dsn, 30) as con:
        with _transaction(con):
            _verify_identity(con, identity)
            if con.run("select exists(select 1 from pg_namespace where nspname='twinctx_v1')") != [[False]]:
                raise ValueError('empty target without a context schema is required')
    initialized = PgSourceIndex(runtime_dsn, experimental=True)
    try:
        with initialized._transaction():
            _verify_identity(initialized.con, identity)
    finally:
        initialized.close()
    report = migrate_offline_guarded(
        source_path=source, snapshot_path=snapshot, dsn=migration_dsn,
        database=identity[0], system_identifier=identity[1], database_oid=identity[2],
        max_snapshot_bytes=max_snapshot_bytes, postgres_storage_paths=postgres_storage_paths,
        min_free_bytes=min_free_bytes, snapshot_timeout_seconds=snapshot_timeout_seconds, offline=True,
    )
    if _canon_digest(canon_path) != before:
        raise RuntimeError('canonical snapshot changed during migration')
    if any(shutil.disk_usage(path).free <= min_free_bytes for path in [parent, *storage]):
        raise RuntimeError('insufficient disk reserve before memory reconciliation')
    store = PgSourceIndex(runtime_dsn, experimental=True)
    try:
        with store._transaction():
            _verify_identity(store.con, identity)
        count = PgMemoryProjection(store, experimental=True).bootstrap_legacy(canonical, offline=True)
    finally:
        store.close()
    if _canon_digest(canon_path) != before:
        raise RuntimeError('canonical snapshot changed during reconciliation')
    return {**report, 'verified_memory': count,
            'legacy_dangling': sum(bool(a.supersedes and a.supersedes not in canonical)
                                   for a in canonical.values()),
            'canonical_sha256': before, 'runtime_cutover_authorized': False}
