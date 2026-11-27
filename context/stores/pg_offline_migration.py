"""Experimental offline assembly; never authorizes a runtime cutover.

Caller must use an isolated, identified destination with all writers fenced,
and independently verify backups and recovery. A failed run may leave a partial
destination and must not be blindly retried. The source is never modified.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import sqlite3
import stat
import struct
import time
from contextlib import closing
from pathlib import Path

from .pg_bulk import _connection, _transaction, _verify_identity, copy_vectors
from .pg_record_copy import copy_records
from .pg_record_verify import verify_records
from .pg_vector_verify import verify_snapshot

_MAX_META_FIELD = 1024 * 1024
_MAX_META_ROWS = 100000


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _create_snapshot(source_path, snapshot_path, max_snapshot_bytes, snapshot_timeout_seconds, min_free_after_bytes):
    source_path = Path(source_path).resolve(strict=True)
    snapshot_path = Path(snapshot_path)
    parent = snapshot_path.parent.resolve(strict=True)
    info = parent.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("snapshot directory must be private and owned by the current user")
    if shutil.disk_usage(parent).free < max_snapshot_bytes + min_free_after_bytes:
        raise RuntimeError("insufficient disk space for bounded snapshot and reserve")
    snapshot_path = parent / snapshot_path.name
    if source_path == snapshot_path:
        raise ValueError("snapshot must differ from source")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(snapshot_path, flags, 0o600)
    os.close(fd)
    try:
        uri = source_path.as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as source, closing(sqlite3.connect(snapshot_path)) as dest:
            source.execute("pragma query_only=on")
            page_size = source.execute("pragma page_size").fetchone()[0]
            deadline = time.monotonic() + snapshot_timeout_seconds

            def progress(_status, _remaining, total):
                if time.monotonic() > deadline:
                    raise TimeoutError("SQLite snapshot exceeded time limit")
                if total * page_size > max_snapshot_bytes:
                    raise ValueError("SQLite snapshot exceeds size limit")

            source.backup(dest, pages=256, sleep=0.01, progress=progress)
            dest.execute("pragma journal_mode=delete")
            dest.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            try:
                integrity = dest.execute("pragma quick_check").fetchall()
            except sqlite3.OperationalError:
                if time.monotonic() > deadline:
                    raise TimeoutError("SQLite snapshot integrity check exceeded time limit") from None
                raise
            finally:
                dest.set_progress_handler(None, 0)
            if integrity != [("ok",)]:
                raise ValueError("SQLite snapshot integrity check failed")
        os.chmod(snapshot_path, 0o400)
        with open(snapshot_path, "rb") as snapshot:
            os.fsync(snapshot.fileno())
        return snapshot_path, _file_hash(snapshot_path)
    except BaseException:
        snapshot_path.unlink(missing_ok=True)
        raise


def _assert_snapshot(path, expected):
    if not hmac.compare_digest(_file_hash(path), expected):
        raise RuntimeError("snapshot changed during offline migration")


def _vectors(snapshot):
    uri = snapshot.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        cursor = source.execute(
            "select case when length(cast(atom_id as blob))<=4096 then atom_id end, "
            "case when length(vec)<=4194304 then vec end, typeof(vec), length(vec) from vectors"
        )
        for key, value, kind, size in cursor:
            if kind not in ("blob", "null") or (size is not None and size > 4194304):
                raise ValueError("invalid or oversized SQLite vector")
            yield key, value


def _meta_digest(rows):
    digest = hashlib.sha256(b"aitwin-legacy-meta-v1\0")
    count = 0
    previous = None
    for key, value in rows:
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ValueError("invalid legacy meta key")
        encoded = key.encode("utf-8")
        if len(encoded) > _MAX_META_FIELD or (previous is not None and encoded <= previous):
            raise ValueError("legacy meta keys must be ordered and bounded")
        digest.update(struct.pack("!Q", len(encoded)) + encoded)
        if value is None:
            digest.update(b"n")
        elif isinstance(value, str) and "\x00" not in value:
            data = value.encode("utf-8")
            if len(data) > _MAX_META_FIELD:
                raise ValueError("legacy meta value exceeds limit")
            digest.update(b"s" + struct.pack("!Q", len(data)) + data)
        else:
            raise ValueError("invalid legacy meta value")
        previous = encoded
        count += 1
        if count > _MAX_META_ROWS:
            raise ValueError("legacy meta exceeds row limit")
    digest.update(struct.pack("!Q", count))
    return count, digest.hexdigest()


def _source_meta(snapshot):
    uri = snapshot.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        cursor = source.execute(
            "select case when length(cast(key as blob))<=1048576 then key end, "
            "case when length(cast(value as blob))<=1048576 then value end, "
            "length(cast(value as blob)) from meta order by key collate binary"
        )
        for key, value, size in cursor:
            if size is not None and size > _MAX_META_FIELD:
                raise ValueError("legacy meta value exceeds limit")
            yield key, value


def _copy_verify_meta(snapshot, dsn, expected_identity):
    with _connection(dsn, 120) as target:
        with _transaction(target):
            _verify_identity(target, expected_identity)
            # Refuse to adopt an unknown previous artifact, even if it is empty.
            target.run("create table twinctx_v1.sqlite_legacy_meta (key text primary key, value text)")
            def copied_rows():
                for key, value in _source_meta(snapshot):
                    target.run("insert into twinctx_v1.sqlite_legacy_meta(key,value) values(:k,:v)", k=key, v=value)
                    yield key, value

            expected = _meta_digest(copied_rows())
        target.run("begin isolation level repeatable read read only")
        try:
            _verify_identity(target, expected_identity)
            target.run('declare legacy_meta_check no scroll cursor for select key,value from twinctx_v1.sqlite_legacy_meta order by key collate "C"')

            def actual_rows():
                while page := target.run("fetch forward 4 from legacy_meta_check"):
                    yield from page

            actual = _meta_digest(actual_rows())
            if expected[0] != actual[0] or not hmac.compare_digest(expected[1], actual[1]):
                raise ValueError("legacy meta content verification failed")
            return actual
        finally:
            target.run("rollback")


def migrate_offline(
    source_path, snapshot_path, dsn, *, database, system_identifier, database_oid, max_snapshot_bytes,
    snapshot_timeout_seconds=300, min_free_after_bytes=1024 * 1024 * 1024, offline=False,
):
    """Prepare and verify an isolated staging destination; never enable runtime.

    Snapshot lives at a new path in a private directory and is retained even
    after successful staging. A full-file SHA check detects later modification,
    but is not a substitute for OS isolation from same-user hostile processes.
    No retry or destination deletion is attempted after partial failure.
    """
    if offline is not True:
        raise RuntimeError("offline migration opt-in is required")
    if type(max_snapshot_bytes) is not int or not 1 <= max_snapshot_bytes <= 1024 ** 4:
        raise ValueError("explicit bounded snapshot size is required")
    if type(snapshot_timeout_seconds) is not int or not 1 <= snapshot_timeout_seconds <= 3600:
        raise ValueError("invalid snapshot timeout")
    if type(min_free_after_bytes) is not int or not 0 <= min_free_after_bytes <= 1024 ** 4:
        raise ValueError("invalid disk reserve")
    if any(not isinstance(value, str) or not value or "\x00" in value for value in (database, system_identifier, database_oid)):
        raise ValueError("explicit destination identity is required")
    if not system_identifier.isdecimal() or not database_oid.isdecimal():
        raise ValueError("invalid PostgreSQL destination identity")
    identity = (database, system_identifier, database_oid)
    with _connection(dsn, 120) as target:
        with _transaction(target):
            _verify_identity(target, identity)
            if target.run("select to_regclass('twinctx_v1.sqlite_legacy_meta')") != [[None]]:
                raise RuntimeError("legacy meta destination already exists")
    snapshot, checksum = _create_snapshot(
        source_path, snapshot_path, max_snapshot_bytes, snapshot_timeout_seconds, min_free_after_bytes
    )
    _assert_snapshot(snapshot, checksum)
    records = copy_records(snapshot, dsn, offline=True, expected_identity=identity)
    _assert_snapshot(snapshot, checksum)
    vectors = copy_vectors(dsn, _vectors(snapshot), offline=True, expected_identity=identity)
    _assert_snapshot(snapshot, checksum)
    meta = _copy_verify_meta(snapshot, dsn, identity)
    _assert_snapshot(snapshot, checksum)
    verified_records = verify_records(snapshot, dsn, offline=True, expected_identity=identity)
    verified_vectors = verify_snapshot(snapshot, dsn, offline=True, expected_identity=identity)
    _assert_snapshot(snapshot, checksum)
    return {
        "snapshot_sha256": checksum,
        "records": records.counts,
        "vectors": vectors.rows,
        "meta": meta[0],
        "verified_records": {name: result.rows for name, result in verified_records.items()},
        "verified_vectors": verified_vectors.rows,
        "runtime_cutover_authorized": False,
    }
