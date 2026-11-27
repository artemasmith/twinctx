"""Offline full-content verification of vector COPY against a SQLite snapshot."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from .pg_bulk import _connection, _verify_identity

_MAX_ID_BYTES = 4096
_MAX_VECTOR_BYTES = 4 * 1024 * 1024
_PAGE_MEMORY_BUDGET = 8 * 1024 * 1024
_MAX_PAGE_ROWS = 512


def _bounded_page_size(requested, max_vector_bytes):
    # pg8000 receives bytea in text format (hex, about twice the raw size).
    # Reserve space for the maximum ID, decoded text and per-row overhead too.
    # One maximum-size row is the irreducible minimum if it exceeds budget.
    per_row = 2 * max_vector_bytes + 4 * _MAX_ID_BYTES + 1024
    return max(1, min(requested, _PAGE_MEMORY_BUDGET // per_row))


@dataclass(frozen=True)
class VectorDigest:
    rows: int
    sha256: str


def digest_vectors(rows):
    """Hash ordered rows with explicit lengths and a distinct NULL marker.

    IDs must be unique, nonempty and in UTF-8 byte order. Input iterators must
    stream bounded pages: this function retains only one row and the prior ID.
    """
    digest = hashlib.sha256(b"aitwin-vector-digest-v1\x00")
    previous = None
    count = 0
    for atom_id, vector in rows:
        if not isinstance(atom_id, str) or not atom_id or "\x00" in atom_id:
            raise ValueError("invalid vector ID")
        key = atom_id.encode("utf-8")
        if len(key) > _MAX_ID_BYTES or (previous is not None and key <= previous):
            raise ValueError("vector IDs must be unique, bounded and ordered by UTF-8 bytes")
        if vector is not None and not isinstance(vector, bytes):
            raise ValueError("vector must be bytes or NULL")
        if vector is not None and len(vector) > _MAX_VECTOR_BYTES:
            raise ValueError("vector exceeds verification limit")
        digest.update(struct.pack("!I", len(key)))
        digest.update(key)
        if vector is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            digest.update(struct.pack("!Q", len(vector)))
            digest.update(vector)
        previous = key
        count += 1
    digest.update(struct.pack("!Q", count))
    return VectorDigest(count, digest.hexdigest())


def verify_snapshot(sqlite_snapshot: Path, dsn: str, *, offline: bool = False, page_size: int = 512, expected_identity=None):
    """Compare every vector from a fenced SQLite snapshot and isolated PG target.

    The caller must establish snapshot identity, fence all writers and verify
    destination identity. The returned digest is valid only for these snapshots.
    """
    if offline is not True:
        raise RuntimeError("offline verification opt-in is required")
    if type(page_size) is not int or not 1 <= page_size <= _MAX_PAGE_ROWS:
        raise ValueError("invalid page size")
    uri = Path(sqlite_snapshot).resolve(strict=True).as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as source:
        source.execute("begin")
        if source.execute("pragma encoding").fetchone()[0].upper() != "UTF-8":
            raise ValueError("SQLite database must use UTF-8 encoding")
        cursor = source.execute(
            "select case when length(cast(atom_id as blob)) <= 4096 then atom_id end, "
            "case when length(vec) <= ? then vec end, typeof(vec), length(vec) "
            "from vectors order by atom_id collate binary",
            (_MAX_VECTOR_BYTES,),
        )

        max_vector_bytes = 0

        def source_rows():
            nonlocal max_vector_bytes
            for key, value, kind, size in cursor:
                if kind not in ("blob", "null") or (size is not None and size > _MAX_VECTOR_BYTES):
                    raise ValueError("invalid or oversized SQLite vector")
                if size is not None:
                    max_vector_bytes = max(max_vector_bytes, size)
                yield key, value

        source_digest = digest_vectors(source_rows())
        actual_page_size = _bounded_page_size(page_size, max_vector_bytes)
        with _connection(dsn, 120) as target:
            target.run("begin isolation level repeatable read read only")
            try:
                _verify_identity(target, expected_identity)
                if target.run("show server_encoding") != [["UTF8"]]:
                    raise ValueError("PostgreSQL database must use UTF8 encoding")
                target.run(
                    "declare verify_vectors no scroll cursor for "
                    "select case when octet_length(atom_id) <= 4096 then atom_id end, "
                    f"case when octet_length(vec) <= {max_vector_bytes} then vec end, octet_length(vec) "
                    'from twinctx_v1.vectors order by atom_id collate "C"'
                )

                def target_rows():
                    while page := target.run(f"fetch forward {actual_page_size} from verify_vectors"):
                        for key, value, size in page:
                            if size is not None and size > _MAX_VECTOR_BYTES:
                                raise ValueError("oversized PostgreSQL vector")
                            if size is not None and size > max_vector_bytes:
                                raise ValueError("vector content verification failed")
                            yield key, value

                target_digest = digest_vectors(target_rows())
                if source_digest.rows != target_digest.rows or not hmac.compare_digest(
                    source_digest.sha256, target_digest.sha256
                ):
                    raise ValueError("vector content verification failed")
                return source_digest
            finally:
                target.run("rollback")
