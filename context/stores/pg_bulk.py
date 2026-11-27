"""Offline, bounded vector copy for experimental PostgreSQL migration.

This is not a runtime Store API. It accepts an iterator over a consistent,
read-only snapshot and transfers binary bytea without per-row round trips. The
caller must use a dedicated offline database and verify backup/restore, identity,
available disk and write fencing before any real migration.
"""

from __future__ import annotations

import ssl
import struct
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from ipaddress import ip_address
from urllib.parse import unquote, urlparse

_COPY_SIGNATURE = b"PGCOPY\n\xff\r\n\x00" + struct.pack("!ii", 0, 0)
_COPY_END = struct.pack("!h", -1)
_MAX_VECTOR_BYTES = 4 * 1024 * 1024
_MAX_ID_BYTES = 4096
_MIN_BATCH_BYTES = 8192
_MAX_BATCH_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class VectorCopyResult:
    rows: int
    batches: int
    max_batch_bytes: int


def _encode_row(atom_id, vector) -> bytes:
    if not isinstance(atom_id, str) or not atom_id or "\x00" in atom_id:
        raise ValueError("invalid vector atom ID")
    identifier = atom_id.encode("utf-8")
    if len(identifier) > _MAX_ID_BYTES:
        raise ValueError("vector atom ID exceeds limit")
    if vector is not None and not isinstance(vector, (bytes, bytearray, memoryview)):
        raise TypeError("vector must be bytes or null")
    vector_bytes = vector.nbytes if isinstance(vector, memoryview) else len(vector) if vector is not None else 0
    if vector_bytes > _MAX_VECTOR_BYTES:
        raise ValueError("vector exceeds limit")
    data = bytes(vector) if vector is not None else None
    return (
        struct.pack("!h", 2)
        + struct.pack("!i", len(identifier))
        + identifier
        + struct.pack("!i", -1 if data is None else len(data))
        + (data or b"")
    )


@contextmanager
def _connection(dsn: str, timeout: int):
    import pg8000.native

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
    connection = pg8000.native.Connection(
        user=unquote(url.username or "postgres"),
        password=unquote(url.password) if url.password else None,
        host=url.hostname,
        port=url.port or 5432,
        database=unquote(url.path.lstrip("/")),
        ssl_context=False if local else ssl.create_default_context(),
        timeout=timeout,
    )
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def _transaction(connection):
    connection.run("begin")
    try:
        yield
        connection.run("commit")
    except BaseException:
        try:
            connection.run("rollback")
        except Exception:
            pass
        raise


def _verify_identity(connection, expected_identity):
    """Check identity within the caller's transaction, if supplied."""
    if expected_identity is None:
        return
    if not isinstance(expected_identity, tuple) or len(expected_identity) != 3:
        raise ValueError("invalid PostgreSQL destination identity")
    actual = connection.run(
        "select current_database(),system_identifier::text, "
        "(select oid::text from pg_database where datname=current_database()) from pg_control_system()"
    )
    if actual != [list(expected_identity)]:
        raise ValueError("PostgreSQL destination identity mismatch")


def copy_vectors(
    dsn: str,
    rows: Iterable[tuple[str, bytes | None]],
    *,
    offline: bool = False,
    batch_bytes: int = _MAX_BATCH_BYTES,
    socket_timeout: int = 120,
    expected_identity=None,
) -> VectorCopyResult:
    """COPY exact vector bytes into an empty isolated PG target as one transaction.

    No existing target vectors are overwritten. NULL values and orphan atom IDs
    are retained; NULL/invalid IDs and duplicate IDs reject the entire transfer.
    Bounded binary batches are at most `batch_bytes` each. Success is returned
    only after commit. A lost commit response has unknown outcome and requires
    reconciliation; the caller must not blindly retry against a live database.
    """
    if offline is not True:
        raise RuntimeError("offline migration opt-in is required")
    if type(batch_bytes) is not int or not (_MIN_BATCH_BYTES <= batch_bytes <= _MAX_BATCH_BYTES):
        raise ValueError("invalid COPY batch limit")
    if type(socket_timeout) is not int or not (1 <= socket_timeout <= 600):
        raise ValueError("invalid offline socket timeout")
    count = batches = largest = 0
    with _connection(dsn, socket_timeout) as con:
        with _transaction(con):
            _verify_identity(con, expected_identity)
            con.run("lock table twinctx_v1.vectors in access exclusive mode")
            if con.run("select count(*) from twinctx_v1.vectors") != [[0]]:
                raise RuntimeError("vector destination must be empty")
            con.run("create temporary table aitwin_vector_stage (atom_id text not null, vec bytea) on commit drop")
            buffer = bytearray(_COPY_SIGNATURE)
            records = 0

            def flush():
                nonlocal buffer, records, batches, largest
                if not records:
                    return
                buffer.extend(_COPY_END)
                size = len(buffer)
                con.run(
                    "copy pg_temp.aitwin_vector_stage(atom_id,vec) from stdin with (format binary)",
                    stream=BytesIO(buffer),
                )
                con.run(
                    "insert into twinctx_v1.vectors(atom_id,vec) select atom_id,vec from pg_temp.aitwin_vector_stage"
                )
                con.run("truncate pg_temp.aitwin_vector_stage")
                batches += 1
                largest = max(largest, size)
                buffer = bytearray(_COPY_SIGNATURE)
                records = 0

            for row in rows:
                if not isinstance(row, (tuple, list)) or len(row) != 2:
                    raise ValueError("invalid vector row")
                encoded = _encode_row(*row)
                if len(_COPY_SIGNATURE) + len(encoded) + len(_COPY_END) > batch_bytes:
                    raise ValueError("vector row exceeds COPY batch limit")
                if len(buffer) + len(encoded) + len(_COPY_END) > batch_bytes:
                    flush()
                buffer.extend(encoded)
                records += 1
                count += 1
            flush()
    return VectorCopyResult(rows=count, batches=batches, max_batch_bytes=largest)
