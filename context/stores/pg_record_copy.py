"""Offline transfer of SQLite records to an isolated, empty PostgreSQL index.

The caller must provide a consistent read-only SQLite snapshot and fence all
writers. This primitive does not copy vectors, metadata, or the memory canon,
verify tenant identity, or authorize a runtime cutover.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from twinctx.contracts import COLS, EDGE_COLS

from .pg_bulk import (
    _COPY_END,
    _COPY_SIGNATURE,
    _MAX_BATCH_BYTES,
    _MIN_BATCH_BYTES,
    _connection,
    _transaction,
    _verify_identity,
)
from .pg_text import project_text

_MAX_TEXT_BYTES = 1024 * 1024
_TABLES = {
    "sources": (("source", "corpus", "sha", "indexed_at", "updated"), {}),
    "atoms": (tuple(COLS), {"start_line": "i", "end_line": "i", "ord": "i", "trust": "f"}),
    "edges": (tuple(EDGE_COLS), {"version": "i", "weight": "f"}),
}


@dataclass(frozen=True)
class RecordCopyResult:
    counts: dict[str, int]
    batches: int
    max_batch_bytes: int


def _encode_record(table, row):
    columns, kinds = _TABLES[table]
    if len(row) != len(columns):
        raise ValueError("invalid record shape")
    raw = None
    if table == "atoms":
        row = list(row)
        idx = columns.index("text")
        row[idx], raw = project_text(row[idx])
    result = bytearray(struct.pack("!h", len(columns) + (1 if table == "atoms" else 0)))
    for column, value in zip(columns, row):
        kind = kinds.get(column, "s")
        if value is None:
            result.extend(struct.pack("!i", -1))
            continue
        if kind == "s" and isinstance(value, str) and "\x00" not in value:
            data = value.encode("utf-8")
            if len(data) > _MAX_TEXT_BYTES:
                raise ValueError("record text exceeds limit")
        elif kind == "i" and type(value) is int and -(1 << 31) <= value < (1 << 31):
            data = struct.pack("!i", value)
        elif kind == "f" and type(value) in (int, float) and math.isfinite(value):
            data = struct.pack("!d", float(value))
        else:
            raise ValueError("invalid record field or type")
        result.extend(struct.pack("!i", len(data)))
        result.extend(data)
    if table == "atoms":
        if raw is None:
            result.extend(struct.pack("!i", -1))
        else:
            result.extend(struct.pack("!i", len(raw)))
            result.extend(raw)
    return result


def copy_records(sqlite_snapshot: Path, dsn: str, *, offline=False, batch_bytes=_MAX_BATCH_BYTES, socket_timeout=120, expected_identity=None):
    """Transfer records in one PG transaction; no runtime or retry side effects.

    An exception while committing may have an unknown outcome. The caller must
    inspect the isolated destination and must not blindly retry a live target.
    """
    if offline is not True:
        raise RuntimeError("offline migration opt-in is required")
    if type(batch_bytes) is not int or not _MIN_BATCH_BYTES <= batch_bytes <= _MAX_BATCH_BYTES:
        raise ValueError("invalid COPY batch limit")
    if type(socket_timeout) is not int or not 1 <= socket_timeout <= 600:
        raise ValueError("invalid offline socket timeout")
    uri = Path(sqlite_snapshot).resolve(strict=True).as_uri() + "?mode=ro"
    counts = {}
    batches = largest = 0
    with closing(sqlite3.connect(uri, uri=True)) as source:
        source.execute("begin")
        if source.execute("pragma encoding").fetchone()[0].upper() != "UTF-8":
            raise ValueError("SQLite database must use UTF-8")
        with _connection(dsn, socket_timeout) as target, _transaction(target):
            _verify_identity(target, expected_identity)
            target.run("lock table twinctx_v1.sources, twinctx_v1.atoms, twinctx_v1.edges, twinctx_v1.vectors in access exclusive mode")
            for table in (*_TABLES, "vectors"):
                if target.run(f"select exists(select 1 from twinctx_v1.{table} limit 1)") != [[False]]:
                    raise RuntimeError("record destination must be empty")
            for table in ("memory_revisions", "memory_operations", "source_operations"):
                if target.run("select to_regclass(:name)", name=f"twinctx_v1.{table}") != [[None]]:
                    target.run(f"lock table twinctx_v1.{table} in access exclusive mode")
                    if target.run(f"select exists(select 1 from twinctx_v1.{table} limit 1)") != [[False]]:
                        raise RuntimeError("record destination must be empty")
            for table, (columns, kinds) in _TABLES.items():
                names = ",".join(f'"{name}"' for name in columns) + (',"text_raw"' if table == "atoms" else "")
                text_cols = [c for c in columns if kinds.get(c, "s") == "s"]
                fields = [
                    f'case when length(cast("{c}" as blob)) <= {_MAX_TEXT_BYTES} then "{c}" end'
                    if c in text_cols else f'"{c}"' for c in columns
                ]
                too_big = " or ".join(f'length(cast("{c}" as blob)) > {_MAX_TEXT_BYTES}' for c in text_cols)
                cursor = source.execute(f"select {','.join(fields)}, ({too_big}) from {table}")
                buffer = bytearray(_COPY_SIGNATURE)
                records = 0
                count = 0

                def flush():
                    nonlocal buffer, records, batches, largest
                    if not records:
                        return
                    buffer.extend(_COPY_END)
                    target.run(f"copy twinctx_v1.{table} ({names}) from stdin with (format binary)", stream=BytesIO(buffer))
                    batches += 1
                    largest = max(largest, len(buffer))
                    buffer = bytearray(_COPY_SIGNATURE)
                    records = 0

                for row in cursor:
                    if row[-1]:
                        raise ValueError("record text exceeds limit")
                    encoded = _encode_record(table, row[:-1])
                    if len(_COPY_SIGNATURE) + len(encoded) + len(_COPY_END) > batch_bytes:
                        raise ValueError("record exceeds COPY batch limit")
                    if len(buffer) + len(encoded) + len(_COPY_END) > batch_bytes:
                        flush()
                    buffer.extend(encoded)
                    records += 1
                    count += 1
                flush()
                counts[table] = count
    return RecordCopyResult(counts, batches, largest)
