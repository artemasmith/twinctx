"""Offline exact verification of twinctx records, independent of runtime adapters.

Only a caller with a consistent SQLite snapshot, a fenced and identified isolated
PostgreSQL target, and a verified backup may use these digests for a migration.
This module does not verify vectors, FTS, meta, JSONL canon, tenant identity, or
ongoing writes; matching digests do not authorize a runtime cutover.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from twinctx.contracts import COLS, EDGE_COLS

from .pg_bulk import _connection, _verify_identity
from .pg_text import restore_text

_MAX_TEXT_BYTES = 1024 * 1024
# Names and types are fixed, never derived from an untrusted database.
_TABLES = {
    "atoms": (tuple(COLS), ("id",), {"start_line": "i", "end_line": "i", "ord": "i", "trust": "f"}),
    "sources": (("source", "corpus", "sha", "indexed_at", "updated"), ("source",), {}),
    "edges": (tuple(EDGE_COLS), ("src", "dst", "kind"), {"version": "i", "weight": "f"}),
}


@dataclass(frozen=True)
class RecordDigest:
    rows: int
    sha256: str


def digest_records(table, rows):
    """Hash all fields with type and length framing in unique UTF-8 key order."""
    if table not in _TABLES:
        raise ValueError("unsupported table")
    cols, keys, types = _TABLES[table]
    key_indexes = tuple(cols.index(k) for k in keys)
    digest = hashlib.sha256(b"aitwin-record-digest-v1\0" + table.encode() + b"\0")
    previous = None
    count = 0
    for row in rows:
        if len(row) != len(cols):
            raise ValueError("invalid row shape")
        key = tuple(row[i].encode("utf-8") if isinstance(row[i], str) else b"" for i in key_indexes)
        if any(not k or len(k) > _MAX_TEXT_BYTES or b"\x00" in k for k in key):
            raise ValueError("invalid primary key")
        if previous is not None and key <= previous:
            raise ValueError("records must be unique and in UTF-8 key order")
        for col, value in zip(cols, row):
            kind = types.get(col, "s")
            if value is None:
                digest.update(b"n")
            elif kind == "s" and isinstance(value, str):
                data = value.encode("utf-8")
                if len(data) > _MAX_TEXT_BYTES:
                    raise ValueError("text exceeds verification limit")
                digest.update(b"s" + struct.pack("!Q", len(data)) + data)
            elif kind == "i" and type(value) is int and -(1 << 63) <= value < (1 << 63):
                digest.update(b"i" + struct.pack("!q", value))
            elif kind == "f" and type(value) in (int, float) and math.isfinite(value):
                digest.update(b"f" + struct.pack("!d", float(value)))
            else:
                raise ValueError("invalid field type or value")
        previous = key
        count += 1
    digest.update(struct.pack("!Q", count))
    return RecordDigest(count, digest.hexdigest())


def verify_records(sqlite_snapshot: Path, dsn: str, *, offline: bool = False, page_size: int = 4, expected_identity=None):
    """Compare records from two fenced snapshots; each field has a bounded size.

    SQLite source MUST be a consistent snapshot (including WAL at creation), not
    a live file. Caller owns fencing, tenant/target identity, backup and restore.
    """
    if offline is not True:
        raise RuntimeError("offline verification opt-in is required")
    if type(page_size) is not int or not 1 <= page_size <= 4:
        raise ValueError("invalid page size")
    uri = Path(sqlite_snapshot).resolve(strict=True).as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as source:
        source.execute("begin")
        if source.execute("pragma encoding").fetchone()[0].upper() != "UTF-8":
            raise ValueError("SQLite database must use UTF-8")
        with _connection(dsn, 120) as target:
            target.run("begin isolation level repeatable read read only")
            try:
                _verify_identity(target, expected_identity)
                if target.run("show server_encoding") != [["UTF8"]]:
                    raise ValueError("PostgreSQL database must use UTF8")
                result = {}
                for table, (cols, keys, types) in _TABLES.items():
                    # Bound potentially large text before transmitting it; flag oversize
                    # independently so it cannot be confused with SQL NULL.
                    text_cols = [c for c in cols if types.get(c, "s") == "s"]
                    src_fields = [
                        f'case when length(cast("{c}" as blob)) <= {_MAX_TEXT_BYTES} then "{c}" end'
                        if c in text_cols
                        else f'"{c}"'
                        for c in cols
                    ]
                    dst_fields = [
                        f'case when octet_length("{c}") <= {_MAX_TEXT_BYTES} then "{c}" end'
                        if c in text_cols
                        else f'cast("{c}" as double precision)'
                        if types.get(c) == "f"
                        else f'"{c}"'
                        for c in cols
                    ]
                    src_big = " or ".join(f'length(cast("{c}" as blob)) > {_MAX_TEXT_BYTES}' for c in text_cols)
                    dst_big = " or ".join(f'octet_length("{c}") > {_MAX_TEXT_BYTES}' for c in text_cols)
                    if table == "atoms":
                        dst_fields.append('case when octet_length("text_raw") <= 1048576 then "text_raw" end')
                        dst_big += ' or octet_length("text_raw") > 1048576'
                    source_order = ",".join(f'"{k}" collate binary' for k in keys)
                    target_order = ",".join(f'"{k}" collate "C"' for k in keys)
                    cur = source.execute(
                        f'select {",".join(src_fields)}, ({src_big}) from "{table}" order by {source_order}'
                    )
                    target.run(
                        f'declare verify_records no scroll cursor for select {",".join(dst_fields)}, ({dst_big}) from twinctx_v1."{table}" order by {target_order}'
                    )

                    def src_rows():
                        for row in cur:
                            if row[-1]:
                                raise ValueError("oversized SQLite text")
                            yield row[:-1]

                    def dst_rows():
                        while page := target.run(f"fetch forward {page_size} from verify_records"):
                            for row in page:
                                if row[-1]:
                                    raise ValueError("oversized PostgreSQL text")
                                if table == "atoms":
                                    values = list(row[:-2])
                                    idx = cols.index("text")
                                    values[idx] = restore_text(values[idx], row[-2])
                                    yield values
                                else:
                                    yield row[:-1]

                    left = digest_records(table, src_rows())
                    right = digest_records(table, dst_rows())
                    target.run("close verify_records")
                    if left.rows != right.rows or not hmac.compare_digest(left.sha256, right.sha256):
                        raise ValueError(f"{table} content verification failed")
                    result[table] = left
                return result
            finally:
                target.run("rollback")
