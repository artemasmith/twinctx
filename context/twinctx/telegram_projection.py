"""Ingest the connector-owned Telegram projection (raw local text; masking is enforced
at the model boundary — ADR §16 amendment, 14.11.2026).

This process runs as ``twin`` and intentionally refuses arbitrary JSONL paths
or fields.  It can only receive the curated projection, never the private
Telethon canon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from privacy import wrap_embedder
from stores import open_store

from .connectors import ingest_canonical, source_reference
from .embed import from_env

PROJECTION = Path("/var/lib/harness-tg/projection/messages.jsonl")
BUSINESS_PROJECTION = Path("/var/lib/harness-tg/projection/business.jsonl")
_FIELDS = {
    "event_id",
    "revision_id",
    "source",
    "tenant_id",
    "connection_id",
    "kind",
    "external_id",
    "source_ts",
    "observed_at",
    "data",
}
_ARCHIVE_BATCH_MESSAGES = 100
_ARCHIVE_BATCH_CHARS = 24_000


def _load(path: Path) -> list[dict]:
    if path != PROJECTION or not path.is_file() or path.is_symlink():
        raise ValueError("invalid Telegram projection")
    result: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != _FIELDS:
                raise ValueError("invalid Telegram projection record")
            if (
                row.get("source") != "telegram"
                or row.get("tenant_id") != "owner"
                or row.get("connection_id") != "telegram-archive"
            ):
                raise ValueError("unexpected Telegram projection identity")
            data = row.get("data")
            if not isinstance(data, dict) or set(data) != {"text"} or not isinstance(data["text"], str):
                raise ValueError("invalid Telegram projection body")
            result.append(row)
    return result


def _load_business(path: Path) -> list[dict]:
    """Load only the fixed Business data projection.

    Business text is deliberately not configuration or an agent command.  The
    connector adds that boundary notice before this function ever sees it; this
    loader merely enforces the file/identity/schema boundary before indexing.
    """
    if path != BUSINESS_PROJECTION or not path.is_file() or path.is_symlink():
        raise ValueError("invalid Telegram Business projection")
    result: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError("invalid Telegram Business projection record")
        if (
            row.get("source") != "telegram-business"
            or row.get("tenant_id") != "owner"
            or row.get("connection_id") != "telegram-business"
            or row.get("kind") != "message"
            or not isinstance(row.get("external_id"), str)
            or not row["external_id"].startswith("tg-business:")
        ):
            raise ValueError("unexpected Telegram Business projection identity")
        data = row.get("data")
        if not isinstance(data, dict) or set(data) != {"text"} or not isinstance(data["text"], str):
            raise ValueError("invalid Telegram Business projection body")
        result.append(row)
    return result


def _archive_batches(rows: list[dict]) -> list[dict]:
    """Coalesce archive messages into small, stable conversation-day documents.

    Indexing one FTS source per historical message makes a large initial archive
    import needlessly expensive.  A batch is bounded by both message count and
    rendered size; a new message changes only the final batch for its dialog and
    day.  The input is the strictly validated projection (raw local text).
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        _, dialog, _ = row["external_id"].split(":", 2)
        timestamp = row["source_ts"]
        day = timestamp[:10] if len(timestamp) >= 10 else "unknown-date"
        grouped[(dialog, day)].append(row)

    batches: list[dict] = []
    for (dialog, day), messages in sorted(grouped.items()):
        messages.sort(key=lambda row: (row["source_ts"], row["external_id"]))
        current: list[dict] = []
        current_size = 0
        ordinal = 0

        def emit() -> None:
            nonlocal ordinal
            if not current:
                return
            revisions = "\n".join(f"{row['external_id']}\0{row['revision_id']}" for row in current)
            revision = hashlib.sha256(revisions.encode("utf-8")).hexdigest()
            text = "\n\n".join(f"[{row['source_ts'] or 'unknown-time'}] {row['data']['text']}" for row in current)
            batches.append(
                {
                    "event_id": f"tg-batch:{revision}",
                    "revision_id": revision,
                    "source": "telegram",
                    "tenant_id": "owner",
                    "connection_id": "telegram-archive",
                    "kind": "message",
                    "external_id": f"tg-batch:{dialog}:{day}:{ordinal}",
                    "source_ts": current[-1]["source_ts"],
                    "observed_at": current[-1]["observed_at"],
                    "data": {"text": text},
                }
            )
            ordinal += 1

        for row in messages:
            rendered_size = len(row["data"]["text"]) + len(row["source_ts"]) + 4
            if current and (
                len(current) >= _ARCHIVE_BATCH_MESSAGES or current_size + rendered_size > _ARCHIVE_BATCH_CHARS
            ):
                emit()
                current = []
                current_size = 0
            current.append(row)
            current_size += rendered_size
        emit()
    return batches


def _legacy_message_sources(store) -> list[str]:
    """Find pre-batch Telegram sources without exposing any message content."""
    sample = {
        "source": "telegram",
        "tenant_id": "owner",
        "connection_id": "telegram-archive",
        "kind": "message",
        "external_id": "tg:0:0",
    }
    prefix = source_reference(sample).rsplit("/", 1)[0] + "/tg-"
    return [
        source
        for source in store.sources("connectors")
        if source.startswith(prefix) and not source.startswith(prefix + "batch-")
    ]


def ingest(path: Path | None = None) -> dict[str, int]:
    path = PROJECTION if path is None else path
    store = open_store()
    # Versions before archive batching indexed one FTS source per message.  Drop
    # that partial, rebuildable index in one bounded transaction before adding
    # the compact documents, so retrieval cannot return duplicate history.
    legacy = _legacy_message_sources(store)
    if legacy:
        drop_many = getattr(store, "drop_sources", None)
        if callable(drop_many):
            drop_many(legacy)
        else:
            for source in legacy:
                store.drop_source(source)
    if path != PROJECTION:
        raise ValueError("invalid Telegram projection")
    embedder = None if os.environ.get("TWINCTX_TELEGRAM_EMBED", "none") == "none" else wrap_embedder(from_env())
    indexed = unchanged = 0
    if path.is_file():
        archive = ingest_canonical(
            store,
            _archive_batches(_load(path)),
            tenant_id="owner",
            connection_id="telegram-archive",
            # Telegram archive text is raw local data (ADR §16 amendment, 14.11.2026);
            # wrap_embedder masks it before any vector call. Operators may explicitly
            # opt in after selecting an approved local/vector provider.
            embedder=embedder,
        )
        indexed += archive["indexed"]
        unchanged += archive["unchanged"]
    if BUSINESS_PROJECTION.is_file():
        business = ingest_canonical(
            store,
            _load_business(BUSINESS_PROJECTION),
            tenant_id="owner",
            connection_id="telegram-business",
            embedder=embedder,
        )
        indexed += business["indexed"]
        unchanged += business["unchanged"]
    return {"indexed": indexed, "unchanged": unchanged}


def ingest_business() -> dict[str, int]:
    """Index only the fixed Business projection without vector calls."""
    store = open_store()
    try:
        return ingest_canonical(
            store, _load_business(BUSINESS_PROJECTION),
            tenant_id="owner", connection_id="telegram-business", embedder=None,
        )
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
        else:
            store.db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", default=str(PROJECTION))
    parser.add_argument("--business-only", action="store_true")
    args = parser.parse_args(argv)
    # An environment override is deliberately not accepted: the path is a
    # security boundary, not an operator convenience flag.
    if Path(args.projection) != PROJECTION:
        raise SystemExit("telegram projection path is fixed")
    print(json.dumps(ingest_business() if args.business_only else ingest(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
