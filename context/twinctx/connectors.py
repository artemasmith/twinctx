"""Projection of connector RawEvents into the rebuildable twinctx index.

The connector canon remains the source of truth.  This module accepts already
normalised events only and never opens ``raw_ref`` or payload files.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .contracts import COLS, Atom, Store, now
from .ingest import Indexer, atom_id, guarded_source_replace, guarded_source_snapshot

_TEXT_FIELDS = ("title", "subject", "text", "body", "description", "status", "attachment_name")
_REQUIRED = ("event_id", "revision_id", "source", "tenant_id", "connection_id", "kind", "external_id")
_BATCH_SIZE = 250


def _event_dict(event: object) -> dict[str, Any]:
    if hasattr(event, "as_dict"):
        value = event.as_dict()
    elif is_dataclass(event):
        value = asdict(event)
    elif isinstance(event, Mapping):
        value = dict(event)
    else:
        raise TypeError("connector event must be RawEvent or mapping")
    missing = [key for key in _REQUIRED if not value.get(key)]
    if missing:
        raise ValueError(f"canonical event missing: {', '.join(missing)}")
    # Raw envelopes are never accepted by the indexing boundary.
    if "raw_payload" in value or "payload" in value:
        raise ValueError("raw payload is not an index input")
    return value


def _segment(value: object) -> str:
    text = str(value)
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")[:48] or "item"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _source_prefix(source: object, tenant: object, connection: object, kind: object) -> str:
    return "/".join((
        "connectors",
        _segment(source),
        _segment(tenant),
        _segment(connection),
        _segment(kind),
        "",
    ))


TELEGRAM_ARCHIVE_MESSAGE_PREFIX = _source_prefix("telegram", "owner", "telegram-archive", "message")
TELEGRAM_BUSINESS_MESSAGE_PREFIX = _source_prefix("telegram-business", "owner", "telegram-business", "message")


def is_raw_telegram_source(source: str) -> bool:
    """Identify the two projected message namespaces, including legacy rows."""
    return source.startswith((TELEGRAM_ARCHIVE_MESSAGE_PREFIX, TELEGRAM_BUSINESS_MESSAGE_PREFIX))


def source_reference(event: Mapping[str, Any]) -> str:
    """Stable, tenant/connection-scoped source for replace-by-source indexing."""
    prefix = _source_prefix(event["source"], event["tenant_id"], event["connection_id"], event["kind"])
    return prefix + _segment(event["external_id"])


def render_canonical(event: Mapping[str, Any]) -> str:
    """Render only the curated canonical representation, never raw_ref content."""
    data = event.get("data")
    data = data if isinstance(data, Mapping) else {}
    lines = [
        f"Источник: {event['source']}",
        f"Тип: {event['kind']}",
        f"Внешний ID: {event['external_id']}",
    ]
    for key in _TEXT_FIELDS:
        value = data.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    participants = event.get("participants")
    if participants:
        lines.append("participants: " + ", ".join(str(item) for item in participants))
    parent = event.get("parent_source_id")
    if parent:
        lines.append(f"parent_source_id: {parent}")
    lines.append(
        "source_ref: "
        f"connector://{event['source']}/{event['tenant_id']}/{event['connection_id']}/"
        f"{event['kind']}/{event['external_id']}?revision={event['revision_id']}"
    )
    return "\n".join(lines)


class ConnectorIndexer:
    """Connection-scoped projection writer for twinctx."""

    def __init__(self, store: Store, *, tenant_id: str, connection_id: str, embedder=None):
        if not tenant_id or not connection_id:
            raise ValueError("tenant_id and connection_id are required")
        self.store = store
        self.tenant_id = tenant_id
        self.connection_id = connection_id
        self.indexer = Indexer(store, embedder)
        self.indexed = 0
        self.unchanged = 0

    def ingest(self, events: Iterable[object]) -> dict[str, int]:
        guarded_batch = getattr(self.store, 'replace_sources_if_current', None)
        missing_vectors = getattr(self.store, 'missing_vector_atoms', None)
        if callable(guarded_batch) and callable(missing_vectors):
            return self._ingest_batched(events, guarded_batch, missing_vectors)
        for raw in events:
            event = _event_dict(raw)
            if (str(event["tenant_id"]), str(event["connection_id"])) != (
                self.tenant_id,
                self.connection_id,
            ):
                raise ValueError("event belongs to another tenant or connection")
            source = source_reference(event)
            revision = str(event["revision_id"])
            snapshot = guarded_source_snapshot(self.store, source)
            previous = snapshot[1][0] if snapshot[0] and snapshot[1] is not None else snapshot[1]
            if previous == revision:
                self.unchanged += 1
                continue
            text = render_canonical(event)
            atom = Atom(
                id=atom_id(source, 0),
                source=source,
                corpus="connectors",
                kind="knowledge",
                doc_type="fact",
                title=f"{event['source']} {event['kind']}",
                heading=str(event["external_id"]),
                text=text,
                observed_at=str(event.get("observed_at") or now()),
                updated=str(event.get("source_ts") or ""),
                trust=0.9,
                ord=0,
            )
            if not guarded_source_replace(self.store, source, "connectors", revision, [atom], atom.updated, snapshot):
                raise RuntimeError("source changed while connector projection was derived")
            self.indexer._embed([atom])
            self.indexed += 1
        return {"indexed": self.indexed, "unchanged": self.unchanged}

    def _ingest_batched(self, events, guarded_batch, missing_vectors):
        rows = []
        expected = {}
        unchanged_ids = {}
        pending_sources = set()

        def flush():
            if not pending_sources:
                return
            if rows:
                request = [
                    {'source': source, 'corpus': corpus, 'sha': sha, 'updated': updated,
                     'expected': expected[source], 'atoms': [{c: getattr(atom, c) for c in COLS} for atom in atoms]}
                    for source, corpus, sha, atoms, updated in rows
                ]
                raw = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode()
                operation_id = 'connector-batch-' + hashlib.sha256(raw).hexdigest()
                outcome = guarded_batch(rows, expected=expected, operation_id=operation_id)
                if outcome == 'conflict':
                    raise RuntimeError('source changed while connector projection was derived')
                if outcome not in ('applied', 'already_applied'):
                    raise RuntimeError('invalid guarded source batch result')
                self.indexed += len(rows)
            atoms_to_embed = [atom for _, _, _, atoms, _ in rows for atom in atoms]
            if unchanged_ids and self.indexer.embedder:
                missing = missing_vectors(list(unchanged_ids))
                if any(atom.id not in unchanged_ids or atom.source != unchanged_ids[atom.id] for atom in missing):
                    raise RuntimeError('connector vector identity differs')
                atoms_to_embed.extend(missing)
            self.indexer._embed(atoms_to_embed)
            rows.clear()
            expected.clear()
            unchanged_ids.clear()
            pending_sources.clear()

        for raw in events:
            event = _event_dict(raw)
            if (str(event['tenant_id']), str(event['connection_id'])) != (self.tenant_id, self.connection_id):
                raise ValueError('event belongs to another tenant or connection')
            source = source_reference(event)
            if source in pending_sources:
                flush()
            revision = str(event['revision_id'])
            snapshot = guarded_source_snapshot(self.store, source)
            if not snapshot[0]:
                raise RuntimeError('guarded source batch requires guarded source snapshots')
            previous = snapshot[1][0] if snapshot[1] is not None else None
            pending_sources.add(source)
            if previous == revision:
                self.unchanged += 1
                unchanged_ids[atom_id(source, 0)] = source
            else:
                atom = Atom(
                    id=atom_id(source, 0), source=source, corpus='connectors', kind='knowledge',
                    doc_type='fact', title=f"{event['source']} {event['kind']}",
                    heading=str(event['external_id']), text=render_canonical(event),
                    # A retry of the same source event must reproduce the same
                    # guarded request; do not inject a new wall-clock value.
                    observed_at=str(event.get('observed_at') or event.get('source_ts') or ''),
                    updated=str(event.get('source_ts') or ''), trust=0.9, ord=0,
                )
                rows.append((source, 'connectors', revision, [atom], atom.updated))
                expected[source] = snapshot[1]
            if len(pending_sources) >= _BATCH_SIZE:
                flush()
        flush()
        return {'indexed': self.indexed, 'unchanged': self.unchanged}


def ingest_canonical(
    store: Store, events: Iterable[object], *, tenant_id: str, connection_id: str, embedder=None,
) -> dict[str, int]:
    return ConnectorIndexer(
        store, tenant_id=tenant_id, connection_id=connection_id, embedder=embedder
    ).ingest(events)
