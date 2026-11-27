"""Core contracts: the Atom model and the store/embedder Protocols (stdlib only).

The core (engine, ingest, connectors projection) never opens a database and never imports an
adapter — it receives a Store. Adapters live outside the core package (`stores/`): today a
SQLite one selected by TWINCTX_STORE; a server-profile adapter is a drop-in addition. The
append-only JSONL canon (sources, and — since the write-through change — memory atoms) is the
truth; every store is a rebuildable index over it.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Atom:
    id: str
    source: str
    corpus: str
    text: str
    kind: str = "knowledge"          # knowledge | memory   (state is never stored — C1)
    doc_type: str = "note"           # note | code | task | incident | session | fact
    title: str = ""
    heading: str = ""
    start_line: int = 0
    end_line: int = 0
    updated: str = ""
    observed_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    trust: float = 1.0
    ord: int = 0
    supersedes: str = ""
    score: float = 0.0
    via: str = "search"
    extra: dict = field(default_factory=dict)


COLS = [f for f in Atom.__dataclass_fields__ if f not in ("score", "via", "extra")]

EDGE_SCHEMA_VERSION = 2   # bump when the edge schema changes; edges are a rebuilt index
EDGE_COLS = ("src", "dst", "kind", "weight", "project", "provenance", "basis",
             "owner", "version", "valid_from", "valid_to", "created_at")


class Store(Protocol):
    """Persistence boundary. One store = one owning deployment zone; several processes of that
    zone may share it (WAL serialises writers on the same host). Memory writes MUST go through
    add_memory/close_memory so the JSONL canon stays authoritative (write-through)."""

    # ── writes ──
    def replace_source(self, source: str, corpus: str, sha: str, atoms: list[Atom], updated: str = "") -> None: ...
    def drop_source(self, source: str) -> None: ...
    def add_memory(self, a: Atom) -> None: ...
    def close_memory(self, atom_id: str, valid_to: str) -> None: ...
    def replace_edges_bulk(self, source_edges: dict[str, list[tuple]]) -> None: ...
    def put_vectors(self, rows: list[tuple[str, bytes]]) -> None: ...
    def put_vectors_if_current(self, rows: list[tuple[Atom, bytes]]) -> int: ...

    # ── reads ──
    def source_sha(self, source: str) -> str | None: ...
    def sources(self, corpus: str | None = None) -> list[str]: ...
    def source_rows(self, source: str) -> list[tuple[str, str]]: ...
    def first_atom_id(self, source: str) -> str | None: ...
    def bm25(self, fts: str, limit: int) -> list[tuple[str, float]]: ...
    def get(self, ids: list[str]) -> dict[str, Atom]: ...
    def by_source(self, source: str) -> list[Atom]: ...
    def neighbors(self, ids: list[str], per_seed: int, project: str = "", as_of: str = "") -> list[dict]: ...
    def dangling_edges(self) -> int: ...
    def atoms_without_vectors(self, limit: int) -> list[Atom]: ...
    def vectors(self) -> list[tuple[str, bytes]]: ...
    def source_ages(self) -> dict[str, str]: ...
    def stats(self) -> dict: ...


class GuardedGraphStore(Protocol):
    """Optional capability: atomically reject a graph derived from stale scope.

    Each opaque version belongs to the captured source tuple. Every source in
    the requested corpora must still be present with the same version at commit.
    """

    def graph_source_versions(self, corpora: tuple[str, ...]) -> dict[str, tuple[str | None, str]]: ...
    def replace_edges_if_current(
        self, source_edges: dict[str, list[tuple]], expected_versions: dict[str, tuple[str | None, str]],
        corpora: tuple[str, ...],
    ) -> None: ...


class GuardedSourceStore(Protocol):
    """Compare-and-swap indexed sources and replay the same operation id safely.

    A conflict leaves the existing source unchanged. The caller must retain the
    same operation id and identical request to reconcile an unknown commit.
    """

    def source_revision(self, source: str) -> tuple[str | None, str] | None: ...
    def replace_source_if_current(
        self, source: str, corpus: str, sha: str, atoms: list[Atom], updated: str = "", *,
        expected: tuple[str | None, str] | None, operation_id: str,
    ) -> str: ...
    def drop_source_if_current(
        self, source: str, *, expected: tuple[str | None, str] | None, operation_id: str,
    ) -> str: ...


class Embedder(Protocol):
    """Vector provider (numpy/fastembed live behind this seam, out of the core)."""

    def embed_passages(self, texts: list[str]) -> list: ...
