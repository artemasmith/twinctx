"""Embedding a replaced/deleted atom must not leave a stale searchable vector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context"))
from stores.sqlite import SqliteStore
from twinctx.contracts import Atom
from twinctx.ingest import Indexer


def test_reindex_during_embedding_does_not_attach_old_vector(tmp_path):
    a = SqliteStore(tmp_path / "index.db")
    b = SqliteStore(tmp_path / "index.db")
    old = Atom(id="same", source="s", corpus="c", title="old", text="old text")
    new = Atom(id="same", source="s", corpus="c", title="new", text="new text")
    a.replace_source("s", "c", "v1", [old])

    class ConcurrentEmbedder:
        def embed_passages(self, _):
            b.replace_source("s", "c", "v2", [new])
            return [b"old-vector"]

    try:
        Indexer(a, ConcurrentEmbedder())._embed([old])
        assert a.get(["same"])["same"].text == "new text"
        assert "same" not in dict(a.vectors())
    finally:
        a.db.close()
        b.db.close()


def test_backfill_does_not_restore_vector_for_deleted_atom(tmp_path):
    a = SqliteStore(tmp_path / "index.db")
    b = SqliteStore(tmp_path / "index.db")
    old = Atom(id="same", source="s", corpus="c", text="old text")
    a.replace_source("s", "c", "v1", [old])

    class ConcurrentEmbedder:
        def embed_passages(self, _):
            b.drop_source("s")
            return [b"old-vector"]

    try:
        Indexer(a, ConcurrentEmbedder()).backfill_vectors()
        assert a.get(["same"]) == {}
        assert "same" not in dict(a.vectors())
    finally:
        a.db.close()
        b.db.close()


def test_current_passage_is_indexed_and_changed_passage_is_skipped(tmp_path):
    store = SqliteStore(tmp_path / "index.db")
    old = Atom(id="same", source="s", corpus="c", title="title", text="text")
    store.replace_source("s", "c", "v1", [old])
    try:
        assert store.put_vectors_if_current([(old, b"correct")]) == 1
        assert dict(store.vectors())["same"] == b"correct"
        changed = Atom(id="same", source="s", corpus="c", title="other title", text="text")
        assert store.put_vectors_if_current([(changed, b"wrong")]) == 0
        assert dict(store.vectors())["same"] == b"correct"
    finally:
        store.db.close()
