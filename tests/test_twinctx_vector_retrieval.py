"""Vector retrieval runs on first search and refreshes after a new memory."""

import importlib
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context"))
from twinctx.engine import Engine


def vec(x, y):
    return struct.pack("ff", x, y)


class Store:
    def __init__(self):
        self.rows = [("first", vec(1, 0))]

    def vectors(self):
        return list(self.rows)

    def add_memory(self, atom):
        self.last = atom

    def put_vectors(self, rows):
        self.rows.extend(rows)


class Embedder:
    def embed_query(self, text):
        return vec(1, 0)

    def embed_passages(self, texts):
        return [vec(2, 0) for _ in texts]


def test_first_search_returns_vector_rank():
    rows, degraded = Engine(Store(), Embedder())._vector("synthetic", 1)
    assert rows == [("first", 1.0)] and degraded == []


def test_remember_invalidates_rank_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("TWINCTX_STORE", "sqlite")
    monkeypatch.setenv("TWINCTX_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("TWINCTX_USAGE", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("TWINCTX_EMBED", "none")
    import twinctx.mcp_server as module

    module = importlib.reload(module)
    store = Store()
    engine = Engine(store, Embedder())
    monkeypatch.setattr(module, "_store", store)
    monkeypatch.setattr(module, "_engine", engine)
    monkeypatch.setattr(module, "_log", lambda *args, **kwargs: None)
    assert engine._vector("synthetic", 1)[0][0][0] == "first"
    new_id = module.context_remember("synthetic new fact")
    assert engine._vector("synthetic", 1)[0][0][0] == new_id
