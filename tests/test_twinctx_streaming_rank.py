"""Streaming ranker matches the existing cosine ordering without materializing the index."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context"))
from stores.sqlite import SqliteStore
from twinctx.engine import Engine
from twinctx.vectorsearch import rank_vectors


def blob(values):
    return np.asarray(values, dtype=np.float32).tobytes()


def test_chunked_scores_and_order_match_original_vectorized_algorithm():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1503, 48)).astype(np.float32)
    q = rng.normal(size=48).astype(np.float32)
    rows = [(f"id-{i:05}", x.tobytes()) for i, x in enumerate(data)]
    expected_scores = data @ q
    expected = np.argsort(-expected_scores)[:35]
    actual = rank_vectors(iter(rows), q.tobytes(), 35)
    assert [x[0] for x in actual] == [rows[i][0] for i in expected]
    np.testing.assert_allclose([x[1] for x in actual], expected_scores[expected], rtol=2e-6, atol=2e-6)


def test_exact_ties_are_deterministic_and_preserve_stream_order():
    rows = [("third", blob([1, 0])), ("first", blob([1, 0])), ("second", blob([1, 0]))]
    assert [x[0] for x in rank_vectors(iter(rows), blob([1, 0]), 2)] == ["third", "first"]


def test_empty_stream_and_zero_result_count():
    assert rank_vectors(iter(()), blob([1, 0]), 10) == []
    assert rank_vectors(iter([("id", blob([1, 0]))]), blob([1, 0]), 0) == []


def test_ranking_never_forms_full_matrix(monkeypatch):
    original = np.vstack

    def bounded(rows, *args, **kwargs):
        assert len(rows) <= 256
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(np, "vstack", bounded)
    rows = ((f"id-{i}", blob([float(i), 1])) for i in range(2049))
    assert rank_vectors(rows, blob([1, 0]), 1)[0][0] == "id-2048"


def test_engine_streams_vectors_and_sees_updates():
    class Embedder:
        def embed_query(self, query):
            return blob([1, 0])

    store = SqliteStore(":memory:")
    try:
        engine = Engine(store, Embedder())
        assert engine._vector("query", 1) == ([], ["no_vectors_indexed"])
        store.put_vectors([("one", blob([1, 0]))])
        first, degraded = engine._vector("query", 1)
        assert first[0][0] == "one" and not degraded
        store.put_vectors([("two", blob([2, 0]))])
        second, degraded = engine._vector("query", 1)
        assert second[0][0] == "two" and not degraded
    finally:
        store.db.close()


def test_engine_does_not_open_stream_when_embedder_fails():
    closed = []

    class Store:
        def iter_vectors(self):
            try:
                yield "id", blob([1, 0])
            finally:
                closed.append(True)

        def vectors(self):
            raise AssertionError("must not materialize")

    class Embedder:
        def embed_query(self, query):
            raise RuntimeError("synthetic")

    assert Engine(Store(), Embedder())._vector("query", 1) == ([], ["embedding_error:RuntimeError"])
    assert closed == []


def test_malformed_vector_degrades_without_leaking_text():
    class Store:
        def iter_vectors(self):
            yield "private-id", b"bad"

    class Embedder:
        def embed_query(self, query):
            return blob([1, 0])

    out, degraded = Engine(Store(), Embedder())._vector("query", 1)
    assert out == [] and degraded == ["embedding_error:ValueError"]


def test_sqlite_stream_has_read_snapshot_during_writer(tmp_path):
    store = SqliteStore(tmp_path / "vectors.sqlite")
    try:
        store.put_vectors([(str(i), blob([i, 1])) for i in range(300)])
        stream = store.iter_vectors(page_size=10)
        first = next(stream)
        store.put_vectors([("new", blob([999, 1]))])
        assert "new" not in [row[0] for row in [first, *stream]]
        assert "new" in [row[0] for row in store.iter_vectors()]
    finally:
        store.db.close()


def test_concurrent_sqlite_streams_are_independent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = SqliteStore(tmp_path / "vectors.sqlite")
    try:
        store.put_vectors([(str(i), blob([i, 1])) for i in range(500)])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda _: Engine(store, type("E", (), {"embed_query": lambda self, q: blob([1, 0])})())._vector(
                        "q", 5
                    ),
                    range(4),
                )
            )
        successful = [(ranked, degraded) for ranked, degraded in results if not degraded]
        assert successful
        assert all([r[0] for r in ranked] == ["499", "498", "497", "496", "495"] for ranked, _ in successful)
        assert all(not degraded or degraded == ["vector_scan_busy"] for _, degraded in results)
    finally:
        store.db.close()


def test_busy_scan_degrades_without_opening_a_second_stream(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import twinctx.vectorsearch as vectorsearch

    entered, release = threading.Event(), threading.Event()
    original = vectorsearch.rank_vectors

    def slow_rank(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(vectorsearch, "rank_vectors", slow_rank)

    class Store:
        def iter_vectors(self):
            yield "id", blob([1, 0])

    class Embedder:
        def embed_query(self, query):
            return blob([1, 0])

    engine = Engine(Store(), Embedder())
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(engine._vector, "query", 1)
        try:
            assert entered.wait(5)
            assert engine._vector("query", 1) == ([], ["vector_scan_busy"])
        finally:
            release.set()
        assert first.result() == ([("id", 1.0)], [])
    assert engine._vector("query", 1) == ([("id", 1.0)], [])
