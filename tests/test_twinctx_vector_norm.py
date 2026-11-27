"""Устойчивость существующего consumer embeddings на синтетических числах."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context"))

from twinctx.embed import _norm  # noqa: E402


@pytest.mark.parametrize(
    "vector",
    [
        [3.0, 4.0],
        [1e40, 0.5],
        [1e20, 1.0],
        [1e-30, 0.0],
        [1.8446741874686296e19] + [1e14] * 16383,
        [1e308, 1e308],
        [1e-320, 0.0],
    ],
)
def test_nonzero_vectors_have_finite_unit_norm(vector):
    original = list(vector)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        raw = _norm(vector)
    assert isinstance(raw, bytes)
    assert len(raw) == 4 * len(vector)
    normalized = np.frombuffer(raw, dtype=np.float32)
    assert np.isfinite(normalized).all()
    assert np.linalg.norm(normalized.astype(np.float64)) == pytest.approx(1.0, abs=1e-6)
    scaled = np.asarray(vector, dtype=np.float64) / max(abs(value) for value in vector)
    expected = scaled / np.linalg.norm(scaled)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-7)
    assert vector == original


def test_zero_vector_remains_zero_and_source_is_unchanged():
    vector = np.asarray([0.0, 0.0], dtype=np.float32)
    before = vector.copy()
    normalized = np.frombuffer(_norm(vector), dtype=np.float32)
    np.testing.assert_array_equal(normalized, [0.0, 0.0])
    np.testing.assert_array_equal(vector, before)


def test_vector_store_blob_and_real_ranking_keep_direction():
    from stores.sqlite import SqliteStore
    from twinctx.vectorsearch import rank_vectors

    store = SqliteStore(":memory:")
    try:
        relevant = _norm([3.0, 4.0])
        orthogonal = _norm([4.0, -3.0])
        store.put_vectors([("relevant", relevant), ("orthogonal", orthogonal)])
        rows = store.vectors()
        assert all(isinstance(blob, bytes) and len(blob) == 8 for _, blob in rows)
        ranked = rank_vectors(rows, _norm([3e200, 4e200]), 2)
        assert [identifier for identifier, _ in ranked] == ["relevant", "orthogonal"]
        assert ranked[0][1] == pytest.approx(1.0, abs=1e-6)
        assert ranked[1][1] == pytest.approx(0.0, abs=1e-6)
    finally:
        store.db.close()


@pytest.mark.parametrize(
    "vector",
    [[], [[1.0, 2.0]], [float("nan"), 1.0], [float("inf"), 1.0], ["1", "2"], [True, 1.0], [1j, 1.0]],
)
def test_invalid_vectors_are_rejected_without_echoing_input(vector):
    with pytest.raises(ValueError) as exc:
        _norm(vector)
    assert "nan" not in str(exc.value).lower()
    assert "inf" not in str(exc.value).lower()


def test_extended_precision_outside_float64_is_rejected():
    if np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp:
        pytest.skip("platform longdouble has no wider exponent range")
    for value in (np.longdouble("1e-4000"), np.longdouble("1e4000")):
        with pytest.raises(ValueError, match="^invalid embedding vector$"):
            _norm(np.asarray([value], dtype=np.longdouble))
