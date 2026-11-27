"""Vector ranking behind a seam: numpy lives HERE, not in the core modules.

The core contract (tests/test_boundaries.py): the BM25 path — engine.assemble without an
embedder — runs on a bare interpreter with no numpy installed. When an embedder is configured,
Engine lazily delegates the matrix math to this module (an optional-dependency seam, not a
store concern).
"""
from __future__ import annotations


def rank_vectors(rows, query_vec: bytes, n: int) -> list[tuple[str, float]]:
    """Rank a stream with bounded memory; ties retain stream order."""
    import heapq
    from itertools import islice

    import numpy as np

    if n < 0:
        raise ValueError("n must be nonnegative")
    if n == 0:
        return []
    q = np.frombuffer(query_vec, dtype=np.float32)
    iterator = iter(rows)
    best = []
    index = 0
    while batch := list(islice(iterator, 256)):
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in batch])
        scores = matrix @ q
        for (identifier, _), score in zip(batch, scores):
            item = (float(score), -index, identifier)
            index += 1
            if len(best) < n:
                heapq.heappush(best, item)
            elif item[:2] > best[0][:2]:
                heapq.heapreplace(best, item)
    return [(identifier, score) for score, _, identifier in sorted(best, reverse=True)]
