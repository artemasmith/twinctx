"""Mmap-backed vector ranking for twinctx (sustainable, in-repo; replaces live server patch).

Why: `store.iter_vectors()` streams 380K×1024 float32 bytea from PostgreSQL — 80–90s
per query. This mixin keeps the same vectors in one mmap'ed numpy matrix on local
disk, so a query is a single BLAS matmul: measured 0.2–0.5s.

The engine calls `rank_mmap(query_vec, n)` via duck typing (see engine._vector).
When TWINCTX_VEC_MATRIX + TWINCTX_VEC_IDS point at valid files AND the mmap is
loadable, the fast path answers; otherwise the method returns None and the engine
falls back to the PG streaming path untouched — zero risk when disabled.

Build the cache: `python -m stores.build_vec_cache --store <PG DSN> --out <dir>`
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_STATE: dict = {"m": None, "ids": None, "matrix_path": None, "ids_path": None, "failed": False}


def _try_load() -> bool:
    """Load mmap matrix + ids once (thread-safe, lazy). Returns True if usable."""
    if _STATE["failed"]:
        return False
    if _STATE["m"] is not None:
        return True
    with _LOCK:
        if _STATE["failed"] or _STATE["m"] is not None:
            return _STATE["m"] is not None
        try:
            import numpy as np

            matrix_path = os.environ.get("TWINCTX_VEC_MATRIX", "")
            ids_path = os.environ.get("TWINCTX_VEC_IDS", "")
            if not matrix_path or not ids_path:
                _STATE["failed"] = True
                return False
            m = np.load(matrix_path, mmap_mode="r")
            ids = Path(ids_path).read_text().splitlines()
            if len(ids) != m.shape[0]:
                _STATE["failed"] = True
                return False
            _STATE.update(m=m, ids=ids, matrix_path=matrix_path, ids_path=ids_path)
            return True
        except Exception:
            _STATE["failed"] = True
            return False


def rank_mmap(query_vec: list[float], n: int) -> list[tuple[str, float]] | None:
    """Fast path: BLAS matmul over mmap'ed matrix. None = cache unusable, fall back.

    query_vec: 1-D list/array of dim D (e.g. 1024). Returns [(atom_id, cosine)] top-n.
    """
    if not _try_load():
        return None
    try:
        import numpy as np

        q = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return None
        q = q / norm
        m = _STATE["m"]
        # Normalized columns assumed (build script L2-normalizes); matmul = cosine
        scores = m @ q  # (rows,) — one BLAS call over mmap pages
        k = min(n * 4, scores.shape[0])  # over-fetch for post-filter
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1][:n]]
        ids = _STATE["ids"]
        return [(ids[i], float(scores[i])) for i in top if scores[i] > 0.0]
    except Exception:
        return None


def mmap_status() -> dict:
    """Diagnostics for the connectors page / health check."""
    loaded = _STATE["m"] is not None
    return {
        "loaded": loaded,
        "failed": _STATE["failed"],
        "matrix": _STATE.get("matrix_path", ""),
        "rows": _STATE["m"].shape[0] if loaded else 0,
        "dim": _STATE["m"].shape[1] if loaded else 0,
    }


def reset_for_tests() -> None:
    """Test helper: clear cached mmap state (to test both fast/slow paths)."""
    global _STATE
    with _LOCK:
        _STATE = {"m": None, "ids": None, "matrix_path": None, "ids_path": None, "failed": False}


def cache_file_size() -> int:
    """Размер mmap-матрицы в байтах — для health/страницы подключений."""
    if _STATE["m"] is not None:
        import numpy as np
        return int(np.asanyarray(_STATE["m"]).nbytes)
    return 0
