"""Search/persistence adapters selected at the twinctx deployment boundary.

The factory lives OUTSIDE the core: entry points (cli, MCP server, the Hermes plugin) call
open_store(); the core receives a Store and never imports this package. Adding a provider
(a server-profile store) is a new module here plus one branch in open_store — nothing in
twinctx/ changes. PostgreSQL is explicit and never falls back to SQLite.
"""

import os
import time
from pathlib import Path


def _memory_recover_due(state_dir: Path) -> bool:
    """Волна 2: recover+import_memory_canon — не чаще интервала (дефолт 3600 с).

    TWINCTX_MEMORY_RECOVER=force — всегда, =never — никогда. Маркер пишется только после
    успешного recover+import, так что сбой не «прячется» пропуском: следующий open повторит.
    """
    mode = os.environ.get("TWINCTX_MEMORY_RECOVER", "").strip().lower()
    if mode == "never":
        return False
    if mode == "force":
        return True
    try:
        interval = float(os.environ.get("TWINCTX_MEMORY_RECOVER_INTERVAL", "").strip() or 3600.0)
    except ValueError:
        interval = 3600.0
    if interval <= 0:
        interval = 3600.0
    marker = state_dir / "open-recover.marker"
    try:
        last = float(marker.read_text().strip())
    except (OSError, ValueError):
        return True
    return (time.time() - last) >= interval


def open_store():
    kind = os.environ.get("TWINCTX_STORE", "")
    if not kind:
        raise ValueError("TWINCTX_STORE must be explicitly set to postgres or sqlite")
    if kind == "sqlite":
        from .sqlite import SqliteStore, default_db_path

        return SqliteStore(default_db_path())
    if kind == "postgres":
        dsn = os.environ.get("TWINCTX_PG_DSN", "")
        state_dir = os.environ.get("TWINCTX_PG_STATE_DIR", "")
        if not dsn:
            raise ValueError("TWINCTX_PG_DSN is required for PostgreSQL")
        if not state_dir:
            raise ValueError("TWINCTX_PG_STATE_DIR is required for PostgreSQL")
        from .pg_memory_canon import PgMemoryCanon
        from .pg_sources import PgSourceIndex

        home = Path(os.environ.get("HARNESS_HOME", Path.home() / "harness"))
        canon = Path(os.environ.get("TWINCTX_MEMORY_CANON", home / "journal" / "memory" / "atoms.jsonl"))
        store = PgSourceIndex(dsn, experimental=True)
        # Cold-start prefetch: BM25-статистики в фоновом треде — первый запрос
        # не платит 10+ секунд за расчёт df/avgdl на 380K строк.
        _warm_bm25 = os.environ.get("TWINCTX_BM25_WARMUP", "1").strip().lower() != "0"
        if _warm_bm25:
            import threading

            def _warm():
                try:
                    store._bm25_corpus_stats()
                except Exception:
                    pass

            threading.Thread(target=_warm, daemon=True, name="bm25-warmup").start()
        try:
            wrapped = PgMemoryCanon(store, canon, state_dir=Path(state_dir))
            state = Path(state_dir)
            if _memory_recover_due(state):
                wrapped.recover()
                wrapped.import_memory_canon(canon)
                temporary = state / ".open-recover.marker.tmp"
                temporary.write_text(str(time.time()))
                os.replace(temporary, state / "open-recover.marker")
            return wrapped
        except BaseException:
            store.close()
            raise
    raise SystemExit(f"unknown TWINCTX_STORE={kind!r} (available: sqlite, postgres)")


def open_store_at(path):
    """Open a specific sqlite file (evals, inspection, restore tooling)."""
    from .sqlite import SqliteStore

    return SqliteStore(path)


def online_search(store):
    """Select the online adapter at the deployment boundary, preserving SQLite."""
    from privacy import wrap_embedder
    from twinctx.embed import from_env
    from twinctx.engine import Engine

    def embedder():
        return wrap_embedder(from_env())

    reader = getattr(store, "online_reader", None)
    if reader is None:
        return Engine(store, embedder())

    from twinctx.online_search import PgOnlineSearch

    return PgOnlineSearch(store, reader(), embedder)
