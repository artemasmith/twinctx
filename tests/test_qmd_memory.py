import hashlib

import pytest
from stores.sqlite import SqliteStore
from twinctx import AssembleRequest, Atom, Engine, qmd_client
from twinctx.qmd_memory import document


def fact(store, id, text, day, *, parent="", trust=1):
    atom = Atom(
        id=id,
        kind="memory",
        corpus="memory",
        source="memory/archives",
        text=text,
        valid_from=day,
        trust=trust,
        supersedes=parent,
    )
    store.add_memory(atom)
    return atom


@pytest.fixture
def decision(tmp_path, monkeypatch):
    store = SqliteStore(tmp_path / "index")
    old = fact(store, "legacy/id", "Храним бэкапы месяц.", "2026-01-01")
    hit = {
        "id": old.id,
        "score": 0.89,
        "sha256": hashlib.sha256(document(old)).hexdigest(),
    }
    monkeypatch.setenv("TWINCTX_QMD_ENABLED", "1")
    monkeypatch.setattr(qmd_client, "hits", lambda path, params, timeout: [hit])
    try:
        yield store, hit
    finally:
        store.db.close()


def ask(store, at="2026-05-01", **kwargs):
    return Engine(store).assemble(
        AssembleRequest(query="Retention policy?", as_of=at, graph=False, **kwargs)
    )


def test_live_version_and_historical_version_from_old_semantic_hit(decision):
    store, hit = decision
    fact(store, "b", "Пять недель.", "2026-02-01", parent=hit["id"])
    fact(store, "c", "Сорок пять дней.", "2026-03-01", parent="b")
    out = ask(store)
    assert "Сорок пять дней" in out["packed"] and "месяц" not in out["packed"]
    assert out["manifest"]["items"][0]["matched_memory_id"] == hit["id"]
    assert "Пять недель" in ask(store, "2026-02-01")["packed"]
    assert ask(store, "2025-12-31")["packed"] == ""
    store.close_memory("c", "2026-04-01")
    assert ask(store)["packed"] == ""


def test_fork_and_stale_document_are_rejected(decision):
    store, hit = decision
    fact(store, "b", "Вторая версия.", "2026-02-01", parent=hit["id"])
    hit["sha256"] = "0" * 64
    assert ask(store)["packed"] == ""
    hit["sha256"] = hashlib.sha256(
        document(store.get([hit["id"]])[hit["id"]])
    ).hexdigest()
    fact(store, "c", "Третья версия.", "2026-03-01", parent=hit["id"])
    assert ask(store)["packed"] == ""


def test_scope_trust_threshold_and_outage(decision, monkeypatch):
    store, hit = decision
    fact(store, "b", "Новое решение.", "2026-02-01", parent=hit["id"], trust=0.2)
    assert ask(store, min_trust=0.5)["packed"] == ""
    hit["score"] = 0.01
    assert ask(store)["packed"] == ""

    def down(*args):
        raise ConnectionRefusedError()

    monkeypatch.setattr(qmd_client, "hits", down)
    for options in (
        {"include_memory": False},
        {"corpora": ["vault"]},
        {"allowed_sources": ("vault/private.md",)},
    ):
        assert (
            "qmd_memory_unavailable"
            not in ask(store, **options)["manifest"]["degraded"]
        )
    assert "qmd_memory_unavailable" in ask(store)["manifest"]["degraded"]
    exact = Engine(store).assemble(AssembleRequest(query="Новое решение", graph=False))
    assert "Новое решение" in exact["packed"]


def test_watch_indexes_decisions_written_after_start(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import time

    state = tmp_path / "qmd"
    # The builder only exercises generation publication; model-backed ranking runs on Linux CI.
    (tmp_path / "node").symlink_to(sys.executable)
    builder = tmp_path / "builder.py"
    builder.write_text(
        "import pathlib, sqlite3, sys\n"
        "with sqlite3.connect(pathlib.Path(sys.argv[2]) / 'index.sqlite') as db:\n"
        "    db.execute('create table generation (id integer)')\n"
    )
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "TWINCTX_STORE": "sqlite",
        "TWINCTX_DB": str(tmp_path / "memory.db"),
        "HARNESS_HOME": str(tmp_path),
        "TWINCTX_EMBED": "none",
    }
    watcher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "twinctx.qmd_refresh",
            "-",
            str(state),
            str(builder),
            "--watch",
            "--interval",
            "1",
        ],
        env=env,
    )

    def manifest_when(predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and watcher.poll() is None:
            try:
                mem = json.loads((state / "current" / "manifest.json").read_bytes())[
                    "memory"
                ]
                if predicate(mem):
                    return mem
            except FileNotFoundError:
                pass
            time.sleep(0.1)
        pytest.fail("watcher did not publish expected memory snapshot")

    try:
        manifest_when(lambda entries: not entries)
        atom_id = subprocess.check_output(
            [sys.executable, "-m", "twinctx", "remember", "Храним журнал месяц."],
            env=env,
            text=True,
        ).strip()
        manifest = manifest_when(
            lambda entries: any(item["id"] == atom_id for item in entries.values())
        )
        name, row = next(iter(manifest.items()))
        content = (state / "current" / "docs" / "@memory" / name).read_bytes()
        assert (
            row["sha256"] == hashlib.sha256(content).hexdigest()
            and "месяц" in content.decode()
        )
    finally:
        watcher.terminate()
        watcher.wait(timeout=5)


def test_old_sqlite_keeps_decisions_when_upgrading_the_index(tmp_path, monkeypatch):
    import sqlite3

    from twinctx.contracts import EDGE_COLS

    monkeypatch.setenv("TWINCTX_MEMORY_CANON", str(tmp_path / "memory.jsonl"))
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            "create table atoms(id text primary key, source text, corpus text, kind text, text text);"
            "insert into atoms values('original', 'memory/policy', 'memory', 'memory', 'Earlier policy');"
            "create table edges(src text, dst text, kind text, weight real, primary key(src,dst,kind));"
        )

    store = SqliteStore(path)
    assert set(EDGE_COLS) <= {
        row[1] for row in store.db.execute("pragma table_info(edges)")
    }
    assert store.get(["original"])["original"].text == "Earlier policy"
    fact(store, "updated", "Current policy", "2026-03-01", parent="original")
    assert {a.id for a in store.memory_family("original")} == {"original", "updated"}
    store.db.close()
    reopened = SqliteStore(path)
    assert reopened.get(["updated"])["updated"].supersedes == "original"
    reopened.db.close()
