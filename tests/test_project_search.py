"""Поиск проектных источников и обновление резервного индекса."""

import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from time import monotonic
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context"))

from twinctx import project_search
from twinctx.contracts import Atom  # noqa: E402

QUERY = "Через сколько часов приходит оповещение?"
PROJECT = "apollo"
OLD_RULE = "Оповещение приходит через 12 часов."
NEW_RULE = "Оповещение приходит через 24 часа."


def write_document(vault, name, rule, *, project=PROJECT, supersedes=None):
    path = vault / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = f"project: {project}\n"
    if supersedes is not None:
        frontmatter += f"supersedes: {supersedes}\n"
    path.write_text(f"---\n{frontmatter}---\n\n# Оповещения\n\n{rule}\n", encoding="utf-8")
    return path


def hit(path, vault):
    return {
        "path": path.relative_to(vault).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def empty_search(_request):
    return [], {"degraded": []}


def expect_uncertain(reply):
    assert re.search(r"(не удалось|не могу|не удалось бы) подтвердить", reply, re.I), reply
    assert re.search(r"действующ\w* редакц\w*|актуальност\w*", reply, re.I), reply


def test_successful_empty_primary_triggers_one_fallback(tmp_path):
    old = write_document(tmp_path, "old.md", OLD_RULE)
    write_document(tmp_path, "new.md", NEW_RULE, supersedes="old")
    order = []

    def primary(request):
        assert request.query == QUERY
        order.append("twinctx")
        return empty_search(request)

    def fallback(project, query, remaining_seconds):
        assert (project, query) == (PROJECT, QUERY)
        assert remaining_seconds > 0
        order.append("qmd")
        return [hit(old, tmp_path)]

    reply = project_search.answer_project(PROJECT, QUERY, search=primary, vault_root=tmp_path, qmd_search=fallback)
    assert order == ["twinctx", "qmd"]
    assert NEW_RULE in reply and "new.md" in reply
    assert OLD_RULE not in reply


def test_primary_hit_and_failure_do_not_trigger_fallback(tmp_path):
    old = write_document(tmp_path, "old.md", OLD_RULE)
    write_document(tmp_path, "new.md", NEW_RULE, supersedes="old")
    # Текст атома намеренно устарел: ответ должен опираться на файл.
    atom = Atom(id="archive", source="vault/old.md", corpus="vault", text=OLD_RULE)
    fallback = Mock(side_effect=AssertionError("QMD cannot run after a primary hit or error"))

    def answer(primary):
        return project_search.answer_project(PROJECT, QUERY, search=primary, vault_root=tmp_path, qmd_search=fallback)

    indexed = {"degraded": [], "source_sha1": {atom.source: hashlib.sha1(old.read_bytes()).hexdigest()}}
    confirmed = answer(lambda _: ([atom], indexed))
    old.write_text(old.read_text() + "\nИсправление: срок изменён.\n")
    expect_uncertain(answer(lambda _: ([atom], indexed)))
    assert NEW_RULE in confirmed and "new.md" in confirmed
    assert OLD_RULE not in confirmed

    interrupted = answer(lambda _: ([], {"degraded": ["online_search_deadline"]}))
    empty = project_search.answer_project(
        PROJECT, QUERY, search=empty_search, vault_root=tmp_path, qmd_search=lambda *_: []
    )
    assert interrupted != empty
    assert re.search("врем|прерв|поиск|недоступ", interrupted, re.I)
    assert NEW_RULE not in interrupted
    fallback.assert_not_called()


def test_fallback_never_returns_source_from_another_project_or_outside_vault(tmp_path):
    vault = tmp_path / "vault"
    ok = write_document(vault, "own-old.md", "Разрешённый источник.")
    write_document(vault, "own.md", NEW_RULE, supersedes="own-old")
    cross = write_document(vault, "cross.md", "Приватное правило Zeus.", project="zeus")
    other = write_document(tmp_path, "external.md", "Правило за пределами Vault.")
    (vault / "escape.md").symlink_to(other)
    captured = []

    def fallback(project, query, _remaining):
        captured.append((project, query))
        return [hit(cross, vault), hit(vault / "escape.md", vault), hit(ok, vault)]

    reply = project_search.answer_project(PROJECT, QUERY, search=empty_search, vault_root=vault, qmd_search=fallback)
    assert captured == [(PROJECT, QUERY)]
    assert "Приватное правило Zeus." not in reply and "Правило за пределами Vault." not in reply
    assert "cross.md" not in reply and "escape.md" not in reply
    isolated_reply = project_search.answer_project(
        PROJECT, QUERY, search=empty_search, vault_root=vault,
        qmd_search=lambda *_: [hit(ok, vault)]
    )
    assert NEW_RULE in isolated_reply and "own.md" in isolated_reply


def test_archive_is_replaced_by_the_active_markdown_source(tmp_path):
    old = write_document(tmp_path, "notes/old.md", OLD_RULE)
    write_document(tmp_path, "notes/middle.md", "Оповещение приходит через 16 часов.", supersedes="old")
    write_document(tmp_path, "notes/new.md", NEW_RULE, supersedes="middle")

    reply = project_search.answer_project(
        PROJECT, QUERY, search=empty_search, vault_root=tmp_path, qmd_search=lambda *_: [hit(old, tmp_path)]
    )
    assert NEW_RULE in reply and "notes/new.md" in reply
    assert OLD_RULE not in reply and "16 часов" not in reply


@pytest.mark.parametrize("failure", ["changed", "deleted", "fork", "cycle", "broken"])
def test_unverifiable_sources_stay_uncertain(tmp_path, failure):
    old = write_document(tmp_path, "old.md", OLD_RULE)
    if failure != "broken":
        write_document(tmp_path, "new.md", NEW_RULE, supersedes="old")
    candidate = hit(old, tmp_path)
    if failure == "changed":
        write_document(tmp_path, "old.md", "Новая правка после индексации.")
    elif failure == "deleted":
        old.unlink()
    elif failure == "fork":
        write_document(tmp_path, "another.md", "Неоднозначная редакция.", supersedes="old")
    elif failure == "cycle":
        old = write_document(tmp_path, "old.md", OLD_RULE, supersedes="new")
        candidate = hit(old, tmp_path)
    elif failure == "broken":
        old = write_document(tmp_path, "old.md", OLD_RULE, supersedes="missing")
        candidate = hit(old, tmp_path)

    reply = project_search.answer_project(
        PROJECT, QUERY, search=empty_search, vault_root=tmp_path, qmd_search=lambda *_: [candidate]
    )
    expect_uncertain(reply)
    assert NEW_RULE not in reply


def test_fallback_outcomes_distinguish_empty_unavailable_and_expired(tmp_path):
    def answer(fallback):
        return project_search.answer_project(
            PROJECT, QUERY, search=empty_search, vault_root=tmp_path, qmd_search=fallback
        )

    empty = answer(lambda *_: [])
    unavailable = answer(Mock(side_effect=URLError("service stopped")))
    qmd_busy = answer(Mock(side_effect=HTTPError("http://127.0.0.1/", 429, "busy", {}, None)))
    assert len({empty, unavailable, qmd_busy}) == 3
    assert re.search("совпадени|ничего", empty, re.I)
    assert re.search("qmd.*недоступ", unavailable, re.I)
    assert re.search("занят|нагруз", qmd_busy, re.I)

    primary = Mock(side_effect=AssertionError("Expired request should not reach primary search"))
    fallback = Mock(side_effect=AssertionError("Expired request should not reach QMD"))
    expiry = project_search.answer_project(
        PROJECT, QUERY, search=primary, vault_root=tmp_path, qmd_search=fallback, admitted_at=monotonic() - 60
    )
    assert re.search("времени|тайм-аут", expiry, re.I)
    primary.assert_not_called()
    fallback.assert_not_called()


def test_busy_fallback_does_not_admit_unbounded_parallel_work(tmp_path):
    arrived = Barrier(3)
    unblock = Event()

    def slow_fallback(*_):
        arrived.wait(timeout=3)
        assert unblock.wait(timeout=4)
        return []

    def search_project():
        return project_search.answer_project(
            PROJECT, QUERY, search=empty_search, vault_root=tmp_path, qmd_search=slow_fallback
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(search_project)
        second = pool.submit(search_project)
        try:
            arrived.wait(timeout=3)
            primary = Mock(side_effect=AssertionError("Busy project search should reject excess work"))
            excess = project_search.answer_project(
                PROJECT, QUERY, search=primary, vault_root=tmp_path, qmd_search=slow_fallback
            )
            assert re.search("занят|нагруз", excess, re.I)
            primary.assert_not_called()
        finally:
            unblock.set()
        assert first.result(timeout=5) == second.result(timeout=5)


def test_refresh_keeps_a_complete_generation_on_repeats_and_failed_build(tmp_path):
    from twinctx.qmd_refresh import refresh

    vault, state = tmp_path / "vault", tmp_path / "qmd"
    current = state / "current"
    calls = tmp_path / "builder-calls"
    builder = tmp_path / "builder.py"
    builder.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "generation = Path(sys.argv[-1])\n"
        f"with Path({str(calls)!r}).open('a') as stream:\n"
        "    stream.write('build\\n')\n"
        "import sqlite3\n"
        "with sqlite3.connect(generation / 'index.sqlite') as db:\n"
        "    db.execute('CREATE TABLE fixture (marker TEXT)')\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(builder)]
    kept = write_document(vault, "keep.md", OLD_RULE)
    removed = write_document(vault, "remove.md", "Будет удалён.")
    third = write_document(vault, "zeus.md", "Документ Zeus.", project="zeus")

    refresh(vault, state, command)
    first = current.resolve(strict=True)
    assert (first / "index.sqlite").read_bytes().startswith(b"SQLite format 3")
    assert json.loads((first / "manifest.json").read_text())["files"] == {
        PROJECT: {"keep.md": hit(kept, vault)["sha256"], "remove.md": hit(removed, vault)["sha256"]},
        "zeus": {"zeus.md": hit(third, vault)["sha256"]},
    }

    removed.unlink()
    kept = write_document(vault, "keep.md", NEW_RULE)
    added = write_document(vault, "added.md", "Ссылка на порядок оповещения.")
    refresh(vault, state, command)
    second = current.resolve(strict=True)
    assert first != second
    expected = {"keep.md": hit(kept, vault)["sha256"], "added.md": hit(added, vault)["sha256"]}
    assert json.loads((second / "manifest.json").read_text())["files"][PROJECT] == expected
    assert sorted(p.name for p in (second / "docs" / PROJECT).iterdir()) == ["added.md", "keep.md"]
    assert (second / "docs" / PROJECT / "keep.md").read_bytes() == kept.read_bytes()

    refresh(vault, state, command)
    assert current.resolve(strict=True) == second
    assert calls.read_text().splitlines() == ["build", "build"]

    write_document(vault, "added.md", "Неполная сборка не станет активной.")
    fail = [
        sys.executable, "-c",
        "import pathlib,sys; (pathlib.Path(sys.argv[-1])/'index.sqlite').write_bytes(b'partial'); sys.exit(7)",
    ]
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        refresh(vault, state, fail)
    assert current.resolve(strict=True) == second
    assert (current / "index.sqlite").read_bytes().startswith(b"SQLite format 3")
    assert json.loads((current / "manifest.json").read_text())["files"][PROJECT] == expected


def test_semantic_hit_without_shared_terms_returns_source(tmp_path, monkeypatch):
    from stores.sqlite import SqliteStore
    from twinctx.engine import Engine

    path = write_document(tmp_path, "archive.md", "Резервные копии хранятся сорок пять дней.")
    fallback = Mock(return_value=[hit(path, tmp_path)])
    monkeypatch.setattr(project_search, "_qmd_search", fallback)
    store = SqliteStore(":memory:")
    try:
        engine = Engine(store)
        monkeypatch.delenv("TWINCTX_QMD_ENABLED", raising=False)
        args = dict(search=engine.search, vault_root=tmp_path)
        assert "archive.md" not in project_search.answer_project(PROJECT, "Когда удаляют бэкапы?", **args)
        fallback.assert_not_called()
        monkeypatch.setenv("TWINCTX_QMD_ENABLED", "1")
        result = project_search.answer_project(PROJECT, "Когда удаляют бэкапы?", **args)
        assert "Возможный источник" in result and "vault/archive.md" in result
        assert "сорок пять" not in result
        fallback.assert_called_once()
    finally:
        store.db.close()
