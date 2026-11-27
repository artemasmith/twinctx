"""Публикует готовое локальное поколение проектного индекса QMD.

## Contract: ProjectIndexRefresh
Inputs: штатный Vault, приватный каталог состояния, команда сборщика.
Outputs: готовый current либо предыдущее готовое поколение при отказе.
State: каталог generations и один writer-lock. Текст остаётся на узле.
Multi-call: неизменившийся снимок пропускается, current меняется атомарно.
Edge cases: пустой Vault, удаление, смена проекта, смена файла во время
сборки, сбой сборщика, конкурентный запуск и отключённое питание.
Dependencies: файловая система, twinctx и локальная сборка QMD.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from twinctx.vault_projects import read_document, scan_projects


def _manifest(documents):
    files = {}
    for path, doc in sorted(documents.items()):
        files.setdefault(doc.project, {})[path] = doc.sha256
    return {"files": files}


def _index_ready(path):
    if not path.is_file() or not path.stat().st_size:
        return False
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            checks = db.execute("PRAGMA quick_check(1)").fetchall()
            return checks == [("ok",)]
    except sqlite3.OperationalError as error:
        # The serving SDK loads sqlite-vec; stdlib SQLite can inspect the
        # rest of the schema but cannot check those virtual tables.
        return str(error) == "no such module: vec0"
    except sqlite3.DatabaseError:
        return False


def _previous_manifest(current):
    try:
        target = current.resolve(strict=True)
        if target.parent != current.parent / "generations" or not _index_ready(target / "index.sqlite"):
            return None
        return json.loads((target / "manifest.json").read_text())
    except (OSError, ValueError, RuntimeError):
        return None


def refresh(vault: Path, state: Path, command) -> bool:
    vault = vault.resolve(strict=True)
    if state.resolve().is_relative_to(vault):
        raise ValueError("QMD state must be outside the vault")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = state.resolve()
    generations = state / "generations"
    generations.mkdir(mode=0o700, exist_ok=True)
    current = state / "current"
    with (state / "refresh.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        documents = scan_projects(vault)
        manifest = _manifest(documents)
        if manifest == _previous_manifest(current):
            return False
        generation = Path(tempfile.mkdtemp(prefix="generation-", dir=generations))
        pending = state / f".{generation.name}.current"
        published = False
        try:
            (generation / "docs").mkdir(mode=0o700)
            for path, doc in documents.items():
                content = read_document(vault, path)
                if hashlib.sha256(content).hexdigest() != doc.sha256:
                    raise ValueError("Vault changed while staging the index")
                destination = generation / "docs" / doc.project / path
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination.write_bytes(content)
                destination.chmod(0o600)
            (generation / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
            subprocess.run([*command, str(generation)], check=True, timeout=540)
            if not _index_ready(generation / "index.sqlite"):
                raise RuntimeError("QMD did not create an index")
            if documents != scan_projects(vault):
                raise ValueError("Vault changed while building the index")
            pending.symlink_to(generation.relative_to(state), target_is_directory=True)
            os.replace(pending, current)
            published = True
        finally:
            pending.unlink(missing_ok=True)
            if not published:
                shutil.rmtree(generation)
        # Keep the previous generation available for a search that opened it
        # immediately before publication. QMD serving uses one slot and timeout.
        previous = sorted(
            (path for path in generations.iterdir() if re.fullmatch(r"generation-[a-z0-9_]+",path.name)
             and path != generation and path.is_dir() and not path.is_symlink()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for obsolete in previous[1:]:
            shutil.rmtree(obsolete)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vault", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("builder", type=Path)
    args = parser.parse_args()
    try:
        published = refresh(args.vault, args.state, ["node", str(args.builder), "build"])
    except BlockingIOError:
        print("QMD refresh already running")
        return 0
    print("QMD index published" if published else "QMD index is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
