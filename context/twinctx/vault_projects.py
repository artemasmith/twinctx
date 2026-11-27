"""Первичные проектные документы Vault и проверяемые цепочки редакций.

## Contract: VaultProjects

### Inputs
Корень Vault, относительный POSIX-путь Markdown и скалярные project/supersedes.
Сканирование может проверять дедлайн через checkpoint без аргументов.

### Outputs
Идентификатор проекта, SHA-256 исходных байтов и явная ссылка на предшественника.
active_revision возвращает путь единственной доказанной редакции или None.

### State
Модуль читает файловую систему и строит снимок заново при каждом сканировании.

### Multi-call invariants
Одинаковые файлы дают одинаковый снимок. Проверка пути применяется и при
индексации, и при прямом чтении. Ошибка чтения не выдаётся за пустой снимок.

### Edge cases
Служебные и скрытые пути, вложенные symlink, выход из корня и файлы без
однозначного проекта исключаются. Нет явной цепочки, цикл, ветвление,
разорванная или неоднозначная ссылка означают неподтверждённую редакцию.

### Dependencies
Локальный Vault и разбор frontmatter из twinctx.ingest.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import stat
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from .ingest import MAX_FILE, SKIP_DIRS, frontmatter

_PROJECT = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY


@dataclass(frozen=True)
class ProjectDocument:
    path: str
    project: str
    sha256: str
    supersedes: str


def normalize_project(value: str) -> str | None:
    return value.lower() if _PROJECT.fullmatch(value) else None


def _excluded(name: str) -> bool:
    return name.startswith(".") or name.casefold() in SKIP_DIRS


def _markdown_parts(relative: str) -> list[str]:
    parts = relative.split("/")
    if (
        "\\" in relative
        or any(ord(char) < 32 or ord(char) == 127 for char in relative)
        or any(not part or _excluded(part) for part in parts)
        or PurePosixPath(parts[-1]).suffix.casefold() != ".md"
    ):
        raise ValueError("path is not an allowed Vault Markdown file")
    return parts


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_bytes(root: Path, parts: list[str]) -> bytes:
    with ExitStack() as stack:
        base = os.open(root, _DIRECTORY_FLAGS)
        stack.callback(os.close, base)
        directories = []
        parent = base
        for name in parts[:-1]:
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
            stack.callback(os.close, child)
            directories.append((parent, name, child))
            parent = child
        name = parts[-1]
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        stream = stack.enter_context(os.fdopen(descriptor, "rb"))
        initial = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError("Vault path is not a plain Markdown file")
        data = stream.read(MAX_FILE + 1)
        latest = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not (
            _file_identity(initial) == _file_identity(os.fstat(stream.fileno())) == _file_identity(latest)
        ):
            raise OSError("Vault file changed while reading")
        for parent, directory, child in directories:
            expected = os.fstat(child)
            current = os.stat(directory, dir_fd=parent, follow_symlinks=False)
            if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
                raise OSError("Vault directory changed while reading")
    return data


def read_document(root: Path, relative: str) -> bytes:
    """Вернуть первичный Markdown в пределах установленного лимита twinctx."""
    data = _read_bytes(root.resolve(strict=True), _markdown_parts(relative))
    if len(data) > MAX_FILE:
        raise ValueError("Vault Markdown file exceeds the configured limit")
    return data


def _metadata(data: bytes) -> tuple[str | None, str]:
    # Сначала ограничиваемся заголовком: частичная или ошибочная кодировка
    # тела не должна превращать неизвестную редакцию в доказанную.
    lines = data.splitlines()
    if not lines or lines[0] != b"---":
        return None, ""
    try:
        end = lines.index(b"---", 1)
        header = b"\n".join(lines[1:end]).decode("utf-8").splitlines()
    except (ValueError, UnicodeDecodeError):
        return None, ""
    metadata = []
    continued_supersedes = False
    previous = ""
    for line in header:
        if line[:1].isspace():
            if previous == "supersedes" and line.strip() and not line.lstrip().startswith("#"):
                continued_supersedes = True
            continue
        previous, separator, _ = line.partition(":")
        previous = previous.strip()
        if separator and previous in {"project", "supersedes"}:
            metadata.append((previous, line))
    fields, _, _ = frontmatter("\n".join(["---", *(line for _, line in metadata), "---"]))
    keys = [key for key, _ in metadata]
    project = normalize_project(fields.get("project", "")) if keys.count("project") == 1 else None
    if keys.count("supersedes") > 1 or continued_supersedes:
        return project, "[]"
    return project, fields.get("supersedes", "")


def _raise_scan_error(error: OSError) -> None:
    raise error


def scan_projects(
    root: Path, checkpoint: Callable[[], None] | None = None,
) -> dict[str, ProjectDocument]:
    """Собрать детерминированный снимок разрешённых проектных документов."""
    resolved = root.resolve(strict=True)
    documents = {}
    for current, directories, files in os.walk(resolved, onerror=_raise_scan_error):
        if checkpoint is not None:
            checkpoint()
        path = Path(current)
        directories[:] = sorted(
            name for name in directories if not _excluded(name) and not (path / name).is_symlink()
        )
        for name in sorted(files):
            if checkpoint is not None:
                checkpoint()
            file = path / name
            if _excluded(name) or file.suffix.casefold() != ".md" or not file.is_file() or file.is_symlink():
                continue
            relative = file.relative_to(resolved).as_posix()
            data = _read_bytes(resolved, _markdown_parts(relative))
            project, supersedes = _metadata(data)
            if project is not None:
                if len(data) > MAX_FILE:
                    raise ValueError("project Markdown file exceeds the configured limit")
                data.decode("utf-8")
                documents[relative] = ProjectDocument(relative, project, hashlib.sha256(data).hexdigest(), supersedes)
    return documents


def _reference_candidates(
    origin: str, reference: str, paths: dict[str, set[str]], stems: dict[str, set[str]],
) -> set[str]:
    name = reference.strip()
    wiki = re.fullmatch(r"\[\[([^\[\]#|]+)\]\]", name)
    if wiki:
        name = wiki[1].strip()
    if not name or name.startswith("/") or re.search(r"[:\[\]#|\\\\\x00-\x1f]", name):
        return set()
    key = name.casefold()
    if "/" not in key:
        stem = key.removesuffix(".md")
        return stems.get(stem, set())
    if not key.endswith(".md"):
        key += ".md"
    roots = (posixpath.normpath(key), posixpath.normpath(posixpath.join(posixpath.dirname(origin), key)))
    return set().union(*(paths.get(relative, set()) for relative in roots))


def active_revision(documents: dict[str, ProjectDocument], relative: str) -> str | None:
    """Подтвердить терминальную редакцию по однозначной цепочке того же проекта."""
    candidate = documents.get(relative)
    if candidate is None:
        return None
    project_files = {path: doc for path, doc in documents.items() if doc.project == candidate.project}
    paths: dict[str, set[str]] = defaultdict(set)
    stems: dict[str, set[str]] = defaultdict(set)
    for path in project_files:
        paths[path.casefold()].add(path)
        stems[PurePosixPath(path).stem.casefold()].add(path)

    previous = {}
    successors: dict[str, list[str]] = defaultdict(list)
    uncertain = {path for alternatives in paths.values() if len(alternatives) > 1 for path in alternatives}
    for path, doc in project_files.items():
        if doc.supersedes:
            choices = _reference_candidates(path.casefold(), doc.supersedes, paths, stems)
            if len(choices) != 1:
                uncertain.add(path)
                uncertain.update(choices)
            else:
                old = next(iter(choices))
                previous[path] = old
                successors[old].append(path)

    # Сначала доходим до начала цепочки, затем проверяем её целиком. Поэтому
    # развилка блокирует также запрос по одной из поздних редакций.
    visited = set()
    oldest = relative
    while oldest in previous:
        if oldest in visited:
            return None
        visited.add(oldest)
        oldest = previous[oldest]
    current = oldest
    while current not in uncertain:
        following = successors[current]
        if not following:
            return current
        if len(following) != 1:
            break
        current = following[0]
    return None
