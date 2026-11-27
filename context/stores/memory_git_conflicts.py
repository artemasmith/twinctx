"""Offline detector for incoming Git changes to the legacy memory canon.

It does not fetch, pull, merge, edit the live journal or integrate a conflict.
Only use with fenced writers, trusted paths and a private conflict directory.
External editors which ignore fencing can still race; this is not a sync engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

_MEMORY_PATH = 'memory/atoms.jsonl'
_MAX_BYTES = 128 * 1024 * 1024
_OID = re.compile(r'^[0-9a-f]{40}(?:[0-9a-f]{24})?$')


@dataclass(frozen=True)
class Inspection:
    status: str
    directory: Path | None = None


class InspectionStaleError(RuntimeError):
    """Inputs changed after preserving a conflict; snapshots remain available."""

    def __init__(self, directory):
        super().__init__('inputs changed during inspection; repeat before reconciliation')
        self.directory = directory


def _git(repo, *args):
    # Do not let a caller's GIT_DIR/GIT_WORK_TREE redirect inspection to a
    # different repository. No command here fetches or writes a worktree.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update({'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0', 'GIT_OPTIONAL_LOCKS': '0'})
    return subprocess.run(['git', '-C', str(repo), *args], env=env, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout


def _blob(repo, commit):
    entry = _git(repo, 'ls-tree', '-z', commit, '--', _MEMORY_PATH)
    if not entry:
        return None
    chunks = entry.split(b'\0')
    if len(chunks) != 2 or chunks[1]:
        raise ValueError('unexpected memory tree entries')
    header, name = chunks[0].split(b'\t', 1)
    mode, kind, _oid = header.split(b' ')
    if name != _MEMORY_PATH.encode() or kind != b'blob' or mode not in (b'100644', b'100755'):
        raise ValueError('incoming memory is not a regular file')
    size = int(_git(repo, 'cat-file', '-s', f'{commit}:{_MEMORY_PATH}').strip())
    if size > _MAX_BYTES:
        raise ValueError('memory blob exceeds inspection limit')
    data = _git(repo, 'cat-file', 'blob', f'{commit}:{_MEMORY_PATH}')
    if len(data) != size:
        raise RuntimeError('git blob length changed')
    return data


def _live(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    try:
        first = os.fstat(fd)
        if not stat.S_ISREG(first.st_mode) or first.st_nlink != 1 or first.st_size > _MAX_BYTES:
            raise ValueError('live memory is not a bounded regular file')
        with os.fdopen(os.dup(fd), 'rb') as source:
            data = source.read(_MAX_BYTES + 1)
        end = os.fstat(fd)
        current = path.lstat()
        def identity(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        if len(data) > _MAX_BYTES or identity(first) != identity(end) or identity(first) != identity(current):
            raise RuntimeError('live memory changed during inspection')
        return data
    finally:
        os.close(fd)


def _write(fd, name, data):
    dest = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(dest, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _preserve(root, local, incoming, base_oid, incoming_oid):
    """Write both available versions and manifest last; never replace a file."""
    rootfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(rootfd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('conflict directory must be private and owned by caller')
        name = 'memory-' + uuid.uuid4().hex
        os.mkdir(name, 0o700, dir_fd=rootfd)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=rootfd)
        try:
            if local is not None:
                _write(fd, 'local.jsonl', local)
            if incoming is not None:
                _write(fd, 'incoming.jsonl', incoming)
            metadata = {
                'version': 1, 'base_commit': base_oid, 'incoming_commit': incoming_oid,
                'local_sha256': hashlib.sha256(local).hexdigest() if local is not None else None,
                'incoming_sha256': hashlib.sha256(incoming).hexdigest() if incoming is not None else None,
            }
            _write(fd, 'manifest.json', (json.dumps(metadata, sort_keys=True) + '\n').encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(rootfd)
        return Path(root) / name
    finally:
        os.close(rootfd)


def inspect_incoming(repo, incoming_commit, conflict_dir, *, offline=False):
    """Compare live canon with one pinned commit; preserve conflict, never merge.

    Caller must fence writers and prevent HEAD/upstream from changing until the
    eventual separate integration; a successful inspection is not authorization
    to execute a later `git pull` or to overwrite the local journal.
    """
    if offline is not True:
        raise RuntimeError('fenced offline inspection required')
    if not isinstance(incoming_commit, str) or not _OID.fullmatch(incoming_commit):
        raise ValueError('a pinned full commit id is required')
    repo = Path(repo)
    root = Path(os.fsdecode(_git(repo, 'rev-parse', '--show-toplevel').rstrip(b'\n')))
    if not os.path.samefile(repo, root):
        raise ValueError('journal repository root required')
    # Conflict snapshots contain raw personal data: never place them under the
    # Git worktree where an ordinary sync could accidentally publish them.
    if Path(conflict_dir).resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError('conflict directory must be outside journal repository')
    head = _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    incoming = _git(repo, 'rev-parse', '--verify', f'{incoming_commit}^{{commit}}').decode().strip()
    if incoming != incoming_commit:
        raise ValueError('incoming object is not the exact commit id')
    bases = _git(repo, 'merge-base', '--all', head, incoming).decode().splitlines()
    if len(bases) != 1:
        raise ValueError('one unambiguous Git merge base is required')
    base = bases[0]
    before = _blob(repo, base)
    remote = _blob(repo, incoming)
    if before == remote:
        return Inspection('unchanged')
    local = _live(root / _MEMORY_PATH)
    if _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip() != head:
        raise RuntimeError('journal HEAD changed during inspection')
    if remote == local:
        return Inspection('identical')
    directory = _preserve(conflict_dir, local, remote, base, incoming)
    try:
        current = _live(root / _MEMORY_PATH)
        current_head = _git(repo, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise InspectionStaleError(directory) from exc
    if current != local or current_head != head:
        raise InspectionStaleError(directory)
    return Inspection('conflict', directory)
