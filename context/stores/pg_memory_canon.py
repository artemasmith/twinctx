"""Fenced PostgreSQL memory writes with a durable JSONL canon and crash replay.

Only cooperating writers sharing the same local filesystem lock are fenced.
Every legacy, external and remote writer must be stopped before activation.
The caller must provide a tenant-specific PostgreSQL database and private canon.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from twinctx.contracts import COLS, Atom

from .memory_canon_recovery import CanonConflictError, _no_duplicate_keys, reconstruct_canon
from .memory_closure_protocol import _fields, _lock, _sync_published
from .pg_memory_projection import PgMemoryProjection, _hash


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _state_digest(state):
    rows = []
    for identity in sorted(state):
        row = _fields(state[identity])
        if row['trust'] is not None:
            row['trust'] = float(row['trust'])
            if row['trust'] == 0:
                row['trust'] = 0.0
        rows.append(row)
    return _digest(_json(rows))


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PgMemoryCanon:
    """Durable cooperative write-through wrapper for an existing PG store.

    One canon file and one PG database form a tenant. The deployer must fence
    non-cooperating writers and ensure the canon resides on a durable local FS.
    Recovery is performed before every read-modify-write and can be invoked at
    process startup. This class does not itself provide cross-host fencing.
    """

    def __init__(self, store, canon, *, state_dir):
        self.store = store
        requested_canon = Path(canon).absolute()
        requested_state = Path(state_dir).absolute()
        if requested_canon.is_symlink() or requested_state.is_symlink():
            raise CanonConflictError('memory paths must not be symlinks')
        self.canon = requested_canon.resolve(strict=True)
        self.state_dir = requested_state.resolve(strict=True)
        identity = _digest(os.fsencode(self.canon))[:32]
        self.pending_path = self.state_dir / (identity + '.pending')
        self.lock_path = self.state_dir / (identity + '.lock')
        self._validate_environment()
        self.projection = PgMemoryProjection(store, experimental=True)

    def __getattr__(self, name):
        return getattr(self.store, name)

    def close(self):
        self.store.close()

    def online_reader(self):
        return self.store.online_reader()

    def _validate_environment(self):
        directory = self.state_dir.lstat()
        if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
                or stat.S_IMODE(directory.st_mode) & 0o077):
            raise CanonConflictError('memory state directory must be private and owned by writer')
        if directory.st_dev != self.canon.parent.stat().st_dev:
            raise CanonConflictError('memory state and canon must share one filesystem')
        info = self.canon.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
            raise CanonConflictError('untrusted memory canon')

    @contextmanager
    def _guard(self):
        self._validate_environment()
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise CanonConflictError('invalid memory journal lock')
            _lock(fd)
            self._validate_environment()
            yield
        finally:
            os.close(fd)

    def _bytes(self):
        fd = os.open(self.canon, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            st = os.fstat(stream.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != os.geteuid():
                raise CanonConflictError('untrusted memory canon')
            if st.st_size > 128 * 1024 * 1024:
                raise CanonConflictError('memory canon exceeds size limit')
            data = stream.read(128 * 1024 * 1024 + 1)
            if len(data) > 128 * 1024 * 1024:
                raise CanonConflictError('memory canon exceeds size limit')
            return data

    def _database_state(self):
        atoms = {a.id: a for a in self.store.memories()}
        with self.store._lock:
            versions = self.store.con.run('select atom_id,revision,content_hash from twinctx_v1.memory_revisions')
        if len(versions) != len(atoms):
            raise CanonConflictError('unregistered PostgreSQL memory')
        for identity, revision, content_hash in versions:
            if identity not in atoms or revision < 1 or _hash(_fields(atoms[identity])) != content_hash:
                raise CanonConflictError('PostgreSQL memory revision differs')
        return atoms

    def _database_identity(self):
        with self.store._lock:
            rows = self.store.con.run(
                "select current_database(), d.oid::text, (pg_control_system()).system_identifier::text "
                "from pg_database d where d.datname=current_database()"
            )
        if len(rows) != 1 or len(rows[0]) != 3 or any(not isinstance(x, str) or not x for x in rows[0]):
            raise CanonConflictError('PostgreSQL database identity unavailable')
        return rows[0]

    def _consistent(self):
        state = reconstruct_canon(self.canon, allow_dangling=True)
        indexed = self._database_state()
        if set(state) != set(indexed) or any(_fields(state[k]) != _fields(indexed[k]) for k in state):
            raise CanonConflictError('canon and PostgreSQL memory differ')
        return state, indexed

    def _read_pending(self):
        try:
            fd = os.open(self.pending_path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise CanonConflictError('untrusted pending memory operation')
            raw = stream.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise CanonConflictError('pending memory operation too large')
        try:
            data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
            required = {'schema', 'operation_id', 'action', 'atom', 'revision', 'database_identity', 'before_file', 'after_file',
                        'before_state', 'after_state'}
            if set(data) != required or data['schema'] != 'aitwin.pg-memory-pending.v1':
                raise ValueError('invalid pending schema')
            if data['action'] not in ('create', 'close') or not isinstance(data['atom'], dict):
                raise ValueError('invalid pending action')
            if not isinstance(data['operation_id'], str) or not re.fullmatch('[0-9a-f]{32}', data['operation_id']):
                raise ValueError('invalid pending operation')
            if set(data['atom']) != set(COLS):
                raise ValueError('invalid pending atom fields')
            if (not isinstance(data['database_identity'], list) or len(data['database_identity']) != 3
                    or any(not isinstance(x, str) or not x for x in data['database_identity'])):
                raise ValueError('invalid pending database identity')
            if data['action'] == 'close' and (type(data['revision']) is not int or data['revision'] < 1):
                raise ValueError('invalid pending revision')
            if any(not isinstance(data[k], str) or not re.fullmatch('[0-9a-f]{64}', data[k]) for k in
                   ('before_file', 'after_file', 'before_state', 'after_state')):
                raise ValueError('invalid pending digest')
            return data
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CanonConflictError('invalid pending memory operation') from exc

    def _write_pending(self, data):
        if self.pending_path.exists() or self.pending_path.is_symlink():
            raise CanonConflictError('pending memory operation already exists')
        payload = _json(data)
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError('pending memory operation too large')
        fd, temp = tempfile.mkstemp(prefix='.memory-pending-', dir=self.state_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.pending_path)
            _sync_dir(self.state_dir)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def _candidate(self, before, atom):
        line = (json.dumps(_fields(atom), ensure_ascii=False, sort_keys=True) + '\n').encode()
        after = before + line
        fd, temp = tempfile.mkstemp(prefix='.memory-check-', dir=self.state_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(after)
            state = reconstruct_canon(Path(temp), allow_dangling=True)
            if atom.id not in state or _fields(state[atom.id]) != _fields(atom):
                raise CanonConflictError('candidate memory differs')
            return after, state
        finally:
            os.unlink(temp)

    def _publish_candidate(self, atom, expected_before, expected_after, checkpoint):
        before = self._bytes()
        if _digest(before) != expected_before:
            raise CanonConflictError('canon changed before publication')
        after, _ = self._candidate(before, atom)
        if _digest(after) != expected_after:
            raise CanonConflictError('pending memory candidate differs')
        fd, temp = tempfile.mkstemp(prefix='.memory-publish-', dir=self.state_dir)
        try:
            source_stat = self.canon.lstat()
            with os.fdopen(fd, 'wb') as stream:
                os.fchown(stream.fileno(), source_stat.st_uid, source_stat.st_gid)
                os.fchmod(stream.fileno(), stat.S_IMODE(source_stat.st_mode))
                stream.write(after)
                stream.flush()
                os.fsync(stream.fileno())
            if checkpoint:
                checkpoint('after_file_sync')
            if _digest(self._bytes()) != expected_before:
                raise CanonConflictError('canon changed during publication')
            os.replace(temp, self.canon)
            if checkpoint:
                checkpoint('after_rename')
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def _recover_locked(self, checkpoint=None):
        operation = self._read_pending()
        if operation is None:
            return False
        try:
            atom = Atom(**operation['atom'])
            self.projection._validate_atom(atom)
            if operation['action'] == 'create':
                if (atom.valid_to or atom.supersedes == atom.id
                        or (atom.supersedes and (not atom.valid_from or type(operation['revision']) is not int
                                                or operation['revision'] < 1))
                        or (not atom.supersedes and operation['revision'] is not None)):
                    raise ValueError('invalid pending create')
            elif not atom.valid_to:
                raise ValueError('invalid pending closure')
        except (TypeError, ValueError) as exc:
            raise CanonConflictError('invalid pending memory atom') from exc
        if self._database_identity() != operation['database_identity']:
            raise CanonConflictError('pending memory database identity differs')
        file_digest = _digest(self._bytes())
        db_digest = _state_digest(self._database_state())
        if file_digest not in (operation['before_file'], operation['after_file']):
            raise CanonConflictError('canon changed during pending memory operation')
        if db_digest not in (operation['before_state'], operation['after_state']):
            raise CanonConflictError('PostgreSQL changed during pending memory operation')
        if file_digest == operation['before_file']:
            if db_digest != operation['before_state']:
                raise CanonConflictError('index committed before canon publication')
            self._publish_candidate(atom, operation['before_file'], operation['after_file'], checkpoint)
        _sync_published(self.canon)
        _sync_dir(self.state_dir)
        if _digest(self._bytes()) != operation['after_file']:
            raise CanonConflictError('published canon differs from pending operation')
        if _state_digest(reconstruct_canon(self.canon, allow_dangling=True)) != operation['after_state']:
            raise CanonConflictError('published memory state differs')
        if checkpoint:
            checkpoint('after_canon_publish')
        if operation['action'] == 'create':
            self.projection.create(operation['operation_id'], atom,
                                   expected_predecessor_revision=operation['revision'])
        else:
            self.projection.close(operation['operation_id'], atom.id,
                                  expected_revision=operation['revision'], valid_to=atom.valid_to)
        if _state_digest(self._database_state()) != operation['after_state']:
            raise CanonConflictError('committed PostgreSQL memory differs')
        if checkpoint:
            checkpoint('after_db_commit')
        self.pending_path.unlink()
        _sync_dir(self.state_dir)
        return True

    def recover(self):
        with self._guard():
            return self._recover_locked()

    def legacy_dangling(self):
        """Report unresolved predecessor identities without changing history."""
        with self._guard():
            self._recover_locked()
            state, _ = self._consistent()
            return sorted((atom.id, atom.supersedes) for atom in state.values()
                          if atom.supersedes and atom.supersedes not in state)

    def _revision(self, identity):
        with self.store._lock:
            rows = self.store.con.run('select revision from twinctx_v1.memory_revisions where atom_id=:id', id=identity)
        if len(rows) != 1:
            raise CanonConflictError('memory revision missing')
        return rows[0][0]

    def _write(self, action, atom, revision, before_state, checkpoint):
        before = self._bytes()
        after, final_state = self._candidate(before, atom)
        operation = {
            'schema': 'aitwin.pg-memory-pending.v1', 'operation_id': uuid.uuid4().hex,
            'action': action, 'atom': _fields(atom), 'revision': revision,
            'database_identity': self._database_identity(),
            'before_file': _digest(before), 'after_file': _digest(after),
            'before_state': _state_digest(before_state), 'after_state': _state_digest(final_state),
        }
        if _digest(self._bytes()) != operation['before_file']:
            raise CanonConflictError('canon changed while preparing operation')
        self._write_pending(operation)
        if checkpoint:
            checkpoint('after_intent')
        self._recover_locked(checkpoint)

    def add_memory(self, atom, *, _checkpoint=None):
        return self._add_memory(atom, allow_reobservation=False, checkpoint=_checkpoint)

    def add_memory_idempotent(self, atom, *, _checkpoint=None):
        """Re-observe an open content-addressed fact without changing its history.

        The caller must derive the id from the fact's identity. Only its
        observation/start timestamps may differ; every persistent content and
        history field must match, and a closed fact is never reopened.
        """
        return self._add_memory(atom, allow_reobservation=True, checkpoint=_checkpoint)

    def _add_memory(self, atom, *, allow_reobservation, checkpoint):
        values = self.projection._validate_atom(atom)
        atom = Atom(**values)
        if atom.valid_to or atom.supersedes == atom.id or (atom.supersedes and not atom.valid_from):
            raise ValueError('invalid new memory validity')
        with self._guard():
            self._recover_locked()
            state, indexed = self._consistent()
            if atom.id in state:
                existing = state[atom.id]
                old_fields, new_fields = _fields(existing), _fields(atom)
                if old_fields == new_fields:
                    return
                if allow_reobservation and not existing.valid_to:
                    for field in ('observed_at', 'valid_from'):
                        old_fields.pop(field)
                        new_fields.pop(field)
                    if old_fields == new_fields:
                        return
                raise CanonConflictError('memory identity differs')
            if self.store.get([atom.id]):
                raise CanonConflictError('memory identity collides with another atom')
            revision = None
            if atom.supersedes:
                predecessor = indexed.get(atom.supersedes)
                if not predecessor or predecessor.valid_to:
                    raise CanonConflictError('memory predecessor missing or closed')
                revision = self._revision(atom.supersedes)
            self._write('create', atom, revision, state, checkpoint)

    def close_memory(self, atom_id, valid_to, *, _checkpoint=None):
        if not isinstance(atom_id, str) or not atom_id or not isinstance(valid_to, str) or not valid_to:
            raise ValueError('memory id and validity end required')
        with self._guard():
            self._recover_locked()
            state, _ = self._consistent()
            atom = state.get(atom_id)
            if atom is None:
                raise CanonConflictError('memory is missing')
            if atom.valid_to:
                if atom.valid_to == valid_to:
                    return
                raise CanonConflictError('memory is already closed differently')
            revision = self._revision(atom_id)
            self._write('close', replace(atom, valid_to=valid_to), revision, state, _checkpoint)

    def import_memory_canon(self, path):
        if Path(path).resolve(strict=True) != self.canon:
            raise ValueError('import only accepts configured canon')
        with self._guard():
            self._recover_locked()
            state, _ = self._consistent()
            _sync_published(self.canon)
            return len(state)

    def export_memory_canon(self, path):
        requested = Path(path).absolute()
        if requested.is_symlink():
            raise ValueError('export destination must not be a symlink')
        destination = requested.resolve(strict=False)
        with self._guard():
            self._recover_locked()
            state, _ = self._consistent()
            _sync_published(self.canon)
            if destination == self.canon:
                return len(state)
            fd, temporary = tempfile.mkstemp(prefix='.memory-export-', dir=destination.parent)
            try:
                with os.fdopen(fd, 'wb') as output, self.canon.open('rb') as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                # Hard-link publication is atomic and refuses an existing name.
                os.link(temporary, destination, follow_symlinks=False)
                os.unlink(temporary)
                _sync_dir(destination.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return len(state)
