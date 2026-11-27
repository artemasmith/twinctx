"""Fenced PostgreSQL memory projection primitive; NOT the Store write-through.

This isolated primitive does not write or recover the JSONL canon and must not be
used by runtime until canon durability, replay, tenant isolation, and writer
fencing are established. It never adopts legacy unversioned memory implicitly.
"""

from __future__ import annotations

import hashlib
import json

from twinctx.contracts import COLS, Atom


class MemoryConflictError(RuntimeError):
    """A version, ownership, or replay identity precondition did not hold."""


def _hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class PgMemoryProjection:
    """Serializes cooperative writers and records idempotency in PG transactions.

    Caller owns an exclusive PgSourceIndex and must fence all other writers;
    there is no canon, tenant, fallback, or cutover support in this component.
    """

    def __init__(self, store, *, experimental=False):
        if experimental is not True:
            raise RuntimeError("memory projection requires experimental opt-in")
        self.store = store
        with store._transaction():
            self._lock()
            store.con.run("""create table if not exists twinctx_v1.memory_revisions (
                atom_id text primary key, revision bigint not null check(revision > 0),
                content_hash text not null)""")
            store.con.run("""create table if not exists twinctx_v1.memory_operations (
                operation_id text primary key, request_hash text not null,
                result_json jsonb not null)""")
            self._verify_schema()

    def _verify_schema(self):
        expected = {
            "memory_revisions": (
                [["atom_id", "text", True], ["revision", "bigint", True], ["content_hash", "text", True]],
                {("p", "PRIMARY KEY (atom_id)"), ("c", "CHECK ((revision > 0))")},
            ),
            "memory_operations": (
                [["operation_id", "text", True], ["request_hash", "text", True], ["result_json", "jsonb", True]],
                {("p", "PRIMARY KEY (operation_id)")},
            ),
        }
        for table, (columns, constraints) in expected.items():
            name = "twinctx_v1." + table
            actual = self.store.con.run(
                """select attname,format_type(atttypid,atttypmod),attnotnull from pg_attribute
                   where attrelid=cast(:table as regclass) and attnum > 0 and not attisdropped order by attnum""",
                table=name,
            )
            actual_constraints = self.store.con.run(
                "select contype,pg_get_constraintdef(oid) from pg_constraint where conrelid=cast(:table as regclass)",
                table=name,
            )
            if actual != columns or set(map(tuple, actual_constraints)) != constraints:
                raise RuntimeError("memory projection schema drift detected")

    def _lock(self):
        self.store.con.run("select pg_advisory_xact_lock(hashtext('aitwin-memory-projection-v1'))")

    def bootstrap_legacy(self, canonical, *, offline=False):
        """Register a fully reconciled, fenced legacy snapshot without changing memories.

        The caller must stop every legacy/remote writer and pass the already
        validated immutable canon snapshot. This does not fence those writers
        or activate the PostgreSQL runtime.
        """
        if offline is not True:
            raise ValueError('offline writer fencing is required')
        if not isinstance(canonical, dict) or len(canonical) > 100_000:
            raise ValueError('invalid bounded canonical snapshot')
        expected = {}
        for identity, atom in canonical.items():
            if not isinstance(atom, Atom) or atom.kind != 'memory' or not identity or identity != atom.id:
                raise ValueError('invalid canonical memory')
            expected[identity] = {c: getattr(atom, c) for c in COLS}
        with self.store._transaction():
            self._lock()
            rows = self.store.con.run(
                f"select {','.join(COLS)} from twinctx_v1.atoms where kind='memory' order by id for update"
            )
            actual = {row[0]: dict(zip(COLS, row)) for row in rows}
            if actual != expected:
                raise MemoryConflictError('canonical and PostgreSQL memory snapshot differ')
            if self.store.con.run('select 1 from twinctx_v1.memory_operations limit 1'):
                raise MemoryConflictError('memory operations exist; bootstrap requires reconciliation')
            revisions = {row[0]: (row[1], row[2]) for row in self.store.con.run(
                'select atom_id,revision,content_hash from twinctx_v1.memory_revisions'
            )}
            expected_revisions = {identity: (1, _hash(value)) for identity, value in actual.items()}
            if revisions and revisions != expected_revisions:
                raise MemoryConflictError('memory revision snapshot differs')
            if not revisions:
                for identity, (_, content_hash) in expected_revisions.items():
                    self.store.con.run(
                        'insert into twinctx_v1.memory_revisions(atom_id,revision,content_hash) values(:id,1,:hash)',
                        id=identity, hash=content_hash,
                    )
            return len(actual)

    def restore_legacy(self, canonical, *, offline=False):
        """Restore missing memory from a validated canon, in one PG transaction.

        Caller must fence every other writer and supply an immutable snapshot
        already validated by reconstruct_canon. Existing differences and atom
        identity collisions are never resolved by overwriting either version.
        """
        if offline is not True:
            raise ValueError('offline writer fencing is required')
        if not isinstance(canonical, dict) or len(canonical) > 100_000:
            raise ValueError('invalid bounded canonical snapshot')
        expected = {}
        for identity, atom in canonical.items():
            if not isinstance(atom, Atom) or atom.kind != 'memory' or not identity or identity != atom.id:
                raise ValueError('invalid canonical memory')
            expected[identity] = {c: getattr(atom, c) for c in COLS}
        with self.store._transaction():
            self._lock()
            if self.store.con.run('select 1 from twinctx_v1.memory_operations limit 1'):
                raise MemoryConflictError('memory operations exist; recovery requires reconciliation')
            rows = self.store.con.run(
                f"select {','.join(COLS)} from twinctx_v1.atoms where kind='memory' order by id for update"
            )
            actual = {row[0]: dict(zip(COLS, row)) for row in rows}
            if any(identity not in expected or expected[identity] != value for identity, value in actual.items()):
                raise MemoryConflictError('canonical and PostgreSQL memory snapshot differ')
            revisions = {row[0]: (row[1], row[2]) for row in self.store.con.run(
                'select atom_id,revision,content_hash from twinctx_v1.memory_revisions'
            )}
            expected_existing = {identity: (1, _hash(value)) for identity, value in actual.items()}
            if revisions and revisions != expected_existing:
                raise MemoryConflictError('memory revision snapshot differs')
            missing = sorted(expected.keys() - actual.keys())
            if missing and self.store.con.run(
                'select id from twinctx_v1.atoms where id=any(cast(:ids as text[])) limit 1', ids=missing
            ):
                raise MemoryConflictError('memory identity collides with another atom')
            columns = ','.join(COLS)
            params = ','.join(':' + c for c in COLS)
            for identity in missing:
                inserted = self.store.con.run(
                    f'insert into twinctx_v1.atoms({columns}) values({params}) returning {columns}',
                    **expected[identity],
                )[0]
                actual[identity] = dict(zip(COLS, inserted))
            for identity, value in actual.items():
                if identity not in revisions:
                    self.store.con.run(
                        'insert into twinctx_v1.memory_revisions(atom_id,revision,content_hash) values(:id,1,:hash)',
                        id=identity, hash=_hash(value),
                    )
            return len(actual)

    @staticmethod
    def _validate_operation(operation_id):
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id.encode()) > 200
            or "\x00" in operation_id
        ):
            raise ValueError("invalid operation id")

    @staticmethod
    def _validate_atom(atom):
        if not isinstance(atom, Atom) or atom.kind != "memory" or not atom.id or not atom.source:
            raise ValueError("a memory atom with id and source is required")
        if any(
            not isinstance(x, str) or "\x00" in x
            for x in (atom.id, atom.source, atom.corpus, atom.text, atom.valid_from, atom.valid_to, atom.supersedes)
        ):
            raise ValueError("invalid memory field")
        return {c: getattr(atom, c) for c in COLS}

    def _request(self, operation_id, request_hash):
        rows = self.store.con.run(
            "select request_hash,result_json::text from twinctx_v1.memory_operations where operation_id=:i",
            i=operation_id,
        )
        if not rows:
            return None
        if rows[0][0] != request_hash:
            raise MemoryConflictError("operation id reused with different request")
        return json.loads(rows[0][1])

    def _record(self, operation_id, request_hash, result):
        self.store.con.run(
            "insert into twinctx_v1.memory_operations(operation_id,request_hash,result_json) values(:i,:h,cast(:r as jsonb))",
            i=operation_id,
            h=request_hash,
            r=json.dumps(result, sort_keys=True),
        )

    def _current(self, atom_id, expected_revision):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("positive expected revision required")
        rows = self.store.con.run(
            f"select {','.join('a.' + c for c in COLS)}, r.revision,r.content_hash "
            "from twinctx_v1.atoms a left join twinctx_v1.memory_revisions r on r.atom_id=a.id "
            "where a.id=:id for update of a",
            id=atom_id,
        )
        if not rows or rows[0][COLS.index("kind")] != "memory":
            raise MemoryConflictError("memory not found")
        row = rows[0]
        values = dict(zip(COLS, row[: len(COLS)]))
        if row[-2] != expected_revision or row[-1] != _hash(values):
            raise MemoryConflictError("memory version or content changed; explicit reconciliation required")
        if values["valid_to"]:
            raise MemoryConflictError("memory is already closed")
        return values

    def _close(self, atom_id, expected_revision, valid_to):
        current = self._current(atom_id, expected_revision)
        self.store.con.run("update twinctx_v1.atoms set valid_to=:to where id=:id", to=valid_to, id=atom_id)
        current["valid_to"] = valid_to
        self.store.con.run(
            "update twinctx_v1.memory_revisions set revision=:revision,content_hash=:h where atom_id=:id",
            id=atom_id,
            revision=expected_revision + 1,
            h=_hash(current),
        )

    def create(self, operation_id, atom, *, expected_predecessor_revision=None):
        """Create memory; optionally close a versioned predecessor in the same transaction."""
        self._validate_operation(operation_id)
        values = self._validate_atom(atom)
        if values["valid_to"]:
            raise ValueError("new memory must be open")
        if values["supersedes"] == values["id"]:
            raise ValueError("memory cannot supersede itself")
        if bool(values["supersedes"]) != (expected_predecessor_revision is not None):
            raise ValueError("predecessor id and revision must be supplied together")
        if values["supersedes"] and not values["valid_from"]:
            raise ValueError("superseding memory requires a validity start")
        request_hash = _hash({"action": "create", "atom": values, "previous_revision": expected_predecessor_revision})
        with self.store._transaction():
            self._lock()
            prior = self._request(operation_id, request_hash)
            if prior is not None:
                return prior
            if values["supersedes"]:
                self._close(values["supersedes"], expected_predecessor_revision, values["valid_from"])
            cols = ",".join(COLS)
            params = ",".join(":" + c for c in COLS)
            inserted = self.store.con.run(
                f"insert into twinctx_v1.atoms({cols}) values({params}) on conflict(id) do nothing returning {cols}",
                **values,
            )
            if not inserted:
                raise MemoryConflictError("atom id already exists")
            self.store.con.run(
                "insert into twinctx_v1.memory_revisions(atom_id,revision,content_hash) values(:id,1,:h)",
                id=values["id"],
                h=_hash(dict(zip(COLS, inserted[0]))),
            )
            result = {"id": values["id"], "revision": 1}
            self._record(operation_id, request_hash, result)
            return result

    def close(self, operation_id, atom_id, *, expected_revision, valid_to):
        """Close exactly the expected open memory version; retry by operation id."""
        self._validate_operation(operation_id)
        if not isinstance(atom_id, str) or not atom_id or not isinstance(valid_to, str) or not valid_to:
            raise ValueError("memory id and validity end required")
        request_hash = _hash({"action": "close", "id": atom_id, "revision": expected_revision, "valid_to": valid_to})
        with self.store._transaction():
            self._lock()
            prior = self._request(operation_id, request_hash)
            if prior is not None:
                return prior
            self._close(atom_id, expected_revision, valid_to)
            result = {"id": atom_id, "revision": expected_revision + 1}
            self._record(operation_id, request_hash, result)
            return result
