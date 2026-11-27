"""SQLite adapter for the twinctx Store contract (stdlib only). Lives OUTSIDE the core package:
the core (twinctx/) receives a Store and never imports this module (import-linter enforces).

One file (`$HARNESS_HOME/data/context/twinctx.db`), rebuildable from sources — the JSONL canon
is the truth and memory atoms are written through to it at add time (RPO ≈ 0), so this file is
an index in the full sense. WAL + busy_timeout: several processes of the owning zone share the
file; writers serialise on the same host.
"""
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

from twinctx.contracts import EDGE_COLS, EDGE_SCHEMA_VERSION, Atom, COLS, now

from .memory_history import family_query

SCHEMA = """
create table if not exists atoms(
  id text primary key, source text not null, corpus text not null, kind text not null,
  doc_type text, title text, heading text, text text not null,
  start_line int, end_line int, updated text, observed_at text,
  valid_from text, valid_to text, trust real default 1.0, ord int default 0, supersedes text);
create index if not exists memory_supersedes on atoms(supersedes) where kind='memory';
create index if not exists atoms_source on atoms(source);
create virtual table if not exists atoms_fts using fts5(
  atom_id unindexed, title, heading, text, tokenize='unicode61 remove_diacritics 2');
create table if not exists edges(
  src text not null, dst text not null, kind text not null, weight real default 1.0,
  project text default '', provenance text not null default '', basis text not null default '',
  owner text default '', version int default 2,
  valid_from text default '', valid_to text default '', created_at text default '',
  primary key(src, dst, kind),
  check(length(basis) between 1 and 200));
create index if not exists edges_dst on edges(dst);
create index if not exists edges_project on edges(project);
create table if not exists sources(source text primary key, corpus text, sha text, indexed_at text, updated text);
create table if not exists vectors(atom_id text primary key, vec blob);
create table if not exists meta(key text primary key, value text);
"""


def memory_canon_path() -> Path:
    """The append-only JSONL canon of memory atoms — the durable truth for memory."""
    home = Path(os.environ.get("HARNESS_HOME", Path.home() / "harness"))
    return Path(os.environ.get("TWINCTX_MEMORY_CANON", home / "journal" / "memory" / "atoms.jsonl"))


def default_db_path() -> Path:
    home = Path(os.environ.get("HARNESS_HOME", Path.home() / "harness"))
    return Path(os.environ.get("TWINCTX_DB", home / "data" / "context" / "twinctx.db"))


class SqliteStore:
    def __init__(self, path: str | Path):
        ro = str(path).startswith("file:")   # "file:...?mode=ro" — read-only inspection of a live db
        if not ro:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, uri=ro, timeout=30)
        self._memory_thread_lock = threading.RLock()
        self.db.row_factory = sqlite3.Row
        self._pragma("pragma busy_timeout=30000")   # multi-process writers on one host: wait, don't fail
        self._pragma("pragma journal_mode=WAL")     # readers never block the writer and vice versa
        self._pragma("pragma synchronous=NORMAL")   # durable across app crashes; power loss risks one checkpoint
        self.edges_ok = True   # False on a read-only db with old edges: search works, graph does not
        try:
            cols = {r[1] for r in self.db.execute("pragma table_info(edges)")}
            if cols and not set(EDGE_COLS) <= cols:
                # v1 edges (src,dst,kind,weight): the edge table is a rebuilt index — drop and let the
                # next indexing pass repopulate it; atoms/vectors/atoms_fts are never touched (A1).
                self.db.execute("drop table edges")
                self.db.commit()
            self.db.executescript(SCHEMA)
        except sqlite3.OperationalError:
            self.edges_ok = False

    def _pragma(self, stmt: str) -> None:
        try:
            self.db.execute(stmt)
        except sqlite3.OperationalError:
            pass   # read-only connections cannot switch the journal mode

    # ── writes ────────────────────────────────────────────────────────────────────────────────
    def replace_source(self, source: str, corpus: str, sha: str, atoms: list[Atom], updated: str = "") -> None:
        """Knowledge is replaced per source (ContextOS C5); memory atoms are never touched here."""
        with self.db:
            self._delete_source(source)
            for a in atoms:
                self._insert(a)
            self.db.execute("insert or replace into sources values(?,?,?,?,?)", (source, corpus, sha, now(), updated))

    def drop_source(self, source: str) -> None:
        with self.db:
            self._delete_source(source)
            self.db.execute("delete from sources where source=?", (source,))

    def _delete_source(self, source: str) -> None:
        ids = [r[0] for r in self.db.execute("select id from atoms where source=? and kind='knowledge'", (source,))]
        for chunk in range(0, len(ids), 500):
            part = ids[chunk:chunk + 500]
            q = ",".join("?" * len(part))
            self.db.execute(f"delete from atoms_fts where atom_id in ({q})", part)
            self.db.execute(f"delete from vectors where atom_id in ({q})", part)
            self.db.execute(f"delete from edges where src in ({q})", part)
            self.db.execute(f"delete from edges where dst in ({q})", part)   # no dangling incoming edges (A5)
            self.db.execute(f"delete from atoms where id in ({q})", part)

    def _insert(self, a: Atom) -> None:
        self.db.execute(f"insert or replace into atoms({','.join(COLS)}) values({','.join('?' * len(COLS))})",
                        [getattr(a, c) for c in COLS])
        self.db.execute("delete from atoms_fts where atom_id=?", (a.id,))
        self.db.execute("insert into atoms_fts(atom_id,title,heading,text) values(?,?,?,?)",
                        (a.id, a.title, a.heading, a.text))

    def add_memory(self, a: Atom) -> None:
        """Memory: append; if it supersedes an atom, close that atom's validity instead of deleting.
        Canon-first: the JSONL line is appended BEFORE the index write, so a crash between the two
        leaves the fact durable in the canon (import-memory replays it). Re-adding the same atom
        may duplicate a canon line — export-memory rewrites the canon canonically and reconciles."""
        if os.environ.get('TWINCTX_EXPERIMENTAL_MEMORY_PROTOCOL') == '1':
            from .memory_runtime_experimental import add_memory
            add_memory(self, memory_canon_path(), a)
            return
        line = json.dumps({c: getattr(a, c) for c in COLS}, ensure_ascii=False, sort_keys=True)
        try:
            canon = memory_canon_path()
            canon.parent.mkdir(parents=True, exist_ok=True)
            with canon.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:   # the index write must still happen; the gap is visible and reconciled nightly
            print(f"twinctx: memory canon append failed ({e}) — atom indexed, not durable until export-memory",
                  file=sys.stderr)
        with self.db:
            if a.supersedes:
                self.db.execute("update atoms set valid_to=? where id=? and kind='memory'", (a.valid_from or now(), a.supersedes))
            self._insert(a)

    def close_memory(self, atom_id: str, valid_to: str) -> None:
        """Close a memory atom's validity without a successor (the plugin's 'remove' action)."""
        if os.environ.get('TWINCTX_EXPERIMENTAL_MEMORY_PROTOCOL') == '1':
            from .memory_closure_protocol import close_memory_offline
            with self._memory_thread_lock:
                if self.db.in_transaction:
                    raise RuntimeError('memory operation requires an idle connection')
                path = self.db.execute('pragma database_list').fetchone()[2]
                if not path:
                    raise RuntimeError('experimental protocol requires a file-backed index')
                close_memory_offline(memory_canon_path(), path, atom_id, valid_to, offline=True)
            return
        with self.db:
            self.db.execute("update atoms set valid_to=? where id=? and kind='memory'", (valid_to, atom_id))

    def memory_family(self, seed: str) -> list[Atom]:
        rows = self.db.execute(family_query("atoms"), {"seed": seed})
        return [Atom(**{k: v for k, v in zip(COLS, row) if v is not None}) for row in rows]

    def memories(self) -> list[Atom]:
        cur = self.db.execute(f"select {','.join(COLS)} from atoms where kind='memory' order by valid_from, id")
        return [Atom(**dict(zip(COLS, r))) for r in cur]

    def replace_source_edges(self, source: str, edges: list[tuple]) -> None:
        self.replace_edges_bulk({source: edges})

    def replace_edges_bulk(self, source_edges: dict[str, list[tuple]]) -> None:
        """Transactional edge replacement (ContextOS C5): for every listed document ALL edges touching
        its atoms — own (src) and incoming (dst) — are deleted and the new sets inserted in ONE
        transaction, so processing order never kills incoming edges and any failing row rolls the
        whole call back to the previous state (A4). Documents outside `source_edges` are not touched;
        the indexer passes the whole vault so incoming edges are re-derived in the same transaction."""
        ids: list[str] = []
        for source in source_edges:
            ids += [r[0] for r in self.db.execute(
                "select id from atoms where source=? and kind='knowledge'", (source,))]
        with self.db:
            for side in ("src", "dst"):
                for chunk in range(0, len(ids), 500):
                    part = ids[chunk:chunk + 500]
                    self.db.execute(f"delete from edges where {side} in ({','.join('?' * len(part))})", part)
            for edges in source_edges.values():
                if edges:
                    self.db.executemany(f"insert or replace into edges({','.join(EDGE_COLS)}) "
                                        f"values({','.join('?' * len(EDGE_COLS))})", edges)

    def put_vectors(self, rows: list[tuple[str, bytes]]) -> None:
        with self.db:
            self.db.executemany("insert or replace into vectors values(?,?)", rows)

    def put_vectors_if_current(self, rows: list[tuple[Atom, bytes]]) -> int:
        """Insert only while the exact passage sent for embedding still exists."""
        written = 0
        with self.db:
            for atom, vector in rows:
                cur = self.db.execute(
                    """insert or replace into vectors(atom_id,vec)
                       select ?,? where exists (
                         select 1 from atoms where id=? and source=? and kind=?
                         and title is ? and heading is ? and text is ? and valid_to is ?)
                    """,
                    (atom.id, vector, atom.id, atom.source, atom.kind, atom.title,
                     atom.heading, atom.text, atom.valid_to),
                )
                written += cur.rowcount
        return written

    def edges_report(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {"edges": one("select count(*) from edges"), "dangling": self.dangling_edges(),
                "by_kind": {r[0]: r[1] for r in self.db.execute("select kind, count(*) from edges group by kind")}}

    def count_edges(self) -> int:
        return self.db.execute("select count(*) from edges").fetchone()[0]

    def source_rows(self, source: str) -> list[tuple[str, str]]:
        """(id, text) in document order — the edge-derivation view (ingest, rebuild)."""
        return [(r[0], r[1]) for r in self.db.execute(
            "select id, text from atoms where source=? order by ord", (source,))]

    def first_atom_id(self, source: str) -> str | None:
        r = self.db.execute("select id from atoms where source=? order by ord limit 1", (source,)).fetchone()
        return r[0] if r else None

    def linked_edge_rows(self) -> list[tuple]:
        """(src_id, dst_id, src_source, dst_source) for every live edge — eval-set generation
        and future adapter migrations."""
        return list(self.db.execute(
            "select e.src, e.dst, s.source, d.source from edges e "
            "join atoms s on s.id = e.src join atoms d on d.id = e.dst order by e.src, e.dst"))

    # ── reads ─────────────────────────────────────────────────────────────────────────────────
    def source_sha(self, source: str) -> str | None:
        r = self.db.execute("select sha from sources where source=?", (source,)).fetchone()
        return r[0] if r else None

    def sources(self, corpus: str | None = None) -> list[str]:
        q = "select source from sources" + (" where corpus=?" if corpus else "")
        return [r[0] for r in self.db.execute(q, (corpus,) if corpus else ())]

    def bm25(self, fts: str, limit: int) -> list[tuple[str, float]]:
        # column weights: atom_id(unindexed), title ×4, heading ×2, text ×1; bm25 is negative-better
        sql = ("select atom_id, bm25(atoms_fts, 0, 4.0, 2.0, 1.0) s from atoms_fts "
               "where atoms_fts match ? order by s limit ?")
        try:
            return [(r[0], -r[1]) for r in self.db.execute(sql, (fts, limit))]
        except sqlite3.OperationalError:
            return []

    def get(self, ids: list[str]) -> dict[str, Atom]:
        out = {}
        for chunk in range(0, len(ids), 500):
            part = ids[chunk:chunk + 500]
            q = ",".join("?" * len(part))
            for r in self.db.execute(f"select {','.join(COLS)} from atoms where id in ({q})", part):
                out[r["id"]] = Atom(**{c: r[c] if r[c] is not None else Atom.__dataclass_fields__[c].default for c in COLS})
        return out

    def by_source(self, source: str) -> list[Atom]:
        ids = [r[0] for r in self.db.execute("select id from atoms where source=? order by ord", (source,))]
        got = self.get(ids)
        return [got[i] for i in ids]

    def neighbors(self, ids: list[str], per_seed: int, project: str = "", as_of: str = "") -> list[dict]:
        """Ranked, bounded edges per seed (ContextOS C6): top-K by weight, deterministic tie-break.
        Validity (edge and target atom), project and dangling-target checks happen in SQL, not
        after the fetch (A6). Both directions are returned; `nb` is the neighbour atom id."""
        as_of = as_of or now()
        out = []
        for i in ids:
            sql = ("select case when src=:i then dst else src end nb, src, dst, kind, weight, "
                   "project, provenance, basis, owner "
                   "from edges where (src=:i or dst=:i) "
                   "and (valid_from is null or valid_from='' or valid_from <= :as_of) "
                   "and (valid_to is null or valid_to='' or valid_to > :as_of) "
                   "and exists (select 1 from atoms a where a.id = edges.dst "
                   "  and (a.valid_to is null or a.valid_to='' or a.valid_to > :as_of)) ")
            params: dict = {"i": i, "as_of": as_of, "lim": per_seed}
            if project:
                sql += "and project = :project "
                params["project"] = project
            sql += "order by weight desc, nb limit :lim"
            for r in self.db.execute(sql, params):
                out.append({"nb": r["nb"], "src": r["src"], "dst": r["dst"], "kind": r["kind"],
                            "weight": r["weight"], "project": r["project"], "provenance": r["provenance"],
                            "basis": r["basis"], "owner": r["owner"]})
        return out

    def dangling_edges(self) -> int:
        one = self.db.execute("select count(*) from edges e where "
                              "not exists (select 1 from atoms a where a.id = e.src) or "
                              "not exists (select 1 from atoms a where a.id = e.dst)").fetchone()
        return one[0]

    def atoms_without_vectors(self, limit: int) -> list[Atom]:
        cur = self.db.execute(f"select {','.join(COLS)} from atoms where (valid_to is null or valid_to='') and "
                              "id not in (select atom_id from vectors) order by source, ord limit ?", (limit,))
        return [Atom(**dict(zip(COLS, r))) for r in cur]

    def vectors(self) -> list[tuple[str, bytes]]:
        return list(self.db.execute("select atom_id, vec from vectors"))

    def iter_vectors(self, page_size: int = 256):
        """Read vector rows in bounded batches, closing the cursor on early exit."""
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        # A separate connection keeps each file-backed stream on its own SQLite
        # read snapshot even when the main connection writes during iteration.
        path = self.db.execute("pragma database_list").fetchone()[2]
        connection = sqlite3.connect(path, timeout=30) if path else self.db
        cursor = None
        try:
            cursor = connection.execute("select atom_id, vec from vectors")
            while batch := cursor.fetchmany(page_size):
                yield from batch
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not self.db:
                connection.close()

    def source_ages(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.db.execute("select corpus, max(indexed_at) from sources group by corpus")}

    def stats(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {
            "atoms": one("select count(*) from atoms"),
            "memory": one("select count(*) from atoms where kind='memory'"),
            "sources": one("select count(*) from sources"),
            "edges": one("select count(*) from edges"),
            "edges_by_kind": {r[0]: r[1] for r in self.db.execute("select kind, count(*) from edges group by kind")},
            "vectors": one("select count(*) from vectors"),
            "by_corpus": {r[0]: r[1] for r in self.db.execute("select corpus, count(*) from atoms group by corpus")},
        }
