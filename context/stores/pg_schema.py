"""Схема/миграции PG-индекса: DDL, FTS-каталог, зонды и warm-start маркер.

Вынесено из pg_sources.py волной 5 техдолга без изменения поведения; pg_sources реэкспортирует _SCHEMA/_SCHEMA_ALL_VERSION.
"""

import json

from .pg_env import _env_number  # noqa: F401 — общий leaf, без циклов

_SCHEMA = """
create schema if not exists twinctx_v1;
create table if not exists twinctx_v1.atoms (
  id text primary key, source text not null, corpus text not null, kind text not null,
  doc_type text, title text, heading text, text text not null,
  start_line integer, end_line integer, updated text, observed_at text,
  valid_from text, valid_to text, trust double precision default 1.0, ord integer default 0, supersedes text
);
create index if not exists atoms_source on twinctx_v1.atoms(source);
create index if not exists atoms_memory_id on twinctx_v1.atoms(id) where kind='memory';
create table if not exists twinctx_v1.sources (
  source text primary key, corpus text, sha text, indexed_at text, updated text
);
create table if not exists twinctx_v1.vectors (atom_id text primary key, vec bytea);
create table if not exists twinctx_v1.edges (
  src text not null, dst text not null, kind text not null, weight real default 1.0,
  project text default '', provenance text not null default '', basis text not null default '',
  owner text default '', version integer default 2, valid_from text default '',
  valid_to text default '', created_at text default '', primary key(src,dst,kind),
  check(length(basis) between 1 and 200)
);
create index if not exists edges_src on twinctx_v1.edges(src);
create index if not exists edges_dst on twinctx_v1.edges(dst);
"""


_SCHEMA_ALL_VERSION = "1"  # поднять при добавлении любой миграции: прогонит полный путь один раз


def _fts_expression(column: str) -> str:
    """Approximate FTS5 unicode61's Latin diacritic folding and separators."""
    if column not in ("title", "heading", "text"):
        raise ValueError("invalid search column")
    folded = (
        f"normalize(regexp_replace(normalize(coalesce({column}, ''), NFD), "
        r"U&'([A-Za-z])[\0300-\036F]+', E'\\1', 'g'), NFC)"
    )
    return f"to_tsvector('simple'::regconfig, regexp_replace({folded}, '[^[:alnum:]]+', ' ', 'g'))"


class PgSchemaMixin:
    def _schema_all_current(self) -> bool:
        """Тёплый старт без миграционной транзакции (advisory lock + 5 зондов DDL).

        True только при полном совпадении: версия schema-all текущая, сохранённый снимок
        каталожных проб равен живому каталогу (fts/trust/text_raw/weight/srcops/индексы)
        и снимок fts совпадает с маркером fts-v1. Любая подделка маркера, утерянный
        GIN-индекс, сменённый тип колонки или посторонний объект уводят на полный путь
        с drift-детекторами — гарантия «не молчать о расхождении» сохранена на каждом открытии.
        """
        try:
            rows = dict(
                self.con.run("select key, value from twinctx_v1.schema_meta where key in ('schema-all','fts-v1')")
            )
        except Exception:
            return False  # холодная база (таблицы нет) или сбой чтения — полный путь
        try:
            combined = json.loads(rows.get("schema-all", ""))
            fts_marker = json.loads(rows.get("fts-v1", ""))
        except (ValueError, TypeError):
            return False
        if not isinstance(combined, dict) or combined.get("v") != _SCHEMA_ALL_VERSION:
            return False
        stored = combined.get("snapshot")
        if not isinstance(stored, dict):
            return False
        live = self._schema_probe_snapshot()
        return stored == live and stored.get("fts") == fts_marker

    def _fts_catalog_signature(self):
        columns = self.con.run(
            """select a.attname, a.attgenerated, pg_get_expr(d.adbin,d.adrelid)
               from pg_attribute a left join pg_attrdef d on d.adrelid=a.attrelid and d.adnum=a.attnum
               where a.attrelid='twinctx_v1.atoms'::regclass and not a.attisdropped
               and a.attname in ('fts_title','fts_heading','fts_text') order by a.attname"""
        )
        indexes = self.con.run(
            """select c.relname,i.indisvalid,i.indisready,pg_get_indexdef(i.indexrelid)
               from pg_index i join pg_class c on c.oid=i.indexrelid
               where i.indrelid='twinctx_v1.atoms'::regclass
               and c.relname in ('atoms_fts_title','atoms_fts_heading','atoms_fts_text') order by c.relname"""
        )
        return {"columns": columns, "indexes": indexes}

    def _probe_column_types(self, relation, names):
        """Типы колонок по каталогу — единая проба для ensure-миграций и тёплой сверки."""
        return self.con.run(
            """select attname, format_type(atttypid,atttypmod) from pg_attribute
               where attrelid=cast(:rel as regclass) and attname=any(cast(:names as text[]))
               and not attisdropped order by attname""",
            rel=relation,
            names=names,
        )

    def _probe_source_operations(self):
        columns = self.con.run(
            """select attname,format_type(atttypid,atttypmod),attnotnull from pg_attribute
            where attrelid='twinctx_v1.source_operations'::regclass and attnum>0 and not attisdropped order by attnum"""
        )
        constraints = self.con.run(
            """select contype,pg_get_constraintdef(oid) from pg_constraint
            where conrelid='twinctx_v1.source_operations'::regclass"""
        )
        return [columns, constraints]

    def _schema_probe_snapshot(self):
        """Полный снимок каталожных проб всех миграционных зон (без DDL)."""
        found = self.con.run("select to_regclass('twinctx_v1.edges_project') is not null")
        return {
            "fts": self._fts_catalog_signature(),
            "atoms_types": self._probe_column_types("twinctx_v1.atoms", ["trust", "text_raw"]),
            "edges_weight": self._probe_column_types("twinctx_v1.edges", ["weight"]),
            "edges_project": bool(found and found[0][0]),
            "source_operations": self._probe_source_operations(),
        }

    def _ensure_fts_schema(self):
        """Apply FTS schema once; reject unexplained or modified objects."""
        self.con.run("create table if not exists twinctx_v1.schema_meta (key text primary key, value text not null)")
        marker = self.con.run("select value from twinctx_v1.schema_meta where key='fts-v1'")
        current = self._fts_catalog_signature()
        if marker:
            try:
                expected = json.loads(marker[0][0])
            except (ValueError, TypeError):
                raise RuntimeError("invalid PostgreSQL FTS schema marker") from None
            if expected != current or not self._fts_schema_complete(current):
                raise RuntimeError("PostgreSQL FTS schema drift detected")
            return
        if current["columns"] or current["indexes"]:
            raise RuntimeError("unmanaged PostgreSQL FTS schema")
        for column in ("title", "heading", "text"):
            self.con.run(
                f"alter table twinctx_v1.atoms add column fts_{column} tsvector "
                f"generated always as ({_fts_expression(column)}) stored"
            )
            self.con.run(f"create index atoms_fts_{column} on twinctx_v1.atoms using gin(fts_{column})")
        current = self._fts_catalog_signature()
        if not self._fts_schema_complete(current):
            raise RuntimeError("PostgreSQL FTS schema initialization failed")
        self.con.run(
            "insert into twinctx_v1.schema_meta(key,value) values('fts-v1',:signature)",
            signature=json.dumps(current, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _fts_schema_complete(signature):
        columns, indexes = signature["columns"], signature["indexes"]
        names = {"title", "heading", "text"}
        if {row[0] for row in columns} != {f"fts_{name}" for name in names}:
            return False
        if any(row[1] != "s" or not row[2] for row in columns):
            return False
        if {row[0] for row in indexes} != {f"atoms_fts_{name}" for name in names}:
            return False
        for name, valid, ready, definition in indexes:
            suffix = name.removeprefix("atoms_fts_")
            if not valid or not ready or f"USING gin (fts_{suffix})" not in definition:
                return False
        return True

    def _ensure_atom_text_raw(self):
        # ALTER TABLE acquires ACCESS EXCLUSIVE even with IF NOT EXISTS.  Avoid
        # blocking active queries on every Store open after initialization.
        rows = self._probe_column_types("twinctx_v1.atoms", ["text_raw"])
        kind = rows[0][1] if rows else None
        if not kind:
            self.con.run("alter table twinctx_v1.atoms add column text_raw bytea")
            kind = self._probe_column_types("twinctx_v1.atoms", ["text_raw"])[0][1]
        if kind != "bytea":
            raise RuntimeError("PostgreSQL atom raw text schema drift detected")

    def _ensure_atom_precision(self):
        rows = self._probe_column_types("twinctx_v1.atoms", ["trust"])
        kind = rows[0][1] if rows else None
        if kind == "real":
            # Widening preserves the existing 32-bit value but cannot restore bits
            # lost by an earlier write. Migration must verify against SQLite.
            self.con.run("alter table twinctx_v1.atoms alter column trust type double precision")
        elif kind != "double precision":
            raise RuntimeError("PostgreSQL atom trust schema drift detected")

    def _ensure_graph_schema(self):
        weights = self._probe_column_types("twinctx_v1.edges", ["weight"])
        kind = weights[0][1] if weights else None
        if kind == "real":
            self.con.run("alter table twinctx_v1.edges alter column weight type double precision")
        elif kind != "double precision":
            raise RuntimeError("PostgreSQL graph schema drift detected")
        self.con.run("create index if not exists edges_project on twinctx_v1.edges(project)")

    def _ensure_source_operation_schema(self):
        self.con.run("""create table if not exists twinctx_v1.source_operations (
            operation_id text primary key, request_hash text not null, source text not null)""")
        columns, constraints = self._probe_source_operations()
        if columns != [
            ["operation_id", "text", True],
            ["request_hash", "text", True],
            ["source", "text", True],
        ] or constraints != [["p", "PRIMARY KEY (operation_id)"]]:
            raise RuntimeError("PostgreSQL source operation schema drift detected")
