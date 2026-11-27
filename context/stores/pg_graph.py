"""Запись и отчёты графа рёбер PG-индекса (чтение neighbors — в _PgSearchReads).

Вынесено из pg_sources.py волной 5 техдолга без изменения поведения.
"""

from twinctx.contracts import EDGE_COLS


class PgGraphMixin:
    def replace_source_edges(self, source: str, edges: list[tuple]) -> None:
        self.replace_edges_bulk({source: edges})

    def graph_source_versions(self, corpora: tuple[str, ...]) -> dict[str, tuple[str | None, str]]:
        """Capture the whole scope in one MVCC snapshot, including tuple versions."""
        if not corpora or any(not isinstance(c, str) or not c for c in corpora):
            raise ValueError("nonempty graph corpora required")
        with self._lock:
            return self._graph_source_versions(corpora)

    def _graph_source_versions(self, corpora):
        rows = self.con.run(
            "select source,sha,xmin::text from twinctx_v1.sources "
            'where corpus=any(cast(:corpora as text[])) order by source collate "C"',
            corpora=list(corpora),
        )
        return {source: (sha, version) for source, sha, version in rows}

    def replace_edges_if_current(self, source_edges, expected_versions, corpora):
        """Commit a full-scope graph only if every source tuple is unchanged."""
        self.replace_edges_bulk(source_edges, expected_graph_versions=expected_versions, expected_corpora=corpora)

    def replace_edges_bulk(
        self,
        source_edges: dict[str, list[tuple]],
        *,
        expected_sources_sha: dict[str, str] | None = None,
        expected_graph_versions=None,
        expected_corpora=None,
    ) -> None:
        """Replace selected edges; callers may guard derivation against source changes.

        This experimental API is not sufficient to enable the PG backend.
        """
        if not source_edges and expected_graph_versions is None:
            return
        if expected_sources_sha is not None and set(expected_sources_sha) != set(source_edges):
            raise ValueError("source version set must match graph rebuild scope")
        if expected_graph_versions is not None:
            if not expected_corpora or set(expected_graph_versions) != set(source_edges):
                raise ValueError("graph version set must match graph rebuild scope")
        owners = {}
        for source, edges in source_edges.items():
            for edge in edges:
                if len(edge) != len(EDGE_COLS) or not isinstance(edge[0], str):
                    raise ValueError("invalid edge shape")
                previous = owners.setdefault(edge[0], source)
                if previous != source:
                    raise ValueError("edge source has multiple owners")
        columns = ",".join(EDGE_COLS)
        bindings = ",".join(":" + column for column in EDGE_COLS)
        update = ",".join(f"{column}=excluded.{column}" for column in EDGE_COLS if column not in ("src", "dst", "kind"))
        with self._transaction():
            self.con.run("select pg_advisory_xact_lock(hashtext('aitwin-twinctx-graph-v1'))")
            if expected_graph_versions is not None:
                if self._graph_source_versions(expected_corpora) != expected_graph_versions:
                    raise ValueError("graph source scope or version changed while graph was derived")
            if expected_sources_sha is not None:
                rows = self.con.run(
                    "select source,sha from twinctx_v1.sources where source=any(cast(:sources as text[]))",
                    sources=list(source_edges),
                )
                if dict(rows) != expected_sources_sha:
                    raise ValueError("source changed while graph was derived")
            if not source_edges:
                return
            if owners:
                rows = self.con.run(
                    "select id,source from twinctx_v1.atoms where kind='knowledge' and id=any(cast(:ids as text[]))",
                    ids=list(owners),
                )
                if dict(rows) != owners:
                    raise ValueError("edge source is missing or belongs to another document")
            self.con.run(
                """delete from twinctx_v1.edges where src in
                       (select id from twinctx_v1.atoms where kind='knowledge' and source=any(cast(:sources as text[])))
                   or dst in
                       (select id from twinctx_v1.atoms where kind='knowledge' and source=any(cast(:sources as text[])))""",
                sources=list(source_edges),
            )
            for edges in source_edges.values():
                for edge in edges:
                    self.con.run(
                        f"insert into twinctx_v1.edges({columns}) values({bindings}) "
                        f"on conflict(src,dst,kind) do update set {update}",
                        **dict(zip(EDGE_COLS, edge)),
                    )

    def count_edges(self) -> int:
        with self._lock:
            return self.con.run("select count(*) from twinctx_v1.edges")[0][0]

    def dangling_edges(self) -> int:
        with self._lock:
            return self.con.run(
                """select count(*) from twinctx_v1.edges e
                   where not exists (select 1 from twinctx_v1.atoms a where a.id=e.src)
                      or not exists (select 1 from twinctx_v1.atoms a where a.id=e.dst)"""
            )[0][0]

    def edges_report(self) -> dict:
        with self._lock:
            by_kind = self.con.run("select kind,count(*) from twinctx_v1.edges group by kind")
            return {
                "edges": self.count_edges(),
                "dangling": self.dangling_edges(),
                "by_kind": {row[0]: row[1] for row in by_kind},
            }

    def linked_edge_rows(self) -> list[tuple]:
        with self._lock:
            rows = self.con.run(
                """select e.src,e.dst,s.source,d.source from twinctx_v1.edges e
                   join twinctx_v1.atoms s on s.id=e.src join twinctx_v1.atoms d on d.id=e.dst
                   order by e.src,e.dst"""
            )
            return [tuple(row) for row in rows]
