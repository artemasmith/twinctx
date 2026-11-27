"""twinctx CLI.

  python -m twinctx index            # vault + workspace repos + tasks + incidents (incremental)
  python -m twinctx query "как архивировать заметки" [--budget 2000] [--corpus vault] [--json]
  python -m twinctx query "сессии" --hops 2 --project apollo   # graph is on by default (TWINCTX_GRAPH); --no-graph turns it off
  python -m twinctx remember "Архив заметок обновляется по пятницам" --about projects/archive.md
  python -m twinctx eval evals/context/golden.jsonl [--k 5]
  python -m twinctx eval evals/context/multihop.jsonl --root tests/fixtures/context-graph --db /tmp/mh.db \
      --modes bm25,hybrid,graph --hops 2                              # mode comparison (multi-hop)
  python -m twinctx edges --check                                     # dangling-edge report
  python -m twinctx edges --rebuild                                   # typed edges from indexed atoms (no re-index)
  python -m twinctx stats
  python -m twinctx export-memory journal/memory/atoms.jsonl   # nightly: memory atoms into git (backup)
  python -m twinctx import-memory journal/memory/atoms.jsonl   # restore on a new VM (idempotent)
Paths come from HARNESS_HOME (default ~/harness): vault/, workspace/*, data/incidents, data/context/twinctx.db.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from stores import open_store, open_store_at

from .contracts import COLS, Atom
from .embed import from_env
from .engine import AssembleRequest, Engine, env_min_cosine
from .ingest import Indexer
from .memory import MemoryDraft, remember

HOME = Path(os.environ.get("HARNESS_HOME", Path.home() / "harness"))


def cmd_index(a):
    t = time.time()
    from privacy import wrap_embedder

    ix = Indexer(open_store(), wrap_embedder(from_env()))
    vault = Path(os.environ.get("VAULT_DIR", HOME / "vault"))
    if vault.is_dir():
        ix.vault(vault)
    journal = HOME / "journal"
    if journal.is_dir():
        ix._index_files(journal, "journal", "journal/", {".md"}, "note")
    ws = HOME / "workspace"
    for repo in sorted(p for p in ws.glob("*") if (p / ".git").exists()) if ws.is_dir() else []:
        ix.code(repo, repo.name)
    if ws.is_dir():
        ix.tasks(ws)
    inc = HOME / "data" / "incidents"
    if inc.is_dir():
        ix.incidents(inc)
    ix.backfill_vectors(int(os.environ.get("TWINCTX_BACKFILL_MAX", "3000")))
    out = {**ix.stats, "seconds": round(time.time() - t, 1), **ix.store.stats()}
    if ix.embed_down:
        out["embedding_error"] = ix.embed_down  # text is indexed (BM25); vectors are added on the next runs
    print(json.dumps(out, ensure_ascii=False))
    if ix.embed_down:
        sys.exit(3)


def cmd_query(a):
    if not (1 <= a.hops <= 2):
        print(f"error: --hops must be 1 or 2 (got {a.hops}); graph traversal depth is 1–2 hops", file=sys.stderr)
        sys.exit(2)
    # embeddings are an outbound provider call: keep wrap_embedder (ADR §16) on this path
    from privacy import wrap_embedder

    eng = Engine(open_store(), wrap_embedder(from_env()))
    res = eng.assemble(
        AssembleRequest(
            query=a.query,
            as_of=a.as_of,
            budget_tokens=a.budget,
            corpora=a.corpus or [],
            graph=a.graph,
            graph_hops=a.hops,
            project=a.project or "",
        )
    )
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    m = res["manifest"]
    print(res["packed"] or "(nothing relevant)")
    print(
        f"\n--- manifest: {len(m['items'])} items, {m['tokens']}/{m['budget']} tokens, "
        f"relevance-filtered {m.get('filtered_by_relevance', 0)}, budget-filtered {m.get('filtered_by_budget', 0)}, "
        f"graph {m['graph_expanded_count']}, degraded {m['degraded']}",
        file=sys.stderr,
    )


def cmd_project_query(a):
    from stores import online_search
    from .project_search import answer_project

    store = open_store()
    try:
        print(answer_project(a.project, a.query, search=online_search(store).search,
                             vault_root=Path(os.environ.get("VAULT_DIR", HOME / "vault"))))
    finally:
        if hasattr(store, "close"):
            store.close()
        else:
            store.db.close()


def cmd_remember(a):
    st = open_store()

    def get_embedder():
        from privacy import wrap_embedder

        return wrap_embedder(from_env())

    atom_id_ = remember(
        st,
        MemoryDraft(
            text=a.text,
            about=a.about,
            trust=a.trust,
            supersedes=a.supersedes or "",
            valid_from=a.valid_from or "",
        ),
        get_embedder=get_embedder,
    )
    print(atom_id_)


def _run_mode(mode: str, store, eng, rows, a) -> dict:
    """One eval mode: bm25 (no embedder), hybrid (kw+vector), graph (hybrid + opt-in traverse)."""
    hit, neg_ok, neg, lines = 0, 0, 0, []
    pos = [r for r in rows if r.get("expect")]
    for r in rows:
        res = eng.assemble(
            AssembleRequest(
                query=r["q"],
                budget_tokens=a.budget,
                k=a.k,
                min_cosine=a.min_cosine,
                min_coverage=a.min_coverage,
                graph=(mode == "graph"),
                graph_hops=a.hops,
            )
        )
        got = [i["source"] for i in res["manifest"]["items"]][: a.k]
        if not r.get("expect"):  # negative: nothing relevant should come back
            neg += 1
            ok = not got
            neg_ok += ok
            lines.append(f"{'PASS' if ok else 'FAIL'} [neg] {r['q'][:60]} → {got[:2]}")
            continue
        ok = any(e in g for e in r["expect"] for g in got)
        hit += ok
        lines.append(f"{'PASS' if ok else 'FAIL'} {r['q'][:60]} → {got[:3]}")
    return {
        "mode": mode,
        "hit": hit,
        "pos": len(pos),
        "neg_ok": neg_ok,
        "neg": neg,
        "recall": hit / max(1, len(pos)),
        "lines": lines,
    }


def cmd_eval(a):
    """recall@k: share of golden queries whose expected source appears in the top-k assembled items (C9).
    --modes bm25,hybrid,graph compares the modes on one set (multi-hop eval, A11); without --modes the
    classic single-mode run (hybrid engine) is kept for the golden baseline."""
    store = open_store_at(a.db) if a.db else open_store()
    from privacy import wrap_embedder

    if a.root:
        Indexer(store, wrap_embedder(from_env())).vault(Path(a.root))  # ADR §16: эмбед-API — тоже модель
    rows = [json.loads(line) for line in Path(a.golden).read_text().splitlines() if line.strip()]
    modes = [m.strip() for m in (a.modes or "").split(",") if m.strip()]
    if not modes:
        results = [_run_mode("hybrid", store, Engine(store, wrap_embedder(from_env())), rows, a)]
    else:
        results = []
        for mode in [m for m in (a.modes or "").split(",") if m.strip()]:
            eng = Engine(store, None if mode == "bm25" else wrap_embedder(from_env()))
            results.append(_run_mode(mode, store, eng, rows, a))
    for r in results:
        print(f"== mode {r['mode']}")
        print("\n".join(r["lines"]))
        print(
            f"recall@{a.k}: {r['hit']}/{r['pos']} = {r['recall']:.2f}; "
            f"negatives empty: {r['neg_ok']}/{r['neg']}; embedder: {os.environ.get('TWINCTX_EMBED', 'none')}"
        )
    if len(results) > 1:
        print("\nmode      recall@%d  negatives" % a.k)
        for r in results:
            print(f"{r['mode']:<9} {r['recall']:<9.2f} {r['neg_ok']}/{r['neg']}")
        by = {r["mode"]: r for r in results}
        if "bm25" in by and "graph" in by:
            d = by["graph"]["recall"] - by["bm25"]["recall"]
            note = "graph helped" if d > 0 else ("graph == bm25" if d == 0 else "graph did NOT help")
            print(f"graph − bm25 = {d:+.2f} ({note})")
    if a.out:
        Path(a.out).write_text(
            json.dumps(
                {
                    r["mode"]: {
                        "recall": r["recall"],
                        "hits": r["hit"],
                        "pos": r["pos"],
                        "negatives_ok": r["neg_ok"],
                        "negatives": r["neg"],
                    }
                    for r in results
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    graph = next((r for r in results if r["mode"] == "graph"), None)
    if graph is not None:  # multi-hop eval: exit 0 when the graph mode produced a non-empty result
        sys.exit(0 if graph["hit"] > 0 and graph["neg_ok"] == graph["neg"] else 1)
    r = results[0]
    sys.exit(0 if r["recall"] >= a.min and r["neg_ok"] == r["neg"] else 1)


def cmd_edges(a):
    """edges --check: dangling-edge report (both endpoints must be live atoms); exit 1 if any (A5).
    edges --rebuild [--corpus PREFIX ...]: re-derive typed edges from already-indexed atoms —
    no re-indexing, no embedding, atoms/vectors untouched (A17)."""
    st = open_store_at(a.db) if a.db else open_store()
    # Old read-only SQLite schemas explicitly report unavailable edges. Stores
    # with an initialized graph need not expose this legacy adapter detail.
    if not getattr(st, "edges_ok", True):
        print(json.dumps({"error": "edges table unavailable (old schema on a read-only db)"}))
        sys.exit(2)
    t0 = time.time()
    if getattr(a, "rebuild", False):
        corpora = tuple(a.corpus) if a.corpus else ("vault",)
        root = Path(os.environ.get("VAULT_DIR", HOME / "vault"))
        Indexer(st).rebuild_edges(root=root if "vault" in corpora else None, corpora=corpora)
    out = st.edges_report()
    if getattr(a, "rebuild", False):
        out["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(1 if out["dangling"] else 0)


def cmd_export_memory(a):
    """Memory atoms exist only in the index db — write them (with validity history) to a JSONL file in git."""
    st = open_store()
    exporter = getattr(st, "export_memory_canon", None)
    if callable(exporter):
        try:
            count = exporter(Path(a.path))
        finally:
            st.close()
        print(f"{count} memory atoms → {a.path}")
        return
    if os.environ.get("TWINCTX_EXPERIMENTAL_MEMORY_PROTOCOL") == "1":
        from stores.memory_runtime_experimental import export_canon
        from stores.sqlite import memory_canon_path

        try:
            count = export_canon(st, memory_canon_path(), Path(a.path))
        finally:
            st.db.close()
        print(f"{count} memory atoms → {a.path}")
        return
    from dataclasses import asdict

    rows = [json.dumps({c: asdict(m)[c] for c in COLS}, ensure_ascii=False, sort_keys=True) for m in st.memories()]
    out = Path(a.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = "\n".join(rows) + ("\n" if rows else "")
    if not out.is_file() or out.read_text() != new:  # unchanged → no git noise
        out.write_text(new)
    print(f"{len(rows)} memory atoms → {out}")


def cmd_import_memory(a):
    st = open_store()
    importer = getattr(st, "import_memory_canon", None)
    if callable(importer):
        try:
            count = importer(Path(a.path))
        finally:
            st.close()
        print(f"{count} memory atoms imported; vectors will be backfilled")
        return
    if os.environ.get("TWINCTX_EXPERIMENTAL_MEMORY_PROTOCOL") == "1":
        from stores.memory_runtime_experimental import import_canon
        from stores.sqlite import memory_canon_path

        canon = memory_canon_path()
        if Path(a.path).absolute() != canon.absolute():
            raise ValueError("experimental import only accepts the configured canon")
        try:
            count = import_canon(st, canon)
        finally:
            st.db.close()
        print(f"{count} memory atoms imported; vectors will be backfilled")
        return
    from privacy import wrap_embedder

    emb, n = wrap_embedder(from_env()), 0  # ADR §16: эмбед-API маскируется
    for line in Path(a.path).read_text().splitlines():
        if line.strip():
            atom = Atom(**json.loads(line))
            with st.db:
                st._insert(atom)
            if emb and not atom.valid_to:
                v = emb.embed_passages([atom.text])[0]
                if v is not None:
                    st.put_vectors([(atom.id, v)])
            n += 1
    print(f"{n} memory atoms imported")


def main():
    ap = argparse.ArgumentParser(prog="twinctx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("--as-of", default="", help="ISO date or timestamp for decision validity")
    q.add_argument("--budget", type=int, default=2500)
    q.add_argument("--corpus", action="append")
    gg = q.add_mutually_exclusive_group()
    gg.add_argument(
        "--graph",
        dest="graph",
        action="store_true",
        default=None,
        help="force graph traversal on (default: TWINCTX_GRAPH, on in the shipped install)",
    )
    gg.add_argument("--no-graph", dest="graph", action="store_false", help="force graph traversal off")
    q.add_argument("--hops", type=int, default=2, help="graph hops (1–2)")
    q.add_argument("--project", default="", help="restrict graph edges to one project")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_query)
    project = sub.add_parser("project-query", help="search current project documents, with optional QMD fallback")
    project.add_argument("project")
    project.add_argument("query")
    project.set_defaults(fn=cmd_project_query)
    r = sub.add_parser("remember")
    r.add_argument("text")
    r.add_argument("--about", default="")
    r.add_argument("--supersedes", default="")
    r.add_argument("--valid-from", default="")
    r.add_argument("--trust", type=float, default=0.7)
    r.set_defaults(fn=cmd_remember)
    e = sub.add_parser("eval")
    e.add_argument("golden")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--budget", type=int, default=4000)
    e.add_argument("--min", type=float, default=0.0)
    # calibrate the relevance floor per embedder: raise until negatives are empty, stop before recall drops
    e.add_argument("--min-cosine", type=float, default=env_min_cosine())
    e.add_argument("--min-coverage", type=float, default=0.34)
    e.add_argument("--modes", default="", help="comma list: bm25,hybrid,graph (multi-hop eval)")
    e.add_argument("--root", default="", help="fixture/vault dir to index into --db before evaluating")
    e.add_argument("--db", default="", help="db path (default: HARNESS_HOME store)")
    e.add_argument("--out", default="", help="write a JSON report to this file")
    e.add_argument("--hops", type=int, default=2, help="graph hops for the graph mode (1–2)")
    e.set_defaults(fn=cmd_eval)
    g = sub.add_parser("edges")
    g.add_argument("--check", action="store_true", help="dangling-edge report (exit 1 if any)")
    g.add_argument("--rebuild", action="store_true", help="rebuild typed edges from indexed atoms (no re-index, A17)")
    g.add_argument("--corpus", action="append", help="corpus prefix to rebuild (default: vault)")
    g.add_argument("--db", default="")
    g.set_defaults(fn=cmd_edges, check=False, rebuild=False, corpus=None)
    for name, fn in (("export-memory", cmd_export_memory), ("import-memory", cmd_import_memory)):
        x = sub.add_parser(name)
        x.add_argument("path")
        x.set_defaults(fn=fn)
    sub.add_parser("stats").set_defaults(fn=lambda a: print(json.dumps(open_store().stats(), ensure_ascii=False)))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
