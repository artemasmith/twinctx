"""Incremental indexing of corpora into the store.

Corpora (C1 knowledge; memory is written by agents, not ingested):
  vault      — markdown notes, chunked by headings; wikilinks/md links + frontmatter supersedes:/
               project: become typed edges (link/supersedes/project, see docs/DECISIONS.md 2026-10-28)
  code:<repo> — source files chunked at def/class/function boundaries
  tasks      — .tasks/*.yaml records
  incidents  — data/incidents/*.jsonl, one atom per week and kind
Unchanged files (same sha1) are skipped; deleted files are dropped (C10: runs from cron/git hooks).
"""
import hashlib
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from .contracts import EDGE_SCHEMA_VERSION, Atom, Store, now
from .text import est_tokens

MAX_CHARS = 1800          # ≈ 500–700 tokens per chunk
MIN_CHARS = 200
SKIP_DIRS = {".git", "node_modules", "dist", "build", ".next", "venv", ".venv", "__pycache__", "vendor",
             ".obsidian", ".trash", "coverage", ".mypy_cache", ".pytest_cache", "target"}
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rb", ".java", ".kt", ".sql", ".sh", ".rs",
            ".yaml", ".yml", ".toml", ".md", ".json"}
MAX_FILE = 200_000
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
MDLINK = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
CODE_BOUNDARY = re.compile(
    r"^(\s{0,4})(def |async def |class |function |export |const \w+ = (async )?\(|func |public |private |protected |"
    r"module |interface |type \w+ = |CREATE |create |#{1,3} )")


def sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def guarded_source_snapshot(store, source):
    """Capture an optional compare-and-swap token before reading source data."""
    capture = getattr(store, "source_revision", None)
    replace = getattr(store, "replace_source_if_current", None)
    if callable(capture) != callable(replace):
        raise RuntimeError("incomplete guarded source store contract")
    return (True, capture(source)) if callable(capture) else (False, store.source_sha(source))


def guarded_source_replace(store, source, corpus, sha, atoms, updated, snapshot):
    guarded, revision = snapshot
    if not guarded:
        store.replace_source(source, corpus, sha, atoms, updated)
        return True
    result = store.replace_source_if_current(source, corpus, sha, atoms, updated,
                                              expected=revision, operation_id=uuid.uuid4().hex)
    if result not in ("applied", "already_applied", "conflict"):
        raise RuntimeError("invalid guarded source result")
    return result != "conflict"


def atom_id(source: str, n: int) -> str:
    return sha1(f"{source}#{n}".encode())[:16]


def frontmatter(text: str) -> tuple[dict, str, int]:
    if not text.startswith("---\n"):
        return {}, text, 0
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text, 0
    fm = {}
    for line in text[4:end].splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip("'\"")
    body = text[end + 4:].lstrip("\n")
    return fm, body, text[: end + 4].count("\n") + 1


def split_md(text: str) -> list[tuple[str, int, int, str]]:
    """→ [(heading_path, start_line, end_line, chunk_text)], merged/split to MIN..MAX chars."""
    fm, body, offset = frontmatter(text)
    lines = body.splitlines()
    sections, path, cur, start = [], [], [], 0
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if m and cur:
            sections.append((" > ".join(path), start, i - 1, "\n".join(cur)))
            cur, start = [], i
        if m:
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
        cur.append(line)
    if cur:
        sections.append((" > ".join(path), start, len(lines) - 1, "\n".join(cur)))
    out = []
    for head, s, e, chunk in sections:
        if out and len(out[-1][3]) + len(chunk) < MIN_CHARS * 3 and len(chunk) < MIN_CHARS:
            h, s0, _, t = out[-1]
            out[-1] = (h, s0, e, t + "\n" + chunk)
            continue
        while len(chunk) > MAX_CHARS:
            cut = chunk.rfind("\n\n", MIN_CHARS, MAX_CHARS)
            cut = cut if cut > 0 else chunk.rfind("\n", MIN_CHARS, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            n = chunk[:cut].count("\n")
            out.append((head, s, s + n, chunk[:cut]))
            chunk, s = chunk[cut:].lstrip("\n"), s + n + 1
        if chunk.strip():
            out.append((head, s, e, chunk))
    return [(h, s + offset + 1, e + offset + 1, t) for h, s, e, t in out]


def split_code(text: str) -> list[tuple[str, int, int, str]]:
    lines = text.splitlines()
    out, start, head = [], 0, ""
    for i, line in enumerate(lines):
        size = sum(len(x) + 1 for x in lines[start:i])
        if i > start and (size > MAX_CHARS or (CODE_BOUNDARY.match(line) and size > MIN_CHARS)):
            out.append((head, start + 1, i, "\n".join(lines[start:i])))
            start = i
        if CODE_BOUNDARY.match(line):
            head = line.strip()[:120]
    if start < len(lines):
        out.append((head, start + 1, len(lines), "\n".join(lines[start:])))
    return out


class Indexer:
    def __init__(self, store: Store, embedder=None):
        self.store = store
        self.embedder = embedder
        self.stats = defaultdict(int)
        self.embed_down = ""   # first embedding failure of this run: later sources skip vectors, backfill later

    # ── generic file corpus ───────────────────────────────────────────────────────────────────
    def _files(self, root: Path, exts: set[str]):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for f in filenames:
                p = Path(dirpath) / f
                try:
                    ok = p.suffix.lower() in exts and p.stat().st_size <= MAX_FILE
                except OSError:   # vanished between listing and stat (parallel pass)
                    continue
                if ok and not f.endswith((".lock", ".min.js")):
                    yield p

    def _index_files(self, root: Path, corpus: str, prefix: str, exts: set[str], doc_type: str) -> list[str]:
        seen, changed = set(), []
        capture = getattr(self.store, "graph_source_versions", None)
        guarded_replace = getattr(self.store, "replace_source_if_current", None)
        guarded_drop = getattr(self.store, "drop_source_if_current", None)
        if callable(guarded_replace) and not (callable(capture) and callable(guarded_drop)):
            raise RuntimeError("incomplete guarded source scope contract")
        versions = capture((corpus,)) if callable(guarded_replace) else None
        if versions is not None and not Path(root).is_dir():
            raise FileNotFoundError("source root is not available")
        for p in self._files(root, exts):
            rel = f"{prefix}{p.relative_to(root).as_posix()}"
            seen.add(rel)
            try:
                raw = p.read_bytes()
            except OSError:   # vanished between walk and read (parallel pass)
                if versions is None:
                    seen.discard(rel)
                else:
                    self.stats["read_errors"] += 1
                continue
            h = sha1(raw)
            snapshot = (True, versions.get(rel)) if versions is not None else (False, self.store.source_sha(rel))
            old_sha = snapshot[1][0] if snapshot[0] and snapshot[1] is not None else snapshot[1]
            if old_sha == h:
                self.stats["unchanged"] += 1
                continue
            text = raw.decode("utf-8", errors="ignore")
            is_md = p.suffix.lower() == ".md"
            fm = frontmatter(text)[0] if is_md else {}
            parts = split_md(text) if is_md else split_code(text)
            title = fm.get("title") or p.stem
            updated = fm.get("updated") or ""
            dtype = doc_type if not is_md or doc_type != "code" else "doc"
            atoms = [Atom(id=atom_id(rel, n), source=rel, corpus=corpus, text=t, doc_type=dtype, title=title,
                          heading=h, start_line=s, end_line=e, updated=updated, observed_at=now(), ord=n)
                     for n, (h, s, e, t) in enumerate(parts)]
            if not guarded_source_replace(self.store, rel, corpus, h, atoms, updated, snapshot):
                self.stats["source_conflicts"] += 1
                continue
            self._embed(atoms)
            changed.append(rel)
            self.stats["indexed"] += 1
            self.stats["atoms"] += len(atoms)
        existing = set(versions) if versions is not None else set(self.store.sources(corpus))
        for gone in existing - seen:
            if versions is not None:
                if not gone.startswith(prefix):
                    continue
                relative = Path(gone[len(prefix):])
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    self.stats["read_errors"] += 1
                    continue
                try:
                    (Path(root) / relative).stat()
                except FileNotFoundError:
                    pass
                except OSError:
                    self.stats["read_errors"] += 1
                    continue
                else:
                    continue
                status = guarded_drop(gone, expected=versions[gone], operation_id=uuid.uuid4().hex)
                if status == "conflict":
                    self.stats["source_conflicts"] += 1
                    continue
                if status not in ("applied", "already_applied"):
                    raise RuntimeError("invalid guarded source result")
            else:
                self.store.drop_source(gone)
            self.stats["dropped"] += 1
        return changed

    def _embed(self, atoms: list[Atom]) -> None:
        """A failed embedding never loses the text: the atoms stay searchable by BM25 and backfill_vectors()
        adds their vectors on a later run (the index must not stall on a provider hiccup). Passages the
        provider cannot take (over its token context) come back as None placeholders — BM25-only by design."""
        if not self.embedder or not atoms or self.embed_down:
            self.stats["embed_deferred"] += len(atoms) if self.embedder else 0
            return
        try:
            vecs = self.embedder.embed_passages([f"{a.title} {a.heading}\n{a.text}" for a in atoms])
        except Exception as e:  # noqa: BLE001 — transient provider errors end up here; oversized ones do not
            self.embed_down = f"{type(e).__name__}: {e}"[:300]
            self.stats["embed_deferred"] += len(atoms)
            return
        rows = [(a, v) for a, v in zip(atoms, vecs) if v is not None]
        self.stats["embed_oversized"] += len(atoms) - len(rows)
        guarded = getattr(self.store, "put_vectors_if_current", None)
        written = guarded(rows) if guarded else (self.store.put_vectors([(a.id, v) for a, v in rows]) or len(rows))
        self.stats["embed_deferred"] += len(rows) - written

    def backfill_vectors(self, limit: int = 3000) -> None:
        """Vectors for atoms that have none yet (first index of a big vault, earlier provider errors).
        Atoms over the embedder context are filtered out: they can never gain a vector, and re-selecting
        them every run is what kept the 400-loop of 2026-10-26 alive."""
        if not self.embedder or self.embed_down:
            return
        todo = self.store.atoms_without_vectors(limit)
        cap = getattr(self.embedder, "max_tokens", 0)
        # The exact check must run after privacy redaction in the embedder.
        # Preserve the legacy early estimate only when no tokenizer is configured.
        if cap and not getattr(self.embedder, "tokenizer_path", ""):
            sendable = [a for a in todo if est_tokens(f"{a.title} {a.heading}\n{a.text}") <= cap]
            self.stats["embed_oversized"] += len(todo) - len(sendable)
            todo = sendable
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            try:
                vecs = self.embedder.embed_passages([f"{a.title} {a.heading}\n{a.text}" for a in chunk])
            except Exception as e:  # noqa: BLE001
                self.embed_down = f"{type(e).__name__}: {e}"[:300]
                return
            rows = [(a, v) for a, v in zip(chunk, vecs) if v is not None]
            self.stats["embed_oversized"] += len(chunk) - len(rows)
            guarded = getattr(self.store, "put_vectors_if_current", None)
            written = guarded(rows) if guarded else (self.store.put_vectors([(a.id, v) for a, v in rows]) or len(rows))
            self.stats["backfilled"] += written
            self.stats["embed_deferred"] += len(rows) - written
        if self.stats["embed_oversized"]:
            print(f"twinctx: {self.stats['embed_oversized']} embeddings unavailable "
                  "(oversized or rejected); original text retained for BM25", file=sys.stderr, flush=True)

    # ── graph: typed explicit edges, rebuilt per source (C6; DECISIONS 2026-10-28) ─────────────
    def vault(self, root: Path) -> None:
        changed = self._index_files(root, "vault", "vault/", {".md"}, "note")
        if changed or not self.store.count_edges():
            self._link_edges(root)   # empty edges (fresh migration) rebuild even without content changes

    # ── corpora ───────────────────────────────────────────────────────────────────────────────
    def code(self, root: Path, name: str) -> None:
        self._index_files(root, f"code:{name}", f"code/{name}/", CODE_EXT, "code")

    def tasks(self, workspace: Path) -> None:
        for d in workspace.glob("*/.tasks"):
            self._index_files(d, "tasks", f"tasks/{d.parent.name}/", {".yaml", ".yml"}, "task")

    def incidents(self, inc_dir: Path) -> None:
        """One atom per (week, kind): incidents are evidence for root-cause, not for chat recall."""
        for f in sorted(inc_dir.glob("*.jsonl")):
            src = f"incidents/{f.name}"
            snapshot = guarded_source_snapshot(self.store, src)
            raw = f.read_bytes()
            old_sha = snapshot[1][0] if snapshot[0] and snapshot[1] is not None else snapshot[1]
            if old_sha == sha1(raw):
                continue
            groups = defaultdict(list)
            for line in raw.decode("utf-8", "ignore").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                groups[r.get("kind", "?")].append(f"{r.get('ts', '')[:16]} {r.get('skill', '')} {r.get('note', '')}")
            atoms = [Atom(id=atom_id(src, n), source=src, corpus="incidents", doc_type="incident", trust=0.8,
                          title=f"incidents {f.stem} {k}", heading=k, text="\n".join(v)[:MAX_CHARS * 2], ord=n)
                     for n, (k, v) in enumerate(sorted(groups.items()))]
            if not guarded_source_replace(self.store, src, "incidents", sha1(raw), atoms, "", snapshot):
                self.stats["source_conflicts"] += 1
                continue
            self._embed(atoms)

    # ── graph: typed edges only (rare-term neighbours are gone: unranked edges were noise, C6) ──
    def _resolve_target(self, t: str, src: str, by_path: dict, by_stem: dict) -> str | None:
        t = t.strip()
        key = t.lower()
        return (by_path.get(key) or by_path.get(key + ".md") or by_stem.get(Path(key).stem)
                or by_path.get(os.path.normpath(os.path.join(os.path.dirname(src[6:]), key)).lower()))

    def _edge_row(self, src_id: str, dst_id: str, kind: str, provenance: str, basis: str,
                  fm: dict) -> tuple:
        return (src_id, dst_id, kind, 1.0, fm.get("project", ""), provenance, basis[:200],
                fm.get("owner", ""), EDGE_SCHEMA_VERSION, fm.get("valid_from", ""),
                fm.get("valid_to", ""), now())

    def _graph_sources(self, corpora):
        capture = getattr(self.store, "graph_source_versions", None)
        commit = getattr(self.store, "replace_edges_if_current", None)
        if callable(capture) != callable(commit):
            raise RuntimeError("incomplete guarded graph store contract")
        if callable(capture):
            versions = capture(tuple(corpora))
            return list(versions), versions
        return [s for corpus in corpora for s in self.store.sources(corpus)], None

    def _graph_frontmatter(self, path, source, versions, *, head_only=False):
        if versions is None:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raw = path.read_bytes()
            if sha1(raw) != versions[source][0]:
                raise ValueError("source file changed while graph was derived")
            text = raw.decode("utf-8", errors="ignore")
        return frontmatter(text[:4000] if head_only else text)[0]

    def _commit_graph(self, bulk, versions, corpora):
        if versions is None:
            self.store.replace_edges_bulk(bulk)
        else:
            self.store.replace_edges_if_current(bulk, versions, tuple(corpora))

    def _link_edges(self, root: Path) -> None:
        """Typed edges per document: wikilink/mdlink → kind 'link', frontmatter supersedes: →
        'supersedes', frontmatter project: → 'project' (only if the target document exists).
        Every source's edge set is replaced in ONE transaction (A3/A4); the table is a derived
        index — a fresh migration repopulates on the next indexing pass."""
        by_stem, by_path = {}, {}
        corpora = ("vault",)
        sources, versions = self._graph_sources(corpora)
        for src in sources:
            rel = src[len("vault/"):]
            by_path[rel.lower()] = src
            by_stem.setdefault(Path(rel).stem.lower(), src)
        total = 0
        bulk: dict[str, list[tuple]] = {}
        for src in sources:
            rel = src[len("vault/"):]
            p = Path(root) / rel
            fm = self._graph_frontmatter(p, src, versions) if p.is_file() else {}
            if versions is not None and not p.is_file():
                raise ValueError("source file missing while graph was derived")
            rows = self.store.source_rows(src)
            bulk[src] = self._edges_for_source(src, rows, fm, by_path, by_stem)
            total += len(bulk[src])
        self._commit_graph(bulk, versions, corpora)
        self.stats["edges"] = total

    # ── graph: rebuild from already-indexed atoms — no re-chunking, no re-embedding (A17) ──────
    def rebuild_edges(self, root: Path | None = None, corpora: tuple[str, ...] = ("vault",)) -> int:
        """`twinctx edges --rebuild`: derives the typed edges from the atoms already in the store
        (wikilinks/mdlinks from atom text, frontmatter from the head of the file on disk when it is
        still there). atoms/atoms_fts/vectors are never touched — safe on the live 1.7 GB db (on a
        copy); the whole set is replaced in one transaction, so a second run is a no-op (A17)."""
        by_stem, by_path = {}, {}
        sources, versions = self._graph_sources(corpora)
        for src in sources:
            rel = src.split("/", 1)[1] if "/" in src else src
            by_path[rel.lower()] = src
            by_stem.setdefault(Path(rel).stem.lower(), src)
        total = 0
        bulk: dict[str, list[tuple]] = {}
        for src in sources:
            fm = {}
            if root is not None:
                rel = src.split("/", 1)[1] if "/" in src else src
                p = Path(root) / rel
                if p.is_file():   # only the head: frontmatter sits at the top of the document
                    fm = self._graph_frontmatter(p, src, versions, head_only=True)
            rows = self.store.source_rows(src)
            bulk[src] = self._edges_for_source(src, rows, fm, by_path, by_stem)
            total += len(bulk[src])
        self._commit_graph(bulk, versions, corpora)
        self.stats["edges"] = total
        return total

    def _edges_for_source(self, src: str, rows: list, fm: dict, by_path: dict, by_stem: dict) -> list[tuple]:
        """Typed edges of one document from (id, text) rows + frontmatter — shared by the indexing
        pass and `edges --rebuild`, so both derivations produce the same set (A17)."""
        first = rows[0][0] if rows else ""
        edges: dict[tuple, tuple] = {}
        for aid, text in rows:
            targets = [(t, "wikilink") for t in WIKILINK.findall(text)]
            targets += [(t, "mdlink") for t in MDLINK.findall(text)]
            for t, prov in targets:
                dst_src = self._resolve_target(t, src, by_path, by_stem)
                if not dst_src or dst_src == src:
                    continue
                dst = self.store.first_atom_id(dst_src)
                if dst:
                    edges[(aid, dst, "link")] = self._edge_row(aid, dst, "link", prov, t, fm)
        if first:
            for field, kind in (("supersedes", "supersedes"), ("project", "project")):
                value = fm.get(field, "")
                if not value:
                    continue
                dst_src = self._resolve_target(value, src, by_path, by_stem)
                if not dst_src or dst_src == src:
                    continue
                dst = self.store.first_atom_id(dst_src)
                if dst:
                    edges[(first, dst, kind)] = self._edge_row(
                        first, dst, kind, f"frontmatter:{field}", value, fm)
        return list(edges.values())
