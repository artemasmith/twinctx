"""assemble(request) -> {packed, manifest} — the ContextOS contract (docs/context-layer.md C2–C8).

Pipeline:
  1 keyword   FTS5 BM25 over stemmed prefix terms
  2 vector    cosine top-N (if an embedder is configured; else degraded=["embedding_unavailable"])
  3 fuse      reciprocal-rank fusion
  4 floor     relevance floor: term coverage OR cosine — the engine can say "nothing relevant" (C3)
  5 filters   kind/corpus, memory validity window (bi-temporal), min trust
  6 graph     seed → traverse over typed edges — on by default, TWINCTX_GRAPH=0 / graph=False turns it off (C6, C7)
  7 pack      by score under the token budget, provenance header per chunk
  8 manifest  what was used, what was filtered and why, degraded modes, source ages
"""
import datetime as dt
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

from .connectors import is_raw_telegram_source
from .contracts import Atom, Store, now
from .reranker import filter_reranked as _filter_reranked
from .text import est_tokens, fts_query, query_terms, retrieval_terms, words

RRF_K = 60
_VECTOR_SCAN = threading.BoundedSemaphore(1)


def env_min_cosine() -> float:
    value = os.environ.get("TWINCTX_MIN_COSINE", "0.78").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    try:
        return float(value)
    except ValueError:
        raise ValueError("TWINCTX_MIN_COSINE must be a number") from None


def _no_deadline():
    pass


@dataclass
class AssembleRequest:
    query: str
    budget_tokens: int = 3000
    corpora: list[str] = field(default_factory=list)   # prefixes: "vault", "code:", "tasks", "incidents", "memory"
    include_memory: bool = True
    as_of: str = ""                                    # ISO time for memory validity; default now
    k: int = 12                                        # seeds kept after fusion
    min_coverage: float = 0.34                         # share of query stems an atom must contain …
    # … or this vector similarity; scale depends on the embedder (e5 ≈ 0.80, MiniLM ≈ 0.55) — calibrate with
    # `twinctx eval --min-cosine X` until negatives are empty and recall stops dropping, then set TWINCTX_MIN_COSINE
    min_cosine: float = field(default_factory=env_min_cosine)
    min_trust: float = 0.0
    graph_hops: int = 2
    graph_max_atoms: int = 6
    per_seed_edges: int = 4
    graph: bool | None = None                          # None = TWINCTX_GRAPH env → default True (see resolve_graph)
    project: str = ""                                  # edges of other projects never mix in (A6/A9)
    allowed_sources: tuple[str, ...] | None = None       # exact project sources, when explicitly scoped


def resolve_graph(req: "AssembleRequest") -> bool:
    """Graph switch: an explicit request value wins → TWINCTX_GRAPH env → default ON.
    The graph is part of the shipped search (DECISIONS 2026-10-28, second entry); opt-in remains
    only for measurements (eval modes pass `graph` explicitly, so env never leaks into eval)."""
    if req.graph is None:
        env = os.environ.get("TWINCTX_GRAPH", "").strip().lower()
        if env:
            return env not in ("0", "false", "no", "off")
        return True
    return req.graph


class Engine:
    def __init__(self, store: Store, embedder=None, redactor=None, *, lexical_only=False, check_deadline=None,
                 rerank=True):
        self.store = store
        self.embedder = embedder
        self._lexical_only = lexical_only
        self._check_deadline = check_deadline or _no_deadline
        self._enable_rerank = rerank
        # `redactor` is a deprecated no-op (ADR §16 amendment, 14.11.2026): the
        # parameter is kept for backward compatibility and IGNORED — the artifact of
        # record is local-only; masking is enforced on the outbound path to model
        # providers (LiteLLM DLP gateway) and in the embeddings adapter.
        self.redactor = None
        self._rows = None

    def memory_embedder(self):
        return self.embedder

    # ── retrieval ─────────────────────────────────────────────────────────────────────────────
    def _vector(self, query: str, n: int) -> tuple[list[tuple[str, float]], list[str]]:
        if not self.embedder:
            return [], ["embedding_unavailable", "bm25_only"]
        try:
            from .vectorsearch import rank_vectors  # numpy seam: optional dependency, not core
            # Network embedding must finish before opening the database snapshot.
            query_vec = self.embedder.embed_query(query)
            # ── mmap fast path (sustainable, in-repo): BLAS matmul over local .npy
            # 0.2–0.5s vs 80–90s streaming from PG. None = cache not loaded → fall through.
            try:
                from .vec_mmap import rank_mmap

                fast = rank_mmap(query_vec, n)
                if fast is not None:
                    return fast, ["vec_mmap"]
            except ImportError:
                pass
            if not _VECTOR_SCAN.acquire(blocking=False):
                return [], ["vector_scan_busy"]
            try:
                source = self.store.iter_vectors() if hasattr(self.store, "iter_vectors") else iter(self.store.vectors())
                rows = iter(source)
                try:
                    first = next(rows, None)
                    if first is None:
                        return [], ["no_vectors_indexed"]
                    return rank_vectors(chain((first,), rows), query_vec, n), []
                finally:
                    close = getattr(rows, "close", None)
                    if close:
                        close()
            finally:
                _VECTOR_SCAN.release()
        except Exception as e:  # never fail the request because of the embedder (ContextOS invariant 2)
            return [], [f"embedding_error:{type(e).__name__}"]

    def search(self, req: AssembleRequest, *, checkpoint=None) -> tuple[list[Atom], dict]:
        self._check_deadline()
        as_of = req.as_of or now()
        counts = Counter()
        online_candidates = getattr(self.store, "online_candidates", None) if self._lexical_only else None
        terms = retrieval_terms(req.query) if online_candidates else query_terms(req.query)
        if online_candidates:
            kw, counts["ranked_candidates"] = online_candidates(terms, req, as_of)
        else:
            kw = self.store.bm25(fts_query(terms), 60) if terms else []
        vec, degraded = ([], ["vector_ranker_unavailable"]) if self._lexical_only else self._vector(req.query, 60)
        fused: dict[str, float] = Counter()
        for rank, (i, _) in enumerate(kw):
            fused[i] += 1 / (RRF_K + rank)
        for rank, (i, _) in enumerate(vec):
            fused[i] += 1 / (RRF_K + rank)
        cos = dict(vec)
        atoms = self.store.get(list(fused))
        kept = []
        lexical_strength = dict(kw) if online_candidates else {}
        online_rank = {}
        for i, score in sorted(fused.items(), key=lambda x: (-x[1], x[0])):
            self._check_deadline()
            a = atoms.get(i)
            if not a or is_raw_telegram_source(a.source):
                continue
            if req.allowed_sources is not None and a.source not in req.allowed_sources:
                counts["filtered_by_source"] += 1
                continue
            if req.corpora and not any(a.corpus.startswith(c) for c in req.corpora):
                counts["filtered_by_corpus"] += 1
                continue
            if a.kind == "memory":
                if not req.include_memory:
                    counts["filtered_by_kind"] += 1
                    continue
                if (a.valid_to and a.valid_to <= as_of) or (a.valid_from and a.valid_from > as_of):
                    counts["filtered_by_time"] += 1
                    continue
            if a.trust < req.min_trust:
                counts["filtered_by_trust"] += 1
                continue
            cov = coverage(terms, a)
            if cov < req.min_coverage and cos.get(i, 0.0) < req.min_cosine:
                counts["filtered_by_relevance"] += 1
                continue
            if online_candidates:
                labels = words(f"{a.title} {a.heading} {a.source.rsplit('/', 1)[-1]}")
                label_coverage = sum(any(label.startswith(term) for label in labels) for term in terms) / len(terms)
                if len(terms) >= 4 and cov * len(terms) <= 2 and not label_coverage:
                    counts["filtered_by_relevance"] += 1
                    continue
                # A named source is stronger evidence than an incidental mention in a note.
                online_rank[i] = lexical_strength[i] + 3 * cov + 4 * label_coverage
            a.score, a.extra = score, {"coverage": round(cov, 2), "cosine": round(cos.get(i, 0.0), 3)}
            kept.append(a)
        counts["candidates"] = len(fused)
        if online_candidates:
            kept.sort(key=lambda atom: online_rank[atom.id], reverse=True)
            by_span = {}
            for atom in kept:
                by_span.setdefault((atom.source, atom.start_line // 40), atom)
            kept = list(by_span.values())[:60]
        if checkpoint is not None and kept:
            checkpoint((kept[: req.k], {"counts": counts.copy(), "degraded": list(degraded), "terms": terms}))

        # ── Реранкер: LLM-as-reranker (главный, ~1с) или cross-encoder (fallback, CPU).
        # Владелец 22.11: «реранкер нам нужен 100% — наивный RAG не даёт буста».
        if self._enable_rerank:
            llm_result = self._llm_rerank(req, kept)
            if llm_result is not None:
                kept = llm_result
                counts["rerank_kept"] = len(kept)
                counts["rerank"] = "llm"
            else:
                reranker_result = self._rerank(req, kept)
                if reranker_result is not None:
                    kept = reranker_result[0]
                    counts["rerank_kept"] = len(kept)
                    counts["rerank"] = "cross-encoder"

        result = kept[: req.k]
        info = {"counts": counts, "degraded": degraded, "terms": terms}
        if req.allowed_sources is not None and result:
            sources = {atom.source for atom in result}
            indexed_hashes = getattr(self.store, "indexed_source_hashes", None)
            info["source_sha1"] = indexed_hashes(sources) if online_candidates else {
                source: self.store.source_sha(source) for source in sources
            }
        return result, info

    def _llm_rerank(self, req: AssembleRequest, kept: list[Atom]) -> Optional[list[Atom]]:
        """LLM реранкинг: один API-вызов оценивает релевантность всех пар (query, doc).

        Возвращает отфильтрованный список (score >= порога) или None (выключен/ошибка).
        Порог: TWINCTX_LLM_RERANKER_MIN (дефолт 5.0 из 10 — ниже = мусор).
        """
        from .llm_reranker import from_env as llm_reranker_enabled
        from .llm_reranker import llm_rerank

        if not llm_reranker_enabled() or not kept:
            return None
        self._check_deadline()
        try:
            texts = [a.text[:500] for a in kept]
            scores = llm_rerank(req.query, texts)
            if scores is None:
                return None
            threshold = float(os.environ.get("TWINCTX_LLM_RERANKER_MIN", "5.0"))
            items = [(a, s) for a, s in zip(kept, scores) if s >= threshold]
            items.sort(key=lambda p: -p[1])
            for a, s in items:
                a.score = s
                a.extra["llm_rerank_score"] = round(s, 1)
            return [a for a, _ in items]
        except Exception:
            return None

    def _rerank(self, req: AssembleRequest, kept: list[Atom]) -> Optional[tuple[list[Atom], list[float]]]:
        """Cross-encoder reranking: (query, doc.text) → relevance score для каждой пары.

        Возвращает (top-k sorted by score, scores) или None (реранкер выключен/ошибка).
        Порог 0.0 (bge-reranker logit): отрицательный = нерелевантно.
        """
        from .reranker import from_env as reranker_from_env

        reranker = reranker_from_env()
        if reranker is None or not kept:
            return None
        self._check_deadline()
        try:
            texts = [a.text[:2048] for a in kept]
            scores = reranker.rerank(req.query, texts)
            if len(scores) != len(kept):
                return None
            items, filtered_scores = _filter_reranked(
                kept, scores, threshold=float(os.environ.get("TWINCTX_RERANK_THRESHOLD", "0.0")),
                k=req.k,
            )
            for a, s in zip(items, filtered_scores):
                a.score = s
                a.extra["rerank_score"] = round(s, 3)
            return items, filtered_scores
        except Exception:
            return None  # реранкер не должен ломать поиск (ContextOS invariant 2)

    # ── graph expansion (seed → traverse; typed edges only; bounded; decayed; default-on) ──────
    DECAY = 0.7   # score × weight × 0.7^(hop-1)

    def expand(self, seeds: list[Atom], req: AssembleRequest, graph: bool) -> list[Atom]:
        """Typed-edge BFS from the surviving seeds, 1..graph_hops hops (A7). Per-hop and total
        limits apply, seeds/taken atoms are never duplicated, and every candidate passes the
        usual corpus/kind/trust checks. Edge validity/project/dangling filters live in SQL (A6)."""
        if not graph or req.graph_hops < 1 or not seeds:
            return []
        as_of = req.as_of or now()
        seen = {a.id for a in seeds}
        frontier = [(s, s.id) for s in seeds[:5]]          # (node atom, origin seed id)
        pool: dict[str, tuple] = {}                        # id -> (score, hop, seed_id, edge)
        considered, hops_used = 0, 0
        for hop in range(1, req.graph_hops + 1):
            self._check_deadline()
            if len(pool) >= req.graph_max_atoms:
                break
            decay = self.DECAY ** (hop - 1)
            found = []
            for node, origin in frontier:
                for e in self.store.neighbors([node.id], req.per_seed_edges,
                                              project=req.project, as_of=as_of):
                    considered += 1
                    if e["nb"] not in seen:
                        found.append((e["nb"], e, origin, node.score * e["weight"] * decay))
            if not found:
                break
            found.sort(key=lambda x: (-x[3], x[0]))
            got = self.store.get([i for i, *_ in found])
            accepted = []
            for i, e, origin, sc in found:
                if len(pool) >= req.graph_max_atoms:
                    break
                a = got.get(i)
                if not a or is_raw_telegram_source(a.source):
                    continue
                if (a.kind == "memory" and not req.include_memory) or a.trust < req.min_trust:
                    continue
                if req.corpora and not any(a.corpus.startswith(c) for c in req.corpora):
                    continue
                seen.add(i)
                pool[i] = (sc, hop, origin, e)
                a.score = sc
                accepted.append((a, origin))
            if not accepted:
                break
            hops_used = hop
            frontier = accepted
        ranked = sorted(pool.items(), key=lambda x: (-x[1][0], x[0]))[: req.graph_max_atoms]
        self.graph_edges_considered, self.graph_hops_used = considered, hops_used
        out = []
        for i, (sc, hop, origin, e) in ranked:
            a = self.store.get([i]).get(i)
            if not a or is_raw_telegram_source(a.source):
                continue
            a.via = "graph"
            a.extra = {"hop": hop, "from": origin,
                       "edge": {"kind": e["kind"], "basis": e["basis"], "provenance": e["provenance"],
                                "owner": e["owner"], "project": e["project"]}}
            out.append(a)
        return out

    # ── assemble ──────────────────────────────────────────────────────────────────────────────
    def assemble(self, req: AssembleRequest, *, checkpoint=None) -> dict:
        def publish_search(result):
            seeds, info = result
            packed = self._pack(req, seeds, [], info, graph_on=False, source_ages={})
            if packed["packed"]:
                checkpoint(packed)

        seeds, info = self.search(req, checkpoint=publish_search if checkpoint is not None else None)
        graph_on = resolve_graph(req)
        extra = self.expand(seeds, req, graph_on)
        return self._pack(req, seeds, extra, info, graph_on=graph_on, source_ages=self.store.source_ages())

    def _pack(self, req, seeds, extra, info, *, graph_on, source_ages):
        parts, used, tokens = [], [], 0
        counts = info["counts"]
        seen_spans = set()
        for a in seeds + extra:
            self._check_deadline()
            span = (a.source, a.start_line // 40)
            if span in seen_spans:
                counts["filtered_as_duplicate"] += 1
                continue
            block = render(a)
            t = est_tokens(block)
            if tokens + t > req.budget_tokens:
                counts["filtered_by_budget"] += 1
                continue
            seen_spans.add(span)
            parts.append(block)
            tokens += t
            used.append({"id": a.id, "source": a.source, "lines": [a.start_line, a.end_line], "kind": a.kind,
                         "via": a.via, "score": round(a.score, 4), "updated": a.updated, **a.extra})
        res = {
            "packed": "\n\n".join(parts),
            "manifest": {
                "query": req.query, "terms": info["terms"], "tokens": tokens, "budget": req.budget_tokens,
                "items": used, "nothing_relevant": not used,
                **{k: v for k, v in counts.items()},
                "graph_enabled": graph_on,
                "graph_expanded_count": sum(1 for u in used if u["via"] == "graph"),
                "graph_hops_used": getattr(self, "graph_hops_used", 0) if graph_on else 0,
                "graph_edges_considered": getattr(self, "graph_edges_considered", 0) if graph_on else 0,
                "degraded": info["degraded"], "source_ages": source_ages,
            },
        }
        # 14.11 (ADR 16 amended): the engine never masks. The local index stores raw text and
        # masking is applied on the outbound path to a model provider (DLP gateway), which keeps
        # its own placeholder-to-name mapping. Masking here re-masked already-masked text with a
        # fresh numbering and lost the mapping, so one person appeared as NAME_3 and NAME_7.
        return res


def coverage(terms: list[str], a: Atom) -> float:
    if not terms:
        return 0.0
    hay = " ".join(words(f"{a.title} {a.heading} {a.text}"))
    return sum(1 for t in terms if re.search(r"(^|\s)" + re.escape(t), hay)) / len(terms)


def render(a: Atom) -> str:
    where = f"{a.source}:{a.start_line}-{a.end_line}" if a.end_line else a.source
    meta = ", ".join(x for x in (a.kind if a.kind == "memory" else "", a.updated and f"updated {a.updated}",
                                 a.via == "graph" and "via link") if x)
    head = f"### [{where}] {a.title}" + (f" › {a.heading}" if a.heading and a.heading != a.title else "")
    return head + (f" ({meta})" if meta else "") + "\n" + a.text.strip()


def age_days(iso: str) -> float:
    try:
        return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(iso)).total_seconds() / 86400
    except ValueError:
        return math.inf
