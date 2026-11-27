"""MCP server `context` for Claude Code / opencode / Hermes workers (stdio).

Tools (names are what agents see — keep them few and non-overlapping):
  context_assemble  — the main entry: packed context under a budget + manifest (ContextOS contract)
  context_search    — cheap ranked list (source, lines, title, snippet) when you only need pointers
  context_read      — a source's chunks by line range, from the index (no filesystem access)
  context_related   — notes linked to/from a source
  context_remember  — write a memory fact (superseding an old one closes its validity, never deletes)
  context_state     — the "state" kind (C1): hot memory + tasks in progress, read directly, never ranked
Every call is appended to data/context/usage.jsonl — C8: a subsystem without a consumer is not delivered.
"""

import json
import logging
import os
import time
from pathlib import Path

from stores import online_search, open_store

from .cli import HOME
from .connectors import is_raw_telegram_source
from .contracts import now
from .engine import AssembleRequest
from .limits import check_budget_tokens, check_k, check_memory_text, check_read_limits
from .memory import MemoryDraft, remember

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server(
    "context",
    instructions=(
        "Search the configured vault, code and memory before reading files. Call context_assemble with the question you "
        "actually need answered; read the manifest: nothing_relevant=true means stop searching, not guess."
    ),
)
_store = open_store()
_engine = online_search(_store)
USAGE = Path(os.environ.get("TWINCTX_USAGE", HOME / "data" / "context" / "usage.jsonl"))
logger = logging.getLogger(__name__)


def _log(tool: str, **kw) -> None:
    try:
        USAGE.parent.mkdir(parents=True, exist_ok=True)
        with USAGE.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"ts": now(), "tool": tool, "agent": os.environ.get("HARNESS_AGENT", "?"), **kw}, ensure_ascii=False
                )
                + "\n"
            )
    except OSError as error:
        logger.warning(
            "twinctx usage log unavailable tool=%s errno=%s degraded=%s", tool, error.errno, kw.get("degraded", [])
        )


@server.tool()
def context_project_search(project: str, query: str) -> str:
    """Find a current Markdown source in one named project. Uses the optional QMD fallback for semantic matches."""
    from .project_search import answer_project

    return answer_project(project, query, search=_engine.search,
                          vault_root=Path(os.environ.get("VAULT_DIR", HOME / "vault")))


@server.tool()
def context_assemble(
    query: str,
    budget_tokens: int = 2500,
    corpora: list[str] | None = None,
    include_memory: bool = True,
    graph: bool | None = None,
    hops: int = 2,
    project: str = "",
) -> str:
    """Relevant context for a question, packed under a token budget, with provenance per chunk.
    corpora: optional prefixes — "vault", "code:<repo>", "tasks", "incidents", "memory".
    graph: typed-edge traversal is ON by default (None = TWINCTX_GRAPH env); pass False to turn it
    off for this call. hops: 1–2. project: restrict graph edges to one project."""
    t = time.monotonic()
    if not (1 <= hops <= 2):
        raise ValueError(f"hops must be 1 or 2, got {hops}")
    check_budget_tokens(budget_tokens)
    res = _engine.assemble(
        AssembleRequest(
            query=query,
            budget_tokens=budget_tokens,
            corpora=corpora or [],
            include_memory=include_memory,
            graph=graph,
            graph_hops=hops,
            project=project or "",
        )
    )
    m = res["manifest"]
    _log(
        "assemble",
        items=len(m["items"]),
        tokens=m["tokens"],
        ms=int((time.monotonic() - t) * 1000),
        nothing=m["nothing_relevant"],
        degraded=m["degraded"],
    )
    summary = {
        k: m[k] for k in ("tokens", "budget", "nothing_relevant", "degraded", "graph_enabled", "graph_expanded_count")
    }
    summary.update({k: v for k, v in m.items() if k.startswith("filtered_")})
    if "ranked_candidates" in m:
        summary["ranked_candidates"] = m["ranked_candidates"]
    summary["sources"] = [f"{i['source']}:{i['lines'][0]}-{i['lines'][1]}" for i in m["items"]]
    if res["packed"]:
        body = res["packed"]
    elif m["nothing_relevant"]:
        body = "NOTHING RELEVANT in the index for this query."
    else:
        body = "CONTEXT SEARCH INTERRUPTED. See degraded in the manifest."
    return body + "\n\n<manifest>" + json.dumps(summary, ensure_ascii=False) + "</manifest>"


@server.tool()
def context_search(query: str, k: int = 8) -> list[dict]:
    """Ranked pointers only (no bodies): use when you will read one or two of them next."""
    check_k(k)
    started = time.monotonic()
    atoms, info = _engine.search(AssembleRequest(query=query, k=k))
    _log(
        "search",
        items=len(atoms),
        degraded=info["degraded"],
        ms=int((time.monotonic() - started) * 1000),
    )
    return [
        {
            "source": a.source,
            "lines": [a.start_line, a.end_line],
            "title": a.title,
            "heading": a.heading,
            "snippet": a.text[:240],
            "score": round(a.score, 4),
        }
        for a in atoms
    ]


@server.tool()
def context_read(source: str, start_line: int = 0, end_line: int = 0, max_chars: int = 8000) -> str:
    """Chunks of one source (as returned by search/assemble), optionally limited to a line range.
    Output is capped at max_chars: a truncation marker names the next line to continue from —
    a silently truncated read makes the model answer from a partial source (and lie about it)."""
    started = time.monotonic()
    check_read_limits(max_chars, start_line, end_line)
    atoms = [] if is_raw_telegram_source(source) else _store.by_source(source)
    parts = [
        a for a in atoms if (not start_line or a.end_line >= start_line) and (not end_line or a.start_line <= end_line)
    ]
    shown, used, last_end = [], 0, 0
    for a in parts:
        block = f"[{a.source}:{a.start_line}-{a.end_line}]\n{a.text}"
        if shown and used + len(block) > max_chars:
            break
        shown.append(block)
        used += len(block)
        last_end = a.end_line
    truncated = len(shown) < len(parts)
    _log("read", chunks=len(parts), shown=len(shown), truncated=truncated, ms=int((time.monotonic() - started) * 1000))
    out = "\n\n".join(shown) or f"no such source: {source}"
    if truncated:
        out += (
            f"\n\n[обрезано: показано {len(shown)} из {len(parts)} чанков (~{used} символов)."
            f" Продолжение: context_read(source, start_line={last_end + 1})]"
        )
    return out


@server.tool()
def context_related(source: str, k: int = 10) -> list[dict]:
    """Notes linked from or to this source (explicit wikilinks/markdown links), strongest first."""
    started = time.monotonic()
    check_k(k)
    ids = [] if is_raw_telegram_source(source) else [a.id for a in _store.by_source(source)]
    edges = _store.neighbors(ids, k)
    got = _store.get([edge["nb"] for edge in edges])
    seen, out = set(), []
    for edge in sorted(edges, key=lambda edge: -edge["weight"]):
        a = got.get(edge["nb"])
        if a and not is_raw_telegram_source(a.source) and a.source not in seen and a.source != source:
            seen.add(a.source)
            out.append({"source": a.source, "title": a.title, "kind": edge["kind"], "weight": edge["weight"]})
    _log("related", items=len(out), ms=int((time.monotonic() - started) * 1000))
    return out[:k]


@server.tool()
def context_remember(text: str, about: str = "", supersedes: str = "") -> str:
    """Store a durable fact/decision as memory. `about` = the vault path it concerns (e.g. projects/demo.md).
    To correct an old fact pass its id in `supersedes`: the old one stays in history with valid_to set."""
    started = time.monotonic()
    check_memory_text(text)
    atom_id_ = remember(
        _store,
        MemoryDraft(text=text, about=about, trust=0.7, supersedes=supersedes),
        get_embedder=_engine.memory_embedder,
        refresh_cache=lambda: setattr(_engine, "_rows", None),
    )
    _log("remember", supersedes=bool(supersedes), ms=int((time.monotonic() - started) * 1000))
    return atom_id_


@server.tool()
def context_state() -> str:
    """Current state (always relevant, never ranked): hot memory and tasks in progress."""
    started = time.monotonic()
    out = []
    hot = HOME / "data" / "memory" / "hot.md"
    if hot.is_file():
        out.append(hot.read_text()[:4000])  # 14.11: raw; masking is the DLP gateway job
    for f in sorted((HOME / "workspace").glob("*/.tasks/*.yaml")):
        head = f.read_text(errors="ignore")[:1500]
        if "status: in_progress" in head or "status: verify" in head or "status: blocked" in head:
            out.append(f"## {f.parent.parent.name}/{f.name}\n{head}")
    _log("state", ms=int((time.monotonic() - started) * 1000))
    return "\n\n".join(out) or "no state recorded"


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
