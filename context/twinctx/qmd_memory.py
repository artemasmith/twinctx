"""Semantic candidates come from QMD. Decision identity, text and validity come from the live store."""
import datetime as dt
import hashlib
import math
import os
from dataclasses import replace
from http.client import HTTPException

from .contracts import Atom


def document(atom: Atom) -> bytes:
    return f"# {atom.title or 'Решение'}\n\n{atom.source}\n\n{atom.text}\n".encode()


def filename(atom_id: str) -> str:
    return hashlib.sha256(atom_id.encode()).hexdigest() + ".md"


def instant(value):
    point = dt.datetime.fromisoformat(value)
    return point if point.tzinfo else point.replace(tzinfo=dt.timezone.utc)


def current_version(family, at):
    """Accept one connected, unbranched history and one effective fact at the requested instant."""
    if not family or len(family) > 256:
        return None
    roots = [a for a in family if not a.supersedes]
    if len(roots) != 1:
        return None
    children = {a.supersedes: a for a in family if a.supersedes}
    if len(children) != len(family) - 1:
        return None
    node = roots[0]
    active = []
    for _ in family:
        if node is None:
            return None
        if (not node.valid_from or instant(node.valid_from) <= at) and (not node.valid_to or at < instant(node.valid_to)):
            active.append(node)
        node = children.get(node.id)
    return active[0] if node is None and len(active) == 1 else None


def is_current(store, atom, as_of):
    if os.environ.get("TWINCTX_QMD_ENABLED") == "1":
        valid = current_version(store.memory_family(atom.id), instant(as_of))
        return valid is not None and atom.id == valid.id
    return (not atom.valid_from or atom.valid_from <= as_of) and (not atom.valid_to or as_of < atom.valid_to)


def candidates(store, req, as_of, degraded):
    if (os.environ.get("TWINCTX_QMD_ENABLED") != "1" or not req.include_memory
        or req.corpora and not any("memory".startswith(c) for c in req.corpora)
        or req.allowed_sources is not None and not any(s.startswith("memory/") for s in req.allowed_sources)
        or not req.query.strip() or len(req.query) > 500):
        return []
    from .qmd_client import hits

    at = instant(as_of)
    threshold = float(os.environ.get("TWINCTX_QMD_MEMORY_MIN_SCORE", "0.45"))
    if not 0 <= threshold <= 1:
        raise ValueError("TWINCTX_QMD_MEMORY_MIN_SCORE must be between 0 and 1")
    found = {}
    try:
        for hit in hits("/memory", {"q": req.query}, 3):
            score, atom_id = hit["score"], hit["id"]
            if not isinstance(atom_id, str) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("invalid memory candidate")
            if not threshold <= score <= 1.000001:
                continue
            family = store.memory_family(atom_id)
            seed = next((a for a in family if a.id == atom_id), None)
            if (seed is None or (seed.valid_from and instant(seed.valid_from) > at) or seed.trust < req.min_trust
                or req.allowed_sources is not None and seed.source not in req.allowed_sources
                or hashlib.sha256(document(seed)).hexdigest() != hit["sha256"]):
                continue
            active = current_version(family, at)
            if (active is None or active.trust < req.min_trust
                or req.allowed_sources is not None and active.source not in req.allowed_sources):
                continue
            found.setdefault(active.id, replace(active, via="qmd_memory", extra={
                "qmd_similarity": round(score, 4), "matched_memory_id": seed.id,
            }))
    except (OSError, HTTPException, TypeError, KeyError, ValueError):
        degraded.append("qmd_memory_unavailable")
    return list(found.values())
