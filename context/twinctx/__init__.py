"""twinctx — context layer for the twin and dev agents.

Contract follows ContextOS: assemble(request) -> {packed, manifest}.
Three kinds (C1): state is read directly by callers and never indexed; knowledge is replaced per
source on reindex; memory is superseded (valid_to), never deleted.
"""
from .engine import Engine, AssembleRequest  # noqa: F401
from .contracts import Atom, Embedder, Store  # noqa: F401
