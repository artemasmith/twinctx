"""Embedding port. TWINCTX_EMBED selects the adapter:
  none                              — BM25 only (engine reports degraded=["embedding_unavailable"])
  fastembed:<model>                 — local CPU ONNX, e.g. fastembed:intfloat/multilingual-e5-large
  openai:<model>[|<query-model>]@<base_url> — any OpenAI-compatible /embeddings; default is the LiteLLM
                                      alias `embed` (Yandex uses separate doc/query models → embed|embed-query)
Vectors are L2-normalised float32 so the engine uses a dot product as cosine.

The OpenAI adapter never lets one bad passage kill the run (incident 2026-10-26: an atom over the
bge-m3 8192-token context 400ed every batch it touched, embedding stalled and the index alarmed
every 15 min):
  * passages over `.max_tokens` (EMBED_MAX_TOKENS, default 8000) are never sent — they come back
    as None placeholders and stay BM25-only (mixing models in one vector space is not an option);
  * a permanent 4xx on a batch degrades to per-item calls: only the offending items become None.
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

from .text import est_tokens

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _norm(vec):
    import numpy as np

    # Scale before taking the norm: converting to float32 or squaring the raw
    # values first can overflow, while tiny but nonzero vectors can underflow.
    if isinstance(vec, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in vec):
        raise ValueError("invalid embedding vector")
    try:
        raw = np.asarray(vec)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
            raise ValueError("invalid embedding vector")
        with np.errstate(over="ignore", invalid="ignore"):
            values = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid embedding vector") from None
    if not np.isfinite(values).all() or np.any((raw != 0) & (values == 0)):
        raise ValueError("invalid embedding vector")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return values.astype(np.float32).tobytes()
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled = values / scale
        norm = float(np.linalg.norm(scaled))
        return (scaled / norm).astype(np.float32).tobytes()


class PermanentEmbedError(Exception):
    """Provider 4xx: retrying the exact same request can never help (oversized input, bad model, bad key)."""


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model, threads=os.cpu_count())
        self.e5 = "e5" in model.lower()

    def embed_passages(self, texts):
        pre = [("passage: " + t) if self.e5 else t for t in texts]
        return [_norm(v) for v in self.model.embed(pre, batch_size=32)]

    def embed_query(self, text):
        return _norm(next(iter(self.model.embed([("query: " + text) if self.e5 else text]))))


class OpenAIEmbedder:
    # The default is below the BGE-M3 8192 token limit. est_tokens is only an
    # estimate; configure the pinned tokenizer to count the actual sent text.
    DEFAULT_MAX_TOKENS = 8000
    BGE_TOKENIZER_SHA256 = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"

    def __init__(self, model: str, base: str, key: str):
        self.model, _, self.query_model = model.partition("|")
        self.query_model = self.query_model or self.model
        self.base, self.key = base.rstrip("/"), key
        self.max_tokens = int(os.environ.get("EMBED_MAX_TOKENS", self.DEFAULT_MAX_TOKENS))
        self.tokenizer_path = os.environ.get("TWINCTX_BGE_TOKENIZER", "")
        self._tokenizer = None
        self._tokenizer_tried = False
        self._tokenizer_lock = threading.Lock()
        self.last_oversized = 0

    def _exact_tokenizer(self):
        """Load the pinned BGE-M3 tokenizer once per process; never download at runtime."""
        if not self.tokenizer_path:
            return None
        with self._tokenizer_lock:
            if self._tokenizer_tried:
                return self._tokenizer
            self._tokenizer_tried = True
            try:
                from tokenizers import Tokenizer

                with open(self.tokenizer_path, "rb") as stream:
                    raw = stream.read(20 * 1024 * 1024 + 1)
                if len(raw) > 20 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != self.BGE_TOKENIZER_SHA256:
                    raise ValueError("tokenizer integrity check failed")
                self._tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
            except Exception as exc:  # optional dependency; preserve existing availability
                print(f"twinctx: BGE tokenizer unavailable ({type(exc).__name__}); "
                      "embedding length guard is approximate", file=sys.stderr, flush=True)
            return self._tokenizer

    def _call(self, texts, model=None, attempts=5):
        """Rate limits and restarts are normal on a fresh provider account: back off (2, 4, 8, 16 s) instead of
        failing the whole index; a permanent error (bad key, bad model, oversized input) raises at once."""
        body = json.dumps({"model": model or self.model, "input": texts}).encode()
        for attempt in range(attempts):
            req = urllib.request.Request(f"{self.base}/embeddings", data=body,
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return [d["embedding"] for d in sorted(json.load(r)["data"], key=lambda d: d["index"])]
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS:
                    raise PermanentEmbedError(f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}") from e
                if attempt == attempts - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == attempts - 1:
                    raise
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError("unreachable")

    def embed_passages(self, texts):
        """Returns one entry per input: a normalised vector, or None for passages the provider
        cannot take (over `.max_tokens`, or a permanent per-item 4xx) — those stay BM25-only."""
        out: list = [None] * len(texts)
        tokenizer = self._exact_tokenizer()
        limit = min(self.max_tokens, 8192) if tokenizer else self.max_tokens
        sendable = [i for i, t in enumerate(texts)
                    if (len(tokenizer.encode(t).ids) if tokenizer else est_tokens(t)) <= limit]
        self.last_oversized = len(texts) - len(sendable)
        for b in range(0, len(sendable), 32):
            batch = sendable[b:b + 32]
            try:
                vecs = self._call([texts[i] for i in batch])
            except PermanentEmbedError:  # isolate: one bad item must not starve the whole batch
                vecs = self._isolate(batch, texts)
            for i, v in zip(batch, vecs):
                if v is not None:
                    out[i] = _norm(v)
        return out

    def _isolate(self, batch, texts):
        vecs = []
        for i in batch:
            try:
                vecs.append(self._call([texts[i]])[0])
            except PermanentEmbedError:
                vecs.append(None)  # keeps zip(batch, vecs) aligned
        return vecs

    def embed_query(self, text):
        tokenizer = self._exact_tokenizer()
        if tokenizer and len(tokenizer.encode(text).ids) > min(self.max_tokens, 8192):
            raise PermanentEmbedError("embedding query exceeds tokenizer limit; lexical search remains available")
        return _norm(self._call([text], self.query_model)[0])


def from_env():
    spec = os.environ.get("TWINCTX_EMBED", "none")
    if spec == "none":
        return None
    kind, _, rest = spec.partition(":")
    if kind == "fastembed":
        return FastEmbedder(rest)
    if kind == "openai":
        model, _, base = rest.partition("@")
        return OpenAIEmbedder(model, base or "http://127.0.0.1:4000/v1",
                              os.environ.get("TWINCTX_EMBED_KEY") or os.environ.get("LITELLM_MASTER_KEY")
                              or os.environ.get("LITELLM_API_KEY", ""))
    raise SystemExit(f"unknown TWINCTX_EMBED={spec}")
