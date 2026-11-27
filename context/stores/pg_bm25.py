"""BM25/FTS-слой: грамматика запроса, паритетный FTS5 реранк и df-кэш.

Вынесено из pg_sources.py волной 5 техдолга без изменения поведения.
"""

import math
import re
import time
import unicodedata

from .pg_env import _env_flag, _env_number

_QUERY_TERM = re.compile(r'"([0-9A-Za-zА-Яа-яЁё_][0-9A-Za-zА-Яа-яЁё_\-]{0,255})"\*')


_LEXEMES = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


_BM25_POOL = 2000  # re-ranked candidates per query (engine asks for 60)


_BM25_K1, _BM25_B = 1.2, 0.75  # FTS5 defaults


_BM25_WEIGHTS = (("title", 4.0), ("heading", 2.0), ("text", 1.0))


_FTS_STATS_TTL = 600.0  # seconds; жёсткий сброс — только порогом изменений или этим TTL


_COMBINING = re.compile(r"[\u0300-\u036f]")


def _fold_tokens(value: str) -> list[str]:
    """Tokenise like the SQL side: NFD fold latin diacritics, lower, split on non-alnum."""
    folded = unicodedata.normalize("NFC", _COMBINING.sub("", unicodedata.normalize("NFD", value or "")))
    return _LEXEMES.findall(folded.lower())


def _bm25_parse(fts: str) -> list[tuple[str, bool]] | None:
    """('term', is_prefix) pairs from the quoted-prefix OR grammar, deduplicated."""
    out, seen = [], set()
    for term in fts.split(" OR "):
        match = _QUERY_TERM.fullmatch(term)
        if not match:
            return None
        lexemes = _LEXEMES.findall(match.group(1))
        if not lexemes:
            return None
        for i, lex in enumerate(lexemes):
            key = (lex.lower(), i == len(lexemes) - 1)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out or None


def _tsquery(fts: str) -> str | None:
    """Convert only the bounded quoted-prefix OR grammar produced by fts_query."""
    if not isinstance(fts, str) or not fts or len(fts) > 4096:
        return None
    terms = fts.split(" OR ")
    if len(terms) > 12:
        return None
    phrases = []
    for term in terms:
        match = _QUERY_TERM.fullmatch(term)
        if not match:
            return None
        lexemes = _LEXEMES.findall(match.group(1))
        if not lexemes:
            return None
        phrase = ["'" + value.lower() + "'" for value in lexemes]
        phrase[-1] += ":*"
        phrases.append("(" + " <-> ".join(phrase) + ")")
    return " | ".join(phrases)


class PgBm25Mixin:
    def _reset_fts_stats_if_stale(self) -> bool:
        """Жёсткий сброс, когда изменения превысили порог доли корпуса. True, если сброшено."""
        n = self._fts_stats.get("n") or 0
        fraction = _env_number("TWINCTX_FTS_DF_STALE_FRACTION", 0.01)
        if n > 0 and self._fts_writes > fraction * n:
            self._fts_writes = 0
            self._fts_stats["ts"] = 0.0
            self._fts_stats["df"] = {}
            return True
        return False

    def _bm25_corpus_stats(self) -> tuple[int, dict[str, float]]:
        """(N atoms, avg per-column lexeme count); sampled, TTL-cached, reset by writes."""
        stats = self._fts_stats
        if stats["n"] > 0 and time.monotonic() - stats["ts"] < _FTS_STATS_TTL:
            return stats["n"], stats["avgdl"]
        with self._lock:
            rows = self._run_read(
                """select c.reltuples::bigint from pg_class c
                   join pg_namespace ns on ns.oid = c.relnamespace
                   where c.relname = 'atoms' and ns.nspname = 'twinctx_v1'"""
            )
            estimate = rows[0][0] if rows else None
            n = (
                int(estimate)
                if estimate and estimate > 0
                else int(self._run_read("select count(*) from twinctx_v1.atoms")[0][0])
            )
            sample = self._run_read(
                """select avg(array_length(tsvector_to_array(fts_title), 1)),
                          avg(array_length(tsvector_to_array(fts_heading), 1)),
                          avg(array_length(tsvector_to_array(fts_text), 1))
                   from (select fts_title, fts_heading, fts_text
                         from twinctx_v1.atoms tablesample system (1)) s"""
            )[0]
            avgdl = {col: max(float(v or 1.0), 1.0) for (col, _), v in zip(_BM25_WEIGHTS, sample)}
            stats.update(ts=time.monotonic(), n=n, avgdl=avgdl)
            return n, avgdl

    def _bm25_df(self, lexeme: str, prefix: bool) -> int:
        """Documents containing the (prefix-)lexeme in any column; GIN count, cached."""
        key = f"{lexeme}:*" if prefix else lexeme
        stats = self._fts_stats
        if stats["n"] <= 0 or self._reset_fts_stats_if_stale():
            pass  # статистика ещё не считалась или только что сброшена порогом — пересчитается ниже
        elif key in stats["df"]:
            return stats["df"][key]
        if stats["n"] <= 0 or time.monotonic() - stats["ts"] >= _FTS_STATS_TTL:
            self._bm25_corpus_stats()
        literal = f"'{lexeme}:*'" if prefix else f"'{lexeme}'"
        with self._lock:
            rows = self._run_read(
                """select count(*) from twinctx_v1.atoms a
                   cross join (select to_tsquery('simple'::regconfig, :q) as query) qq
                   where a.fts_title @@ qq.query
                      or a.fts_heading @@ qq.query
                      or a.fts_text @@ qq.query""",
                q=literal,
            )
        df = int(rows[0][0])
        stats["df"][key] = df
        return df

    def bm25(self, fts: str, limit: int) -> list[tuple[str, float]]:
        """Exact BM25 ranking with FTS5 parity (IDF + length normalisation, 4/2/1 columns).

        GIN + ts_rank select the candidate pool; the final order is computed in
        Python from per-lexeme document frequencies so rare distinctive terms
        beat long repetitive documents — the SQLite backend always ranked this way.
        """
        query = _tsquery(fts)
        terms = _bm25_parse(fts)
        if query is None or terms is None or type(limit) is not int or limit <= 0:
            return []
        pool = max(limit, min(_BM25_POOL, limit * 8))
        with self._lock:
            rows = self._run_read(
                """select a.id,
                          4 * ts_rank(a.fts_title, q.query)
                          + 2 * ts_rank(a.fts_heading, q.query)
                          + ts_rank(a.fts_text, q.query) as score
                   from twinctx_v1.atoms a
                   cross join (select to_tsquery('simple'::regconfig, :query) as query) q
                   where a.fts_title @@ q.query or a.fts_heading @@ q.query or a.fts_text @@ q.query
                   order by score desc, a.id limit :lim""",
                query=query,
                lim=pool,
            )
        if not rows:
            return []
        atoms = self.get([row[0] for row in rows])
        n, avgdl = self._bm25_corpus_stats()
        scored: list[tuple[str, float]] = []
        for atom in atoms.values():
            self._check_search_deadline()
            tokens = {col: _fold_tokens(getattr(atom, col) or "") for col, _ in _BM25_WEIGHTS}
            total, idf_sum = 0.0, 0.0
            for lexeme, prefix in terms:
                df = self._bm25_df(lexeme, prefix)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5)) if n else 0.0
                idf_sum += idf
                for col, weight in _BM25_WEIGHTS:
                    column = tokens[col]
                    tf = sum(1 for token in column if (token.startswith(lexeme) if prefix else token == lexeme))
                    if not tf:
                        continue
                    norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * len(column) / avgdl.get(col, 1.0))
                    total += weight * idf * tf * (_BM25_K1 + 1.0) / (tf + norm)
            scored.append((atom.id, total if idf_sum > 0 else 0.0))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def _invalidate_fts_stats(self, changed: int = 1) -> None:
        """Статистика BM25 (N, avgdl, df) устаревает от записей.

        Мягкий режим (волна 2, по умолчанию): кэш живёт, пока изменения не превысили долю
        корпуса (TWINCTX_FTS_DF_STALE_FRACTION, дефолт 1%) — одиночные записи не сбрасывают
        df, и первый после тика индексации запрос не делает GIN-count по каждому лексему.
        TWINCTX_FTS_DF_SOFT_INVALIDATION=0 возвращает прежнюю семантику «сброс на каждой записи».
        """
        changed = max(int(changed), 0)
        if _env_flag("TWINCTX_FTS_DF_SOFT_INVALIDATION", True):
            self._fts_writes += changed
            self._fts_stats["n"] = (self._fts_stats.get("n") or 0) + changed  # оценка без запроса
            self._reset_fts_stats_if_stale()
            return
        self._fts_stats["ts"] = 0.0
        self._fts_stats["df"] = {}
