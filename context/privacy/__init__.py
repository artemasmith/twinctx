"""PII masking seam (ADR §16): nothing personal leaves for the LLM API unmasked.

Policy (measured decision 2026-11-06, AMENDED 2026-11-07 by owner):
- EVERYTHING outbound to model providers is masked, INCLUDING the owner's own typed input
  (logins, passwords, API keys, personal data). Enforcement moved to the LiteLLM gateway
  (deploy/litellm/dlp.py) — the only chokepoint that also sees Hermes/Claude Code input.
- Hard classes (phones, cards, emails, INN/SNILS/passport, IPv4, secrets, credential pairs)
  always; names/geo/orgs via the optional natasha NER extra.
- Placeholders are deterministic per call ([ИМЯ_1] → mapping); the mapping never leaves the
  host. The gateway persists its request mappings 0600 with a TTL (unmask on retries).
- mask_texts()/mask_messages() share ONE mapping across a whole request so placeholder
  numbering stays consistent between the owner's message and the recalled context.

Core is stdlib-only (the zero-dep contract); natasha lives behind `names=True` and degrades
to hard classes with a single stderr warning when not installed (extra `privacy`).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder → original
    ner_active: bool = False


# ── hard classes (stdlib, ordered: specific before generic) ──────────────────────────────────────
def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _inn_ok(s: str) -> bool:
    """10- or 12-digit INN checksum (filters random digit runs)."""
    digits = [int(c) for c in s]
    if len(digits) == 10:
        w = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return digits[-1] == sum(a * b for a, b in zip(digits, w)) % 11 % 10
    if len(digits) == 12:
        w1, w2 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8], [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        return (
            digits[10] == sum(a * b for a, b in zip(digits, w1)) % 11 % 10
            and digits[11] == sum(a * b for a, b in zip(digits, w2)) % 11 % 10
        )
    return False


def _card_repl(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return m.group(0) if not (_luhn_ok(digits) and len(digits) == 16) else "\x00CARD\x00"


def _inn_repl(m: re.Match) -> str:
    return m.group(0) if not _inn_ok(m.group(0)) else "\x00INN\x00"


_CLASSES: list[tuple[str, re.Pattern, Callable | None]] = [
    # secrets first: never let keys through in any direction (top formats from gitleaks/detect-secrets)
    (
        "СЕКРЕТ",
        re.compile(
            r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
            r"xox[bapos]-[A-Za-z0-9-]{10,}|age1[a-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
            r"gsk_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{30,}|"
            r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}|"
            r"\d{8,10}:[A-Za-z0-9_-]{30,40}|Bearer\s+[A-Za-z0-9._-]{16,})"
        ),
        None,
    ),
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), None),
    ("ТЕЛЕФОН", re.compile(r"(?<!\d)(?:\+7|8)[\s(-]{0,2}\d{3}[\s)-]{0,2}\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), None),
    ("СНИЛС", re.compile(r"\b\d{3}-\d{3}-\d{3}[ ]?\d{2}\b"), None),
    ("КАРТА", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), _card_repl),
    ("ИНН", re.compile(r"\b\d{10}(?:\d{2})?\b"), _inn_repl),
    ("ПАСПОРТ", re.compile(r"\b\d{4}\s?\d{6}\b"), None),  # после ИНН: валидный ИНН уже замаскирован
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None),
]

# КРЕД-пары: ключевое слово («пароль:», «api_key=») остаётся модели, маскируется только значение.
_CRED_RE = re.compile(
    r"(?i)\b(парол(?:ь|я)|password|pwd|логин|login|токен|token|api[_-]?key|apikey|"
    r"секрет|secret)\b(\s*[:=]\s*)([^\s,;.!?]+)"
)

_SENTINEL = "\x00"


_DIGIT_CLASSES = {"ТЕЛЕФОН", "КАРТА", "ИНН", "СНИЛС", "ПАСПОРТ"}


def _norm(label: str, value: str) -> str:
    """Dedup key: formats don't matter (+7 912… == +7 (912)…), the value does."""
    if label in _DIGIT_CLASSES:
        return re.sub("[^0-9]", "", value)
    return value.strip().lower()


def _mask_classes(text: str, out: "MaskResult", seen: dict[str, str]) -> str:
    for label, pattern, repl in _CLASSES:

        def sub(m, _l=label):
            original = m.group(0)
            if repl:
                probe = repl(m)
                if probe == original:
                    return original  # checksum failed → not really this class
            key = _l + "|" + _norm(_l, original)
            if key not in seen:
                idx = sum(1 for k in seen if k.split("|", 1)[0] == _l) + 1
                seen[key] = f"[{_l}_{idx}]"
                out.mapping[seen[key]] = original
            return seen[key]

        text = pattern.sub(sub, text)
    return text


def _mask_cred(text: str, out: "MaskResult", seen: dict[str, str]) -> str:
    """КРЕД-пары: сохраняем ключевое слово («пароль:»), маскируем только значение."""

    def sub(m):
        value = m.group(3)
        if value.startswith("[") and value.endswith("]"):
            return m.group(0)  # уже закрыто классом выше (EMAIL/СЕКРЕТ/…)
        key = "КРЕД|" + value.strip().lower()
        if key not in seen:
            idx = sum(1 for k in seen if k.split("|", 1)[0] == "КРЕД") + 1
            seen[key] = f"[КРЕД_{idx}]"
            out.mapping[seen[key]] = value
        return m.group(1) + m.group(2) + seen[key]

    return _CRED_RE.sub(sub, text)


def _clean(text: str) -> str:
    return text.replace(_SENTINEL, "")


# ── optional NER (natasha extra) ──────────────────────────────────────────────────────────────────
_ner = None
_ner_warned = False
_ner_last_used = 0.0


def _touch_ner() -> None:
    global _ner_last_used
    import time

    _ner_last_used = time.monotonic()


def release_ner(max_idle_seconds: float = 600.0) -> None:
    """Выгрузить natasha-модели после простоя (RAM-гигиена долгоживущих процессов:
    эмбеддинги ~600-700 МБ; VM 4 ГБ — держать их вечно нельзя, замер 09.11 — swap-storm).
    Дешёвый no-op, если модели не загружены/недавно использовались; следующий mask()
    при необходимости перезагрузит (~2-4 с)."""
    global _ner
    if _ner and _ner_last_used:
        import time

        if time.monotonic() - _ner_last_used >= max_idle_seconds:
            _ner = None


def _load_ner():
    """Lazy natasha: 1.6 (NewsNERTagger) or classic (NewsNerModel); False when unavailable."""
    global _ner
    if _ner is not None:
        _touch_ner()
        return _ner
    try:
        from natasha import NewsEmbedding, Segmenter

        emb, seg = NewsEmbedding(), Segmenter()
        try:  # natasha 1.6
            from natasha import NewsNERTagger

            _ner = ("tagger", seg, emb, NewsNERTagger(emb))
        except ImportError:  # older releases
            from natasha import NER, NewsNerModel

            _ner = ("model", seg, emb, NER(emb, NewsNerModel()))
    except Exception:
        _ner = False
    return _ner


_NER_LABELS = {"PER": "ИМЯ", "LOC": "ГЕО", "ORG": "ОРГ", "PERSON": "ИМЯ"}
_PLACEHOLDER_BODY = re.compile(r"[A-ZА-ЯЁ_]+_\d+")


def _mask_ner(text: str, out: "MaskResult", *, strict: bool = False) -> str:
    global _ner_warned
    ner = _load_ner()
    if not ner:
        if strict:
            raise RuntimeError("NER unavailable")
        if not _ner_warned:
            print(
                "privacy: natasha NER not installed — names/geo/orgs are NOT masked "
                "(pip install 'twinctx[privacy]' or natasha)",
                file=sys.stderr,
            )
            _ner_warned = True
        return text
    try:
        _, seg, emb, tagger = ner
        if ner[0] == "tagger":  # natasha 1.6: tagger(text) → markup
            markup = tagger(text)
            spans = [(s.start, s.stop, s.type) for s in markup.spans]
        else:  # classic: Doc + segment + NER(emb, model)
            from natasha import Doc

            doc = Doc(text)
            doc.segment(seg)
            tagger(doc)
            spans = [(s.start, s.stop, s.type) for s in doc.spans]
        spans = sorted({(a, b, _NER_LABELS.get(t, t)) for a, b, t in spans if t in _NER_LABELS}, reverse=True)
        for a, b, label in spans:
            original = text[a:b]
            if len(original) < 2:
                continue
            # NER работает ПОСЛЕ регэксп-классов и принимает их плейсхолдеры за организации
            # (замер на VM 09.11: natasha даёт спан «ТЕЛЕФОН_1» БЕЗ скобок → двойные скобки,
            # сломанный round-trip). Артефакт по уже-маскированному тексту: спан содержит
            # скобку, обрамлён скобками или совпадает с телом плейсхолдера — пропускаем.
            if (
                "[" in original
                or "]" in original
                or (a > 0 and b < len(text) and text[a - 1] == "[" and text[b] == "]")
                or _PLACEHOLDER_BODY.fullmatch(original)
            ):
                continue
            idx = len({k for k in out.mapping if k.startswith(f"[{label}_")}) + 1
            ph = f"[{label}_{idx}]"
            if original in out.mapping.values():
                ph = next(k for k, v in out.mapping.items() if v == original)
            else:
                out.mapping[ph] = original
            text = text[:a] + ph + text[b:]
    except Exception as e:  # NER must never break the turn
        if strict:
            raise RuntimeError("NER failed") from e
        if not _ner_warned:
            print(f"privacy: NER failed ({type(e).__name__}) — text passes with hard classes only", file=sys.stderr)
            _ner_warned = True
    return text


def _ner_cap() -> int:
    """NER only on texts shorter than this (regex classes always run full-length); CPU guard."""
    import os

    try:
        return int(os.environ.get("DLP_NER_MAX_CHARS", "32000"))
    except ValueError:
        return 32000


def mask_texts(
    texts: list, *, names: bool = True, require_ner: bool = False, fail_on_ner_limit: bool = False
) -> tuple[list, dict[str, str]]:
    """Mask a list of texts sharing ONE mapping (consistent numbering across a whole request)."""
    shared = MaskResult(text="")
    seen: dict[str, str] = {}
    ner = bool(names and _load_ner())
    if require_ner and not ner:
        raise RuntimeError("NER unavailable")
    masked = []
    for t in texts:
        if not isinstance(t, str):
            masked.append(t)
            continue
        t = _mask_classes(t, shared, seen)
        t = _mask_cred(t, shared, seen)
        if t and ner:
            if len(t) <= _ner_cap():
                t = _mask_ner(t, shared, strict=True) if require_ner else _mask_ner(t, shared)
            elif fail_on_ner_limit:
                raise RuntimeError("NER text limit exceeded")
            elif require_ner:
                print("privacy: NER skipped for oversized text; hard-class masking only", file=sys.stderr, flush=True)
        masked.append(_clean(t))
    return masked, shared.mapping


def mask_messages(messages: list) -> tuple[list, dict[str, str]]:
    """chat-completions/anthropic messages → (masked copy, mapping); non-text parts pass through."""
    texts: list[str] = []

    def _texts_of(m):
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
        elif isinstance(m, str):
            texts.append(m)

    for m in messages or []:
        _texts_of(m)
    masked, mapping = mask_texts(texts)
    it = iter(masked)

    def _apply(m):
        if isinstance(m, dict):
            m = dict(m)
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = next(it)
            elif isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        p = dict(part)
                        p["text"] = next(it)
                        parts.append(p)
                    else:
                        parts.append(part)
                m["content"] = parts
            return m
        return next(it)

    return [_apply(m) for m in messages], mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Обратная замена плейсхолдеров; longest-first, чтобы [ИМЯ_10] не съел [ИМЯ_1]."""
    for ph in sorted(mapping, key=len, reverse=True):
        if ph in text:
            text = text.replace(ph, mapping[ph])
    return text


def mask(text: str, *, names: bool = True) -> MaskResult:
    """Mask hard-PII classes (always) + names/geo/orgs (natasha extra, names=True)."""
    masked, mapping = mask_texts([text], names=names)
    return MaskResult(text=masked[0], mapping=mapping, ner_active=bool(names and _load_ner()))


def enabled_in_env() -> bool:
    import os

    return os.environ.get("TWINCTX_PII", "on") != "off"


def redactor():
    """Text→text callable for the Engine seam; None under TWINCTX_PII=off. Entry points only."""
    import os

    if os.environ.get("TWINCTX_PII", "on") == "off":
        return None
    try:
        return lambda s: mask(s).text
    except Exception:
        return None


def wrap_embedder(emb):
    """Embed-API is a model too (ADR §16, amended 07.11): passage AND query text are masked
    BEFORE they leave for vectors. Local index keeps raw text (BM25/search stay functional);
    only the outbound call is masked."""
    r = redactor()
    if emb is None or r is None:
        return emb

    class _MaskedEmbedder:
        def embed_passages(self, texts):
            return emb.embed_passages([r(t) for t in texts])

        def embed_query(self, text):
            return emb.embed_query(r(text))  # owner's query masked too — policy amended 07.11

        def __getattr__(self, name):
            return getattr(emb, name)

    return _MaskedEmbedder()
