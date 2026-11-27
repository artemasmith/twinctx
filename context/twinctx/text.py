"""Tokenization shared by indexing and querying.

Russian is inflected and FTS5 has no Russian stemmer, so terms are reduced to a crude stem
(strip 1–3 trailing letters) and queried as prefixes: «задачи/задачу/задачам» → «задач*».
Vectors (when enabled) cover what stemming misses.
"""
import re

WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё_][0-9A-Za-zА-Яа-яЁё_\-]{1,}")
CYR = re.compile(r"[А-Яа-яЁё]")
STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот
от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять
уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после
над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между это как что где кто
the a an and or of to in on for with is are was were be been it this that at by from as not no do does
""".split())
ENDINGS = sorted("""
ами ями ого его ому ему ыми ими ией ях ах ам ям ом ем ой ей ую юю ая яя ое ее ые ие ый ий ов ев ов
ть ла ло ли ет ют ат ят ит ешь ишь ем им ете ите ся сь а я о е ы и у ю ь й
ing ed es s
""".split(), key=len, reverse=True)


def words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in WORD.findall(text)]


def stem(w: str) -> str:
    if len(w) <= 4:
        return w
    for e in ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 4:
            return w[: -len(e)]
    return w


def query_terms(q: str, limit: int = 12) -> list[str]:
    out, seen = [], set()
    for w in words(q):
        if w in STOP or len(w) < 3:
            continue
        s = stem(w)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:limit]


def fts_query(terms: list[str]) -> str:
    """OR of prefix terms; quotes keep FTS5 syntax characters (-, :) literal."""
    return " OR ".join(f'"{t}"*' for t in terms)


def retrieval_terms(query: str) -> list[str]:
    """Keep the question while removing a trailing request about answer formatting."""
    question, separator, tail = query.partition("?")
    if separator and re.fullmatch(
        r"\s*(?:(?:пожалуйста[, ]+)?ответь(?:\s+(?:одной\s+фразой|коротко|кратко|по\s+нашему\s+рабочему\s+контексту))+)\s*[.!]?\s*",
        tail,
        re.IGNORECASE,
    ):
        return query_terms(question)
    return query_terms(query)


def quoted_subject_terms(query: str) -> list[str]:
    """Return one short explicitly named subject if the question specifies it."""
    subjects = re.findall(r'«([^«»\n]{1,100})»|"([^"\n]{1,100})"', query)
    if len(subjects) != 1:
        return []
    terms = query_terms(subjects[0][0] or subjects[0][1])
    return terms if 2 <= len(terms) <= 5 else []


def est_tokens(text: str) -> int:
    """Rough: Cyrillic ≈ 2.6 chars/token, Latin/code ≈ 3.8 (dev standard: Russian is costlier)."""
    cyr = len(CYR.findall(text))
    return int(cyr / 2.6 + (len(text) - cyr) / 3.8) + 1
