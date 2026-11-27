"""Жёсткие границы аргументов MCP-инструментов: один вызов агента не должен ронять stdio-сервер по OOM и раздувать ответ (аудит 19.11.2026)."""

MAX_BUDGET_TOKENS = 32768  # context_assemble: бюджет packed-ответа; выше — раздутие контекста вызванного агента
MAX_READ_CHARS = 65536  # context_read: max_chars=10**9 выгружает весь индекс одним вызовом — OOM stdio-сервера
MAX_K = 100  # context_search / context_related: число возвращаемых указателей
MAX_MEMORY_CHARS = 64000  # context_remember: факт памяти, а не документ; длинное — файлом через документ-канал


def _check_int(name: str, value: int, low: int, high: int) -> None:
    if not isinstance(value, int) or not (low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}, got {value!r}")


def check_budget_tokens(value: int) -> None:
    _check_int("budget_tokens", value, 1, MAX_BUDGET_TOKENS)


def check_k(value: int) -> None:
    _check_int("k", value, 1, MAX_K)


def check_read_limits(max_chars: int, start_line: int = 0, end_line: int = 0) -> None:
    _check_int("max_chars", max_chars, 1, MAX_READ_CHARS)
    for name, line in (("start_line", start_line), ("end_line", end_line)):
        if not isinstance(line, int) or line < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {line!r}")
    if start_line > 0 and end_line > 0 and start_line > end_line:
        raise ValueError(f"start_line must be <= end_line, got {start_line} > {end_line}")


def check_memory_text(text: str) -> None:
    if len(text) > MAX_MEMORY_CHARS:
        raise ValueError(
            f"memory text too long: {len(text)} chars (max {MAX_MEMORY_CHARS}); "
            "save it as a document/file instead of a memory fact"
        )
