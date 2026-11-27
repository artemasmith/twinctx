"""File masking CLI: sanitized copies for sharing/teaching material/model input.

  python -m privacy mask FILE [FILE...]           # → FILE.masked рядом с исходником
  python -m privacy mask --ext md,txt,log DIR     # рекурсивно по расширениям

Placeholder mapping is NOT written anywhere (mapping stays in-process); the summary prints
class counts only. Binary files are skipped; originals are never modified.
"""
import argparse
import sys
from pathlib import Path

from . import mask


def mask_file(path: Path) -> tuple[int, dict[str, int]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return -1, {}
    res = mask(text)
    out = path.with_name(path.name + ".masked")
    out.write_text(res.text, encoding="utf-8")
    counts: dict[str, int] = {}
    for ph in res.mapping:
        label = ph[1:ph.rindex("_")]
        counts[label] = counts.get(label, 0) + 1
    return len(res.mapping), counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privacy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mask")
    m.add_argument("paths", nargs="+")
    m.add_argument("--ext", default="", help="для каталогов: расширения через запятую (md,txt,log)")
    a = ap.parse_args(argv)
    files: list[Path] = []
    for p in map(Path, a.paths):
        if p.is_dir():
            exts = {f".{e.strip().lstrip('.')}" for e in a.ext.split(",") if e.strip()} or {".txt", ".md", ".log"}
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix in exts and not f.name.endswith(".masked")]
        elif p.is_file():
            files.append(p)
    total = 0
    for f in files:
        n, counts = mask_file(f)
        if n < 0:
            print(f"SKIP (binary): {f}")
            continue
        total += n
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "чисто"
        print(f"{f} → {f.name}.masked  [{detail}]")
    print(f"итого: {total} плейсхолдеров в {len(files)} файлах")
    return 0


if __name__ == "__main__":
    sys.exit(main())
