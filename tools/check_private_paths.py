#!/usr/bin/env python3
"""Fail if the repository contains machine-specific paths, personal identifiers or secrets.

Run before every commit (it is wired into pre-commit and CI):

    python tools/check_private_paths.py            # scan the whole repo
    python tools/check_private_paths.py FILE ...   # scan given files (pre-commit passes these)

A line can be exempted with the marker  "private-path-ok"  (use sparingly).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Patterns are assembled from pieces so that this file does not match itself.
_D = "Drive"
PATTERNS = [
    ("Google Drive path", re.compile(r"/content/g" + r"drive|My" + _D)),
    ("project folder name", re.compile(r"2024" + r"SUMMER")),
    ("macOS home directory", re.compile(r"/Users/(?!you\b|name\b|username\b)[A-Za-z0-9._-]+")),
    ("Linux home directory", re.compile(r"/home/(?!runner\b|you\b|user\b|username\b|name\b)[A-Za-z0-9._-]+")),
    ("Windows user directory", re.compile(r"[A-Za-z]:\\+Users\\+")),
    ("e-mail address", re.compile(
        r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org)\b|users\.noreply\.github\.com\b)"
        r"[A-Za-z0-9-]+\.[A-Za-z]{2,}\b")),
    ("hard-coded credential", re.compile(
        r"(?i)(api[_-]?key|token|secret|passwd|password)\s*[:=]\s*['\"](?!YOUR_|xxx|<)[A-Za-z0-9_\-]{16,}['\"]")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("API secret key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", ".venv", "venv", ".eggs"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg", ".pt", ".pth", ".npz", ".npy",
                 ".pkl", ".gz", ".zip", ".ico"}
MARKER = "private-path-ok"


def iter_files(paths):
    if paths:
        for p in paths:
            p = Path(p)
            if p.is_file():
                yield p.resolve()
        return
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
            yield p


def scan(paths=None):
    hits = []
    for f in iter_files(paths):
        if f == SELF or f.suffix.lower() in SKIP_SUFFIXES or f.name.startswith("test_hygiene"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if MARKER in line:
                continue
            for label, rx in PATTERNS:
                m = rx.search(line)
                if m:
                    try:
                        rel = f.relative_to(ROOT)
                    except ValueError:
                        rel = f
                    hits.append((str(rel), n, label, m.group(0)))
    return hits


def main(argv):
    hits = scan(argv[1:])
    if not hits:
        print("check_private_paths: clean")
        return 0
    print("check_private_paths: found machine-specific or sensitive strings:\n")
    for rel, n, label, frag in hits:
        print(f"  {rel}:{n}: {label}: {frag!r}")
    print("\nReplace them with environment variables or relative paths (see .env.example),")
    print("or add the marker 'private-path-ok' to a line that is genuinely safe.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
