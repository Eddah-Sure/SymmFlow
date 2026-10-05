"""Repository hygiene: no private paths or credentials, and every script at least compiles."""
import importlib.util
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_private_paths", ROOT / "tools" / "check_private_paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_private_paths_or_secrets():
    hits = _load_checker().scan()
    assert not hits, "\n".join(f"{r}:{n}: {label}: {frag}" for r, n, label, frag in hits)


def test_all_python_files_compile():
    for sub in ("src", "scripts", "tools", "tests"):
        for f in (ROOT / sub).rglob("*.py"):
            py_compile.compile(str(f), doraise=True)


def test_checker_detects_a_planted_path(tmp_path):
    mod = _load_checker()
    bad = tmp_path / "x.py"
    bad.write_text('P = "/content/g' + 'drive/My' + 'Drive/data"\n', encoding="utf-8")
    assert mod.scan([str(bad)])
