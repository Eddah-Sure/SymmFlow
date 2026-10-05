"""The model module must resolve its data/checkpoint locations from the environment, not from a baked-in path."""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "src" / "_").read_text(encoding="utf-8")


def test_data_root_comes_from_environment():
    assert re.search(r'DEFAULT_DATA_ROOT\s*=\s*os\.environ\.get\("MP20_ROOT"', SRC)


def test_no_absolute_default_paths():
    # /content/drive is Colab's fixed, public mount point (used only if the user opts in to mounting)
    hits = [m.group(0) for m in re.finditer(r"""['"]/(?:content|home|Users|mnt)/[^'"]*""", SRC)
            if not m.group(0).lstrip("'\"").startswith("/content/drive")]
    assert not hits, hits
