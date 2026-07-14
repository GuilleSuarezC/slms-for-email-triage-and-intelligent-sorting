"""Smoke test: every Python file under src/ and scripts/ byte-compiles.

This catches syntax errors without importing the heavy ML dependencies
(scikit-learn, torch, transformers, ...), so it runs in a bare environment.

Runnable directly (`python tests/test_imports.py`) or via pytest.
"""
import py_compile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["src", "scripts"]


def iter_python_files():
    for d in SOURCE_DIRS:
        yield from (REPO_ROOT / d).rglob("*.py")


def test_all_sources_compile():
    failures = []
    checked = 0
    for path in iter_python_files():
        checked += 1
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - only on error
            failures.append(f"{path.relative_to(REPO_ROOT)}: {exc.msg}")
    assert checked > 0, "no Python sources found to compile"
    assert not failures, "byte-compile failures:\n" + "\n".join(failures)
    return checked


if __name__ == "__main__":
    n = test_all_sources_compile()
    print(f"OK  test_imports: {n} Python source files byte-compile cleanly")
