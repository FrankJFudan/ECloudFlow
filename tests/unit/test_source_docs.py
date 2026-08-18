"""Tests for the Python source documentation policy checker."""

from pathlib import Path

from tools.check_python_docs import check_paths


def test_checker_rejects_cjk_comment_and_incomplete_core_docstring(tmp_path: Path):
    source = tmp_path / "bad.py"
    source.write_text(
        '# \u4e2d\u6587\u6ce8\u91ca\n'
        'def core_step(x):\n'
        '    """Return x."""\n'
        '    return x\n',
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"bad.core_step"})
    assert any("English-only" in error for error in errors)
    assert any(":param x:" in error for error in errors)
