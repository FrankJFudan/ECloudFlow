from pathlib import Path

REQUIRED = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "docs/theory.md",
    "docs/data.md",
    "docs/training.md",
    "docs/sampling.md",
    "docs/evaluation.md",
    "docs/configuration.md",
    "docs/distributed.md",
    "docs/visualization.md",
    "docs/reproducibility.md",
    "examples/python_api.py",
)
FORBIDDEN = ("TODO", "TBD", "FIXME", "implement later")


def test_required_docs_exist_and_are_substantive():
    for name in REQUIRED:
        path = Path(name)
        assert path.is_file(), f"missing documentation deliverable: {name}"
        text = path.read_text(encoding="utf-8")
        minimum = 30 if name.endswith(".md") else 12
        assert len(text.splitlines()) >= minimum, f"too little content in {name}"
        assert not any(marker.lower() in text.lower() for marker in FORBIDDEN), name


def test_markdown_has_equations_workflows_and_limitations():
    theory = Path("docs/theory.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "SE(3)" in theory and "z_t" in theory
    assert "ecloudflow sample" in readme
    assert "Limitations" in readme
