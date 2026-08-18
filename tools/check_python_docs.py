"""Check Python source files for English documentation policy compliance."""

import argparse
import ast
import io
import re
import tokenize
from collections.abc import Iterable, Sequence
from pathlib import Path

CJK = re.compile(r"[\u3400-\u9fff]")
REQUIRED_FIELDS = (":param ", ":return:", ":rtype:")


def check_paths(
    paths: Iterable[Path], designated: set[str] | None = None
) -> list[str]:
    """Validate English-only comments and designated API docstrings.

    :param paths: Python source files to inspect.
    :param designated: Fully qualified ``module.function`` names requiring
        detailed Sphinx fields.
    :return: Human-readable policy violations.
    :rtype: list[str]
    """
    designated = designated or set()
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and CJK.search(token.string):
                errors.append(
                    f"{path}:{token.start[0]} English-only comments required"
                )
        tree = ast.parse(text, filename=str(path))
        module = path.stem
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{module}.{node.name}"
                doc = ast.get_docstring(node) or ""
                if CJK.search(doc):
                    errors.append(
                        f"{path}:{node.lineno} English-only docstrings required"
                    )
                if name in designated:
                    parameter_names = [argument.arg for argument in node.args.args]
                    for parameter_name in parameter_names:
                        field = f":param {parameter_name}:"
                        if field not in doc:
                            errors.append(f"{path}:{node.lineno} missing {field}")
                    for field in REQUIRED_FIELDS[1:]:
                        if field not in doc:
                            errors.append(f"{path}:{node.lineno} missing {field}")
    return errors


def _expand_paths(paths: Sequence[Path]) -> list[Path]:
    """Expand Python files and directories into deterministic source paths.

    :param paths: Files or directories supplied on the command line.
    :return: Sorted Python source files beneath the supplied paths.
    :rtype: list[pathlib.Path]
    """
    expanded: set[Path] = set()
    for path in paths:
        if path.is_dir():
            expanded.update(candidate for candidate in path.rglob("*.py") if candidate.is_file())
        elif path.is_file() and path.suffix == ".py":
            expanded.add(path)
    return sorted(expanded)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the documentation-policy checker as a command-line program.

    :param argv: Optional command-line arguments excluding the program name.
    :return: Zero when all inspected sources conform, otherwise one.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Check Python comments and designated API docstrings."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    errors = check_paths(_expand_paths(arguments.paths))
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
