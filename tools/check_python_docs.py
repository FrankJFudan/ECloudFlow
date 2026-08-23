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
DESIGNATED_APIS = frozenset(
    {
        "types.QMProvenance.__reduce__",
        "types.SampleProvenance.__reduce__",
        "types.ComplexSample.__reduce__",
        "splits.build_grouped_split",
        "splits._global_alignment_identity",
        "manifest.DatasetManifest.write",
        "manifest.DatasetManifest.read",
        "shards.ShardWriter.write",
        "shards._recover_ready_generation",
        "shards.stream_samples",
        "shards._resolve_cached_shard",
        "shards.sample_ids_for_partition",
        "shards.bucketed_batches",
        "datamodule._ShardBatchDataset.__iter__",
        "datamodule.ECloudDataModule.setup",
        "datamodule.ECloudDataModule.set_epoch",
        "datamodule.ECloudDataModule.state_dict",
        "datamodule.ECloudDataModule.load_state_dict",
        "diffgui_lmdb.DiffGuiLMDBImporter.from_config",
        "diffgui_lmdb.DiffGuiLMDBImporter.__iter__",
        "diffgui_lmdb.DiffGuiLMDBImporter.iter_samples",
    }
)


def check_paths(paths: Iterable[Path], designated: set[str] | None = None) -> list[str]:
    """Validate English-only comments and designated API docstrings.

    :param paths: Python source files to inspect.
    :param designated: Fully qualified ``module.function`` names requiring
        detailed Sphinx fields.
    :return: Human-readable policy violations.
    :rtype: list[str]
    """
    selected_designated = (
        DESIGNATED_APIS if designated is None else frozenset(designated)
    )
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and CJK.search(token.string):
                errors.append(f"{path}:{token.start[0]} English-only comments required")
        tree = ast.parse(text, filename=str(path))
        module = path.stem
        qualified_names = _qualified_function_names(tree, module)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = qualified_names[id(node)]
                doc = ast.get_docstring(node) or ""
                if CJK.search(doc):
                    errors.append(
                        f"{path}:{node.lineno} English-only docstrings required"
                    )
                if name in selected_designated:
                    parameter_names = [
                        argument.arg
                        for argument in (
                            *node.args.posonlyargs,
                            *node.args.args,
                            *node.args.kwonlyargs,
                        )
                        if argument.arg not in {"self", "cls"}
                    ]
                    if node.args.vararg is not None:
                        parameter_names.append(node.args.vararg.arg)
                    if node.args.kwarg is not None:
                        parameter_names.append(node.args.kwarg.arg)
                    for parameter_name in parameter_names:
                        field = f":param {parameter_name}:"
                        if field not in doc:
                            errors.append(f"{path}:{node.lineno} missing {field}")
                    for field in REQUIRED_FIELDS[1:]:
                        if field not in doc:
                            errors.append(f"{path}:{node.lineno} missing {field}")
    return errors


def _qualified_function_names(tree: ast.AST, module: str) -> dict[int, str]:
    """Map function nodes to stable module/class-qualified API names.

    :param tree: Parsed Python abstract syntax tree.
    :param module: Source module stem used by the documentation policy.
    :return: Function-node identity mapped to dotted qualified name.
    :rtype: dict[int, str]

    Nested classes and functions retain their lexical path, preventing methods
    such as multiple ``__reduce__`` implementations from colliding.
    """
    names: dict[int, str] = {}

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        next_parents = parents
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            next_parents = (*parents, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names[id(node)] = ".".join((module, *next_parents))
        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree, ())
    return names


def _expand_paths(paths: Sequence[Path]) -> list[Path]:
    """Expand Python files and directories into deterministic source paths.

    :param paths: Files or directories supplied on the command line.
    :return: Sorted Python source files beneath the supplied paths.
    :rtype: list[pathlib.Path]
    """
    expanded: set[Path] = set()
    for path in paths:
        if path.is_dir():
            expanded.update(
                candidate for candidate in path.rglob("*.py") if candidate.is_file()
            )
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
