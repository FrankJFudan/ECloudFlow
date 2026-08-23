"""Check Python source files for English documentation policy compliance."""

import argparse
import ast
import io
import re
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CJK = re.compile(r"[\u3400-\u9fff]")
REQUIRED_FIELDS = (":param ", ":return:", ":rtype:")


@dataclass(frozen=True)
class SemanticTopic:
    """Define one semantic documentation topic through accepted regex patterns."""

    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class APIDocContract:
    """Define substantive documentation requirements for one important API."""

    require_raises: bool
    topics: tuple[SemanticTopic, ...]
    min_words: int = 35
    forbidden_patterns: tuple[str, ...] = ()


def _topic(name: str, *patterns: str) -> SemanticTopic:
    """Build one immutable semantic-topic requirement."""
    return SemanticTopic(name=name, patterns=patterns)


_DEVICE = _topic("device", r"\bcpu\b", r"\bdevices?\b", r"\baccelerator\b")
_DTYPE = _topic("dtype", r"\bdtype\b")
_FRAME = _topic("frame", r"\bframes?\b")
_MASK = _topic("mask", r"\bmask(?:s)?\b")
_MUTATION = _topic(
    "mutation",
    r"\bmutat",
    r"\bimmutable\b",
    r"read-only",
    r"does not alter",
    r"never modif",
)
_DETERMINISM = _topic(
    "determinism", r"determin", r"\bstable\b", r"byte-sorted", r"exact(?:ly)? once"
)
_RESUME = _topic("resume", r"resum", r"recover", r"checkpoint", r"\bready\b")
_FAILURE = _topic("failure", r"fail", r"invalid", r"reject", r"error")
_POWER_LOSS = _topic(
    "power-loss",
    r"power.loss",
    r"directory.*fsync",
    r"flushfilebuffers",
    r"write-through",
)
_DISTRIBUTED = _topic("distributed", r"\brank\b", r"\bworker\b", r"distributed")
_SHAPE = _topic("shape", r"\[n,\s*3\]", r"\bshape\b")
_UNITS = _topic("units", r"angstrom", r"\bunits\b")
_CACHE = _topic("cache", r"\bcache\b", r"sha-256", r"content-addressed")
_ALIGNMENT = _topic("alignment", r"global alignment", r"affine-gap", r"gap column")
_CHECKPOINT = _topic("checkpoint", r"checkpoint", r"manifest hash", r"restore")
_GRADIENT = _topic("gradient", r"gradient", r"autograd", r"differentiab")
_IRREP = _topic("irrep", r"irrep", r"0e", r"1o", r"spherical-harmonic")
_CHUNK = _topic("chunk", r"chunk")
_LATENT_LAYOUT = _topic(
    "latent-layout", r"19x0e\s*\+\s*8x1o\s*\+\s*1x2e.*every configured"
)
_PLACEMENT_RETENTION_VERB = (
    r"(?:preserv\w*|retain\w*|keep\w*|kept|maintain\w*|remain\w*|stay\w*)"
)
_ORIGINAL_DEVICE_PLACEMENT = (
    r"(?:(?:(?:their|its|the)\s+)?(?:original|input|source)\s+"
    r"(?:accelerator|gpu|cuda|device)(?:\s+placement)?|"
    r"(?:accelerator|gpu|cuda|device)\s+placement)"
)
_FALSE_STORAGE_DEVICE_CLAIMS = (
    (
        rf"\b{_PLACEMENT_RETENTION_VERB}\b"
        rf"(?:\s+[A-Za-z-]+){{0,6}}\s+{_ORIGINAL_DEVICE_PLACEMENT}\b"
    ),
    (
        rf"\b{_ORIGINAL_DEVICE_PLACEMENT}\b"
        rf"(?:\s+[A-Za-z-]+){{0,6}}\s+{_PLACEMENT_RETENTION_VERB}\b"
    ),
    r"preserv\w*.{0,80}(?:original\s+)?device",
    r"retain\w*.{0,80}\bdevice\b",
    r"never.{0,80}(?:mov\w*|transfer\w*).{0,40}(?:cpu|device)",
    r"does not.{0,80}(?:move|transfer).{0,40}(?:tensor|device|cpu)",
)

API_DOC_CONTRACTS: dict[str, APIDocContract] = {
    "tokenizer.EquivariantFieldTokenizer.forward": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
            _GRADIENT,
            _IRREP,
            _LATENT_LAYOUT,
        ),
        min_words=100,
    ),
    "tokenizer.EquivariantFieldTokenizer.encode": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
            _GRADIENT,
            _IRREP,
            _LATENT_LAYOUT,
        ),
        min_words=55,
    ),
    "tokenizer.EquivariantFieldTokenizer.decode": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
            _GRADIENT,
            _IRREP,
            _LATENT_LAYOUT,
            _CHUNK,
        ),
        min_words=70,
    ),
    "decoder.ElectronFieldDecoder.forward": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
            _GRADIENT,
            _IRREP,
            _LATENT_LAYOUT,
            _CHUNK,
        ),
        min_words=110,
    ),
    "decoder.ElectronFieldDecoder.decode": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
            _GRADIENT,
            _IRREP,
            _LATENT_LAYOUT,
            _CHUNK,
        ),
        min_words=60,
    ),
    "tokenizer.EquivariantFieldTokenizer.latent_irreps": APIDocContract(
        False, (_IRREP, _LATENT_LAYOUT), min_words=40
    ),
    "decoder.ElectronFieldDecoder.latent_irreps": APIDocContract(
        False, (_IRREP, _LATENT_LAYOUT), min_words=40
    ),
    "durability.sync_file": APIDocContract(True, (_MUTATION, _FAILURE, _POWER_LOSS)),
    "durability.flush_directory": APIDocContract(
        True, (_MUTATION, _FAILURE, _POWER_LOSS)
    ),
    "durability.durable_replace": APIDocContract(
        True, (_MUTATION, _FAILURE, _POWER_LOSS)
    ),
    "durability.durable_unlink": APIDocContract(
        True, (_MUTATION, _FAILURE, _POWER_LOSS)
    ),
    "durability.durable_mkdir": APIDocContract(
        True, (_MUTATION, _FAILURE, _POWER_LOSS)
    ),
    "types.QMProvenance.__reduce__": APIDocContract(
        True, (_DEVICE, _MUTATION, _FAILURE)
    ),
    "types.SampleProvenance.__reduce__": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _FRAME, _UNITS, _MUTATION, _FAILURE)
    ),
    "types.ComplexSample.__reduce__": APIDocContract(
        True,
        (_DEVICE, _DTYPE, _SHAPE, _FRAME, _UNITS, _MASK, _MUTATION, _FAILURE),
    ),
    "splits.build_grouped_split": APIDocContract(
        True, (_DETERMINISM, _ALIGNMENT, _MUTATION, _FAILURE)
    ),
    "splits._global_alignment_identity": APIDocContract(
        True, (_DETERMINISM, _ALIGNMENT, _MUTATION, _FAILURE)
    ),
    "manifest.DatasetManifest.write": APIDocContract(
        True, (_MUTATION, _FAILURE, _POWER_LOSS)
    ),
    "manifest.DatasetManifest.read": APIDocContract(
        True, (_DETERMINISM, _MUTATION, _FAILURE)
    ),
    "shards.ShardWriter.write": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _RESUME,
            _FAILURE,
            _POWER_LOSS,
        ),
        min_words=90,
        forbidden_patterns=_FALSE_STORAGE_DEVICE_CLAIMS,
    ),
    "shards._recover_ready_generation": APIDocContract(
        True, (_DEVICE, _MUTATION, _RESUME, _FAILURE, _POWER_LOSS)
    ),
    "shards.stream_samples": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _DISTRIBUTED,
            _CACHE,
            _FAILURE,
        ),
        forbidden_patterns=_FALSE_STORAGE_DEVICE_CLAIMS,
    ),
    "shards._resolve_cached_shard": APIDocContract(
        True, (_DEVICE, _MUTATION, _DETERMINISM, _CACHE, _FAILURE)
    ),
    "shards.sample_ids_for_partition": APIDocContract(
        True, (_DEVICE, _MUTATION, _DETERMINISM, _DISTRIBUTED, _FAILURE)
    ),
    "shards.bucketed_batches": APIDocContract(
        True,
        (_DEVICE, _DTYPE, _FRAME, _UNITS, _MASK, _MUTATION, _DETERMINISM, _FAILURE),
    ),
    "datamodule._ShardBatchDataset.__iter__": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _DISTRIBUTED,
            _FAILURE,
        ),
    ),
    "datamodule.ECloudDataModule.setup": APIDocContract(
        True, (_DEVICE, _MUTATION, _CHECKPOINT, _FAILURE)
    ),
    "datamodule.ECloudDataModule.set_epoch": APIDocContract(
        True, (_DEVICE, _MUTATION, _DETERMINISM, _DISTRIBUTED, _FAILURE)
    ),
    "datamodule.ECloudDataModule.state_dict": APIDocContract(
        False, (_DEVICE, _MUTATION, _DETERMINISM, _CHECKPOINT)
    ),
    "datamodule.ECloudDataModule.load_state_dict": APIDocContract(
        True, (_DEVICE, _MUTATION, _DETERMINISM, _CHECKPOINT, _FAILURE)
    ),
    "diffgui_lmdb.DiffGuiLMDBImporter.from_config": APIDocContract(
        True, (_DEVICE, _MUTATION, _DETERMINISM, _FAILURE)
    ),
    "diffgui_lmdb.DiffGuiLMDBImporter.__iter__": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
        ),
    ),
    "diffgui_lmdb.DiffGuiLMDBImporter.iter_samples": APIDocContract(
        True,
        (
            _DEVICE,
            _DTYPE,
            _SHAPE,
            _FRAME,
            _UNITS,
            _MASK,
            _MUTATION,
            _DETERMINISM,
            _FAILURE,
        ),
    ),
    "continuous.ContinuousPath.sample": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "continuous.ContinuousPath.targets": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _FAILURE, _GRADIENT)
    ),
    "continuous.ContinuousPath.velocity_loss": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MASK, _MUTATION, _FAILURE, _GRADIENT)
    ),
    "continuous.ContinuousPath.sample_times": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "categorical.CategoricalPath.sample": APIDocContract(
        True,
        (_DEVICE, _DTYPE, _SHAPE, _MASK, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT),
    ),
    "categorical.CategoricalPath.endpoint_loss": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MASK, _MUTATION, _FAILURE, _GRADIENT)
    ),
    "schedules.InterpolantSchedule.data_weight": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "schedules.InterpolantSchedule.data_weight_derivative": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "schedules.InterpolantSchedule.noise_scale": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "schedules.InterpolantSchedule.noise_scale_derivative": APIDocContract(
        True, (_DEVICE, _DTYPE, _SHAPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
    "continuous._normalize_shape": APIDocContract(
        True, (_DEVICE, _DTYPE, _MUTATION, _DETERMINISM, _FAILURE, _GRADIENT)
    ),
}
DESIGNATED_APIS = frozenset(API_DOC_CONTRACTS)


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
    found_designated: set[str] = set()
    inspected_modules: dict[str, Path] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and CJK.search(token.string):
                errors.append(f"{path}:{token.start[0]} English-only comments required")
        tree = ast.parse(text, filename=str(path))
        module = path.stem
        inspected_modules[module] = path
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
                    found_designated.add(name)
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
                    contract = API_DOC_CONTRACTS.get(
                        name,
                        APIDocContract(require_raises=False, topics=(), min_words=20),
                    )
                    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", doc))
                    if word_count < contract.min_words:
                        errors.append(
                            f"{path}:{node.lineno} substantive documentation required "
                            f"({word_count} < {contract.min_words} words)"
                        )
                    if contract.require_raises and not re.search(
                        r"(?m)^\s*:raises\s+[^:]+:", doc
                    ):
                        errors.append(f"{path}:{node.lineno} missing :raises semantics")
                    for topic in contract.topics:
                        if not any(
                            re.search(pattern, doc, flags=re.IGNORECASE | re.DOTALL)
                            for pattern in topic.patterns
                        ):
                            errors.append(
                                f"{path}:{node.lineno} missing semantic topic {topic.name}"
                            )
                    if any(
                        re.search(pattern, doc, flags=re.IGNORECASE | re.DOTALL)
                        for pattern in contract.forbidden_patterns
                    ):
                        errors.append(
                            f"{path}:{node.lineno} false canonical CPU/device claim"
                        )
    for name in sorted(selected_designated - found_designated):
        module = name.split(".", 1)[0]
        if module in inspected_modules:
            errors.append(
                f"{inspected_modules[module]}: designated API not found: {name}"
            )
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
