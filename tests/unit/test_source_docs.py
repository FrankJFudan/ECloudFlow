"""Tests for the Python source documentation policy checker."""

import inspect
from pathlib import Path

import pytest

from ecloudflow.process import ContinuousPath
from tools.check_python_docs import API_DOC_CONTRACTS, DESIGNATED_APIS, check_paths


def test_checker_rejects_cjk_comment_and_incomplete_core_docstring(tmp_path: Path):
    source = tmp_path / "bad.py"
    source.write_text(
        "# \u4e2d\u6587\u6ce8\u91ca\n"
        "def core_step(x):\n"
        '    """Return x."""\n'
        "    return x\n",
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"bad.core_step"})
    assert any("English-only" in error for error in errors)
    assert any(":param x:" in error for error in errors)


def test_task7_designated_apis_keep_detailed_sphinx_contracts() -> None:
    """Task 7 scientific and distributed public contracts stay documented."""
    required = {
        "durability.sync_file",
        "durability.flush_directory",
        "durability.durable_replace",
        "durability.durable_unlink",
        "durability.durable_mkdir",
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
    assert required <= DESIGNATED_APIS
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/ecloudflow/data/durability.py",
        root / "src/ecloudflow/core/types.py",
        root / "src/ecloudflow/data/splits.py",
        root / "src/ecloudflow/data/manifest.py",
        root / "src/ecloudflow/data/shards.py",
        root / "src/ecloudflow/data/datamodule.py",
        root / "src/ecloudflow/data/diffgui_lmdb.py",
    ]
    assert check_paths(paths, designated=required) == []


def test_task8_designated_apis_keep_scientific_tensor_contracts() -> None:
    """Task 8 encoding and decoding APIs retain complete semantic contracts."""
    required = {
        "tokenizer.EquivariantFieldTokenizer.forward",
        "tokenizer.EquivariantFieldTokenizer.encode",
        "tokenizer.EquivariantFieldTokenizer.decode",
        "tokenizer.EquivariantFieldTokenizer.latent_irreps",
        "decoder.ElectronFieldDecoder.forward",
        "decoder.ElectronFieldDecoder.decode",
        "decoder.ElectronFieldDecoder.latent_irreps",
    }
    assert required <= DESIGNATED_APIS
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/ecloudflow/ecloud/tokenizer.py",
        root / "src/ecloudflow/ecloud/decoder.py",
    ]
    assert check_paths(paths, designated=required) == []


def test_task9_designated_apis_keep_stochastic_path_contracts() -> None:
    """Task 9 paths retain endpoint, mask, and gradient API semantics."""
    required = {
        "continuous.ContinuousPath.sample",
        "continuous.ContinuousPath.targets",
        "continuous.ContinuousPath.velocity_loss",
        "continuous.ContinuousPath.sample_times",
        "categorical.CategoricalPath.sample",
        "categorical.CategoricalPath.endpoint_loss",
        "schedules.InterpolantSchedule.data_weight",
        "schedules.InterpolantSchedule.data_weight_derivative",
        "schedules.InterpolantSchedule.noise_scale",
        "schedules.InterpolantSchedule.noise_scale_derivative",
        "continuous._normalize_shape",
    }
    assert required <= DESIGNATED_APIS
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/ecloudflow/process/continuous.py",
        root / "src/ecloudflow/process/categorical.py",
        root / "src/ecloudflow/process/schedules.py",
    ]
    assert check_paths(paths, designated=required) == []
    assert ":raises TypeError:" in inspect.getdoc(ContinuousPath.sample_times)


def test_task10_designated_apis_keep_equivariance_and_cache_contracts() -> None:
    """Task 10 joint model APIs retain tensor and distributed semantics."""
    required = {
        "pocket_encoder.PocketEncoding.__post_init__",
        "pocket_encoder.PocketEncoder.encode",
        "count_predictor.AtomCountPredictor.forward",
        "ecloudflow.ECloudFlowModel.encode_pocket",
        "ecloudflow.ECloudFlowModel.forward",
        "ecloudflow.ModelPrediction.__post_init__",
    }
    assert required <= DESIGNATED_APIS
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/ecloudflow/models/pocket_encoder.py",
        root / "src/ecloudflow/models/count_predictor.py",
        root / "src/ecloudflow/models/ecloudflow.py",
    ]
    assert check_paths(paths, designated=required) == []


def test_checker_rejects_missing_designated_api(tmp_path: Path) -> None:
    """A registry entry must resolve to a real module-qualified callable."""
    source = tmp_path / "missing.py"
    source.write_text('"""No designated function exists."""\n', encoding="utf-8")
    errors = check_paths([source], designated={"missing.core_step"})
    assert any("designated API not found" in error for error in errors)


def test_checker_rejects_placeholder_and_missing_semantic_contracts(
    tmp_path: Path,
) -> None:
    """Formal fields alone cannot satisfy a scientific API contract."""
    source = tmp_path / "shards.py"
    source.write_text(
        "class ShardWriter:\n"
        "    def write(self, samples, output_dir, *, split=None):\n"
        '        """Placeholder.\n\n'
        "        :param samples: Value.\n"
        "        :param output_dir: Value.\n"
        "        :param split: Value.\n"
        "        :return: Value.\n"
        "        :rtype: object\n"
        '        """\n'
        "        return object()\n",
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"shards.ShardWriter.write"})
    assert any("substantive documentation" in error for error in errors)
    assert any("missing :raises" in error for error in errors)
    for topic in ("device", "dtype", "mutation", "determinism", "resume", "failure"):
        assert any(f"missing semantic topic {topic}" in error for error in errors)


@pytest.mark.parametrize(
    "claim",
    [
        "Tensors preserve their original device placement.",
        "The reader retains tensors on their input GPU.",
        "Serialization keeps tensors on their original accelerator.",
        "Original CUDA placement is retained by the stored payload.",
        "The tensors remain on the source device.",
    ],
)
def test_checker_rejects_false_canonical_device_claim(
    tmp_path: Path, claim: str
) -> None:
    """Shard persistence may not claim to preserve accelerator placement."""
    assert API_DOC_CONTRACTS["shards.ShardWriter.write"].forbidden_patterns
    assert API_DOC_CONTRACTS["shards.stream_samples"].forbidden_patterns
    body = (
        "CPU dtype shape frame mask mutation deterministic resume failure power loss "
        "validation checkpoint cache worker publication recovery angstrom metadata "
    )
    source = tmp_path / "shards.py"
    source.write_text(
        "class ShardWriter:\n"
        "    def write(self, samples, output_dir, *, split=None):\n"
        '        """Persist deterministic data.\n\n'
        "        :param samples: Canonical samples.\n"
        "        :param output_dir: Dataset path.\n"
        "        :param split: Optional split.\n"
        "        :return: Manifest.\n"
        "        :rtype: object\n"
        "        :raises RuntimeError: On failure.\n\n"
        f"        {body} {claim} "
        "This detailed placeholder describes validation and publication behavior repeatedly.\n"
        '        """\n'
        "        return object()\n",
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"shards.ShardWriter.write"})
    assert any("false canonical CPU/device claim" in error for error in errors)


def test_checker_allows_explicit_caller_accelerator_transfer(tmp_path: Path) -> None:
    """A correct later-transfer instruction is not a placement-preservation claim."""
    body = (
        "CPU dtype shape frame mask mutation deterministic resume failure power loss "
        "validation checkpoint cache worker publication recovery angstrom metadata "
    )
    source = tmp_path / "shards.py"
    source.write_text(
        "class ShardWriter:\n"
        "    def write(self, samples, output_dir, *, split=None):\n"
        '        """Persist deterministic canonical CPU data.\n\n'
        "        :param samples: Canonical samples.\n"
        "        :param output_dir: Dataset path.\n"
        "        :param split: Optional split.\n"
        "        :return: Manifest.\n"
        "        :rtype: object\n"
        "        :raises RuntimeError: On failure.\n\n"
        f"        {body} Stored CPU copies preserve dtype and shape; the caller "
        "explicitly transfers reconstructed batches to an accelerator for training. "
        "This detailed contract describes validation and publication behavior.\n"
        '        """\n'
        "        return object()\n",
        encoding="utf-8",
    )
    errors = check_paths([source], designated={"shards.ShardWriter.write"})
    assert not any("false canonical CPU/device claim" in error for error in errors)
