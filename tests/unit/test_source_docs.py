"""Tests for the Python source documentation policy checker."""

from pathlib import Path

from tools.check_python_docs import DESIGNATED_APIS, check_paths


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
        root / "src/ecloudflow/core/types.py",
        root / "src/ecloudflow/data/splits.py",
        root / "src/ecloudflow/data/manifest.py",
        root / "src/ecloudflow/data/shards.py",
        root / "src/ecloudflow/data/datamodule.py",
        root / "src/ecloudflow/data/diffgui_lmdb.py",
    ]
    assert check_paths(paths, designated=required) == []
