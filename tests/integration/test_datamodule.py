"""Integration tests for DataModule and read-only DiffGui ingestion."""

import hashlib
import pickle
from dataclasses import replace
from pathlib import Path

import lmdb
import pytest
from pydantic import ValidationError

from ecloudflow.config.schema import DataConfig
from ecloudflow.data.datamodule import ECloudDataModule
from ecloudflow.data.diffgui_lmdb import DiffGuiLMDBImporter
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.data.shards import ShardWriter
from ecloudflow.data.splits import GroupedSplit


def _template():
    """Return one graph-only canonical sample for integration fixtures."""
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures"
    return build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="template",
        build_fields=False,
    )


def test_datamodule_streams_bucketed_batches(tmp_path: Path) -> None:
    """The configured loader yields every local sample in node-count buckets."""
    template = _template()
    samples = [replace(template, source_id=f"dm-{index}") for index in range(5)]
    ShardWriter(max_samples_per_shard=2).write(samples, tmp_path)
    config = DataConfig(
        shard_dir=str(tmp_path),
        batch_size=2,
        num_workers=0,
        shuffle_buffer=1,
        bucket_width=8,
    )
    module = ECloudDataModule(config)
    module.setup("fit")
    batches = list(module.train_dataloader())
    assert sorted(sample.source_id for batch in batches for sample in batch) == [
        f"dm-{index}" for index in range(5)
    ]
    assert all(1 <= len(batch) <= 2 for batch in batches)


def test_datamodule_filters_production_samples_from_split_metadata(
    tmp_path: Path,
) -> None:
    """Entity audit lookups never replace sample-level loader filtering."""
    template = _template()
    samples = [replace(template, source_id=f"split-{index}") for index in range(3)]
    split = GroupedSplit(
        sample_partitions={"split-0": "train", "split-1": "val", "split-2": "test"},
        entity_partitions={"protein-0": "train"},
        sample_groups={
            sample.source_id: f"group-{index}" for index, sample in enumerate(samples)
        },
        entity_groups={"protein-0": "group-0"},
        hash="sha256:" + "0" * 64,
    )
    ShardWriter().write(samples, tmp_path, split=split)
    config = DataConfig(
        shard_dir=str(tmp_path),
        partition="train",
        batch_size=4,
        num_workers=0,
        shuffle_buffer=0,
    )
    module = ECloudDataModule(config)
    observed = [
        sample.source_id for batch in module.train_dataloader() for sample in batch
    ]
    assert observed == ["split-0"]
    assert module.manifest is not None
    assert module.manifest.entity_partitions["protein-0"] == "train"
    with pytest.raises(ValidationError):
        config.batch_size = 99  # type: ignore[misc]


def test_diffgui_importer_is_read_only_and_requires_canonical_output(
    tmp_path: Path,
) -> None:
    """Compatibility conversion leaves the source LMDB byte-for-byte unchanged."""
    database = tmp_path / "diffgui.lmdb"
    environment = lmdb.open(str(database), subdir=False, map_size=1 << 20)
    with environment.begin(write=True) as transaction:
        transaction.put(
            b"000",
            pickle.dumps(
                {
                    "protein_filename": "complex/toy_pocket.pdb",
                    "ligand_filename": "complex/toy_ligand.sdf",
                }
            ),
        )
    environment.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    importer = DiffGuiLMDBImporter(
        database,
        source_root=Path(__file__).resolve().parents[1] / "fixtures",
    )
    imported = list(importer)
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert [sample.source_id for sample in imported] == ["diffgui-000"]
    assert before == after
