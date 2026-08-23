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
from ecloudflow.data.splits import GroupedSplit, SplitAudit
from ecloudflow.exceptions import DataValidationError


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
        entity_partitions={"protein:protein-0": "train"},
        sample_groups={
            sample.source_id: f"group-{index}" for index, sample in enumerate(samples)
        },
        entity_groups={"protein:protein-0": "group-0"},
        audit=SplitAudit(
            grouping_method="test.v1",
            sequence_identity=0.4,
            ligand_tanimoto=0.8,
            seed=0,
            fractions=(0.8, 0.1, 0.1),
            input_hashes={sample.source_id: "sha256:" + "1" * 64 for sample in samples},
            source_identifiers={
                sample.source_id: sample.source_id for sample in samples
            },
        ),
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
    assert module.manifest.entity_partitions["protein:protein-0"] == "train"
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


def test_persistent_worker_observes_shared_epoch_and_reproduces_order(
    tmp_path: Path,
) -> None:
    """The same persistent worker changes and restores seeded epoch ordering."""
    template = _template()
    samples = [replace(template, source_id=f"epoch-{index}") for index in range(12)]
    ShardWriter(max_samples_per_shard=3).write(samples, tmp_path)
    module = ECloudDataModule(
        DataConfig(
            shard_dir=str(tmp_path),
            partition="all",
            batch_size=1,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=1,
            shuffle_buffer=12,
            pin_memory=False,
        )
    )
    loader = module.train_dataloader()

    def order() -> list[str]:
        return [sample.source_id for batch in loader for sample in batch]

    module.set_epoch(0)
    epoch_zero = order()
    module.set_epoch(1)
    epoch_one = order()
    module.set_epoch(0)
    repeated_zero = order()
    assert epoch_zero != epoch_one
    assert repeated_zero == epoch_zero
    iterator = loader._iterator
    if iterator is not None:
        iterator._shutdown_workers()


def test_checkpoint_hash_loaded_before_setup_is_validated_later(
    tmp_path: Path,
) -> None:
    """Lightning restore-before-setup cannot bypass dataset identity checks."""
    template = _template()
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    ShardWriter().write([replace(template, source_id="first")], first_root)
    ShardWriter().write([replace(template, source_id="second")], second_root)
    first = ECloudDataModule(
        DataConfig(shard_dir=str(first_root), num_workers=0, pin_memory=False)
    )
    first.setup()
    state = first.state_dict()
    second = ECloudDataModule(
        DataConfig(shard_dir=str(second_root), num_workers=0, pin_memory=False)
    )
    second.load_state_dict(state)
    with pytest.raises(DataValidationError, match="manifest hash mismatch"):
        second.setup()
