"""Tests for typed, leakage-controlled local dataset importers."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ecloudflow.cli.main import app
from ecloudflow.data.importers import (
    LocalImportOptions,
    discover_crossdocked,
    discover_pdbbind,
    import_local_dataset,
    parse_pdbbind_affinity,
)
from ecloudflow.data.shards import stream_samples


def _copy_complex(
    fixture_dir: Path,
    directory: Path,
    identifier: str,
    *,
    pocket: bool = True,
) -> None:
    """Create one minimal PDBBind-style source directory from canonical fixtures."""
    directory.mkdir(parents=True)
    shutil.copyfile(
        fixture_dir / "complex/toy_pocket.pdb",
        directory / f"{identifier}_protein.pdb",
    )
    if pocket:
        shutil.copyfile(
            fixture_dir / "complex/toy_pocket.pdb",
            directory / f"{identifier}_pocket.pdb",
        )
    shutil.copyfile(
        fixture_dir / "complex/toy_ligand.sdf",
        directory / f"{identifier}_ligand.sdf",
    )


def _pdbbind_root(fixture_dir: Path, tmp_path: Path) -> Path:
    """Build a two-record local PDBBind tree with affinity INDEX metadata."""
    root = tmp_path / "pdbbind"
    (root / "index").mkdir(parents=True)
    shutil.copyfile(
        fixture_dir / "data/pdbbind_index.txt",
        root / "index/INDEX_general_PL_data.2020",
    )
    _copy_complex(fixture_dir, root / "1abc", "1abc")
    _copy_complex(fixture_dir, root / "2def", "2def")
    return root


def test_pdbbind_affinity_parser_preserves_assay_censoring_and_units() -> None:
    """Published pK and raw measurement semantics must both remain available."""
    metadata = parse_pdbbind_affinity(6.0, "Ki<1uM")
    assert metadata.measurement == "Ki"
    assert metadata.relation == "<"
    assert metadata.value_molar == pytest.approx(1.0e-6)
    assert metadata.properties() == {
        "affinity": 6.0,
        "affinity_measurement": "Ki",
        "affinity_relation": "<",
        "affinity_raw": "Ki<1uM",
        "pki": 6.0,
        "affinity_raw_value": 1.0,
        "affinity_unit": "uM",
        "affinity_value_molar": pytest.approx(1.0e-6),
    }


def test_pdbbind_batch_import_round_trips_metadata_and_grouped_split(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """PDBBind labels and textual assay metadata survive canonical shards."""
    root = _pdbbind_root(fixture_dir, tmp_path)
    discovery = discover_pdbbind(root)
    assert [record.sample_id for record in discovery.records] == ["1ABC", "2DEF"]
    output = tmp_path / "processed"
    result = import_local_dataset(
        LocalImportOptions(
            dataset="pdbbind",
            source_root=root,
            output_dir=output,
            build_fields=False,
            workers=2,
            strict_sources=True,
            target_shard_size_gb=0.5,
            max_samples_per_shard=1,
        )
    )
    assert result.manifest.partition_mode == "grouped"
    assert set(result.manifest.sample_partitions) == {"1ABC", "2DEF"}
    restored = {
        sample.source_id: sample
        for sample in stream_samples(result.manifest.shard_paths(output))
    }
    assert restored["1ABC"].properties["affinity"] == pytest.approx(7.30103)
    assert restored["1ABC"].properties["affinity_measurement"] == "Kd"
    assert restored["1ABC"].properties["affinity_raw"] == "Kd=50nM"
    assert restored["2DEF"].properties["affinity_relation"] == "<"
    assert restored["2DEF"].properties["pki"] == pytest.approx(6.0)
    assert (
        restored["2DEF"]
        .provenance.source_paths["index"]
        .endswith("INDEX_general_PL_data.2020")
    )


def test_crossdocked_import_extracts_indexed_gzip_record_and_filters_rmsd(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Official virtual pose names resolve to base receptor and gzip SDF record."""
    root = tmp_path / "crossdocked"
    (root / "types").mkdir(parents=True)
    (root / "target").mkdir()
    shutil.copyfile(
        fixture_dir / "data/crossdocked.types",
        root / "types/it2_tt_v1.1_completeset_train0.types",
    )
    shutil.copyfile(
        fixture_dir / "complex/toy_pocket.pdb", root / "target/receptor.pdb"
    )
    ligand_text = (fixture_dir / "complex/toy_ligand.sdf").read_text()
    with gzip.open(root / "target/ligand.sdf.gz", "wt", encoding="utf-8") as stream:
        stream.write(ligand_text)
        stream.write(ligand_text)
    discovery = discover_crossdocked(root, rmsd_threshold=1.0)
    assert len(discovery.records) == 1
    assert discovery.filtered_count == 1
    assert discovery.records[0].ligand_record_index == 1
    output = tmp_path / "processed-crossdocked"
    result = import_local_dataset(
        LocalImportOptions(
            dataset="crossdocked",
            source_root=root,
            output_dir=output,
            build_fields=False,
            strict_sources=True,
            target_shard_size_gb=0.5,
        )
    )
    sample = next(iter(stream_samples(result.manifest.shard_paths(output))))
    assert sample.properties["pose_rmsd"] == pytest.approx(0.5)
    assert sample.properties["source_dataset"] == "crossdocked"
    assert sample.provenance.source_paths["ligand"].endswith("ligand.sdf.gz")
    assert sample.provenance.tool_versions["ligand_record_index"] == "1"
    assert "pocket_extractor" in sample.provenance.tool_versions


def test_non_strict_source_issue_is_hashed_into_dataset_manifest(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """An indexed missing complex remains visible in the content-addressed manifest."""
    root = _pdbbind_root(fixture_dir, tmp_path)
    (root / "2def/2def_ligand.sdf").unlink()
    output = tmp_path / "processed-with-skip"
    result = import_local_dataset(
        LocalImportOptions(
            dataset="pdbbind",
            source_root=root,
            output_dir=output,
            build_fields=False,
            strict_sources=False,
            target_shard_size_gb=0.5,
        )
    )
    assert result.manifest.sample_ids == ("1ABC",)
    assert [
        (record.sample_id, record.category) for record in result.manifest.skips
    ] == [("2DEF", "MissingComplex")]
    serialized = json.loads((output / "manifest.json").read_text())
    assert serialized["skips"][0]["sample_id"] == "2DEF"


def test_import_local_cli_publishes_manifest_and_accounting_summary(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """The public CLI exposes a complete local-source-to-training-data path."""
    root = _pdbbind_root(fixture_dir, tmp_path)
    output = tmp_path / "cli-output"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "import-local",
            "--dataset",
            "pdbbind",
            "--source-root",
            str(root),
            "--output-dir",
            str(output),
            "--no-fields",
            "--workers",
            "2",
            "--strict-sources",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "import-summary.json").read_text())
    assert (output / "manifest.json").is_file()
    assert summary["dataset"] == "pdbbind"
    assert summary["serialized_count"] == 2
    assert summary["issue_count"] == 0
    assert sum(summary["partition_counts"].values()) == 2
