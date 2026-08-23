"""Tests for leakage-controlled protein and ligand splits."""

import pytest

from ecloudflow.data.splits import build_grouped_split


def _records() -> list[dict[str, str]]:
    """Return records containing protein-cluster and scaffold leakage."""
    return [
        {
            "sample_id": "complex-a",
            "protein_id": "protein-A",
            "sequence_cluster": "cluster-1",
            "ligand_id": "ligand-A",
            "ligand_smiles": "CCO",
        },
        {
            "sample_id": "complex-b",
            "protein_id": "protein-A-homolog",
            "sequence_cluster": "cluster-1",
            "ligand_id": "ligand-B",
            "ligand_smiles": "CCN",
        },
        {
            "sample_id": "complex-c",
            "protein_id": "protein-C",
            "sequence_cluster": "cluster-2",
            "ligand_id": "ligand-X",
            "ligand_smiles": "Cc1ccccc1",
        },
        {
            "sample_id": "complex-d",
            "protein_id": "protein-D",
            "sequence_cluster": "cluster-3",
            "ligand_id": "ligand-X-analog",
            "ligand_smiles": "Oc1ccccc1",
        },
        {
            "sample_id": "complex-e",
            "protein_id": "protein-E",
            "sequence_cluster": "cluster-4",
            "ligand_id": "ligand-E",
            "ligand_smiles": "C1CCCCC1",
        },
    ]


def test_grouped_split_keeps_homologs_and_similar_ligands_together() -> None:
    """Known protein clusters and similar scaffolds must never cross splits."""
    split = build_grouped_split(
        _records(), sequence_identity=0.4, ligand_tanimoto=0.8, seed=7
    )
    assert split.partition_of("protein-A") == split.partition_of("protein-A-homolog")
    assert split.partition_of("ligand-X") == split.partition_of("ligand-X-analog")
    assert split.partition_of("complex-a") == split.partition_of("protein-A")
    assert split.sample_partitions.keys() == {
        "complex-a",
        "complex-b",
        "complex-c",
        "complex-d",
        "complex-e",
    }
    assert split.hash.startswith("sha256:")


def test_grouped_split_is_order_independent_and_strict() -> None:
    """Input order cannot change assignments and invalid thresholds fail early."""
    forward = build_grouped_split(_records(), seed=17)
    reverse = build_grouped_split(reversed(_records()), seed=17)
    assert forward.hash == reverse.hash
    assert forward.sample_partitions == reverse.sample_partitions
    with pytest.raises(ValueError, match="ligand_tanimoto"):
        build_grouped_split(_records(), ligand_tanimoto=1.1)
    with pytest.raises(KeyError, match="unknown split identifier"):
        forward.partition_of("missing")
