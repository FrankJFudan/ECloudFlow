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


def test_global_sequence_alignment_groups_one_residue_indel() -> None:
    """An insertion must not shift homolog identity below the split threshold."""
    records = [
        {
            "sample_id": "indel-a",
            "protein_id": "protein-indel-a",
            "protein_sequence": "ACDEFGHIKLMN",
            "ligand_id": "ligand-indel-a",
            "ligand_group": "ligand-group-a",
        },
        {
            "sample_id": "indel-b",
            "protein_id": "protein-indel-b",
            "protein_sequence": "ACDDEFGHIKLMN",
            "ligand_id": "ligand-indel-b",
            "ligand_group": "ligand-group-b",
        },
        {
            "sample_id": "unrelated",
            "protein_id": "protein-unrelated",
            "protein_sequence": "YYYYYYYYYYYY",
            "ligand_id": "ligand-unrelated",
            "ligand_group": "ligand-group-c",
        },
    ]
    split = build_grouped_split(records, sequence_identity=0.9, seed=9)
    assert split.partition_of(
        "protein-indel-a", entity_kind="protein"
    ) == split.partition_of("protein-indel-b", entity_kind="protein")


def test_split_rejects_missing_identifiers_or_grouping_evidence() -> None:
    """Leakage-controlled labels require traceable entities and comparisons."""
    base = {
        "sample_id": "missing",
        "protein_id": "protein",
        "sequence_cluster": "cluster",
        "ligand_id": "ligand",
        "ligand_smiles": "CCO",
    }
    missing_identifier = dict(base)
    del missing_identifier["protein_id"]
    with pytest.raises(ValueError, match="protein_id is required"):
        build_grouped_split([missing_identifier])
    missing_evidence = dict(base)
    del missing_evidence["sequence_cluster"]
    with pytest.raises(ValueError, match="protein grouping evidence"):
        build_grouped_split([missing_evidence])


def test_equal_text_entity_ids_require_qualified_lookup() -> None:
    """Protein and ligand namespaces cannot collide in persisted audit metadata."""
    records = [
        {
            "sample_id": "namespace-a",
            "protein_id": "shared",
            "sequence_cluster": "protein-a",
            "ligand_id": "ligand-a",
            "ligand_group": "ligand-a",
        },
        {
            "sample_id": "namespace-b",
            "protein_id": "protein-b",
            "sequence_cluster": "protein-b",
            "ligand_id": "shared",
            "ligand_group": "ligand-b",
        },
    ]
    split = build_grouped_split(records, fractions=(0.5, 0.0, 0.5), seed=3)
    assert "protein:shared" in split.entity_partitions
    assert "ligand:shared" in split.entity_partitions
    assert split.partition_of("shared", entity_kind="protein") in {"train", "test"}
    assert split.audit.input_hashes.keys() == split.sample_partitions.keys()
