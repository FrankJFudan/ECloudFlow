"""Leakage-controlled train/validation/test grouping utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

_PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class GroupedSplit:
    """Store sample partitions together with leakage-audit metadata.

    :param sample_partitions: Production sample identifier to partition lookup.
    :param entity_partitions: Protein and ligand identifiers to partitions.
    :param sample_groups: Sample identifiers to connected leakage groups.
    :param entity_groups: Protein and ligand identifiers to leakage groups.
    :param hash: Content-addressed fingerprint prefixed by ``"sha256:"``.
    :return: Immutable grouped split result.
    :rtype: GroupedSplit

    Entity lookups exist for auditing only. Data loading remains sample based,
    which avoids treating a protein or ligand identifier as a training record.
    """

    sample_partitions: Mapping[str, str]
    entity_partitions: Mapping[str, str]
    sample_groups: Mapping[str, str]
    entity_groups: Mapping[str, str]
    hash: str

    def __post_init__(self) -> None:
        """Defensively copy all lookup tables into read-only mappings."""
        for name in (
            "sample_partitions",
            "entity_partitions",
            "sample_groups",
            "entity_groups",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if set(self.sample_partitions) != set(self.sample_groups):
            raise ValueError("sample partition and group lookups must share keys")
        if set(self.entity_partitions) != set(self.entity_groups):
            raise ValueError("entity partition and group lookups must share keys")
        if any(value not in _PARTITIONS for value in self.sample_partitions.values()):
            raise ValueError("sample partitions contain an unknown partition")
        if any(value not in _PARTITIONS for value in self.entity_partitions.values()):
            raise ValueError("entity partitions contain an unknown partition")
        if not self.hash.startswith("sha256:") or len(self.hash) != 71:
            raise ValueError("split hash must be a prefixed SHA-256 digest")

    @property
    def assignments(self) -> Mapping[str, str]:
        """Return the production sample partition mapping."""
        return self.sample_partitions

    @property
    def groups(self) -> Mapping[str, str]:
        """Return the production sample leakage-group mapping."""
        return self.sample_groups

    def partition_of(self, identifier: str) -> str:
        """Return the partition for a sample or audited entity.

        :param identifier: Sample, protein, or ligand identifier.
        :return: One of ``"train"``, ``"val"``, or ``"test``.
        :rtype: str
        :raises KeyError: If the identifier was absent from the split inputs.
        """
        if identifier in self.sample_partitions:
            return self.sample_partitions[identifier]
        if identifier in self.entity_partitions:
            return self.entity_partitions[identifier]
        raise KeyError(f"unknown split identifier: {identifier}")

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable sample and entity split metadata."""
        return {
            "hash": self.hash,
            "sample_partitions": dict(self.sample_partitions),
            "entity_partitions": dict(self.entity_partitions),
            "sample_groups": dict(self.sample_groups),
            "entity_groups": dict(self.entity_groups),
        }


class _DisjointSet:
    """Deterministic union-find structure with lexical roots."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return and path-compress the canonical root for ``item``."""
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """Join two sets while selecting the lexical root deterministically."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def build_grouped_split(
    records: Iterable[Mapping[str, Any]],
    *,
    sequence_identity: float = 0.4,
    ligand_tanimoto: float = 0.8,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> GroupedSplit:
    """Build deterministic connected-component partitions without leakage.

    :param records: Complex records containing ``sample_id`` (or
        ``source_id``), protein and ligand identifiers, and optionally
        ``sequence_cluster``, ``protein_sequence``, ``ligand_scaffold``, or
        ``ligand_smiles`` values.
    :param sequence_identity: Pairwise positional identity threshold used only
        when raw protein sequences are supplied. Precomputed sequence-cluster
        identifiers take precedence and are always honored.
    :param ligand_tanimoto: Morgan fingerprint Tanimoto threshold for ligand
        Murcko scaffolds (or the complete molecule for acyclic ligands).
    :param seed: Seed incorporated into deterministic component ordering.
    :param fractions: Train, validation, and test fractions summing to one.
    :return: Sample partitions plus entity/group audit lookups and a hash.
    :rtype: GroupedSplit
    :raises ValueError: If thresholds, fractions, identifiers, SMILES, or
        duplicate sample records are invalid.

    Protein and ligand relationships are joined in one disjoint-set graph.
    Consequently, transitive leakage chains are kept together even when one
    pair shares a protein cluster and another pair shares a ligand scaffold.
    """
    _validate_split_parameters(sequence_identity, ligand_tanimoto, fractions)
    rows = sorted((dict(record) for record in records), key=_sample_identifier)
    sample_ids = [_sample_identifier(row) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("split records contain duplicate sample identifiers")
    disjoint = _DisjointSet()
    normalized: list[dict[str, str | None]] = []
    for row, sample_id in zip(rows, sample_ids):
        protein_id = _required_entity_id(row, "protein_id", "protein", sample_id)
        ligand_id = _required_entity_id(row, "ligand_id", "ligand", sample_id)
        sample_node = f"sample:{sample_id}"
        disjoint.find(sample_node)
        disjoint.union(sample_node, f"protein-id:{protein_id}")
        disjoint.union(sample_node, f"ligand-id:{ligand_id}")
        protein_cluster = _optional_text(
            row.get("sequence_cluster", row.get("protein_group"))
        )
        ligand_group = _optional_text(
            row.get("ligand_scaffold", row.get("ligand_group"))
        )
        if protein_cluster:
            disjoint.union(sample_node, f"protein-cluster:{protein_cluster}")
        if ligand_group:
            disjoint.union(sample_node, f"ligand-group:{ligand_group}")
        normalized.append(
            {
                "sample_id": sample_id,
                "protein_id": protein_id,
                "ligand_id": ligand_id,
                "protein_sequence": _optional_text(row.get("protein_sequence")),
                "ligand_smiles": _optional_text(row.get("ligand_smiles")),
            }
        )

    _union_similar_sequences(normalized, disjoint, sequence_identity)
    _union_similar_ligands(normalized, disjoint, ligand_tanimoto)
    components: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        root = disjoint.find(f"sample:{sample_id}")
        components.setdefault(root, []).append(sample_id)
    component_order = sorted(
        components,
        key=lambda root: hashlib.sha256(f"{seed}:{root}".encode()).hexdigest(),
    )
    partitions = _allocate_components(components, component_order, fractions)
    sample_groups: dict[str, str] = {}
    entity_groups: dict[str, str] = {}
    entity_partitions: dict[str, str] = {}
    for row in normalized:
        sample_id = str(row["sample_id"])
        root = disjoint.find(f"sample:{sample_id}")
        group_id = "group:sha256:" + hashlib.sha256(root.encode()).hexdigest()
        sample_groups[sample_id] = group_id
        for entity_key in ("protein_id", "ligand_id"):
            entity_id = str(row[entity_key])
            previous = entity_partitions.get(entity_id)
            if previous is not None and previous != partitions[sample_id]:
                raise RuntimeError("entity leakage remained after component grouping")
            entity_partitions[entity_id] = partitions[sample_id]
            entity_groups[entity_id] = group_id
    payload = {
        "seed": seed,
        "sequence_identity": sequence_identity,
        "ligand_tanimoto": ligand_tanimoto,
        "fractions": fractions,
        "sample_partitions": partitions,
        "entity_partitions": entity_partitions,
        "sample_groups": sample_groups,
        "entity_groups": entity_groups,
    }
    digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return GroupedSplit(
        sample_partitions=partitions,
        entity_partitions=entity_partitions,
        sample_groups=sample_groups,
        entity_groups=entity_groups,
        hash=digest,
    )


def _validate_split_parameters(
    sequence_identity: float,
    ligand_tanimoto: float,
    fractions: tuple[float, float, float],
) -> None:
    """Validate public leakage thresholds and partition fractions."""
    if not 0.0 <= sequence_identity <= 1.0:
        raise ValueError("sequence_identity must lie in [0, 1]")
    if not 0.0 <= ligand_tanimoto <= 1.0:
        raise ValueError("ligand_tanimoto must lie in [0, 1]")
    if (
        len(fractions) != 3
        or any(value < 0.0 for value in fractions)
        or abs(sum(fractions) - 1.0) > 1e-8
    ):
        raise ValueError(
            "fractions must contain three nonnegative values summing to one"
        )


def _sample_identifier(record: Mapping[str, Any]) -> str:
    """Extract one non-empty sample identifier for deterministic sorting."""
    value = record.get("sample_id", record.get("source_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("split records require a non-empty sample_id or source_id")
    return value.strip()


def _required_entity_id(
    record: Mapping[str, Any], primary: str, fallback: str, sample_id: str
) -> str:
    """Extract an entity identifier with a stable sample-scoped fallback."""
    value = record.get(primary, record.get(fallback, f"{fallback}:{sample_id}"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{primary} must be a non-empty string when supplied")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    """Normalize optional textual metadata without inventing a category."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _union_similar_sequences(
    rows: list[dict[str, str | None]],
    disjoint: _DisjointSet,
    threshold: float,
) -> None:
    """Join sample nodes whose supplied sequences exceed positional identity."""
    for left_index, left in enumerate(rows):
        left_sequence = left["protein_sequence"]
        if not left_sequence:
            continue
        for right in rows[left_index + 1 :]:
            right_sequence = right["protein_sequence"]
            if (
                right_sequence
                and _positional_identity(left_sequence, right_sequence) >= threshold
            ):
                disjoint.union(
                    f"sample:{left['sample_id']}", f"sample:{right['sample_id']}"
                )


def _positional_identity(left: str, right: str) -> float:
    """Return conservative ungapped positional sequence identity."""
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / denominator


def _union_similar_ligands(
    rows: list[dict[str, str | None]],
    disjoint: _DisjointSet,
    threshold: float,
) -> None:
    """Join ligand Murcko-scaffold fingerprints above the Tanimoto threshold."""
    fingerprints: list[Any | None] = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row in rows:
        smiles = row["ligand_smiles"]
        if smiles is None:
            fingerprints.append(None)
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid ligand_smiles for sample {row['sample_id']}")
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        target = scaffold if scaffold.GetNumAtoms() else molecule
        fingerprints.append(generator.GetFingerprint(target))
    for left_index, left_fingerprint in enumerate(fingerprints):
        if left_fingerprint is None:
            continue
        for right_index in range(left_index + 1, len(fingerprints)):
            right_fingerprint = fingerprints[right_index]
            if (
                right_fingerprint is not None
                and DataStructs.TanimotoSimilarity(left_fingerprint, right_fingerprint)
                >= threshold
            ):
                disjoint.union(
                    f"sample:{rows[left_index]['sample_id']}",
                    f"sample:{rows[right_index]['sample_id']}",
                )


def _allocate_components(
    components: Mapping[str, list[str]],
    component_order: list[str],
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    """Assign whole components while tracking sample-count fraction targets."""
    total = sum(len(values) for values in components.values())
    targets = [fraction * total for fraction in fractions]
    counts = [0, 0, 0]
    assignments: dict[str, str] = {}
    for root in component_order:
        size = len(components[root])
        deficits = [targets[index] - counts[index] for index in range(3)]
        partition_index = max(range(3), key=lambda index: (deficits[index], -index))
        for sample_id in components[root]:
            assignments[sample_id] = _PARTITIONS[partition_index]
        counts[partition_index] += size
    return dict(sorted(assignments.items()))
