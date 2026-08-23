"""Leakage-controlled train/validation/test grouping utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from Bio.Align import PairwiseAligner
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

_PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitAudit:
    """Persist every input needed to reproduce and audit a grouped split.

    :param grouping_method: Versioned grouping algorithm identifier.
    :param sequence_identity: Global-alignment protein identity threshold.
    :param ligand_tanimoto: Murcko/Morgan ligand similarity threshold.
    :param seed: Deterministic component allocation seed.
    :param fractions: Train, validation, and test fractions.
    :param input_hashes: Canonical leakage-input hash per sample.
    :param source_identifiers: Traceable source identifier per sample.
    :return: Immutable, JSON-serializable audit contract.
    :rtype: SplitAudit
    """

    grouping_method: str
    sequence_identity: float
    ligand_tanimoto: float
    seed: int
    fractions: tuple[float, float, float]
    input_hashes: Mapping[str, str]
    source_identifiers: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate thresholds and freeze input/source lookup mappings."""
        if not self.grouping_method:
            raise ValueError("grouping_method must be non-empty")
        _validate_split_parameters(
            self.sequence_identity, self.ligand_tanimoto, self.fractions
        )
        for name in ("input_hashes", "source_identifiers"):
            values = dict(getattr(self, name))
            if any(not key or not value for key, value in values.items()):
                raise ValueError(f"{name} must contain non-empty string pairs")
            object.__setattr__(self, name, MappingProxyType(values))
        if set(self.input_hashes) != set(self.source_identifiers):
            raise ValueError(
                "split input hashes and source identifiers must share keys"
            )
        if any(not _is_sha256(value) for value in self.input_hashes.values()):
            raise ValueError("split input hashes must be prefixed SHA-256 digests")

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON primitives for manifests and split hashes.

        :return: Complete recoverable audit fields with ordinary mappings/lists.
        :rtype: dict[str, Any]

        Mapping proxies are copied without mutating this record. Sorting occurs
        only at the JSON/hash boundary, so scientific thresholds remain exact.
        """
        return {
            "grouping_method": self.grouping_method,
            "sequence_identity": self.sequence_identity,
            "ligand_tanimoto": self.ligand_tanimoto,
            "seed": self.seed,
            "fractions": list(self.fractions),
            "input_hashes": dict(self.input_hashes),
            "source_identifiers": dict(self.source_identifiers),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SplitAudit:
        """Reconstruct and validate an audit record from manifest primitives.

        :param values: Mapping emitted by :meth:`to_dict`.
        :return: Frozen audit metadata with recovered tuple/mapping types.
        :rtype: SplitAudit
        :raises KeyError: If a required recoverability field is absent.
        :raises ValueError: If thresholds, fractions, hashes, or sources fail.
        """
        return cls(
            grouping_method=str(values["grouping_method"]),
            sequence_identity=float(values["sequence_identity"]),
            ligand_tanimoto=float(values["ligand_tanimoto"]),
            seed=int(values["seed"]),
            fractions=tuple(float(value) for value in values["fractions"]),  # type: ignore[arg-type]
            input_hashes=dict(values["input_hashes"]),
            source_identifiers=dict(values["source_identifiers"]),
        )


@dataclass(frozen=True)
class GroupedSplit:
    """Store sample partitions together with leakage-audit metadata.

    :param sample_partitions: Production sample identifier to partition lookup.
    :param entity_partitions: Protein and ligand identifiers to partitions.
    :param sample_groups: Sample identifiers to connected leakage groups.
    :param entity_groups: Protein and ligand identifiers to leakage groups.
    :param audit: Recoverable algorithms, thresholds, seed, fractions, and
        canonical input/source hashes.
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
    audit: SplitAudit
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
        if not _is_sha256(self.hash):
            raise ValueError("split hash must be a prefixed SHA-256 digest")

    @property
    def assignments(self) -> Mapping[str, str]:
        """Return the immutable production sample partition mapping.

        :return: Sample IDs mapped to train, validation, or test.
        :rtype: Mapping[str, str]
        """
        return self.sample_partitions

    @property
    def groups(self) -> Mapping[str, str]:
        """Return immutable sample-to-connected-component audit groups.

        :return: Sample IDs mapped to stable hashed group identifiers.
        :rtype: Mapping[str, str]
        """
        return self.sample_groups

    def partition_of(self, identifier: str, *, entity_kind: str | None = None) -> str:
        """Return the partition for a sample or audited entity.

        :param identifier: Sample, protein, or ligand identifier.
        :param entity_kind: Optional ``"protein"`` or ``"ligand"`` qualifier.
        :return: One of ``"train"``, ``"val"``, or ``"test``.
        :rtype: str
        :raises KeyError: If the identifier was absent from the split inputs.
        """
        if entity_kind is not None:
            if entity_kind not in {"protein", "ligand"}:
                raise ValueError("entity_kind must be 'protein' or 'ligand'")
            qualified = f"{entity_kind}:{identifier}"
            if qualified in self.entity_partitions:
                return self.entity_partitions[qualified]
            raise KeyError(f"unknown split identifier: {qualified}")
        if identifier in self.sample_partitions:
            return self.sample_partitions[identifier]
        matches = [
            self.entity_partitions[qualified]
            for qualified in (f"protein:{identifier}", f"ligand:{identifier}")
            if qualified in self.entity_partitions
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"ambiguous entity identifier {identifier!r}; supply entity_kind"
            )
        raise KeyError(f"unknown split identifier: {identifier}")

    def to_metadata(self) -> dict[str, Any]:
        """Return complete JSON-serializable split publication metadata.

        :return: Hash, sample/entity partitions and groups, and full audit data.
        :rtype: dict[str, Any]

        The returned dictionaries are copies; callers cannot mutate this split.
        """
        return {
            "hash": self.hash,
            "sample_partitions": dict(self.sample_partitions),
            "entity_partitions": dict(self.entity_partitions),
            "sample_groups": dict(self.sample_groups),
            "entity_groups": dict(self.entity_groups),
            "audit": self.audit.to_dict(),
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
    :param sequence_identity: Global-alignment identity threshold used only
        when raw protein sequences are supplied. Precomputed sequence-cluster
        identifiers take precedence and are always honored.
    :param ligand_tanimoto: Morgan fingerprint Tanimoto threshold for ligand
        Murcko scaffolds (or the complete molecule for acyclic ligands).
    :param seed: Seed incorporated into deterministic component ordering.
    :param fractions: Train, validation, and test fractions summing to one.
    :return: Sample partitions plus entity/group audit lookups and a hash.
    :rtype: GroupedSplit
    :raises ValueError: If thresholds, fractions, identifiers, SMILES, or
        duplicate sample records are invalid, or if explicit protein/ligand
        grouping evidence is missing.

    Protein and ligand relationships are joined in one disjoint-set graph.
    Consequently, transitive leakage chains are kept together even when one
    pair shares a protein cluster and another pair shares a ligand scaffold.
    Raw protein fallback edges use deterministic affine-gap global alignment;
    raw ligand edges use CPU Morgan fingerprints over Murcko scaffolds. The
    resulting connected components, not individual samples, are assigned by a
    seeded stable ordering. Inputs are read but never mutated, and all audit
    thresholds, source identifiers, normalized-input hashes, groups, and
    qualified entity namespaces are preserved in the returned frozen record.
    Pairwise fallbacks are quadratic and intended for audit-sized datasets;
    production preprocessing should provide explicit clusters.
    """
    _validate_split_parameters(sequence_identity, ligand_tanimoto, fractions)
    rows = sorted((dict(record) for record in records), key=_sample_identifier)
    sample_ids = [_sample_identifier(row) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("split records contain duplicate sample identifiers")
    disjoint = _DisjointSet()
    normalized: list[dict[str, str | None]] = []
    for row, sample_id in zip(rows, sample_ids):
        protein_id = _required_entity_id(row, "protein_id", "protein")
        ligand_id = _required_entity_id(row, "ligand_id", "ligand")
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
        protein_sequence = _optional_text(row.get("protein_sequence"))
        ligand_smiles = _optional_text(row.get("ligand_smiles"))
        if protein_cluster is None and protein_sequence is None:
            raise ValueError(f"sample {sample_id} lacks protein grouping evidence")
        if ligand_group is None and ligand_smiles is None:
            raise ValueError(f"sample {sample_id} lacks ligand grouping evidence")
        if protein_cluster:
            disjoint.union(sample_node, f"protein-cluster:{protein_cluster}")
        if ligand_group:
            disjoint.union(sample_node, f"ligand-group:{ligand_group}")
        normalized.append(
            {
                "sample_id": sample_id,
                "protein_id": protein_id,
                "ligand_id": ligand_id,
                "protein_sequence": protein_sequence,
                "ligand_smiles": ligand_smiles,
                "protein_cluster": protein_cluster,
                "ligand_group": ligand_group,
                "source_identifier": _optional_text(row.get("source_identifier"))
                or sample_id,
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
        for entity_kind, entity_key in (
            ("protein", "protein_id"),
            ("ligand", "ligand_id"),
        ):
            entity_id = f"{entity_kind}:{row[entity_key]}"
            previous = entity_partitions.get(entity_id)
            if previous is not None and previous != partitions[sample_id]:
                raise RuntimeError("entity leakage remained after component grouping")
            entity_partitions[entity_id] = partitions[sample_id]
            entity_groups[entity_id] = group_id
    input_hashes = {
        str(row["sample_id"]): "sha256:"
        + hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for row in normalized
    }
    source_identifiers = {
        str(row["sample_id"]): str(row["source_identifier"]) for row in normalized
    }
    audit = SplitAudit(
        grouping_method="ecloudflow.connected-components.global-align.murcko-morgan.v1",
        sequence_identity=sequence_identity,
        ligand_tanimoto=ligand_tanimoto,
        seed=seed,
        fractions=fractions,
        input_hashes=input_hashes,
        source_identifiers=source_identifiers,
    )
    payload = {
        "audit": audit.to_dict(),
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
        audit=audit,
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


def _required_entity_id(record: Mapping[str, Any], primary: str, fallback: str) -> str:
    """Extract a required entity identifier without leakage-hiding fallbacks."""
    value = record.get(primary, record.get(fallback))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{primary} is required and must be a non-empty string")
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
                and _global_alignment_identity(left_sequence, right_sequence)
                >= threshold
            ):
                disjoint.union(
                    f"sample:{left['sample_id']}", f"sample:{right['sample_id']}"
                )


def _global_alignment_identity(left: str, right: str) -> float:
    """Return deterministic global-alignment identity including gap columns.

    :param left: Non-empty protein sequence using residue character codes.
    :param right: Non-empty protein sequence in the same residue alphabet.
    :return: Exact-match count divided by alignment columns, in ``[0, 1]``.
    :rtype: float
    :raises ValueError: If Biopython cannot align an empty/invalid sequence.

    A global affine-gap alignment is selected with fixed scores. Identity is
    exact character matches divided by the full alignment length, including
    insertions/deletions. This prevents a single indel from shifting every
    downstream comparison as an ungapped ``zip`` calculation would. Scores are
    dimensionless (match ``2``, mismatch ``-1``, gap open ``-2``, extension
    ``-0.5``); execution is deterministic on CPU and does not mutate either
    sequence. Complexity is quadratic in sequence length for this audit fallback.
    """
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(left.upper(), right.upper())[0]
    indices = alignment.indices
    matches = 0
    for left_index, right_index in zip(indices[0], indices[1]):
        if (
            left_index >= 0
            and right_index >= 0
            and left[left_index].upper() == right[right_index].upper()
        ):
            matches += 1
    return matches / indices.shape[1] if indices.shape[1] else 0.0


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


def _is_sha256(value: str) -> bool:
    """Return whether a value is a canonical prefixed hexadecimal digest."""
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True
