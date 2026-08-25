"""Production local-source importers for PDBBind and CrossDocked datasets."""

from __future__ import annotations

import gzip
import hashlib
import math
import re
import tempfile
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from Bio.PDB import PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ecloudflow.chemistry.standardize import CANONICAL_ISOMERIC_SMILES_PROPERTY
from ecloudflow.core.types import ComplexSample, SampleProvenance
from ecloudflow.data.manifest import DatasetManifest, SkipRecord
from ecloudflow.data.parsers import (
    build_complex_sample,
    parse_ligand_sdf,
    parse_pocket_pdb,
)
from ecloudflow.data.shards import ShardWriter
from ecloudflow.data.splits import build_grouped_split
from ecloudflow.exceptions import DataValidationError

DatasetFamily = Literal["pdbbind", "crossdocked"]

_PDBBIND_MEASUREMENT = re.compile(
    r"(?P<kind>IC50|Kd|Ki|Ka)\s*(?P<relation><=|>=|=|<|>|~)?\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)?\s*"
    r"(?P<unit>fM|pM|nM|uM|mM|M)?",
    re.IGNORECASE,
)
_UNIT_TO_MOLAR = {
    "fm": 1.0e-15,
    "pm": 1.0e-12,
    "nm": 1.0e-9,
    "um": 1.0e-6,
    "mm": 1.0e-3,
    "m": 1.0,
}
_AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass(frozen=True)
class AffinityMetadata:
    """Represent one PDBBind affinity observation without erasing its assay type.

    :param pk_value: Published negative base-10 logarithmic affinity column.
    :param measurement: Canonical ``Kd``, ``Ki``, ``IC50``, ``Ka``, or
        ``unknown`` measurement family.
    :param relation: Published equality/censoring operator, such as ``=`` or ``<``.
    :param raw_expression: Unmodified fifth INDEX token.
    :param raw_value: Optional numeric value parsed from the expression.
    :param raw_unit: Optional concentration unit parsed from the expression.
    :param value_molar: Optional concentration converted to molar units.
    :return: Immutable metadata that can be serialized into sample properties.
    :rtype: AffinityMetadata

    The generic ``affinity`` property always stores the published pK column so
    training does not mix linear concentrations with logarithmic targets. Raw
    values and censoring metadata remain available for assay-specific filtering.
    """

    pk_value: float
    measurement: str
    relation: str
    raw_expression: str
    raw_value: float | None = None
    raw_unit: str | None = None
    value_molar: float | None = None

    def properties(self) -> dict[str, float | str]:
        """Return stable sample-property names for this affinity observation."""
        values: dict[str, float | str] = {
            "affinity": self.pk_value,
            "affinity_measurement": self.measurement,
            "affinity_relation": self.relation,
            "affinity_raw": self.raw_expression,
        }
        alias = {
            "Kd": "pkd",
            "Ki": "pki",
            "IC50": "pic50",
            "Ka": "pka",
        }.get(self.measurement)
        if alias is not None:
            values[alias] = self.pk_value
        if self.raw_value is not None:
            values["affinity_raw_value"] = self.raw_value
        if self.raw_unit is not None:
            values["affinity_unit"] = self.raw_unit
        if self.value_molar is not None:
            values["affinity_value_molar"] = self.value_molar
        return values


@dataclass(frozen=True)
class LocalComplexSource:
    """Describe one immutable local protein-ligand source record.

    :param family: Dataset family controlling source interpretation.
    :param sample_id: Stable unique identifier written to shards and manifests.
    :param protein_path: Full protein/receptor PDB used for grouping and, when
        necessary, pocket extraction.
    :param ligand_path: Direct one-record SDF or compressed multi-record SDF.
    :param index_path: PDBBind INDEX or CrossDocked ``.types`` source file.
    :param protein_id: Entity identity used by leakage-controlled splitting.
    :param ligand_id: Ligand/pose identity used by splitting and audit metadata.
    :param source_identifier: Dataset-qualified origin retained in split audit.
    :param properties: Numeric labels and textual measurement metadata.
    :param pocket_path: Optional existing pocket PDB; otherwise one is extracted.
    :param ligand_record_index: Optional zero-based record in a compressed SDF.
    :param protein_cluster: Optional externally supplied sequence-cluster ID.
    :return: Frozen, path-validated source descriptor.
    :rtype: LocalComplexSource
    """

    family: DatasetFamily
    sample_id: str
    protein_path: Path
    ligand_path: Path
    index_path: Path
    protein_id: str
    ligand_id: str
    source_identifier: str
    properties: Mapping[str, float | int | str] = field(default_factory=dict)
    pocket_path: Path | None = None
    ligand_record_index: int | None = None
    protein_cluster: str | None = None

    def __post_init__(self) -> None:
        """Validate source paths/identifiers and freeze property metadata."""
        if self.family not in {"pdbbind", "crossdocked"}:
            raise ValueError(f"unsupported dataset family: {self.family}")
        for name in ("sample_id", "protein_id", "ligand_id", "source_identifier"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in ("protein_path", "ligand_path", "index_path"):
            path = Path(getattr(self, name))
            if not path.is_file():
                raise DataValidationError(f"{name} does not exist: {path}")
            object.__setattr__(self, name, path.resolve())
        if self.pocket_path is not None:
            pocket = Path(self.pocket_path)
            if not pocket.is_file():
                raise DataValidationError(f"pocket_path does not exist: {pocket}")
            object.__setattr__(self, "pocket_path", pocket.resolve())
        if self.ligand_record_index is not None and self.ligand_record_index < 0:
            raise ValueError("ligand_record_index must be nonnegative")
        properties = dict(self.properties)
        if any(not isinstance(key, str) or not key for key in properties):
            raise ValueError("source property keys must be non-empty strings")
        object.__setattr__(self, "properties", MappingProxyType(properties))


@dataclass(frozen=True)
class ImportIssue:
    """Record one transparently excluded local-source record."""

    sample_id: str
    category: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Return JSON-safe issue fields."""
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "message": self.message,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """Return deterministic discovery records, issues, and filtered pose count."""

    records: tuple[LocalComplexSource, ...]
    issues: tuple[ImportIssue, ...] = ()
    filtered_count: int = 0


@dataclass(frozen=True)
class LocalImportOptions:
    """Configure bounded, deterministic local dataset conversion.

    :param dataset: ``pdbbind`` or ``crossdocked``.
    :param source_root: Existing extracted dataset root; no downloader is called.
    :param output_dir: Destination consumed by :class:`ShardWriter`.
    :param index_path: Optional explicit PDBBind INDEX or CrossDocked types file.
    :param protein_clusters: Optional two-column ``protein_id cluster_id`` file.
    :param build_fields: Whether to build pocket and xTB ligand electron fields.
    :param workers: Ordered bounded preprocessing concurrency.
    :param strict_sources: Fail before publication if any source is invalid.
    :param rmsd_threshold: Maximum accepted CrossDocked pose RMSD in angstroms.
    :param pocket_radius: Protein residue cutoff around ligand atoms in angstroms.
    :param sequence_identity: Fallback protein grouping threshold.
    :param ligand_tanimoto: Split-audit ligand similarity threshold.
    :param train_fraction: Target train fraction assigned by connected component.
    :param val_fraction: Target validation fraction; remainder is test.
    :param split_seed: Stable connected-component allocation seed.
    :param max_pairwise_sequences: Maximum records allowed to use quadratic raw
        sequence fallback; larger imports must provide precomputed clusters.
    :param limit: Optional deterministic discovery prefix for smoke runs.
    :param target_shard_size_gb: Atomic tar shard size target.
    :param max_samples_per_shard: Optional deterministic shard sample bound.
    :return: Immutable conversion settings.
    :rtype: LocalImportOptions

    Source validation is a graph-only first pass. It establishes the exact set
    of samples used to build the leakage split before the second, field-enabled
    pass enters ``ShardWriter``. Consequently a published grouped manifest can
    never contain incomplete or stale split coverage.
    """

    dataset: DatasetFamily
    source_root: Path
    output_dir: Path
    index_path: Path | None = None
    protein_clusters: Path | None = None
    build_fields: bool = True
    workers: int = 1
    strict_sources: bool = False
    rmsd_threshold: float = 1.0
    pocket_radius: float = 10.0
    sequence_identity: float = 0.4
    ligand_tanimoto: float = 0.8
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    split_seed: int = 2026
    max_pairwise_sequences: int = 5000
    limit: int | None = None
    target_shard_size_gb: float = 1.0
    max_samples_per_shard: int | None = None

    def __post_init__(self) -> None:
        """Validate paths, thresholds, fractions, and bounded worker settings."""
        root = Path(self.source_root)
        if not root.is_dir():
            raise DataValidationError(f"dataset source root does not exist: {root}")
        object.__setattr__(self, "source_root", root.resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.index_path is not None:
            object.__setattr__(self, "index_path", Path(self.index_path))
        if self.protein_clusters is not None:
            clusters = Path(self.protein_clusters)
            if not clusters.is_file():
                raise DataValidationError(
                    f"protein cluster mapping does not exist: {clusters}"
                )
            object.__setattr__(self, "protein_clusters", clusters.resolve())
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive when supplied")
        if self.max_pairwise_sequences < 1:
            raise ValueError("max_pairwise_sequences must be positive")
        if not math.isfinite(self.rmsd_threshold) or self.rmsd_threshold < 0.0:
            raise ValueError("rmsd_threshold must be finite and nonnegative")
        if not math.isfinite(self.pocket_radius) or self.pocket_radius <= 0.0:
            raise ValueError("pocket_radius must be finite and positive")
        test_fraction = 1.0 - self.train_fraction - self.val_fraction
        if (
            self.train_fraction < 0.0
            or self.val_fraction < 0.0
            or test_fraction < -1.0e-8
        ):
            raise ValueError("train/validation fractions must leave a nonnegative test")

    @property
    def fractions(self) -> tuple[float, float, float]:
        """Return train, validation, and derived test fractions."""
        return (
            self.train_fraction,
            self.val_fraction,
            max(0.0, 1.0 - self.train_fraction - self.val_fraction),
        )


@dataclass(frozen=True)
class LocalImportResult:
    """Describe one published local dataset conversion."""

    manifest: DatasetManifest
    issues: tuple[ImportIssue, ...]
    discovered_count: int
    validated_count: int
    filtered_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-safe conversion summary."""
        return {
            "manifest_hash": self.manifest.hash,
            "generation_id": self.manifest.generation_id,
            "discovered_count": self.discovered_count,
            "validated_count": self.validated_count,
            "serialized_count": len(self.manifest.sample_ids),
            "filtered_count": self.filtered_count,
            "issue_count": len(self.issues),
            "issues": [issue.as_dict() for issue in self.issues],
            "partition_counts": {
                partition: sum(
                    value == partition
                    for value in self.manifest.sample_partitions.values()
                )
                for partition in ("train", "val", "test")
            },
        }


@dataclass(frozen=True)
class _ValidatedSource:
    """Retain exact split evidence from the graph-only validation pass."""

    source: LocalComplexSource
    properties: Mapping[str, float | int | str]
    split_record: Mapping[str, str]


@dataclass(frozen=True)
class _ValidationOutcome:
    """Return either one validated source or one transparent issue."""

    validated: _ValidatedSource | None = None
    issue: ImportIssue | None = None


def parse_pdbbind_affinity(pk_value: float, expression: str) -> AffinityMetadata:
    """Parse one PDBBind raw affinity token while retaining its exact text.

    :param pk_value: Finite published ``-log10`` affinity column.
    :param expression: Fifth INDEX token, for example ``Kd=50nM`` or ``Ki<1uM``.
    :return: Typed normalized and raw affinity metadata.
    :rtype: AffinityMetadata
    :raises ValueError: If the pK value is non-finite or expression is empty.

    Unknown or partially specified expressions are not discarded. They retain
    ``measurement='unknown'`` and the exact raw expression while the numeric pK
    target remains usable. Concentration conversion occurs only when both a
    numeric value and a recognized molar unit are present.
    """
    if not math.isfinite(pk_value):
        raise ValueError("PDBBind pK value must be finite")
    if not expression:
        raise ValueError("PDBBind affinity expression must be non-empty")
    match = _PDBBIND_MEASUREMENT.search(expression)
    if match is None:
        return AffinityMetadata(pk_value, "unknown", "=", expression)
    raw_kind = match.group("kind").lower()
    kind = {"kd": "Kd", "ki": "Ki", "ic50": "IC50", "ka": "Ka"}[raw_kind]
    relation = match.group("relation") or "="
    raw_value = float(match.group("value")) if match.group("value") else None
    raw_unit = match.group("unit")
    value_molar = None
    if raw_value is not None and raw_unit is not None:
        value_molar = raw_value * _UNIT_TO_MOLAR[raw_unit.lower()]
    return AffinityMetadata(
        pk_value=pk_value,
        measurement=kind,
        relation=relation,
        raw_expression=expression,
        raw_value=raw_value,
        raw_unit=raw_unit,
        value_molar=value_molar,
    )


def read_protein_clusters(path: str | Path) -> dict[str, str]:
    """Read a strict two-column ``protein_id cluster_id`` mapping.

    :param path: UTF-8 whitespace-delimited mapping; blank/comment lines and one
        optional ``protein_id cluster_id`` header are accepted.
    :return: Identifier-to-cluster mapping including case-folded lookup aliases.
    :rtype: dict[str, str]
    :raises DataValidationError: If a row is malformed or assigns one member to
        conflicting clusters.
    """
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"protein cluster mapping does not exist: {source}")
    values: dict[str, str] = {}
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if [token.lower() for token in tokens[:2]] == [
                "protein_id",
                "cluster_id",
            ]:
                continue
            if len(tokens) != 2:
                raise DataValidationError(
                    f"invalid protein cluster row {line_number}: expected two columns"
                )
            member, cluster = tokens
            for key in {member, member.casefold()}:
                previous = values.get(key)
                if previous is not None and previous != cluster:
                    raise DataValidationError(
                        f"protein {member!r} has conflicting cluster assignments"
                    )
                values[key] = cluster
    if not values:
        raise DataValidationError("protein cluster mapping contains no assignments")
    return values


def discover_pdbbind(
    source_root: str | Path,
    *,
    index_path: str | Path | None = None,
    protein_clusters: Mapping[str, str] | None = None,
) -> DiscoveryResult:
    """Discover PDBBind complexes and typed affinity labels from local files.

    :param source_root: Extracted PDBBind root containing complex directories.
    :param index_path: Optional explicit ``INDEX_general_PL_data.*`` path.
    :param protein_clusters: Optional complete protein-to-cluster lookup.
    :return: Sorted valid local records plus transparent missing-source issues.
    :rtype: DiscoveryResult
    :raises DataValidationError: If the root/index is absent, index rows are
        malformed, cluster coverage is incomplete, or no complex can be found.

    Discovery never downloads, converts, or mutates licensed PDBBind files. It
    indexes ligand directories once, preferring an existing ``*_pocket*.pdb``;
    a missing pocket is derived from ``*_protein.pdb`` only during validation.
    Both the full protein and affinity INDEX are retained as source provenance.
    """
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise DataValidationError(f"PDBBind root does not exist: {root}")
    index = _resolve_pdbbind_index(root, index_path)
    affinity_rows = _read_pdbbind_index(index)
    directories: dict[str, list[Path]] = {}
    for ligand in root.rglob("*_ligand.sdf"):
        pdb_id = ligand.name.removesuffix("_ligand.sdf").casefold()
        directories.setdefault(pdb_id, []).append(ligand.parent)
    records: list[LocalComplexSource] = []
    issues: list[ImportIssue] = []
    for pdb_id, affinity, resolution, year in affinity_rows:
        candidates = sorted(
            set(directories.get(pdb_id.casefold(), [])),
            key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
        )
        if not candidates:
            issues.append(
                ImportIssue(pdb_id.upper(), "MissingComplex", "ligand SDF not found")
            )
            continue
        directory = candidates[0]
        ligand = directory / f"{pdb_id}_ligand.sdf"
        if not ligand.is_file():
            matching = sorted(directory.glob("*_ligand.sdf"))
            ligand = matching[0] if matching else ligand
        protein = directory / f"{pdb_id}_protein.pdb"
        if not protein.is_file():
            matching = sorted(directory.glob("*_protein.pdb"))
            protein = matching[0] if matching else protein
        if not protein.is_file():
            issues.append(ImportIssue(pdb_id.upper(), "MissingProtein", str(protein)))
            continue
        pockets = sorted(directory.glob(f"{pdb_id}_pocket*.pdb"))
        cluster = _cluster_lookup(
            protein_clusters,
            pdb_id,
            pdb_id.upper(),
            protein.stem,
            _relative_path(protein, root),
        )
        if protein_clusters is not None and cluster is None:
            issues.append(
                ImportIssue(
                    pdb_id.upper(),
                    "MissingProteinCluster",
                    "no cluster mapping matched PDB ID or protein path",
                )
            )
            continue
        properties: dict[str, float | int | str] = {
            "source_dataset": "pdbbind",
            **affinity.properties(),
        }
        if resolution is not None:
            properties["resolution_angstrom"] = resolution
        if year is not None:
            properties["release_year"] = year
        records.append(
            LocalComplexSource(
                family="pdbbind",
                sample_id=pdb_id.upper(),
                protein_path=protein,
                ligand_path=ligand,
                pocket_path=pockets[0] if pockets else None,
                index_path=index,
                protein_id=pdb_id.upper(),
                ligand_id=pdb_id.upper(),
                source_identifier=f"pdbbind:{pdb_id.lower()}",
                properties=properties,
                protein_cluster=cluster,
            )
        )
    if not records:
        raise DataValidationError("PDBBind discovery found no usable source records")
    return DiscoveryResult(
        tuple(sorted(records, key=lambda item: item.sample_id)), tuple(issues)
    )


def discover_crossdocked(
    source_root: str | Path,
    *,
    types_path: str | Path | None = None,
    rmsd_threshold: float = 1.0,
    protein_clusters: Mapping[str, str] | None = None,
) -> DiscoveryResult:
    """Discover CrossDocked poses from the official local ``.types`` index.

    :param source_root: Extracted CrossDocked2020 v1.1 root.
    :param types_path: Optional explicit six-column completeset types file.
    :param rmsd_threshold: Inclusive maximum pose RMSD in angstroms.
    :param protein_clusters: Optional complete protein-to-cluster lookup.
    :return: Deterministically sorted pose records, issues, and RMSD filter count.
    :rtype: DiscoveryResult
    :raises DataValidationError: If the root/types file is missing or no record
        survives source validation and RMSD filtering.

    Official types rows name virtual ``protein_N.pdb``/``ligand_N.sdf`` records.
    When those direct files are absent, discovery resolves the corresponding
    base receptor PDB and compressed multi-record ``.sdf.gz`` without unpacking
    the dataset. Record extraction happens in an isolated temporary directory.
    Unsafe absolute or parent-traversing index paths are rejected.
    """
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise DataValidationError(f"CrossDocked root does not exist: {root}")
    index = _resolve_crossdocked_types(root, types_path)
    records: list[LocalComplexSource] = []
    issues: list[ImportIssue] = []
    filtered = 0
    seen: set[str] = set()
    with index.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 6:
                issues.append(
                    ImportIssue(
                        f"line-{line_number}",
                        "InvalidTypesRow",
                        "expected at least six whitespace-delimited columns",
                    )
                )
                continue
            try:
                rmsd = float(tokens[2])
            except ValueError:
                issues.append(
                    ImportIssue(f"line-{line_number}", "InvalidRMSD", tokens[2])
                )
                continue
            if not math.isfinite(rmsd) or rmsd < 0.0:
                issues.append(
                    ImportIssue(f"line-{line_number}", "InvalidRMSD", tokens[2])
                )
                continue
            if rmsd > rmsd_threshold:
                filtered += 1
                continue
            try:
                virtual_protein = _safe_dataset_path(root, tokens[3])
                virtual_ligand = _safe_dataset_path(root, tokens[4])
                protein = _resolve_crossdocked_protein(virtual_protein)
                ligand, record_index = _resolve_crossdocked_ligand(virtual_ligand)
            except (DataValidationError, ValueError) as error:
                issues.append(
                    ImportIssue(f"line-{line_number}", type(error).__name__, str(error))
                )
                continue
            identity = f"{_relative_path(protein, root)}|{tokens[4]}|{record_index}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            sample_id = f"crossdocked-{digest}"
            if sample_id in seen:
                issues.append(
                    ImportIssue(sample_id, "DuplicatePose", f"types line {line_number}")
                )
                continue
            seen.add(sample_id)
            protein_relative = _relative_path(protein, root)
            cluster = _cluster_lookup(
                protein_clusters,
                protein_relative,
                tokens[3],
                protein.stem,
            )
            if protein_clusters is not None and cluster is None:
                issues.append(
                    ImportIssue(
                        sample_id,
                        "MissingProteinCluster",
                        "no cluster mapping matched receptor path or stem",
                    )
                )
                continue
            records.append(
                LocalComplexSource(
                    family="crossdocked",
                    sample_id=sample_id,
                    protein_path=protein,
                    ligand_path=ligand,
                    ligand_record_index=record_index,
                    index_path=index,
                    protein_id=protein_relative,
                    ligand_id=f"{tokens[4]}#{record_index if record_index is not None else 0}",
                    source_identifier=f"crossdocked:{index.name}:{line_number}",
                    properties={
                        "source_dataset": "crossdocked",
                        "pose_rmsd": rmsd,
                        "types_line": line_number,
                    },
                    protein_cluster=cluster,
                )
            )
    if not records:
        raise DataValidationError(
            "CrossDocked discovery found no source records within the RMSD threshold"
        )
    return DiscoveryResult(
        tuple(sorted(records, key=lambda item: item.sample_id)),
        tuple(issues),
        filtered,
    )


def import_local_dataset(options: LocalImportOptions) -> LocalImportResult:
    """Validate, split, and stream a complete local dataset to atomic shards.

    :param options: Strict local conversion, split, field, and shard settings.
    :return: Published manifest plus transparent discovery/validation accounting.
    :rtype: LocalImportResult
    :raises DataValidationError: If discovery yields no valid records, strict
        mode observes an issue, large raw-sequence grouping lacks clusters, or
        a source changes/fails between validation and serialization.
    :raises ShardWriteError: If atomic dataset generation cannot be published.

    The importer intentionally performs two deterministic streaming passes.
    Pass one materializes one record at a time, builds a graph-only
    ``ComplexSample``, extracts protein/scaffold grouping evidence, and discards
    tensors. The exact accepted identifiers then form one connected-component
    split. Pass two rebuilds those same sources, optionally including physical
    fields, through a bounded ordered worker queue feeding ``ShardWriter``.
    Memory is therefore ``O(workers * sample_size)`` rather than dataset-sized,
    while split assignments exactly cover every published sample.
    """
    clusters = (
        read_protein_clusters(options.protein_clusters)
        if options.protein_clusters is not None
        else None
    )
    if options.dataset == "pdbbind":
        discovery = discover_pdbbind(
            options.source_root,
            index_path=options.index_path,
            protein_clusters=clusters,
        )
    else:
        discovery = discover_crossdocked(
            options.source_root,
            types_path=options.index_path,
            rmsd_threshold=options.rmsd_threshold,
            protein_clusters=clusters,
        )
    records = discovery.records[: options.limit]
    if clusters is None and len(records) > options.max_pairwise_sequences:
        raise DataValidationError(
            f"raw sequence grouping for {len(records)} records exceeds "
            f"max_pairwise_sequences={options.max_pairwise_sequences}; provide "
            "--protein-clusters from MMseqs2/CD-HIT"
        )
    outcomes = _bounded_ordered_map(
        lambda source: _validate_source(source, options.pocket_radius),
        records,
        options.workers,
    )
    validated: list[_ValidatedSource] = []
    issues = list(discovery.issues)
    for outcome in outcomes:
        if outcome.validated is not None:
            validated.append(outcome.validated)
        elif outcome.issue is not None:
            issues.append(outcome.issue)
    if options.strict_sources and issues:
        preview = "; ".join(
            f"{issue.sample_id}:{issue.category}" for issue in issues[:5]
        )
        raise DataValidationError(
            f"strict local import rejected {len(issues)} source records: {preview}"
        )
    if not validated:
        raise DataValidationError("local import validation accepted no samples")
    split = build_grouped_split(
        (value.split_record for value in validated),
        sequence_identity=options.sequence_identity,
        ligand_tanimoto=options.ligand_tanimoto,
        seed=options.split_seed,
        fractions=options.fractions,
    )
    unique_paths = {
        path
        for value in validated
        for path in (
            value.source.protein_path,
            value.source.ligand_path,
            value.source.index_path,
            value.source.pocket_path,
        )
        if path is not None
    }
    source_hashes = {
        path: digest
        for path, digest in _bounded_ordered_map(
            lambda path: (path, _sha256_file(path)),
            tuple(sorted(unique_paths)),
            options.workers,
        )
    }
    samples = _bounded_ordered_map(
        lambda value: _build_validated_source(
            value,
            build_fields=options.build_fields,
            pocket_radius=options.pocket_radius,
            source_hashes=source_hashes,
        ),
        validated,
        options.workers,
    )
    manifest = ShardWriter(
        target_shard_size_gb=options.target_shard_size_gb,
        max_samples_per_shard=options.max_samples_per_shard,
        preprocessing_version="local-complex-v1",
    ).write(
        samples,
        options.output_dir,
        split=split,
        source_skips=tuple(_issue_to_skip(issue) for issue in issues),
    )
    return LocalImportResult(
        manifest=manifest,
        issues=tuple(issues),
        discovered_count=len(records),
        validated_count=len(validated),
        filtered_count=discovery.filtered_count,
    )


def _read_pdbbind_index(
    path: Path,
) -> list[tuple[str, AffinityMetadata, float | None, int | None]]:
    """Parse strict data columns from a PDBBind general protein-ligand INDEX."""
    rows: list[tuple[str, AffinityMetadata, float | None, int | None]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split("//", 1)[0].split()
            if len(tokens) < 5:
                raise DataValidationError(
                    f"invalid PDBBind INDEX row {line_number}: expected five columns"
                )
            pdb_id = tokens[0].lower()
            if pdb_id in seen:
                raise DataValidationError(f"duplicate PDBBind INDEX ID: {pdb_id}")
            seen.add(pdb_id)
            try:
                pk_value = float(tokens[3])
            except ValueError as error:
                raise DataValidationError(
                    f"invalid PDBBind pK value on row {line_number}: {tokens[3]}"
                ) from error
            resolution = _optional_float(tokens[1])
            year = int(tokens[2]) if tokens[2].isdigit() else None
            rows.append(
                (
                    pdb_id,
                    parse_pdbbind_affinity(pk_value, tokens[4]),
                    resolution,
                    year,
                )
            )
    if not rows:
        raise DataValidationError(f"PDBBind INDEX contains no data rows: {path}")
    return rows


def _validate_source(
    source: LocalComplexSource, pocket_radius: float
) -> _ValidationOutcome:
    """Build one graph-only sample and retain exact leakage-grouping evidence."""
    try:
        with _materialized_source(source, pocket_radius) as (pocket, ligand):
            build_complex_sample(
                pocket,
                ligand,
                source.sample_id,
                build_fields=False,
                properties=source.properties,
            )
            molecule = parse_ligand_sdf(ligand)
        canonical_smiles = (
            molecule.GetProp(CANONICAL_ISOMERIC_SMILES_PROPERTY)
            if molecule.HasProp(CANONICAL_ISOMERIC_SMILES_PROPERTY)
            else Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
        properties = dict(source.properties)
        properties[CANONICAL_ISOMERIC_SMILES_PROPERTY] = canonical_smiles
        split_record: dict[str, str] = {
            "sample_id": source.sample_id,
            "source_identifier": source.source_identifier,
            "protein_id": source.protein_id,
            "ligand_id": source.ligand_id,
            "ligand_scaffold": _ligand_scaffold(molecule),
        }
        if source.protein_cluster is not None:
            split_record["sequence_cluster"] = source.protein_cluster
        else:
            split_record["protein_sequence"] = _protein_sequence(source.protein_path)
        return _ValidationOutcome(
            validated=_ValidatedSource(
                source=source,
                properties=MappingProxyType(properties),
                split_record=MappingProxyType(split_record),
            )
        )
    except Exception as error:  # noqa: BLE001 - per-record batch accounting
        return _ValidationOutcome(
            issue=ImportIssue(source.sample_id, type(error).__name__, str(error))
        )


def _build_validated_source(
    value: _ValidatedSource,
    *,
    build_fields: bool,
    pocket_radius: float,
    source_hashes: Mapping[Path, str],
) -> ComplexSample:
    """Rebuild a validated source with fields and stable original provenance.

    :param value: Source and metadata accepted by the graph-only first pass.
    :param build_fields: Whether to invoke physical field builders.
    :param pocket_radius: Radius used when no existing pocket file is present.
    :param source_hashes: Precomputed hashes for every immutable original file.
    :return: Canonical sample whose provenance never references temporary paths.
    :rtype: ComplexSample
    :raises DataValidationError: If deterministic rebuilding unexpectedly fails.

    CrossDocked compressed ligand records and generated pockets exist only in a
    private temporary directory. Before return, their paths/hashes are replaced
    by the original receptor, archive, and types-file identities plus explicit
    extraction metadata. The serialized sample therefore remains auditable
    after temporary files are removed.
    """
    source = value.source
    try:
        with _materialized_source(source, pocket_radius) as (pocket, ligand):
            sample = build_complex_sample(
                pocket,
                ligand,
                source.sample_id,
                build_fields=build_fields,
                properties=value.properties,
            )
    except Exception as error:
        raise DataValidationError(
            f"validated source changed or failed during field build "
            f"({source.sample_id}): {error}"
        ) from error
    original_paths = {
        "protein": str(source.protein_path),
        "ligand": str(source.ligand_path),
        "index": str(source.index_path),
    }
    original_hashes = {
        role: source_hashes[path]
        for role, path in (
            ("protein", source.protein_path),
            ("ligand", source.ligand_path),
            ("index", source.index_path),
        )
    }
    if source.pocket_path is not None:
        original_paths["pocket"] = str(source.pocket_path)
        original_hashes["pocket"] = source_hashes[source.pocket_path]
    tool_versions = dict(sample.provenance.tool_versions)
    tool_versions["local_dataset_importer"] = "local-complex-v1"
    if source.pocket_path is None:
        tool_versions["pocket_extractor"] = f"residue-distance-{pocket_radius:g}A-v1"
    if source.ligand_record_index is not None:
        tool_versions["ligand_record_index"] = str(source.ligand_record_index)
    provenance = SampleProvenance(
        source_paths=original_paths,
        file_hashes=original_hashes,
        tool_versions=tool_versions,
        preprocessing_status="complete",
        original_ligand_positions=sample.provenance.original_ligand_positions,
        qm=sample.provenance.qm,
    )
    return replace(sample, provenance=provenance)


@contextmanager
def _materialized_source(
    source: LocalComplexSource, pocket_radius: float
) -> Iterator[tuple[Path, Path]]:
    """Yield direct or temporary one-record ligand and pocket paths."""
    with tempfile.TemporaryDirectory(
        prefix=f"ecloudflow-{source.family}-"
    ) as directory:
        work = Path(directory)
        ligand = source.ligand_path
        if ligand.suffix.lower() == ".gz":
            ligand = work / "ligand.sdf"
            _extract_sdf_record(
                source.ligand_path,
                source.ligand_record_index or 0,
                ligand,
            )
        pocket = source.pocket_path
        if pocket is None:
            pocket = work / "pocket.pdb"
            _extract_pocket(source.protein_path, ligand, pocket, pocket_radius)
        yield pocket, ligand


def _extract_sdf_record(source: Path, index: int, destination: Path) -> None:
    """Extract one zero-based molecule from a gzip SDF without unpacking all poses."""
    molecule: Chem.Mol | None = None
    try:
        with gzip.open(source, "rb") as stream:
            supplier = Chem.ForwardSDMolSupplier(
                stream, sanitize=False, removeHs=False, strictParsing=True
            )
            for record_index, candidate in enumerate(supplier):
                if record_index == index:
                    molecule = candidate
                    break
    except (OSError, RuntimeError) as error:
        raise DataValidationError(
            f"failed to read compressed SDF {source}: {error}"
        ) from error
    if molecule is None:
        raise DataValidationError(
            f"compressed SDF has no valid record {index}: {source}"
        )
    writer = Chem.SDWriter(str(destination))
    try:
        writer.write(molecule)
    finally:
        writer.close()


class _ResidueSelect(Select):
    """Select a frozen set of Biopython residue full identifiers."""

    def __init__(self, identifiers: set[tuple[Any, ...]]) -> None:
        self.identifiers = identifiers

    def accept_residue(self, residue: Any) -> bool:
        """Return whether one residue is inside the ligand distance cutoff."""
        return residue.get_full_id() in self.identifiers


def _extract_pocket(
    protein_path: Path,
    ligand_path: Path,
    destination: Path,
    radius: float,
) -> None:
    """Write complete protein residues having any atom within ligand cutoff."""
    structure = parse_pocket_pdb(protein_path)
    ligand = parse_ligand_sdf(ligand_path)
    ligand_positions = np.asarray(ligand.GetConformer().GetPositions(), dtype=float)
    selected: set[tuple[Any, ...]] = set()
    radius_squared = radius * radius
    for residue in structure.get_residues():
        if any(
            np.square(ligand_positions - np.asarray(atom.get_coord(), dtype=float))
            .sum(axis=1)
            .min()
            <= radius_squared
            for atom in residue.get_atoms()
        ):
            selected.add(residue.get_full_id())
    if not selected:
        raise DataValidationError(
            f"no protein residue lies within {radius:g} A of the ligand"
        )
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(destination), _ResidueSelect(selected))


def _protein_sequence(path: Path) -> str:
    """Extract deterministic chain-order amino-acid sequence from a receptor PDB."""
    structure = parse_pocket_pdb(path)
    residues: list[str] = []
    for residue in structure.get_residues():
        if not is_aa(residue, standard=False):
            continue
        residues.append(_AMINO_ACIDS.get(residue.get_resname().upper(), "X"))
    sequence = "".join(residues)
    if not sequence:
        raise DataValidationError(f"protein contains no amino-acid sequence: {path}")
    return sequence


def _ligand_scaffold(molecule: Chem.Mol) -> str:
    """Return canonical Murcko scaffold, or full identity for acyclic ligands."""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    target = scaffold if scaffold.GetNumAtoms() else molecule
    value = Chem.MolToSmiles(target, canonical=True, isomericSmiles=True)
    if not value:
        raise DataValidationError("ligand scaffold is empty")
    return value


def _resolve_pdbbind_index(root: Path, explicit: str | Path | None) -> Path:
    """Resolve one explicit or conventional PDBBind general INDEX file."""
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            raise DataValidationError(f"PDBBind INDEX does not exist: {candidate}")
        return candidate.resolve()
    candidates = sorted(
        {
            *root.glob("index/INDEX_general_PL_data.*"),
            *root.glob("INDEX_general_PL_data.*"),
        }
    )
    if not candidates:
        raise DataValidationError(
            "PDBBind INDEX_general_PL_data.* was not found under source root"
        )
    return candidates[-1].resolve()


def _resolve_crossdocked_types(root: Path, explicit: str | Path | None) -> Path:
    """Resolve one explicit or conventional CrossDocked completeset types file."""
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            raise DataValidationError(
                f"CrossDocked types file does not exist: {candidate}"
            )
        return candidate.resolve()
    preferred = root / "types" / "it2_tt_v1.1_completeset_train0.types"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(root.glob("types/*completeset*.types"))
    if not candidates:
        raise DataValidationError("CrossDocked completeset .types file was not found")
    return candidates[0].resolve()


def _safe_dataset_path(root: Path, value: str) -> Path:
    """Resolve an index path below the dataset root without traversal."""
    portable = PurePosixPath(value.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise DataValidationError(f"unsafe dataset-relative path: {value}")
    candidate = (root / Path(*portable.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise DataValidationError(
            f"dataset path escapes source root: {value}"
        ) from error
    return candidate


def _resolve_crossdocked_protein(virtual: Path) -> Path:
    """Resolve a direct or official suffix-indexed CrossDocked receptor path."""
    if virtual.is_file():
        return virtual
    match = re.match(r"^(?P<base>.+)_[0-9]+$", virtual.stem)
    candidate = virtual.with_name(f"{match.group('base')}.pdb") if match else virtual
    if not candidate.is_file():
        raise DataValidationError(f"CrossDocked receptor does not exist: {candidate}")
    return candidate


def _resolve_crossdocked_ligand(virtual: Path) -> tuple[Path, int | None]:
    """Resolve a direct SDF or compressed official multi-record ligand source."""
    if virtual.is_file():
        return virtual, None
    match = re.match(r"^(?P<base>.+)_(?P<index>[0-9]+)$", virtual.stem)
    if match is None:
        raise DataValidationError(f"cannot derive CrossDocked ligand record: {virtual}")
    archive = virtual.with_name(f"{match.group('base')}.sdf.gz")
    if not archive.is_file():
        raise DataValidationError(
            f"CrossDocked ligand archive does not exist: {archive}"
        )
    return archive, int(match.group("index"))


def _cluster_lookup(
    clusters: Mapping[str, str] | None, *identifiers: str
) -> str | None:
    """Return the first exact or case-folded external cluster match."""
    if clusters is None:
        return None
    for identifier in identifiers:
        for key in (identifier, identifier.casefold()):
            value = clusters.get(key)
            if value:
                return value
    return None


def _relative_path(path: Path, root: Path) -> str:
    """Return a stable POSIX dataset-relative source identity."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _optional_float(value: str) -> float | None:
    """Parse a finite optional numeric metadata field."""
    try:
        candidate = float(value)
    except ValueError:
        return None
    return candidate if math.isfinite(candidate) else None


def _issue_to_skip(issue: ImportIssue) -> SkipRecord:
    """Convert one discovery issue into a bounded manifest skip record."""
    message = " ".join(issue.message.split())[:240] or issue.category
    return SkipRecord(
        sample_id=issue.sample_id or "unknown",
        category=issue.category,
        message=message,
    )


def _sha256_file(path: Path) -> str:
    """Hash one immutable source file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_ordered_map(
    function: Callable[[Any], Any],
    values: Sequence[Any],
    workers: int,
) -> Iterator[Any]:
    """Apply work concurrently with deterministic order and bounded submission.

    :param function: Pure per-record callable.
    :param values: Stable input sequence.
    :param workers: Positive thread count.
    :return: Results in input order while at most ``2 * workers`` are pending.
    :rtype: Iterator[Any]

    ``Executor.map`` may eagerly submit an entire dataset on supported Python
    versions. This explicit queue bounds futures and their captured path/sample
    state, which is important when tens of thousands of complexes are imported.
    xTB work executes in subprocesses, so threads also permit useful parallelism
    without copying a large Python process for every worker.
    """
    if workers == 1:
        for value in values:
            yield function(value)
        return
    iterator = iter(values)
    pending: deque[Future[Any]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ecloudflow-import"
    ) as executor:
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


__all__ = [
    "AffinityMetadata",
    "DatasetFamily",
    "DiscoveryResult",
    "ImportIssue",
    "LocalComplexSource",
    "LocalImportOptions",
    "LocalImportResult",
    "discover_crossdocked",
    "discover_pdbbind",
    "import_local_dataset",
    "parse_pdbbind_affinity",
    "read_protein_clusters",
]
