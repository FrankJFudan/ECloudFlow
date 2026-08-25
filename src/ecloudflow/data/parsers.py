"""Strict protein/ligand parsing and complex tensor construction."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from Bio.PDB import PDBParser
from rdkit import Chem

from ecloudflow.chemistry.standardize import standardize_molecule
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import (
    ComplexSample,
    ElectronField,
    QMProvenance,
    SampleProvenance,
)
from ecloudflow.data.features import (
    ligand_graph_from_molecule,
    pocket_graph_from_entity,
)
from ecloudflow.ecloud.provenance import FieldBuilderBundle
from ecloudflow.exceptions import DataValidationError


def parse_pocket_pdb(path: str | Path) -> Any:
    """Parse one protein pocket PDB without selecting a fallback structure.

    :param path: Existing PDB path containing at least one atom.
    :return: Biopython ``Structure`` object with the source hierarchy intact.
    :rtype: Bio.PDB.Structure.Structure
    :raises DataValidationError: If the file is unreadable, malformed, or empty.
    """
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"pocket file does not exist: {source}")
    try:
        structure = PDBParser(QUIET=True).get_structure(source.stem, str(source))
        atom_count = sum(1 for _ in structure.get_atoms())
    except Exception as error:
        raise DataValidationError(
            f"failed to parse pocket PDB {source}: {error}"
        ) from error
    if atom_count == 0:
        raise DataValidationError("pocket contains no atoms")
    return structure


def parse_ligand_sdf(path: str | Path) -> Chem.Mol:
    """Parse exactly one sanitized ligand conformer from an SDF file.

    :param path: Existing SDF path containing exactly one bonded molecule.
    :return: Defensive standardized RDKit molecule retaining its conformer.
    :rtype: rdkit.Chem.Mol
    :raises DataValidationError: If the file has zero or multiple records,
        missing 3D coordinates, or unsanitizable chemistry.
    """
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"ligand file does not exist: {source}")
    try:
        records = list(Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False))
    except Exception as error:
        raise DataValidationError(
            f"failed to read ligand SDF {source}: {error}"
        ) from error
    molecules = [molecule for molecule in records if molecule is not None]
    if len(molecules) != 1 or len(records) != 1:
        raise DataValidationError("ligand SDF must contain exactly one valid molecule")
    molecule = molecules[0]
    try:
        standardized = standardize_molecule(molecule)
    except (TypeError, ValueError, RuntimeError) as error:
        raise DataValidationError(
            f"ligand chemistry is not sanitizable: {error}"
        ) from error
    if standardized.GetNumConformers() != 1 or not standardized.GetConformer().Is3D():
        raise DataValidationError("ligand must contain exactly one 3D conformer")
    return standardized


def build_complex_sample(
    pocket_path: str | Path,
    ligand_path: str | Path,
    sample_id: str,
    field_builders: FieldBuilderBundle | None = None,
    *,
    build_fields: bool = True,
    properties: Mapping[str, float | int | str | torch.Tensor] | None = None,
) -> ComplexSample:
    """Parse one cocrystal pair into the canonical model data contract.

    :param pocket_path: Protein-pocket PDB path in the original coordinate frame.
    :param ligand_path: SDF path containing exactly one bonded ligand conformer.
    :param sample_id: Stable source identifier stored in every artifact.
    :param field_builders: Optional pocket/ligand physical-field bundle.
    :param build_fields: Whether to build deterministic pocket and optional xTB
        ligand fields. Defaults to true per the physical-field contract; set it
        to false for graph-only preprocessing. xTB failures leave the ligand
        field absent with typed provenance.
    :param properties: Optional finite labels and non-empty measurement metadata
        copied into the immutable sample contract without interpretation.
    :return: Centered graph, optional fields, inverse frame, and provenance.
    :rtype: ComplexSample
    :raises DataValidationError: If files are unreadable or chemistry/geometry
        cannot form a valid sample. No alternate sample is ever selected.
    """
    if not sample_id:
        raise DataValidationError("sample_id must be non-empty")
    pocket_source, ligand_source = Path(pocket_path), Path(ligand_path)
    structure = parse_pocket_pdb(pocket_source)
    ligand = parse_ligand_sdf(ligand_source)
    pocket_atoms = list(structure.get_atoms())
    global_pocket = torch.stack(
        [
            torch.as_tensor(atom.get_coord(), dtype=torch.float32)
            for atom in pocket_atoms
        ]
    )
    frame = CoordinateFrame.from_pocket(global_pocket)
    pocket = pocket_graph_from_entity(structure, frame)
    ligand_global = torch.tensor(
        ligand.GetConformer().GetPositions(), dtype=torch.float32
    )
    ligand_graph = ligand_graph_from_molecule(ligand, frame)
    pocket_field: ElectronField | None = None
    ligand_field: ElectronField | None = None
    tool_versions: dict[str, str] = {}
    qm_provenance: QMProvenance | None = None
    if build_fields:
        bundle = field_builders or FieldBuilderBundle.default()
        pocket_field = _reframe_field(bundle.pocket_builder.build(pocket), frame)
        try:
            calculation_molecule = Chem.AddHs(Chem.Mol(ligand), addCoords=True)
            result = bundle.ligand_builder.calculate_ligand(
                calculation_molecule,
                charge=int(sum(atom.GetFormalCharge() for atom in ligand.GetAtoms())),
                multiplicity=1,
            )
            if getattr(result, "density", None) is not None:
                ligand_field = _reframe_field(result.density, frame)
            tool_versions["xTB"] = str(getattr(result, "status", "available"))
            tool = result.provenance
            qm_provenance = QMProvenance(
                status=str(result.status.value),
                qm_mask=bool(result.qm_mask),
                tool=tool.tool,
                version=tool.version,
                executable=tool.executable,
                command=tool.command,
                charge=tool.charge,
                multiplicity=tool.multiplicity,
                failure_category=str(result.failure_category.value),
                source_hashes=tool.source_hashes,
                integrated_electron_count=tool.integrated_electron_count,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            tool_versions["xTB"] = f"unavailable:{type(error).__name__}"
            runner = bundle.ligand_builder
            qm_provenance = QMProvenance(
                status="unavailable",
                qm_mask=False,
                tool="xTB",
                version="unavailable",
                executable=str(getattr(runner, "executable", "xtb")),
                command=(str(getattr(runner, "executable", "xtb")),),
                charge=int(sum(atom.GetFormalCharge() for atom in ligand.GetAtoms())),
                multiplicity=1,
                failure_category=type(error).__name__,
            )
    provenance = SampleProvenance(
        source_paths={
            "pocket": str(pocket_source.resolve()),
            "ligand": str(ligand_source.resolve()),
        },
        file_hashes={
            "pocket": _sha256(pocket_source),
            "ligand": _sha256(ligand_source),
        },
        tool_versions=tool_versions,
        preprocessing_status="complete",
        original_ligand_positions=ligand_global,
        qm=qm_provenance,
    )
    return ComplexSample(
        source_id=sample_id,
        pocket=pocket,
        ligand=ligand_graph,
        pocket_field=pocket_field,
        ligand_field=ligand_field,
        properties=dict(properties or {}),
        frame=frame,
        provenance=provenance,
    )


def _sha256(path: Path) -> str:
    """Compute a content hash for immutable source provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reframe_field(field: ElectronField, frame: CoordinateFrame) -> ElectronField:
    """Convert a field's coordinates into the sample frame and tensor contract.

    :param field: Electron field with an explicit source frame.
    :param frame: Target sample pocket frame.
    :return: Field with global positions transformed to the target frame, while
        preserving values, masks, batches, and channel names.
    :rtype: ElectronField
    :raises DataValidationError: If the source field omits a frame or uses an
        incompatible batch layout.
    """
    if field.frame is None:
        raise DataValidationError("field builder returned a field without a frame")
    global_positions = field.frame.to_global(field.positions)
    global_positions = global_positions.to(
        device=frame.origin.device, dtype=frame.origin.dtype
    )
    positions = frame.to_local(global_positions)
    return ElectronField(
        positions=positions,
        values=field.values.to(device=frame.origin.device, dtype=frame.origin.dtype),
        mask=field.mask.to(dtype=torch.bool),
        batch=field.batch.to(dtype=torch.long),
        channel_names=field.channel_names,
        frame=frame,
    )
