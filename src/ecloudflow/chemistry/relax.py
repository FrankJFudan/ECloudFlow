"""Optional constrained pose relaxation with strict raw/relaxed separation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem


class RelaxationStatus(str):
    """String status values for the optional force-field boundary."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class RelaxationResult:
    """Return a relaxed molecule and explicit force-field status.

    :param molecule: Defensive RDKit molecule copy containing the relaxed
        conformer.  The input raw molecule is never mutated.
    :param status: ``converged``, ``max_iterations``, ``unavailable``, or
        ``failed``.
    :param method: Force-field method used, normally ``MMFF`` or ``UFF``.
    :param iterations: Requested maximum iteration count.
    :param reason: Optional diagnostic message.
    :return: Immutable relaxation result.
    :rtype: RelaxationResult
    """

    molecule: Chem.Mol
    status: str
    method: str
    iterations: int
    reason: str = ""


def relax_molecule(
    molecule: Chem.Mol,
    *,
    fixed_atom_mask: torch.Tensor | None = None,
    method: str = "MMFF",
    max_iterations: int = 200,
    return_result: bool = False,
) -> Chem.Mol | RelaxationResult:
    """Relax a pose while pinning an optional fixed fragment.

    :param molecule: RDKit molecule with a 3D conformer.  It is copied before
        force-field setup and is never modified in place.
    :param fixed_atom_mask: Optional boolean ``[N]`` mask.  Fixed atoms are
        registered as force-field fixed points and their coordinates are
        restored bitwise after optimization as a defensive final clamp.
    :param method: ``"MMFF"`` (default) or ``"UFF"``.
    :param max_iterations: Positive optimizer iteration bound.
    :param return_result: Return :class:`RelaxationResult` with status instead
        of only the molecule.  The simple molecule return is the public default.
    :return: Relaxed defensive molecule, or a typed status result when requested.
    :rtype: rdkit.Chem.Mol | RelaxationResult
    :raises TypeError: If ``molecule`` is not an RDKit molecule.
    :raises ValueError: If the method, mask, iteration bound, or conformer is
        invalid.

    Raw and relaxed poses are separate objects.  A missing force field or a
    failed optimization is reported explicitly and still returns a copy, so a
    caller can preserve the raw artifact and decide whether to reject the
    relaxed artifact.
    """
    if not isinstance(molecule, Chem.Mol):
        raise TypeError("molecule must be an RDKit Mol.")
    if molecule.GetNumConformers() == 0:
        raise ValueError("molecule must contain a 3D conformer.")
    if not isinstance(method, str) or method.upper() not in {"MMFF", "UFF"}:
        raise ValueError("method must be 'MMFF' or 'UFF'.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    mask = _validate_mask(fixed_atom_mask, molecule.GetNumAtoms())
    raw_positions = _get_positions(molecule)
    relaxed = Chem.Mol(molecule)
    status = RelaxationStatus.FAILED
    reason = ""
    try:
        force_field = _build_force_field(relaxed, method.upper(), max_iterations)
    except (RuntimeError, ValueError) as error:
        force_field = None
        reason = str(error)
    if force_field is None:
        status = (
            RelaxationStatus.UNAVAILABLE
            if method.upper() == "MMFF"
            else RelaxationStatus.FAILED
        )
    else:
        if mask is not None:
            for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                force_field.AddFixedPoint(int(index))
        try:
            result_code = int(force_field.Minimize(maxIts=max_iterations))
            status = (
                RelaxationStatus.CONVERGED
                if result_code == 0
                else RelaxationStatus.MAX_ITERATIONS
            )
        except (RuntimeError, ValueError) as error:
            status = RelaxationStatus.FAILED
            reason = str(error)
    if mask is not None:
        _set_positions(relaxed, raw_positions, mask)
    result = RelaxationResult(relaxed, status, method.upper(), max_iterations, reason)
    return result if return_result else result.molecule


def write_raw_and_relaxed(
    molecule: Chem.Mol,
    output_dir: str | Path,
    *,
    fixed_atom_mask: torch.Tensor | None = None,
    method: str = "MMFF",
    max_iterations: int = 200,
) -> tuple[Path, Path]:
    """Write immutable ``raw.sdf`` and separate ``relaxed.sdf`` artifacts.

    :param molecule: Generated RDKit molecule with its raw binding pose.
    :param output_dir: Directory created when necessary for the two artifacts.
    :param fixed_atom_mask: Optional exact fixed-fragment atom mask passed to
        :func:`relax_molecule`.
    :param method: Force-field method used for the separate relaxed artifact.
    :param max_iterations: Positive force-field iteration bound.
    :return: ``(raw_path, relaxed_path)`` in deterministic file-name order.
    :rtype: tuple[pathlib.Path, pathlib.Path]
    :raises ValueError: If the molecule or relaxation configuration is invalid.

    The raw molecule is serialized before optimization and from a defensive
    copy.  The relaxed writer uses a separate copy and can therefore never
    overwrite or mutate the raw SDF, even when optimization fails.
    """
    if not isinstance(molecule, Chem.Mol):
        raise TypeError("molecule must be an RDKit Mol.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw.sdf"
    relaxed_path = destination / "relaxed.sdf"
    _write_molecule(Chem.Mol(molecule), raw_path)
    relaxed = relax_molecule(
        molecule,
        fixed_atom_mask=fixed_atom_mask,
        method=method,
        max_iterations=max_iterations,
    )
    assert isinstance(relaxed, Chem.Mol)
    _write_molecule(relaxed, relaxed_path)
    return raw_path, relaxed_path


def _build_force_field(molecule: Chem.Mol, method: str, max_iterations: int):
    """Construct a force field, returning ``None`` when MMFF parameters miss."""
    if method == "MMFF":
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        if properties is None:
            return None
        return AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0)
    return AllChem.UFFGetMoleculeForceField(molecule, confId=0)


def _validate_mask(mask: torch.Tensor | None, atom_count: int) -> torch.Tensor | None:
    """Validate and detach a fixed atom mask."""
    if mask is None:
        return None
    if mask.dtype != torch.bool or mask.shape != (atom_count,):
        raise ValueError("fixed_atom_mask must have boolean shape [N].")
    return mask.detach().to(device="cpu")


def _get_positions(molecule: Chem.Mol) -> torch.Tensor:
    """Copy conformer coordinates as float64 CPU values."""
    return torch.tensor(molecule.GetConformer(0).GetPositions(), dtype=torch.float64)


def _set_positions(molecule: Chem.Mol, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Restore fixed coordinates exactly after force-field optimization."""
    conformer = molecule.GetConformer(0)
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        x, y, z = (float(value) for value in raw[index].tolist())
        conformer.SetAtomPosition(int(index), (x, y, z))


def _write_molecule(molecule: Chem.Mol, path: Path) -> None:
    """Write one molecule with an explicit writer close boundary."""
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()


__all__ = [
    "RelaxationResult",
    "RelaxationStatus",
    "relax_molecule",
    "write_raw_and_relaxed",
]
