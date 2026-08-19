"""Deterministic, physically typed protein-pocket scalar fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdPartialCharges

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import ElectronField, PocketGraph

POCKET_CHANNELS = (
    "density",
    "partial_charge",
    "donor",
    "acceptor",
    "hydrophobic",
    "aromatic",
)

_KYTE_DOOLITTLE = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}
_AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})
_DONOR_ELEMENTS = frozenset({7, 8, 16})
_ACCEPTOR_ELEMENTS = frozenset({7, 8, 16})


@dataclass(frozen=True)
class _PocketAtoms:
    """Store normalized atom-level physical inputs in global coordinates."""

    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    partial_charges: torch.Tensor
    donors: torch.Tensor
    acceptors: torch.Tensor
    hydrophobicity: torch.Tensor
    aromatic: torch.Tensor


@dataclass(frozen=True)
class PocketFieldBuilder:
    """Build six smooth physical channels on a centered symmetric grid.

    :param spacing: Positive grid spacing in Å.
    :param padding: Non-negative radial padding beyond every atom in Å.
    :param chunk_size: Positive number of grid points evaluated per chunk.
    :param max_grid_points: Positive allocation guard for the flattened grid.
    :param gaussian_width_scale: Positive scale applied to half the element van
        der Waals radius when defining each Gaussian sigma.
    :return: Immutable deterministic builder configuration.
    :rtype: PocketFieldBuilder
    :raises ValueError: If a numerical configuration value is invalid.

    The density and partial-charge channels have units e/Å³. Density integrates
    to the sum of atomic numbers in an unbounded domain; partial charge integrates
    to the available Gasteiger/formal charges. Donor, acceptor, hydrophobic, and
    aromatic channels are normalized Gaussian feature densities in Å⁻³. The
    hydrophobic amplitude is normalized Kyte-Doolittle residue hydrophobicity
    when residue metadata exist and a conservative atom-type signal otherwise.
    """

    spacing: float = 1.0
    padding: float = 2.0
    chunk_size: int = 8192
    max_grid_points: int = 1_000_000
    gaussian_width_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the deterministic grid and kernel configuration.

        :return: None.
        :rtype: None
        :raises ValueError: If spacing, padding, counts, or width scale is invalid.
        """
        for name, value, positive in (
            ("spacing", self.spacing, True),
            ("padding", self.padding, False),
            ("gaussian_width_scale", self.gaussian_width_scale, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (value <= 0 if positive else value < 0)
            ):
                comparator = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be a {comparator} finite number.")
        for name, value in (
            ("chunk_size", self.chunk_size),
            ("max_grid_points", self.max_grid_points),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def default(
        cls,
        *,
        spacing: float = 1.0,
        padding: float = 2.0,
        chunk_size: int = 8192,
        max_grid_points: int = 1_000_000,
        gaussian_width_scale: float = 1.0,
    ) -> PocketFieldBuilder:
        """Construct a physical pocket-field builder with optional overrides.

        :param spacing: Positive grid spacing in Å.
        :param padding: Non-negative atom-to-boundary padding in Å.
        :param chunk_size: Positive grid evaluation chunk size.
        :param max_grid_points: Positive flattened-grid allocation guard.
        :param gaussian_width_scale: Positive element-width multiplier.
        :return: Validated deterministic builder.
        :rtype: PocketFieldBuilder
        :raises TypeError: If an unknown configuration key is supplied.
        :raises ValueError: If a supplied configuration value is invalid.
        """
        return cls(
            spacing=spacing,
            padding=padding,
            chunk_size=chunk_size,
            max_grid_points=max_grid_points,
            gaussian_width_scale=gaussian_width_scale,
        )

    def build(self, pocket: Chem.Mol | PocketGraph | Any) -> ElectronField:
        """Build a centered six-channel pocket field.

        :param pocket: One of: an RDKit molecule with exactly one finite 3D
            conformer; a validated :class:`PocketGraph` with atomic numbers;
            or a Biopython entity exposing ``get_atoms()``. RDKit inputs use
            Gasteiger charges and feature-factory donor/acceptor/aromatic flags.
            Biopython inputs use element, residue, atom-name, and optional
            ``pqr_charge`` metadata. Metals receive robust default widths and
            never require Gasteiger support.
        :return: ``ElectronField`` on a deterministic symmetric local grid,
            with frame origin at the global pocket-atom centroid, all points
            valid, batch zero, and channels ordered as :data:`POCKET_CHANNELS`.
        :rtype: ElectronField
        :raises TypeError: If the pocket type is unsupported.
        :raises ValueError: If atoms, atomic numbers, conformers, coordinates,
            or the requested grid allocation are invalid.

        The caller's molecule/structure is never modified. Gaussian evaluation
        is chunked, deterministic, device-local CPU float64 work. No files,
        subprocesses, caches containing molecular data, or approximate QM
        provenance are produced.
        """
        atoms = _normalize_pocket(pocket)
        output_dtype = torch.float64
        output_device = torch.device("cpu")
        if isinstance(pocket, PocketGraph):
            output_dtype = pocket.positions.dtype
            output_device = pocket.positions.device
            if pocket.frame is None:
                frame = CoordinateFrame.from_pocket(pocket.positions)
                local_atoms = (
                    (pocket.positions - frame.origin)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                )
            else:
                frame = pocket.frame
                local_atoms = atoms.positions
        else:
            frame = CoordinateFrame.from_pocket(atoms.positions)
            local_atoms = frame.to_local(atoms.positions)
        grid = self._symmetric_grid(local_atoms)
        widths = _element_widths(atoms.atomic_numbers) * self.gaussian_width_scale
        amplitudes = torch.stack(
            (
                atoms.atomic_numbers.to(torch.float64),
                atoms.partial_charges,
                atoms.donors,
                atoms.acceptors,
                atoms.hydrophobicity,
                atoms.aromatic,
            ),
            dim=-1,
        )
        values = torch.empty((grid.shape[0], len(POCKET_CHANNELS)), dtype=torch.float64)
        normalization = (2.0 * math.pi) ** -1.5 / widths.pow(3)
        for start in range(0, grid.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, grid.shape[0])
            squared_distances = (
                (grid[start:stop, None, :] - local_atoms[None, :, :])
                .square()
                .sum(dim=-1)
            )
            kernels = torch.exp(
                -0.5 * squared_distances / widths.square().unsqueeze(0)
            ) * normalization.unsqueeze(0)
            values[start:stop] = kernels @ amplitudes
        return ElectronField(
            positions=grid.to(device=output_device, dtype=output_dtype),
            values=values.to(device=output_device, dtype=output_dtype),
            mask=torch.ones(grid.shape[0], dtype=torch.bool, device=output_device),
            batch=torch.zeros(grid.shape[0], dtype=torch.long, device=output_device),
            channel_names=POCKET_CHANNELS,
            frame=frame,
        )

    def _symmetric_grid(self, local_atoms: torch.Tensor) -> torch.Tensor:
        """Create a centered odd-sized Cartesian grid around local atoms."""
        extents = local_atoms.abs().amax(dim=0) + self.padding
        half_counts = torch.ceil(extents / self.spacing).to(torch.long)
        shape = tuple(int(2 * count + 1) for count in half_counts)
        point_count = math.prod(shape)
        if point_count > self.max_grid_points:
            raise ValueError(
                f"pocket grid requires {point_count} points, exceeding "
                f"max_grid_points={self.max_grid_points}."
            )
        axes = [
            torch.arange(-count, count + 1, dtype=torch.float64) * self.spacing
            for count in half_counts.tolist()
        ]
        return torch.cartesian_prod(*axes)


def _normalize_pocket(pocket: Chem.Mol | PocketGraph | Any) -> _PocketAtoms:
    """Convert supported chemistry structures to atom-level physical inputs."""
    if isinstance(pocket, Chem.Mol):
        return _from_rdkit(pocket)
    if isinstance(pocket, PocketGraph):
        return _from_pocket_graph(pocket)
    if hasattr(pocket, "get_atoms") and callable(pocket.get_atoms):
        return _from_biopython(pocket)
    raise TypeError("pocket must be an RDKit Mol, PocketGraph, or Biopython entity.")


def _from_rdkit(molecule: Chem.Mol) -> _PocketAtoms:
    """Extract chemistry-aware fields from an RDKit 3D molecule copy."""
    if molecule.GetNumAtoms() == 0 or molecule.GetNumConformers() != 1:
        raise ValueError("RDKit pocket must contain atoms and exactly one conformer.")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError("RDKit pocket conformer must be three-dimensional.")
    positions = torch.tensor(conformer.GetPositions(), dtype=torch.float64)
    _validate_positions(positions)
    copy = Chem.Mol(molecule)
    if Chem.SanitizeMol(copy, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError("RDKit pocket must be sanitizable.")
    try:
        rdPartialCharges.ComputeGasteigerCharges(copy, throwOnParamFailure=False)
    except (RuntimeError, ValueError):
        pass
    charges = []
    for atom in copy.GetAtoms():
        value = (
            atom.GetDoubleProp("_GasteigerCharge")
            if atom.HasProp("_GasteigerCharge")
            else float(atom.GetFormalCharge())
        )
        charges.append(value if math.isfinite(value) else float(atom.GetFormalCharge()))
    donor_indices: set[int] = set()
    acceptor_indices: set[int] = set()
    for feature in _feature_factory().GetFeaturesForMol(copy):
        if feature.GetFamily() == "Donor":
            donor_indices.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptor_indices.update(feature.GetAtomIds())
    hydrophobicity = []
    for atom in copy.GetAtoms():
        info = atom.GetPDBResidueInfo()
        residue = "" if info is None else info.GetResidueName().strip().upper()
        if residue in _KYTE_DOOLITTLE:
            hydrophobicity.append(_KYTE_DOOLITTLE[residue] / 4.5)
        else:
            hydrophobicity.append(
                1.0
                if atom.GetAtomicNum() in {6, 16} and atom.GetFormalCharge() == 0
                else 0.0
            )
    atom_count = copy.GetNumAtoms()
    return _PocketAtoms(
        positions=positions,
        atomic_numbers=torch.tensor(
            [atom.GetAtomicNum() for atom in copy.GetAtoms()], dtype=torch.long
        ),
        partial_charges=torch.tensor(charges, dtype=torch.float64),
        donors=_index_flags(atom_count, donor_indices),
        acceptors=_index_flags(atom_count, acceptor_indices),
        hydrophobicity=torch.tensor(hydrophobicity, dtype=torch.float64),
        aromatic=torch.tensor(
            [float(atom.GetIsAromatic()) for atom in copy.GetAtoms()],
            dtype=torch.float64,
        ),
    )


def _from_pocket_graph(pocket: PocketGraph) -> _PocketAtoms:
    """Extract a conservative physical default from the canonical graph."""
    if pocket.atom_numbers is None:
        raise ValueError("PocketGraph.atom_numbers is required for a physical field.")
    if torch.unique(pocket.batch).numel() != 1:
        raise ValueError("PocketFieldBuilder.build accepts one pocket graph at a time.")
    positions = pocket.positions.detach().to(device="cpu", dtype=torch.float64)
    atomic_numbers = pocket.atom_numbers.detach().to(device="cpu", dtype=torch.long)
    _validate_atomic_numbers(atomic_numbers)
    donors = torch.isin(atomic_numbers, torch.tensor(tuple(_DONOR_ELEMENTS))).to(
        torch.float64
    )
    acceptors = torch.isin(atomic_numbers, torch.tensor(tuple(_ACCEPTOR_ELEMENTS))).to(
        torch.float64
    )
    hydrophobicity = torch.isin(atomic_numbers, torch.tensor([6, 16])).to(torch.float64)
    count = positions.shape[0]
    return _PocketAtoms(
        positions=positions,
        atomic_numbers=atomic_numbers,
        partial_charges=torch.zeros(count, dtype=torch.float64),
        donors=donors,
        acceptors=acceptors,
        hydrophobicity=hydrophobicity,
        aromatic=torch.zeros(count, dtype=torch.float64),
    )


def _from_biopython(entity: Any) -> _PocketAtoms:
    """Extract element and residue physics from a Biopython entity."""
    atoms = list(entity.get_atoms())
    if not atoms:
        raise ValueError("Biopython pocket must contain at least one atom.")
    periodic_table = Chem.GetPeriodicTable()
    positions = []
    atomic_numbers = []
    charges = []
    donors = []
    acceptors = []
    hydrophobicity = []
    aromatic = []
    for atom in atoms:
        element = str(getattr(atom, "element", "") or "").strip().title()
        try:
            atomic_number = int(periodic_table.GetAtomicNumber(element))
        except RuntimeError:
            atomic_number = 0
        if atomic_number <= 0:
            raise ValueError(
                f"unsupported or missing Biopython atom element: {element!r}."
            )
        coordinate = torch.as_tensor(atom.get_coord(), dtype=torch.float64)
        positions.append(coordinate)
        atomic_numbers.append(atomic_number)
        charge = getattr(atom, "pqr_charge", None)
        charges.append(
            float(charge)
            if charge is not None and math.isfinite(float(charge))
            else 0.0
        )
        residue = atom.get_parent()
        residue_name = str(residue.get_resname()).strip().upper()
        atom_name = str(atom.get_name()).strip().upper()
        donors.append(
            float(atomic_number in _DONOR_ELEMENTS and atom_name not in {"O", "OXT"})
        )
        acceptors.append(
            float(atomic_number in _ACCEPTOR_ELEMENTS and atom_name not in {"NZ"})
        )
        hydrophobicity.append(_KYTE_DOOLITTLE.get(residue_name, 0.0) / 4.5)
        aromatic.append(
            float(residue_name in _AROMATIC_RESIDUES and atomic_number in {6, 7})
        )
    position_tensor = torch.stack(positions)
    _validate_positions(position_tensor)
    return _PocketAtoms(
        positions=position_tensor,
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        partial_charges=torch.tensor(charges, dtype=torch.float64),
        donors=torch.tensor(donors, dtype=torch.float64),
        acceptors=torch.tensor(acceptors, dtype=torch.float64),
        hydrophobicity=torch.tensor(hydrophobicity, dtype=torch.float64),
        aromatic=torch.tensor(aromatic, dtype=torch.float64),
    )


@lru_cache(maxsize=1)
def _feature_factory() -> ChemicalFeatures.MolChemicalFeatureFactory:
    """Load RDKit's built-in pharmacophore feature definitions once."""
    return ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )


def _index_flags(count: int, indices: set[int]) -> torch.Tensor:
    """Convert an atom-index set to deterministic float64 indicator values."""
    flags = torch.zeros(count, dtype=torch.float64)
    if indices:
        flags[torch.tensor(sorted(indices), dtype=torch.long)] = 1.0
    return flags


def _element_widths(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return robust element Gaussian sigmas from van der Waals radii."""
    _validate_atomic_numbers(atomic_numbers)
    table = Chem.GetPeriodicTable()
    widths = []
    for atomic_number in atomic_numbers.tolist():
        try:
            radius = float(table.GetRvdw(int(atomic_number)))
        except RuntimeError:
            radius = 1.8
        if not math.isfinite(radius) or radius <= 0.0:
            radius = 1.8
        widths.append(max(0.35, radius / 2.0))
    return torch.tensor(widths, dtype=torch.float64)


def _validate_positions(positions: torch.Tensor) -> None:
    """Validate a non-empty finite Cartesian coordinate tensor."""
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 3:
        raise ValueError("pocket positions must have shape [P, 3] for P >= 1.")
    if not torch.isfinite(positions).all():
        raise ValueError("pocket positions must be finite.")


def _validate_atomic_numbers(atomic_numbers: torch.Tensor) -> None:
    """Validate supported positive atomic numbers including metals."""
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() == 0:
        raise ValueError("atomic numbers must be a non-empty vector.")
    if bool((atomic_numbers <= 0).any()) or bool((atomic_numbers > 118).any()):
        raise ValueError("atomic numbers must be in [1, 118].")
