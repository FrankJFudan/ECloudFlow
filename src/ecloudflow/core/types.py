"""Immutable, validated tensor contracts shared across ECloudFlow components."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.exceptions import ContractValidationError

TensorProperty: TypeAlias = float | int | torch.Tensor
SampleProperty: TypeAlias = TensorProperty | str


@dataclass(frozen=True)
class QMProvenance:
    """Store serializable, credential-free metadata for one QM attempt.

    :param status: Stable status string such as ``"success"`` or ``"tool_missing"``.
    :param qm_mask: True only when a genuine QM density was accepted.
    :param tool: External tool name.
    :param version: Tool version or ``"unavailable"``.
    :param executable: Executable name/path supplied to the runner.
    :param command: Exact argument-list command without environment values.
    :param charge: Molecular charge.
    :param multiplicity: Spin multiplicity.
    :param failure_category: Sanitized failure category.
    :param source_hashes: Immutable hashes of inputs and captured streams.
    :param integrated_electron_count: Accepted electron count, if available.
    :return: Immutable provenance record safe for manifests and reports.
    :rtype: QMProvenance
    """

    status: str
    qm_mask: bool
    tool: str
    version: str
    executable: str
    command: tuple[str, ...]
    charge: int
    multiplicity: int
    failure_category: str
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    integrated_electron_count: float | None = None

    def __post_init__(self) -> None:
        """Validate and freeze credential-free QM metadata."""
        if not all(
            isinstance(value, str) and value
            for value in (
                self.status,
                self.tool,
                self.version,
                self.executable,
                self.failure_category,
            )
        ):
            raise ContractValidationError(
                "QM provenance string fields must be non-empty."
            )
        if not self.command or any(not value for value in self.command):
            raise ContractValidationError(
                "QM provenance command must be a non-empty argument list."
            )
        if (
            not isinstance(self.charge, int)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity <= 0
        ):
            raise ContractValidationError(
                "QM provenance charge/multiplicity are invalid."
            )
        object.__setattr__(
            self, "source_hashes", _freeze_string_mapping(self.source_hashes)
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Describe lossless CPU-process transport for QM provenance.

        :return: The validated record type and its complete constructor argument
            tuple, with the read-only source-hash view copied to a plain mapping.
        :rtype: tuple[Any, tuple[Any, ...]]
        :raises ContractValidationError: During receiver reconstruction if
            transported QM metadata violates the canonical provenance contract.

        Pickle invokes this hook only for process/checkpoint transport. It does
        not run xTB, alter ``qm_mask``, move tensors between devices, or mutate
        this frozen record. Reconstruction calls :class:`QMProvenance`, which
        revalidates command, charge, multiplicity, status, and source hashes and
        restores an immutable mapping. Malformed transported values therefore
        fail through the normal provenance contract rather than bypassing it.
        """
        return (
            type(self),
            (
                self.status,
                self.qm_mask,
                self.tool,
                self.version,
                self.executable,
                self.command,
                self.charge,
                self.multiplicity,
                self.failure_category,
                dict(self.source_hashes),
                self.integrated_electron_count,
            ),
        )


@dataclass(frozen=True)
class PocketGraph:
    """Represent a flattened protein-pocket graph in a local pocket frame.

    :param positions: Pocket atom coordinates with shape ``[P, 3]`` in
        angstroms and the centered pocket frame, floating dtype/device.
    :param features: Per-pocket-atom features with shape ``[P, F]``, floating
        dtype/device matching ``positions``.
    :param batch: Complex index per pocket atom with shape ``[P]``,
        ``torch.long`` dtype/device matching ``positions``.
    :param atom_numbers: Optional atomic numbers with shape ``[P]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param frame: Optional centered pocket coordinate frame. When supplied it
        must have the same floating dtype/device as ``positions``.
    :return: Immutable canonical pocket graph.
    :rtype: PocketGraph
    :raises ContractValidationError: If ranks, dtypes, devices, finite values,
        or shared leading dimensions are invalid.

    Edge neighborhoods are constructed by model components from coordinates;
    this contract intentionally stores no stale radius or kNN edge cache.
    """

    positions: torch.Tensor
    features: torch.Tensor
    batch: torch.Tensor
    atom_numbers: torch.Tensor | None = None
    frame: CoordinateFrame | None = None

    def __post_init__(self) -> None:
        """Validate pocket tensor shapes and compatibility.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If the graph does not satisfy its
            flattened tensor contract.
        """
        count = _validate_positions(self.positions, "positions", nonempty=True)
        _validate_float_matrix(self.features, "features", count, self.positions)
        _validate_index_vector(self.batch, "batch", count, self.positions)
        if self.atom_numbers is not None:
            _validate_index_vector(
                self.atom_numbers, "atom_numbers", count, self.positions
            )
        if self.frame is not None and (
            self.frame.origin.dtype != self.positions.dtype
            or self.frame.origin.device != self.positions.device
        ):
            raise ContractValidationError(
                "frame must have the same dtype and device as pocket positions."
            )


@dataclass(frozen=True)
class LigandGraph:
    """Represent a discrete ligand graph using canonical unordered halfedges.

    :param positions: Ligand coordinates with shape ``[N, 3]`` in angstroms
        and a centered pocket frame, floating dtype/device.
    :param atom_types: Atom vocabulary indices with shape ``[N]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param formal_charges: Formal charge values with shape ``[N]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param halfedge_index: One unordered pair per ligand bond candidate with
        shape ``[2, E]``, ``torch.long`` dtype/device, and strictly increasing
        endpoint rows ``halfedge_index[0] < halfedge_index[1]``.
    :param bond_types: Bond vocabulary indices with shape ``[E]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param batch: Complex index per ligand atom with shape ``[N]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :return: Immutable canonical discrete ligand graph.
    :rtype: LigandGraph
    :raises ContractValidationError: If graph ranks, index topology, or batch
        membership disagree.

    A halfedge is stored once for each unordered candidate pair. Dense
    ``[N, N, B]`` bond tensors and duplicate reversed edges are forbidden.
    """

    positions: torch.Tensor
    atom_types: torch.Tensor
    formal_charges: torch.Tensor
    halfedge_index: torch.Tensor
    bond_types: torch.Tensor
    batch: torch.Tensor

    def __post_init__(self) -> None:
        """Validate ligand graph tensors and canonical bond topology.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If the ligand graph violates a
            flattened canonical halfedge invariant.
        """
        count = _validate_positions(self.positions, "positions", nonempty=True)
        _validate_index_vector(self.atom_types, "atom_types", count, self.positions)
        _validate_signed_long_vector(
            self.formal_charges, "formal_charges", count, self.positions
        )
        _validate_index_vector(self.batch, "batch", count, self.positions)
        edge_count = _validate_halfedge_index(
            self.halfedge_index, count, self.positions
        )
        _validate_index_vector(
            self.bond_types, "bond_types", edge_count, self.positions
        )
        _validate_halfedge_batches(self.halfedge_index, self.batch, None)


@dataclass(frozen=True)
class ElectronField:
    """Represent sampled or coefficient electron-field channels on a point set.

    :param positions: Field point coordinates with shape ``[G, 3]`` in
        angstroms in ``frame`` when provided, floating dtype/device.
    :param values: Field-channel values or coefficients with shape ``[G, C]``,
        floating dtype/device matching ``positions``.
    :param mask: Valid field-point mask with shape ``[G]``, ``torch.bool``
        dtype/device matching ``positions``.
    :param batch: Complex index per field point with shape ``[G]``,
        ``torch.long`` dtype/device matching ``positions``.
    :param channel_names: Immutable names for ``C`` field channels; empty names
        are permitted when a learned latent has no semantic channels.
    :param frame: Optional coordinate frame that defines ``positions``.
    :return: Immutable field tensor contract.
    :rtype: ElectronField
    :raises ContractValidationError: If ranks, dtypes, devices, channels, or
        finite values are invalid.

    Masked points remain allocated for batching and must be ignored by losses
    and aggregations. The contract does not prescribe a grid topology.
    """

    positions: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    batch: torch.Tensor
    channel_names: tuple[str, ...] = ()
    frame: CoordinateFrame | None = None

    def __post_init__(self) -> None:
        """Validate field tensors and channel metadata.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If field tensors have inconsistent
            leading dimensions, channel names, or coordinate metadata.
        """
        count = _validate_positions(self.positions, "positions", nonempty=False)
        channels = _validate_float_matrix(self.values, "values", count, self.positions)
        _validate_bool_vector(self.mask, "mask", count, self.positions)
        _validate_index_vector(self.batch, "batch", count, self.positions)
        if self.channel_names and len(self.channel_names) != channels:
            raise ContractValidationError(
                "channel_names must be empty or have one name per value channel."
            )
        if any(not name for name in self.channel_names):
            raise ContractValidationError("channel_names must not contain empty names.")
        if self.frame is not None and (
            self.frame.origin.device != self.positions.device
            or self.frame.origin.dtype != self.positions.dtype
        ):
            raise ContractValidationError(
                "frame must have the same dtype and device as field positions."
            )


@dataclass(frozen=True)
class MolecularState:
    """Represent one batched ligand state along a generative trajectory.

    :param positions: Flattened coordinates with shape ``[N, 3]`` in angstroms
        in centered pocket frames, floating dtype/device.
    :param atom_logits: Flattened atom-type values with shape ``[N, A]`` and a
        floating dtype/device matching ``positions``.
    :param charge_logits: Flattened formal-charge values with shape ``[N, Q]``
        and a floating dtype/device matching ``positions``.
    :param halfedge_index: Canonical unordered pairs with shape ``[2, E]`` and
        ``torch.long`` dtype/device; row-zero indices are strictly smaller than
        row-one indices.
    :param bond_logits: One bond-class vector per halfedge with shape ``[E, B]``
        and a floating dtype/device matching ``positions``.
    :param electron_latent: Equivariant field tokens with shape ``[N, C]`` and
        a floating dtype/device matching ``positions``.
    :param node_batch: Complex index per node with shape ``[N]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param halfedge_batch: Complex index per halfedge with shape ``[E]`` and
        ``torch.long`` dtype/device matching ``positions``.
    :param frame: Optional centered pocket coordinate frame for ``positions``.
        When supplied, it must share the positions floating dtype/device.
    :return: Immutable molecular state used by training and sampling.
    :rtype: MolecularState
    :raises ContractValidationError: If node counts, ranks, finite values,
        devices, batches, or canonical bond topology disagree.

    The canonical bond representation stores one shared value for every
    unordered pair. It never materializes a dense ``[N, N, B]`` tensor;
    symmetric dense matrices are deferred to final decoding or reporting.
    """

    positions: torch.Tensor
    atom_logits: torch.Tensor
    charge_logits: torch.Tensor
    halfedge_index: torch.Tensor
    bond_logits: torch.Tensor
    electron_latent: torch.Tensor
    node_batch: torch.Tensor
    halfedge_batch: torch.Tensor
    frame: CoordinateFrame | None = None

    def __post_init__(self) -> None:
        """Validate state tensor ranks, batches, and halfedge invariants.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If flattened state tensors cannot
            represent a valid batched molecular trajectory state.
        """
        count = _validate_positions(self.positions, "positions", nonempty=False)
        _validate_float_matrix(self.atom_logits, "atom_logits", count, self.positions)
        _validate_float_matrix(
            self.charge_logits, "charge_logits", count, self.positions
        )
        _validate_float_matrix(
            self.electron_latent, "electron_latent", count, self.positions
        )
        _validate_index_vector(self.node_batch, "node_batch", count, self.positions)
        edge_count = _validate_halfedge_index(
            self.halfedge_index, count, self.positions
        )
        _validate_float_matrix(
            self.bond_logits, "bond_logits", edge_count, self.positions
        )
        _validate_index_vector(
            self.halfedge_batch,
            "halfedge_batch",
            edge_count,
            self.positions,
        )
        _validate_halfedge_batches(
            self.halfedge_index, self.node_batch, self.halfedge_batch
        )
        if self.frame is not None and (
            self.frame.origin.dtype != self.positions.dtype
            or self.frame.origin.device != self.positions.device
        ):
            raise ContractValidationError(
                "frame must have the same dtype and device as state positions."
            )

    def replace(
        self, **changes: torch.Tensor | CoordinateFrame | None
    ) -> MolecularState:
        """Return a revalidated state with selected tensor fields replaced.

        :param changes: Keyword tensor or coordinate-frame replacements for
            dataclass fields.
        :return: New immutable molecular state after all invariants are checked.
        :rtype: MolecularState
        :raises TypeError: If a replacement is not a molecular-state field.
        :raises ContractValidationError: If replacements break tensor, batch, or
            canonical unordered-halfedge invariants.

        This method preserves autograd connectivity because it neither detaches
        nor copies tensors. The original state remains bound to its original
        tensor attributes.
        """
        return dataclasses.replace(self, **cast(Any, changes))


@dataclass(frozen=True)
class FragmentCondition:
    """Specify exact fixed fields for a positioned fragment generation task.

    :param reference: Clean canonical molecular state supplying immutable values
        in centered pocket frames.
    :param fixed_atom_mask: Boolean atom-identity and charge mask with shape
        ``[N]`` on the reference device.
    :param fixed_bond_mask: Boolean internal-halfedge mask with shape ``[E]``
        on the reference device. It selects immutable bonds within each fixed
        component. When ``component_ids`` is omitted, every fixed-fixed pair
        is internal; when component labels are supplied, fixed-fixed pairs
        across components remain editable so link/merge tasks can connect them.
    :param fixed_coord_mask: Boolean coordinate mask with shape ``[N]`` on the
        reference device. It must exactly equal ``fixed_atom_mask`` so fixed
        coordinates are restored and free coordinates remain generative.
    :param attachment_mask: Boolean atom mask with shape ``[N]`` identifying
        allowed fixed-to-free attachment sites on the reference device.
    :param component_ids: Optional ``torch.long`` component labels with shape
        ``[N]`` on the reference device.
    :param task_id: Stable fragment task identifier such as ``"grow"``.
    :return: Immutable exact fragment preservation contract.
    :rtype: FragmentCondition
    :raises ContractValidationError: If masks, batches, devices, or exact
        internal-bond semantics are invalid.

    Clamping restores fixed atom identities, charges, coordinates, and every
    internal unordered halfedge from ``reference`` after each sampling substep.
    """

    reference: MolecularState
    fixed_atom_mask: torch.Tensor
    fixed_bond_mask: torch.Tensor
    fixed_coord_mask: torch.Tensor
    attachment_mask: torch.Tensor
    component_ids: torch.Tensor | None = None
    task_id: str = "fixed_pose"

    def __post_init__(self) -> None:
        """Validate fragment mask shapes and exact internal bond selection.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If masks cannot be applied exactly to
            the reference molecular state.
        """
        node_count = self.reference.positions.shape[0]
        edge_count = self.reference.halfedge_index.shape[1]
        _validate_bool_vector(
            self.fixed_atom_mask,
            "fixed_atom_mask",
            node_count,
            self.reference.positions,
        )
        _validate_bool_vector(
            self.fixed_coord_mask,
            "fixed_coord_mask",
            node_count,
            self.reference.positions,
        )
        _validate_bool_vector(
            self.attachment_mask,
            "attachment_mask",
            node_count,
            self.reference.positions,
        )
        _validate_bool_vector(
            self.fixed_bond_mask,
            "fixed_bond_mask",
            edge_count,
            self.reference.positions,
        )
        if self.component_ids is not None:
            _validate_index_vector(
                self.component_ids,
                "component_ids",
                node_count,
                self.reference.positions,
            )
        if not self.task_id:
            raise ContractValidationError("task_id must be a non-empty string.")
        expected_bond_mask = _internal_bond_mask(
            self.reference.halfedge_index,
            self.fixed_atom_mask,
            self.component_ids,
        )
        if not torch.equal(self.fixed_bond_mask, expected_bond_mask):
            raise ContractValidationError(
                "fixed_bond_mask must select exactly the internal fixed halfedges."
            )
        if not torch.equal(self.fixed_coord_mask, self.fixed_atom_mask):
            raise ContractValidationError(
                "fixed_coord_mask must exactly equal fixed_atom_mask."
            )
        if bool((self.attachment_mask & ~self.fixed_atom_mask).any()):
            raise ContractValidationError(
                "attachment_mask may contain only fixed fragment atoms."
            )

    @classmethod
    def from_atom_mask(
        cls,
        fixed_atom_mask: torch.Tensor,
        reference: MolecularState,
        *,
        fixed_coord_mask: torch.Tensor | None = None,
        attachment_mask: torch.Tensor | None = None,
        component_ids: torch.Tensor | None = None,
        task_id: str = "fixed_pose",
    ) -> FragmentCondition:
        """Create exact halfedge masks directly from an atom preservation mask.

        :param fixed_atom_mask: Boolean atom mask with shape ``[N]`` on the
            reference device; true entries preserve atom type and charge.
        :param reference: Clean canonical state that supplies values to clamp.
        :param fixed_coord_mask: Optional boolean coordinate mask with shape
            ``[N]``; omitted values default to ``fixed_atom_mask`` and supplied
            values must equal it exactly.
        :param attachment_mask: Optional boolean mask with shape ``[N]``;
            omitted values default to no marked attachment sites.
        :param component_ids: Optional long fragment-component labels with shape
            ``[N]`` on the reference device.
        :param task_id: Non-empty stable task identifier.
        :return: Validated fragment condition with direct unordered-halfedge
            internal-bond mask and no dense bond allocation.
        :rtype: FragmentCondition
        :raises ContractValidationError: If masks do not match the reference or
            an attachment site is not a fixed atom.

        Without component labels, the fixed-bond mask is
        ``fixed[src] & fixed[dst] & (src != dst)``. With ``component_ids``, an
        additional same-component predicate leaves cross-component pairs
        editable for link/merge decoding. The final predicate explicitly
        excludes diagonals even though a canonical ``MolecularState`` already
        rejects self-edges.
        """
        node_count = reference.positions.shape[0]
        _validate_bool_vector(
            fixed_atom_mask, "fixed_atom_mask", node_count, reference.positions
        )
        coordinate_mask = (
            fixed_atom_mask if fixed_coord_mask is None else fixed_coord_mask
        )
        attachments = (
            torch.zeros_like(fixed_atom_mask)
            if attachment_mask is None
            else attachment_mask
        )
        fixed_bond_mask = _internal_bond_mask(
            reference.halfedge_index, fixed_atom_mask, component_ids
        )
        return cls(
            reference=reference,
            fixed_atom_mask=fixed_atom_mask,
            fixed_bond_mask=fixed_bond_mask,
            fixed_coord_mask=coordinate_mask,
            attachment_mask=attachments,
            component_ids=component_ids,
            task_id=task_id,
        )


@dataclass(frozen=True)
class GenerationCondition:
    """Collect explicit pocket, field, fragment, and target sampling context.

    :param pocket: Centered pocket graph defining the conditioning structure.
    :param pocket_field: Optional electron-field representation in the same
        centered coordinate frame as ``pocket``.
    :param fragment: Optional exact fixed-fragment condition.
    :param property_targets: Immutable mapping from property names to finite
        scalar values or tensors. Tensor values must be finite.
    :param interaction_targets: Optional finite interaction target tensor on
        the same device as ``pocket.positions``.
    :return: Immutable condition passed to model and sampler components.
    :rtype: GenerationCondition
    :raises ContractValidationError: If field/frame dtype-device-batch context,
        targets, or fragment references are incompatible with the pocket.

    The mapping is copied into a read-only proxy so conditions cannot be
    silently changed through a caller-owned dictionary after validation.
    """

    pocket: PocketGraph
    pocket_field: ElectronField | None = None
    fragment: FragmentCondition | None = None
    property_targets: Mapping[str, TensorProperty] = field(default_factory=dict)
    interaction_targets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate condition devices and freeze property target metadata.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If target tensors or context devices
            cannot be jointly consumed by a model.
        """
        device = self.pocket.positions.device
        dtype = self.pocket.positions.dtype
        if self.pocket_field is not None:
            if (
                self.pocket_field.positions.device != device
                or self.pocket_field.positions.dtype != dtype
            ):
                raise ContractValidationError(
                    "pocket_field must share the pocket coordinate dtype and device."
                )
            if self.pocket.frame is None or self.pocket_field.frame is None:
                raise ContractValidationError(
                    "pocket and pocket_field must both declare a coordinate frame."
                )
            if self.pocket.frame != self.pocket_field.frame:
                raise ContractValidationError(
                    "pocket_field frame must equal the pocket coordinate frame."
                )
            _validate_batch_membership(
                self.pocket.batch, self.pocket_field.batch, "pocket_field"
            )
        if self.fragment is not None:
            reference = self.fragment.reference
            if (
                reference.positions.device != device
                or reference.positions.dtype != dtype
            ):
                raise ContractValidationError(
                    "fragment reference must share the pocket coordinate dtype and device."
                )
            if self.pocket.frame is None or reference.frame is None:
                raise ContractValidationError(
                    "pocket and fragment reference must both declare a coordinate frame."
                )
            if self.pocket.frame != reference.frame:
                raise ContractValidationError(
                    "fragment reference frame must equal the pocket coordinate frame."
                )
            _validate_batch_membership(
                self.pocket.batch, reference.node_batch, "fragment reference"
            )
        frozen_targets = _freeze_properties(self.property_targets)
        object.__setattr__(self, "property_targets", frozen_targets)
        if self.interaction_targets is not None:
            _validate_finite_tensor(
                self.interaction_targets, "interaction_targets", device=device
            )


@dataclass(frozen=True)
class SampleProvenance:
    """Store typed source and preprocessing provenance for one complex sample.

    :param source_paths: Immutable mapping of named source roles to paths.
    :param file_hashes: Immutable mapping of source roles to content hashes.
    :param tool_versions: Immutable mapping of preprocessing tool names to
        version strings.
    :param preprocessing_status: Stable status such as ``"complete"``.
    :param original_ligand_positions: Optional global-frame ligand coordinates
        with shape ``[N, 3]`` in angstroms for inverse-frame verification.
    :return: Immutable provenance record.
    :rtype: SampleProvenance
    :raises ContractValidationError: If metadata is malformed or original
        coordinates are not finite floating Cartesian points.
    """

    source_paths: Mapping[str, str] = field(default_factory=dict)
    file_hashes: Mapping[str, str] = field(default_factory=dict)
    tool_versions: Mapping[str, str] = field(default_factory=dict)
    preprocessing_status: str = "complete"
    original_ligand_positions: torch.Tensor | None = None
    qm: QMProvenance | None = None

    def __post_init__(self) -> None:
        """Freeze provenance mappings and validate optional source coordinates.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If a mapping key/value or source
            coordinate tensor does not meet the provenance contract.
        """
        object.__setattr__(
            self, "source_paths", _freeze_string_mapping(self.source_paths)
        )
        object.__setattr__(
            self, "file_hashes", _freeze_string_mapping(self.file_hashes)
        )
        object.__setattr__(
            self, "tool_versions", _freeze_string_mapping(self.tool_versions)
        )
        if not self.preprocessing_status:
            raise ContractValidationError("preprocessing_status must be non-empty.")
        if self.original_ligand_positions is not None:
            _validate_positions(
                self.original_ligand_positions,
                "original_ligand_positions",
                nonempty=False,
            )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Describe lossless DataLoader transport for sample provenance.

        :return: The validated provenance type and complete constructor values;
            immutable metadata views are represented as ordinary dictionaries.
        :rtype: tuple[Any, tuple[Any, ...]]
        :raises ContractValidationError: During receiver reconstruction if a
            mapping or ligand-coordinate tensor violates provenance invariants.

        The optional ``original_ligand_positions`` tensor keeps its ``[N, 3]``
        shape, Cartesian angstrom frame, floating dtype, device, and values.
        This hook performs no device transfer, source-file access, hash update,
        or in-place mutation. The receiving constructor rechecks finite
        coordinates and freezes every metadata mapping, so persistent worker
        transport preserves the canonical provenance invariants.
        """
        return (
            type(self),
            (
                dict(self.source_paths),
                dict(self.file_hashes),
                dict(self.tool_versions),
                self.preprocessing_status,
                self.original_ligand_positions,
                self.qm,
            ),
        )


@dataclass(frozen=True)
class ComplexSample:
    """Collect one traceable pocket-ligand complex in canonical local tensors.

    :param source_id: Stable non-empty identifier propagated to all artifacts.
    :param pocket: Centered protein-pocket graph.
    :param ligand: Centered discrete ligand graph.
    :param pocket_field: Optional pocket electron-field channels or coefficients.
    :param ligand_field: Optional ligand electron-field channels or coefficients.
    :param properties: Immutable mapping of available affinity or
        physicochemical property values.
    :param frame: Inverse transform from centered local coordinates to global
        source coordinates.
    :param provenance: Typed source files, hashes, tool versions, and status.
    :param fragment: Optional exact fragment condition sampled for this complex.
    :return: Immutable, traceable data-system sample.
    :rtype: ComplexSample
    :raises ContractValidationError: If tensors, frames, provenance, or
        fragment reference geometry are incompatible.

    The contract uses explicit fields rather than arbitrary payload dictionaries.
    Local pocket, ligand, and present field tensors share batch identities and
    the same centered frame, with global restoration through ``frame``.
    """

    source_id: str
    pocket: PocketGraph
    ligand: LigandGraph
    pocket_field: ElectronField | None
    ligand_field: ElectronField | None
    properties: Mapping[str, SampleProperty]
    frame: CoordinateFrame
    provenance: SampleProvenance
    fragment: FragmentCondition | None = None

    def __post_init__(self) -> None:
        """Validate sample coordinate compatibility and freeze properties.

        :return: None.
        :rtype: None
        :raises ContractValidationError: If a sample's tensor devices/dtypes,
            fields, provenance coordinates, or fragment state disagree.
        """
        if not self.source_id:
            raise ContractValidationError("source_id must be a non-empty string.")
        device = self.pocket.positions.device
        dtype = self.pocket.positions.dtype
        _validate_batch_membership(self.pocket.batch, self.ligand.batch, "ligand")
        for name, positions in (
            ("ligand", self.ligand.positions),
            ("frame", self.frame.origin),
        ):
            if positions.device != device or positions.dtype != dtype:
                raise ContractValidationError(
                    f"{name} must share the pocket coordinate dtype and device."
                )
        if self.pocket.frame is None or self.pocket.frame != self.frame:
            raise ContractValidationError(
                "pocket frame must equal the sample coordinate frame."
            )
        for name, field_value in (
            ("pocket_field", self.pocket_field),
            ("ligand_field", self.ligand_field),
        ):
            if field_value is not None:
                if (
                    field_value.positions.device != device
                    or field_value.positions.dtype != dtype
                ):
                    raise ContractValidationError(
                        f"{name} must share the pocket coordinate dtype and device."
                    )
                _validate_batch_membership(self.pocket.batch, field_value.batch, name)
                if field_value.frame is None or field_value.frame != self.frame:
                    raise ContractValidationError(
                        f"{name} frame must equal the sample coordinate frame."
                    )
        if self.provenance.original_ligand_positions is not None and (
            self.provenance.original_ligand_positions.shape
            != self.ligand.positions.shape
        ):
            raise ContractValidationError(
                "original_ligand_positions must match ligand position shape."
            )
        if self.fragment is not None and not torch.equal(
            self.fragment.reference.positions, self.ligand.positions
        ):
            raise ContractValidationError(
                "fragment reference positions must match the sample ligand positions."
            )
        if self.fragment is not None:
            if self.fragment.reference.frame is None:
                raise ContractValidationError(
                    "fragment reference must declare the sample coordinate frame."
                )
            if self.fragment.reference.frame != self.frame:
                raise ContractValidationError(
                    "fragment reference frame must equal the sample coordinate frame."
                )
            _validate_batch_membership(
                self.ligand.batch,
                self.fragment.reference.node_batch,
                "fragment reference",
            )
        object.__setattr__(
            self, "properties", _freeze_sample_properties(self.properties)
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Transport canonical samples while restoring immutable mappings.

        :return: Constructor and primitive argument tuple understood by pickle.
        :rtype: tuple[Any, tuple[Any, ...]]
        :raises ContractValidationError: During receiver reconstruction if any
            transported graph, frame, field, provenance, or fragment is invalid.

        PyTorch DataLoader uses multiprocessing pickle between persistent CPU
        workers and the training process. ``MappingProxyType`` is intentionally
        not picklable, so mappings are copied to ordinary dictionaries only for
        transport; the receiving constructor revalidates and freezes them.
        Pocket/ligand/field coordinates retain ``[N, 3]`` shapes, local binding
        frame, angstrom units, floating dtype and device; graph features, masks,
        batch indices, formal charges, bond orders, and optional fragment fixed
        masks are unchanged. The method performs no sampling, tensor transfer,
        filesystem access, or mutation. Any incompatible reconstructed tensor
        fails the ordinary :class:`ComplexSample` contract.
        """
        return (
            type(self),
            (
                self.source_id,
                self.pocket,
                self.ligand,
                self.pocket_field,
                self.ligand_field,
                dict(self.properties),
                self.frame,
                self.provenance,
                self.fragment,
            ),
        )


def _validate_positions(value: torch.Tensor, name: str, *, nonempty: bool) -> int:
    """Validate one flattened finite Cartesian coordinate tensor.

    :param value: Candidate coordinates with expected shape ``[N, 3]``.
    :param name: Human-readable tensor name for errors.
    :param nonempty: Whether the leading dimension must be positive.
    :return: Validated leading count ``N``.
    :rtype: int
    :raises ContractValidationError: If rank, dtype, finite values, or count
        violates the coordinate contract.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ContractValidationError(f"{name} must have shape [N, 3].")
    if nonempty and value.shape[0] == 0:
        raise ContractValidationError(f"{name} must contain at least one point.")
    if not value.is_floating_point():
        raise ContractValidationError(f"{name} must have a floating dtype.")
    if not torch.isfinite(value).all():
        raise ContractValidationError(f"{name} must contain only finite values.")
    return value.shape[0]


def _validate_float_matrix(
    value: torch.Tensor, name: str, leading_count: int, reference: torch.Tensor
) -> int:
    """Validate a finite floating matrix with a shared leading dimension.

    :param value: Candidate tensor with expected shape ``[N, C]``.
    :param name: Human-readable tensor name for errors.
    :param leading_count: Required leading dimension ``N``.
    :param reference: Tensor supplying required dtype/device.
    :return: Validated channel count ``C``.
    :rtype: int
    :raises ContractValidationError: If rank, dtype/device, or finite values
        violate the flattened feature contract.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or value.shape[0] != leading_count:
        raise ContractValidationError(f"{name} must have shape [{leading_count}, C].")
    if value.shape[1] == 0:
        raise ContractValidationError(f"{name} must include at least one channel.")
    if not value.is_floating_point():
        raise ContractValidationError(f"{name} must have a floating dtype.")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ContractValidationError(
            f"{name} must have the same dtype and device as positions."
        )
    if not torch.isfinite(value).all():
        raise ContractValidationError(f"{name} must contain only finite values.")
    return value.shape[1]


def _validate_index_vector(
    value: torch.Tensor, name: str, leading_count: int, reference: torch.Tensor
) -> None:
    """Validate a long index vector with a shared leading dimension.

    :param value: Candidate tensor with expected shape ``[N]``.
    :param name: Human-readable tensor name for errors.
    :param leading_count: Required leading dimension ``N``.
    :param reference: Tensor supplying required device.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If rank, dtype, device, or sign is invalid.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1 or value.shape[0] != leading_count:
        raise ContractValidationError(f"{name} must have shape [{leading_count}].")
    if value.dtype != torch.long:
        raise ContractValidationError(f"{name} must have torch.long dtype.")
    if value.device != reference.device:
        raise ContractValidationError(f"{name} must be on the positions device.")
    if bool((value < 0).any()):
        raise ContractValidationError(f"{name} must not contain negative indices.")


def _validate_signed_long_vector(
    value: torch.Tensor, name: str, leading_count: int, reference: torch.Tensor
) -> None:
    """Validate a signed long value vector with a shared leading dimension.

    :param value: Candidate tensor with expected shape ``[N]`` and long dtype.
    :param name: Human-readable tensor name for errors.
    :param leading_count: Required leading dimension ``N``.
    :param reference: Tensor supplying required device.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If rank, dtype, or device is invalid.

    Formal charges are signed chemistry values, unlike atom, bond, and batch
    indices, so negative values are intentionally accepted.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1 or value.shape[0] != leading_count:
        raise ContractValidationError(f"{name} must have shape [{leading_count}].")
    if value.dtype != torch.long:
        raise ContractValidationError(f"{name} must have torch.long dtype.")
    if value.device != reference.device:
        raise ContractValidationError(f"{name} must be on the positions device.")


def _validate_bool_vector(
    value: torch.Tensor, name: str, leading_count: int, reference: torch.Tensor
) -> None:
    """Validate a boolean mask vector with a shared leading dimension.

    :param value: Candidate tensor with expected shape ``[N]``.
    :param name: Human-readable tensor name for errors.
    :param leading_count: Required leading dimension ``N``.
    :param reference: Tensor supplying required device.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If rank, dtype, or device is invalid.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError(f"{name} must be a torch.Tensor.")
    if value.ndim != 1 or value.shape[0] != leading_count:
        raise ContractValidationError(f"{name} must have shape [{leading_count}].")
    if value.dtype != torch.bool:
        raise ContractValidationError(f"{name} must have torch.bool dtype.")
    if value.device != reference.device:
        raise ContractValidationError(f"{name} must be on the positions device.")


def _validate_halfedge_index(
    value: torch.Tensor, node_count: int, reference: torch.Tensor
) -> int:
    """Validate one canonical unordered halfedge index tensor.

    :param value: Candidate indices with shape ``[2, E]`` and long dtype.
    :param node_count: Number of available nodes for endpoint bounds.
    :param reference: Tensor supplying required device.
    :return: Validated halfedge count ``E``.
    :rtype: int
    :raises ContractValidationError: If rank, dtype/device, bounds, duplicate
        pairs, self-edges, or unordered endpoint ordering is invalid.
    """
    if not isinstance(value, torch.Tensor):
        raise ContractValidationError("halfedge_index must be a torch.Tensor.")
    if value.ndim != 2 or value.shape[0] != 2:
        raise ContractValidationError("halfedge_index must have shape [2, E].")
    if value.dtype != torch.long:
        raise ContractValidationError("halfedge_index must have torch.long dtype.")
    if value.device != reference.device:
        raise ContractValidationError("halfedge_index must be on the positions device.")
    if value.numel() and (bool((value < 0).any()) or bool((value >= node_count).any())):
        raise ContractValidationError(
            "halfedge_index contains endpoints outside [0, N)."
        )
    if not bool((value[0] < value[1]).all()):
        raise ContractValidationError(
            "halfedge_index must have row-zero indices strictly smaller than row-one indices."
        )
    if value.shape[1] != torch.unique(value, dim=1).shape[1]:
        raise ContractValidationError(
            "halfedge_index must not contain duplicate pairs."
        )
    return value.shape[1]


def _validate_halfedge_batches(
    halfedge_index: torch.Tensor,
    node_batch: torch.Tensor,
    halfedge_batch: torch.Tensor | None,
) -> None:
    """Validate that every halfedge joins nodes in one complex batch.

    :param halfedge_index: Canonical endpoints with shape ``[2, E]``.
    :param node_batch: Node complex indices with shape ``[N]``.
    :param halfedge_batch: Optional edge complex indices with shape ``[E]``.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If endpoints cross complexes or edge batch
        assignments do not equal their endpoint batch.
    """
    source_batch = node_batch[halfedge_index[0]]
    target_batch = node_batch[halfedge_index[1]]
    if not torch.equal(source_batch, target_batch):
        raise ContractValidationError(
            "halfedges must connect nodes in the same complex."
        )
    if halfedge_batch is not None and not torch.equal(halfedge_batch, source_batch):
        raise ContractValidationError(
            "halfedge_batch must equal the complex index of every halfedge endpoint."
        )


def _internal_bond_mask(
    halfedge_index: torch.Tensor,
    fixed_atom_mask: torch.Tensor,
    component_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the direct immutable-component mask for canonical halfedges.

    :param halfedge_index: Canonical unordered pairs with shape ``[2, E]``.
    :param fixed_atom_mask: Boolean atom mask with shape ``[N]``.
    :param component_ids: Optional non-negative component labels with shape
        ``[N]``. If supplied, only fixed endpoints sharing a label are
        immutable; cross-component pairs remain editable for linking.
    :return: Boolean mask with shape ``[E]`` true exactly for fixed endpoint
        pairs, on the same device as the input tensors.
    :rtype: torch.Tensor

    The direct endpoint calculation requires ``O(E)`` memory and explicitly
    excludes diagonal pairs. It does not create any dense ``[N, N]`` mask.
    """
    source, target = halfedge_index
    mask = fixed_atom_mask[source] & fixed_atom_mask[target] & (source != target)
    if component_ids is not None:
        if component_ids.shape != fixed_atom_mask.shape:
            raise ContractValidationError(
                "component_ids must match fixed_atom_mask shape."
            )
        mask = mask & (component_ids[source] == component_ids[target])
    return mask


def _validate_batch_membership(
    expected: torch.Tensor, actual: torch.Tensor, name: str
) -> None:
    """Require two flattened tensors to represent the same complex batches.

    :param expected: Reference complex indices with shape ``[N]``.
    :param actual: Candidate complex indices with shape ``[M]``.
    :param name: Human-readable candidate name for diagnostic messages.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If unique complex identities differ.

    Node counts may differ across pockets, ligands, and fields, so this checks
    the sorted unique batch identities instead of elementwise equality.
    """
    if not torch.equal(torch.unique(expected), torch.unique(actual)):
        raise ContractValidationError(
            f"{name} must have the same complex batch membership as the pocket."
        )


def _freeze_properties(
    values: Mapping[str, TensorProperty],
) -> Mapping[str, TensorProperty]:
    """Validate and copy scalar property targets into a read-only mapping.

    :param values: Property values keyed by non-empty strings.
    :return: Read-only mapping with the same validated scalar or tensor values.
    :rtype: Mapping[str, TensorProperty]
    :raises ContractValidationError: If a key or value is not finite and typed.
    """
    copied: dict[str, TensorProperty] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ContractValidationError(
                "property target keys must be non-empty strings."
            )
        if isinstance(value, torch.Tensor):
            _validate_finite_tensor(value, f"property_targets[{key!r}]")
        elif not isinstance(value, (float, int)) or not torch.isfinite(
            torch.tensor(value, dtype=torch.float64)
        ):
            raise ContractValidationError(
                "property target values must be finite floats, ints, or tensors."
            )
        copied[key] = value
    return MappingProxyType(copied)


def _freeze_sample_properties(
    values: Mapping[str, SampleProperty],
) -> Mapping[str, SampleProperty]:
    """Validate numeric targets and textual source metadata for one sample.

    :param values: Sample properties keyed by stable non-empty names. Numeric
        values are available to training, while strings preserve measurement
        semantics such as ``Kd``, ``<``, and the original PDBBind expression.
    :return: Defensive read-only copy of every validated property.
    :rtype: Mapping[str, SampleProperty]
    :raises ContractValidationError: If a key is empty, a string is empty, or
        a numeric/tensor value is non-finite or otherwise invalid.

    Generation conditions deliberately continue to use
    :func:`_freeze_properties`, which accepts numeric values only. This sample-
    specific validator therefore cannot accidentally route textual metadata
    into model conditioning tensors. :class:`TrainingBatchBuilder` selects
    finite scalar values and ignores these strings during collation.
    """
    numeric: dict[str, TensorProperty] = {}
    copied: dict[str, SampleProperty] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ContractValidationError(
                "sample property keys must be non-empty strings."
            )
        if isinstance(value, str):
            if not value:
                raise ContractValidationError(
                    "textual sample property values must be non-empty."
                )
            copied[key] = value
        else:
            numeric[key] = value
    copied.update(_freeze_properties(numeric))
    return MappingProxyType(copied)


def _freeze_string_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    """Validate and copy textual metadata into a read-only mapping.

    :param values: Metadata mapping with non-empty string keys and values.
    :return: Read-only copy of validated metadata.
    :rtype: Mapping[str, str]
    :raises ContractValidationError: If any key or value is empty or non-string.
    """
    copied: dict[str, str] = {}
    for key, value in values.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise ContractValidationError(
                "provenance metadata keys and values must be non-empty strings."
            )
        copied[key] = value
    return MappingProxyType(copied)


def _validate_finite_tensor(
    value: torch.Tensor, name: str, *, device: torch.device | None = None
) -> None:
    """Validate one finite floating tensor and optional device compatibility.

    :param value: Candidate finite floating tensor of any non-empty shape.
    :param name: Human-readable tensor name for errors.
    :param device: Optional required torch device.
    :return: None.
    :rtype: None
    :raises ContractValidationError: If dtype, finite values, or device is invalid.
    """
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ContractValidationError(f"{name} must be a floating torch.Tensor.")
    if not torch.isfinite(value).all():
        raise ContractValidationError(f"{name} must contain only finite values.")
    if device is not None and value.device != device:
        raise ContractValidationError(f"{name} must be on the required device.")
