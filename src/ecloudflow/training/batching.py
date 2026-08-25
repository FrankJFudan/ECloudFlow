"""Online construction of stochastic training batches from clean complexes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.config import ModelConfig, TrainerConfig
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.masks import clamp_fragment
from ecloudflow.core.types import (
    ComplexSample,
    ElectronField,
    FragmentCondition,
    GenerationCondition,
    MolecularState,
    PocketGraph,
)
from ecloudflow.ecloud import (
    SphericalFieldBasis,
    multipole_moments,
    project_density_to_atoms,
    reconstruct_density,
)
from ecloudflow.process import ContinuousPath, LinearBridge
from ecloudflow.training.types import (
    ElectronDecoderContext,
    TrainingBatch,
    TrainingTargets,
)


@dataclass(frozen=True)
class _FieldObservation:
    """Hold one projected field and optional genuine-QM supervision."""

    coefficients: torch.Tensor
    query_positions: torch.Tensor | None
    density: torch.Tensor | None
    density_gradient: torch.Tensor | None
    electron_count: torch.Tensor | None
    dipole: torch.Tensor | None
    qm: bool


class TrainingBatchBuilder:
    """Build exact hybrid flow/score targets from canonical clean complexes.

    :param model_config: Frozen architecture and electron-basis configuration.
    :param trainer_config: Frozen runtime configuration containing path noise.
    :return: Stateless callable suitable for
        :class:`~ecloudflow.training.module.ECloudFlowTrainingModule`.
    :rtype: TrainingBatchBuilder

    The DataModule deliberately yields immutable :class:`ComplexSample` values
    so rank/worker sharding remains independent of model code. This builder is
    the explicit bridge to optimization. It runs after Lightning transfers a
    rank-local clean batch to the strategy device, projects genuine ligand
    density into atom-centered irreps, tokenizes it with the trainable field
    encoder, and draws one coupled stochastic interpolant. Coordinates remain
    in their centered pocket binding frames; batching relabels those local
    frames with one identity transform and never rotates or translates values.

    Ligand bonds remain canonical sparse unordered halfedges. Nonbonded pairs
    used by the clash loss are generated separately and never become bond
    candidates. Fixed fragment atoms, charges, coordinates, and internal bonds
    are copied exactly into the noisy state after every path construction step.
    Genuine-QM masks come only from provenance; density gradients are derivatives
    of the documented atom-centered Galerkin reconstruction rather than invented
    labels. Random draws use the active device's global PyTorch generator, whose
    per-rank state is captured by the reproducible-checkpoint callback.
    """

    def __init__(
        self, model_config: ModelConfig, trainer_config: TrainerConfig
    ) -> None:
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be ModelConfig.")
        if not isinstance(trainer_config, TrainerConfig):
            raise TypeError("trainer_config must be TrainerConfig.")
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.vocabulary = ChemicalVocabulary.default_ligand()
        self.basis = SphericalFieldBasis(
            model_config.field_n_radial,
            model_config.lmax,
            model_config.field_cutoff,
            model_config.field_chunk_size,
        )
        self.path = ContinuousPath(
            LinearBridge(interior_noise=trainer_config.interior_noise)
        )

    def __call__(
        self,
        samples: Sequence[ComplexSample],
        field_tokenizer: nn.Module,
    ) -> TrainingBatch:
        """Convert one rank-local clean sample sequence into training tensors.

        :param samples: Non-empty canonical complexes already transferred to the
            Lightning-selected device. Every complex retains local angstrom
            coordinates, graph categories, optional fields, and provenance.
        :param field_tokenizer: Trainable tokenizer whose packed output layout
            must equal ``model_config.electron_latent_dim``.
        :return: Noisy molecular state, conditioning, exact path targets,
            sparse chemical targets, and optional genuine-QM decoder context.
        :rtype: TrainingBatch
        :raises TypeError: If inputs do not implement the canonical contracts.
        :raises ValueError: If a batch mixes devices/dtypes, exceeds the atom
            limit, contains unsupported chemistry, or marks incomplete data as QM.

        All categorical endpoint channels follow the immutable ligand
        vocabulary. Continuous position/electron paths share one per-complex
        antithetic time draw, while independent Gaussian couplings retain exact
        velocity and score targets. The clean input objects are read only. No
        CPU/GPU transfer, dense bond tensor, distributed collective, filesystem
        operation, or external QM calculation occurs here.
        """
        clean = tuple(samples)
        if not clean or not all(isinstance(sample, ComplexSample) for sample in clean):
            raise TypeError("samples must be a non-empty ComplexSample sequence.")
        if not isinstance(field_tokenizer, nn.Module):
            raise TypeError("field_tokenizer must be a torch.nn.Module.")
        device = clean[0].ligand.positions.device
        dtype = clean[0].ligand.positions.dtype
        if any(
            sample.ligand.positions.device != device
            or sample.ligand.positions.dtype != dtype
            or sample.pocket.positions.device != device
            or sample.pocket.positions.dtype != dtype
            for sample in clean
        ):
            raise ValueError("all clean complexes must share one floating dtype/device.")

        counts = torch.tensor(
            [sample.ligand.positions.shape[0] for sample in clean],
            dtype=torch.long,
            device=device,
        )
        if bool((counts > self.model_config.max_atoms).any()):
            raise ValueError(
                f"ligand atom count exceeds configured maximum {self.model_config.max_atoms}."
            )
        frame = CoordinateFrame(origin=torch.zeros(3, dtype=dtype, device=device))
        atom_classes = torch.cat([sample.ligand.atom_types for sample in clean])
        charge_classes = torch.cat(
            [self._charge_classes(sample) for sample in clean]
        )
        positions = torch.cat([sample.ligand.positions for sample in clean])
        node_batch = torch.repeat_interleave(
            torch.arange(len(clean), device=device), counts
        )
        halfedge_index, bond_classes, halfedge_batch = self._batch_bonds(clean)
        atom_count = positions.shape[0]
        edge_count = halfedge_index.shape[1]

        atom_endpoint = functional.one_hot(
            atom_classes, len(self.vocabulary.atom_symbols)
        ).to(dtype=dtype)
        charge_endpoint = functional.one_hot(
            charge_classes, len(self.vocabulary.formal_charges)
        ).to(dtype=dtype)
        bond_endpoint = functional.one_hot(
            bond_classes, len(self.vocabulary.bond_classes)
        ).to(dtype=dtype)

        observations, padded_latent, atom_mask = self._tokenize_fields(
            clean,
            atom_classes,
            charge_classes,
            field_tokenizer,
        )
        electron_endpoint = torch.cat(
            [padded_latent[index, : int(count)] for index, count in enumerate(counts)]
        )
        if electron_endpoint.shape != (
            atom_count,
            self.model_config.electron_latent_dim,
        ):
            raise ValueError("field tokenizer returned an incompatible latent layout.")

        clean_state = MolecularState(
            positions=positions,
            atom_logits=atom_endpoint,
            charge_logits=charge_endpoint,
            halfedge_index=halfedge_index,
            bond_logits=bond_endpoint,
            electron_latent=electron_endpoint,
            node_batch=node_batch,
            halfedge_batch=halfedge_batch,
            frame=frame,
        )
        fragment = self._batch_fragment(clean, clean_state)
        condition = GenerationCondition(
            pocket=self._batch_pocket(clean, frame),
            pocket_field=self._batch_pocket_field(clean, frame),
            fragment=fragment,
            property_targets=self._common_properties(clean, dtype, device),
        )

        time = self.path.sample_times(
            len(clean),
            device=device,
            dtype=dtype,
            antithetic=True,
            mode="score",
        )
        position_prior = self._coordinate_prior(clean)
        electron_prior = torch.randn_like(electron_endpoint)
        if fragment is not None:
            fixed = fragment.fixed_atom_mask
            position_prior = torch.where(
                fixed[:, None], positions, position_prior
            )
            electron_prior = torch.where(
                fixed[:, None], electron_endpoint, electron_prior
            )
        position_sample = self.path.sample(
            position_prior, positions, time[node_batch]
        )
        electron_sample = self.path.sample(
            electron_prior, electron_endpoint, time[node_batch]
        )
        position_velocity, position_score = self.path.targets(
            position_prior, positions, position_sample
        )
        electron_velocity, electron_score = self.path.targets(
            electron_prior, electron_endpoint, electron_sample
        )
        fixed_atoms = (
            fragment.fixed_atom_mask
            if fragment is not None
            else torch.zeros(atom_count, dtype=torch.bool, device=device)
        )
        fixed_bonds = (
            fragment.fixed_bond_mask
            if fragment is not None
            else torch.zeros(edge_count, dtype=torch.bool, device=device)
        )
        noisy_state = MolecularState(
            positions=position_sample.value,
            atom_logits=self._categorical_state(
                atom_classes,
                time[node_batch],
                len(self.vocabulary.atom_symbols),
                fixed_atoms,
            ),
            charge_logits=self._categorical_state(
                charge_classes,
                time[node_batch],
                len(self.vocabulary.formal_charges),
                fixed_atoms,
            ),
            halfedge_index=halfedge_index,
            bond_logits=self._categorical_state(
                bond_classes,
                time[halfedge_batch],
                len(self.vocabulary.bond_classes),
                fixed_bonds,
            ),
            electron_latent=electron_sample.value,
            node_batch=node_batch,
            halfedge_batch=halfedge_batch,
            frame=frame,
        )
        if fragment is not None:
            noisy_state = clamp_fragment(noisy_state, fragment)

        affinity, affinity_mask = self._property_target(
            clean,
            ("affinity", "pkd", "pki", "pka", "binding_affinity"),
            dtype,
            device,
        )
        interaction, interaction_mask = self._property_target(
            clean,
            ("interaction", "interaction_probability", "contact_probability"),
            dtype,
            device,
        )
        decoder_context, field_targets = self._decoder_targets(
            clean,
            observations,
            padded_latent,
            atom_mask,
        )
        bond_lengths = (
            (positions[halfedge_index[0]] - positions[halfedge_index[1]]).norm(dim=-1)
            if edge_count
            else positions.new_empty((0,))
        )
        targets = TrainingTargets(
            position_velocity=position_velocity,
            position_score=position_score,
            electron_velocity=electron_velocity,
            electron_score=electron_score,
            atom_classes=atom_classes,
            charge_classes=charge_classes,
            bond_classes=bond_classes,
            count_classes=counts,
            editable_atom_mask=~fixed_atoms,
            editable_bond_mask=~fixed_bonds,
            node_batch=node_batch,
            halfedge_index=halfedge_index,
            halfedge_batch=halfedge_batch,
            count_mask=torch.ones(len(clean), dtype=torch.bool, device=device),
            qm_mask=field_targets["qm_mask"],
            density=field_targets["density"],
            density_gradient=field_targets["density_gradient"],
            field_mask=field_targets["field_mask"],
            electron_count=field_targets["electron_count"],
            dipole=field_targets["dipole"],
            latent_cycle=field_targets["latent_cycle"],
            latent_cycle_mask=field_targets["latent_cycle_mask"],
            valence_limits=self._valence_limits(atom_classes, dtype),
            bond_order_values=torch.tensor(
                self.vocabulary.bond_orders, dtype=dtype, device=device
            ),
            bond_length_mean=bond_lengths,
            bond_length_std=torch.full_like(bond_lengths, 0.1),
            nonbonded_halfedge_index=self._nonbonded_pairs(clean),
            protein_positions=torch.cat(
                [sample.pocket.positions for sample in clean]
            ),
            protein_batch=torch.cat(
                [
                    torch.full(
                        (sample.pocket.positions.shape[0],),
                        index,
                        dtype=torch.long,
                        device=device,
                    )
                    for index, sample in enumerate(clean)
                ]
            ),
            ring_triplets=torch.empty((3, 0), dtype=torch.long, device=device),
            ring_angle_mean=torch.empty(0, dtype=dtype, device=device),
            ring_angle_std=torch.empty(0, dtype=dtype, device=device),
            interaction=interaction,
            interaction_mask=interaction_mask,
            affinity=affinity,
            affinity_mask=affinity_mask,
        )
        return TrainingBatch(
            state=noisy_state,
            time=time,
            condition=condition,
            targets=targets,
            decoder_context=decoder_context,
        )

    def _charge_classes(self, sample: ComplexSample) -> torch.Tensor:
        """Map signed formal charges to stable vocabulary class indices."""
        indices = [
            self.vocabulary.charge_index(int(value))
            for value in sample.ligand.formal_charges.tolist()
        ]
        return torch.tensor(
            indices,
            dtype=torch.long,
            device=sample.ligand.formal_charges.device,
        )

    def _batch_bonds(
        self, samples: Sequence[ComplexSample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate canonical ligand bonds with node and batch offsets."""
        device = samples[0].ligand.positions.device
        edges: list[torch.Tensor] = []
        classes: list[torch.Tensor] = []
        batches: list[torch.Tensor] = []
        offset = 0
        for batch_index, sample in enumerate(samples):
            edge = sample.ligand.halfedge_index
            if edge.shape[1]:
                edges.append(edge + offset)
                classes.append(sample.ligand.bond_types)
                batches.append(
                    torch.full(
                        (edge.shape[1],),
                        batch_index,
                        dtype=torch.long,
                        device=device,
                    )
                )
            offset += sample.ligand.positions.shape[0]
        return (
            torch.cat(edges, dim=1)
            if edges
            else torch.empty((2, 0), dtype=torch.long, device=device),
            torch.cat(classes)
            if classes
            else torch.empty(0, dtype=torch.long, device=device),
            torch.cat(batches)
            if batches
            else torch.empty(0, dtype=torch.long, device=device),
        )

    def _tokenize_fields(
        self,
        samples: Sequence[ComplexSample],
        atom_classes: torch.Tensor,
        charge_classes: torch.Tensor,
        tokenizer: nn.Module,
    ) -> tuple[tuple[_FieldObservation, ...], torch.Tensor, torch.Tensor]:
        """Project density, pad atoms, and execute the trainable tokenizer."""
        device = atom_classes.device
        dtype = samples[0].ligand.positions.dtype
        counts = [sample.ligand.positions.shape[0] for sample in samples]
        max_count = max(counts)
        coefficients = torch.zeros(
            (
                len(samples),
                max_count,
                self.basis.n_radial,
                self.basis.harmonic_dim,
            ),
            dtype=dtype,
            device=device,
        )
        features = torch.zeros(
            (
                len(samples),
                max_count,
                len(self.vocabulary.atom_symbols)
                + len(self.vocabulary.formal_charges),
            ),
            dtype=dtype,
            device=device,
        )
        atom_mask = torch.zeros(
            (len(samples), max_count), dtype=torch.bool, device=device
        )
        observations: list[_FieldObservation] = []
        offset = 0
        for index, (sample, count) in enumerate(zip(samples, counts, strict=True)):
            observation = self._field_observation(sample)
            observations.append(observation)
            coefficients[index, :count] = observation.coefficients.to(dtype=dtype)
            features[index, :count] = torch.cat(
                (
                    functional.one_hot(
                        atom_classes[offset : offset + count],
                        len(self.vocabulary.atom_symbols),
                    ),
                    functional.one_hot(
                        charge_classes[offset : offset + count],
                        len(self.vocabulary.formal_charges),
                    ),
                ),
                dim=-1,
            ).to(dtype=dtype)
            atom_mask[index, :count] = True
            offset += count
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("field_tokenizer must expose an encode method.")
        latent = encode(coefficients, features, atom_mask)
        if not isinstance(latent, torch.Tensor):
            raise TypeError("field_tokenizer.encode must return a tensor.")
        return tuple(observations), latent, atom_mask

    def _field_observation(self, sample: ComplexSample) -> _FieldObservation:
        """Project one available density and derive provenance-safe QM targets."""
        positions = sample.ligand.positions
        coefficients = positions.new_zeros(
            (
                positions.shape[0],
                self.basis.n_radial,
                self.basis.harmonic_dim,
            )
        )
        provenance = sample.provenance.qm
        genuine_qm = bool(provenance is not None and provenance.qm_mask)
        field = sample.ligand_field
        if field is None:
            if genuine_qm:
                raise ValueError(
                    f"sample {sample.source_id!r} marks QM true without ligand density."
                )
            return _FieldObservation(coefficients, None, None, None, None, None, False)
        channel = self._density_channel(field)
        if channel is None:
            if genuine_qm:
                raise ValueError(
                    f"sample {sample.source_id!r} marks QM true without density channel."
                )
            return _FieldObservation(coefficients, None, None, None, None, None, False)
        selected = field.mask
        query = field.positions[selected]
        density = field.values[selected, channel]
        if query.shape[0] == 0:
            if genuine_qm:
                raise ValueError(
                    f"sample {sample.source_id!r} marks QM true with no valid field points."
                )
            return _FieldObservation(coefficients, None, None, None, None, None, False)
        if bool((density < 0.0).any()):
            raise ValueError("ligand density used for tokenization must be non-negative.")
        weights = self._integration_weights(sample, query, density, genuine_qm)
        coefficients = project_density_to_atoms(
            density,
            query,
            positions,
            weights,
            self.basis,
        ).to(dtype=positions.dtype)
        if not genuine_qm:
            return _FieldObservation(coefficients, None, None, None, None, None, False)
        moments = multipole_moments(density, query, weights)
        raw_query = query.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            represented = reconstruct_density(
                coefficients.detach(), raw_query, positions.detach(), self.basis
            )
            gradient = torch.autograd.grad(represented.sum(), raw_query)[0]
        return _FieldObservation(
            coefficients=coefficients,
            query_positions=query,
            density=density,
            density_gradient=gradient.detach(),
            electron_count=moments.electron_count.detach(),
            dipole=moments.dipole.detach(),
            qm=True,
        )

    @staticmethod
    def _density_channel(field: ElectronField) -> int | None:
        """Return an explicit density channel or the sole unnamed channel."""
        if "density" in field.channel_names:
            return field.channel_names.index("density")
        if not field.channel_names and field.values.shape[1] == 1:
            return 0
        return None

    @staticmethod
    def _integration_weights(
        sample: ComplexSample,
        query: torch.Tensor,
        density: torch.Tensor,
        genuine_qm: bool,
    ) -> torch.Tensor:
        """Recover uniform quadrature volume from physical provenance or bounds."""
        provenance = sample.provenance.qm
        electron_count = (
            provenance.integrated_electron_count if provenance is not None else None
        )
        density_sum = density.sum()
        if electron_count is not None and electron_count > 0.0 and density_sum > 0.0:
            volume = density.new_tensor(float(electron_count)) / density_sum
        elif genuine_qm:
            raise ValueError(
                "genuine-QM density requires integrated electron-count provenance."
            )
        else:
            extent = (query.amax(dim=0) - query.amin(dim=0)).clamp_min(1.0e-3)
            volume = extent.prod() / max(query.shape[0], 1)
        if not bool(torch.isfinite(volume)) or bool(volume <= 0.0):
            raise ValueError("density quadrature volume must be finite and positive.")
        return volume.expand_as(density)

    @staticmethod
    def _categorical_state(
        target: torch.Tensor,
        time: torch.Tensor,
        class_count: int,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate an affine prior-to-endpoint simplex with exact fixed rows."""
        endpoint = functional.one_hot(target, class_count).to(dtype=time.dtype)
        prior = torch.full_like(endpoint, 1.0 / class_count)
        probabilities = (1.0 - time[:, None]) * prior + time[:, None] * endpoint
        return torch.where(fixed[:, None], endpoint, probabilities)

    @staticmethod
    def _batch_pocket(
        samples: Sequence[ComplexSample], frame: CoordinateFrame
    ) -> PocketGraph:
        """Concatenate pocket graphs while preserving local binding coordinates."""
        device = samples[0].pocket.positions.device
        atom_numbers = (
            torch.cat([sample.pocket.atom_numbers for sample in samples])
            if all(sample.pocket.atom_numbers is not None for sample in samples)
            else None
        )
        return PocketGraph(
            positions=torch.cat([sample.pocket.positions for sample in samples]),
            features=torch.cat([sample.pocket.features for sample in samples]),
            batch=torch.cat(
                [
                    torch.full(
                        (sample.pocket.positions.shape[0],),
                        index,
                        dtype=torch.long,
                        device=device,
                    )
                    for index, sample in enumerate(samples)
                ]
            ),
            atom_numbers=atom_numbers,
            frame=frame,
        )

    @staticmethod
    def _batch_pocket_field(
        samples: Sequence[ComplexSample], frame: CoordinateFrame
    ) -> ElectronField | None:
        """Concatenate pocket fields only when every complex supplies one."""
        if not all(sample.pocket_field is not None for sample in samples):
            return None
        fields = [sample.pocket_field for sample in samples]
        assert all(field is not None for field in fields)
        typed = [field for field in fields if field is not None]
        names = typed[0].channel_names
        if any(field.channel_names != names for field in typed):
            raise ValueError("batched pocket fields must use one channel schema.")
        device = typed[0].positions.device
        return ElectronField(
            positions=torch.cat([field.positions for field in typed]),
            values=torch.cat([field.values for field in typed]),
            mask=torch.cat([field.mask for field in typed]),
            batch=torch.cat(
                [
                    torch.full(
                        (field.positions.shape[0],),
                        index,
                        dtype=torch.long,
                        device=device,
                    )
                    for index, field in enumerate(typed)
                ]
            ),
            channel_names=names,
            frame=frame,
        )

    @staticmethod
    def _batch_fragment(
        samples: Sequence[ComplexSample], reference: MolecularState
    ) -> FragmentCondition | None:
        """Merge per-complex fragment masks against the batched clean state."""
        device = reference.positions.device
        fixed: list[torch.Tensor] = []
        attachment: list[torch.Tensor] = []
        components: list[torch.Tensor] = []
        task_ids: list[str] = []
        component_offset = 0
        has_components = False
        for sample in samples:
            count = sample.ligand.positions.shape[0]
            fragment = sample.fragment
            if fragment is None:
                fixed.append(torch.zeros(count, dtype=torch.bool, device=device))
                attachment.append(torch.zeros(count, dtype=torch.bool, device=device))
                components.append(torch.zeros(count, dtype=torch.long, device=device))
                task_ids.append("de_novo")
                continue
            fixed.append(fragment.fixed_atom_mask)
            attachment.append(fragment.attachment_mask)
            task_ids.append(fragment.task_id)
            if fragment.component_ids is None:
                components.append(torch.zeros(count, dtype=torch.long, device=device))
            else:
                has_components = True
                shifted = fragment.component_ids + component_offset
                components.append(shifted)
                component_offset = int(shifted.max().item()) + 1
        fixed_mask = torch.cat(fixed)
        if not bool(fixed_mask.any()):
            return None
        task_id = task_ids[0] if len(set(task_ids)) == 1 else "mixed"
        return FragmentCondition.from_atom_mask(
            fixed_mask,
            reference,
            attachment_mask=torch.cat(attachment),
            component_ids=torch.cat(components) if has_components else None,
            task_id=task_id,
        )

    @staticmethod
    def _coordinate_prior(samples: Sequence[ComplexSample]) -> torch.Tensor:
        """Draw each ligand prior inside its observed pocket bounding box."""
        chunks: list[torch.Tensor] = []
        for sample in samples:
            pocket = sample.pocket.positions
            lower = pocket.amin(dim=0)
            upper = pocket.amax(dim=0)
            extent = (upper - lower).clamp_min(1.0)
            margin = 0.05 * extent
            count = sample.ligand.positions.shape[0]
            chunks.append(
                lower
                - margin
                + torch.rand(
                    (count, 3), dtype=pocket.dtype, device=pocket.device
                )
                * (extent + 2.0 * margin)
            )
        return torch.cat(chunks)

    @staticmethod
    def _common_properties(
        samples: Sequence[ComplexSample],
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Stack only finite scalar properties present in every complex."""
        names = set(samples[0].properties)
        for sample in samples[1:]:
            names.intersection_update(sample.properties)
        values: dict[str, torch.Tensor] = {}
        for name in sorted(names):
            scalars = [TrainingBatchBuilder._scalar(sample.properties[name]) for sample in samples]
            if all(value is not None for value in scalars):
                values[name] = torch.tensor(
                    [float(value) for value in scalars], dtype=dtype, device=device
                )
        return values

    @staticmethod
    def _property_target(
        samples: Sequence[ComplexSample],
        aliases: tuple[str, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract one optional per-complex scalar label and availability mask."""
        values = torch.zeros(len(samples), dtype=dtype, device=device)
        mask = torch.zeros(len(samples), dtype=torch.bool, device=device)
        for index, sample in enumerate(samples):
            by_name = {name.lower(): value for name, value in sample.properties.items()}
            for alias in aliases:
                if alias in by_name:
                    scalar = TrainingBatchBuilder._scalar(by_name[alias])
                    if scalar is not None:
                        values[index] = scalar
                        mask[index] = True
                    break
        return values, mask

    @staticmethod
    def _scalar(value: object) -> float | None:
        """Convert a finite numeric scalar or one-element tensor to float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                return None
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            candidate = float(value)
            return candidate if torch.isfinite(torch.tensor(candidate)) else None
        return None

    def _decoder_targets(
        self,
        samples: Sequence[ComplexSample],
        observations: Sequence[_FieldObservation],
        padded_latent: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[ElectronDecoderContext | None, dict[str, torch.Tensor | None]]:
        """Pad genuine-QM fields and construct the flattened decoder mapping."""
        device = padded_latent.device
        dtype = padded_latent.dtype
        qm_mask = torch.tensor(
            [observation.qm for observation in observations],
            dtype=torch.bool,
            device=device,
        )
        empty: dict[str, torch.Tensor | None] = {
            "qm_mask": qm_mask,
            "density": None,
            "density_gradient": None,
            "field_mask": None,
            "electron_count": None,
            "dipole": None,
            "latent_cycle": None,
            "latent_cycle_mask": None,
        }
        if not bool(qm_mask.any()):
            return None, empty
        max_points = max(
            observation.query_positions.shape[0]
            for observation in observations
            if observation.query_positions is not None
        )
        batch_size = len(samples)
        queries = torch.zeros((batch_size, max_points, 3), dtype=dtype, device=device)
        density = torch.zeros((batch_size, max_points), dtype=dtype, device=device)
        gradient = torch.zeros(
            (batch_size, max_points, 3), dtype=dtype, device=device
        )
        field_mask = torch.zeros(
            (batch_size, max_points), dtype=torch.bool, device=device
        )
        electron_count = torch.zeros(batch_size, dtype=dtype, device=device)
        dipole = torch.zeros((batch_size, 3), dtype=dtype, device=device)
        for index, observation in enumerate(observations):
            if not observation.qm:
                continue
            assert observation.query_positions is not None
            assert observation.density is not None
            assert observation.density_gradient is not None
            assert observation.electron_count is not None
            assert observation.dipole is not None
            count = observation.query_positions.shape[0]
            queries[index, :count] = observation.query_positions
            density[index, :count] = observation.density
            gradient[index, :count] = observation.density_gradient
            field_mask[index, :count] = True
            electron_count[index] = observation.electron_count
            dipole[index] = observation.dipole
        flat_index = torch.full(atom_mask.shape, -1, dtype=torch.long, device=device)
        offset = 0
        for index, sample in enumerate(samples):
            count = sample.ligand.positions.shape[0]
            flat_index[index, :count] = torch.arange(
                offset, offset + count, dtype=torch.long, device=device
            )
            offset += count
        context = ElectronDecoderContext(
            query_grid=queries,
            atom_mask=atom_mask,
            flat_index=flat_index,
        )
        return context, {
            "qm_mask": qm_mask,
            "density": density,
            "density_gradient": gradient,
            "field_mask": field_mask,
            "electron_count": electron_count,
            "dipole": dipole,
            "latent_cycle": padded_latent.detach(),
            "latent_cycle_mask": atom_mask,
        }

    @staticmethod
    def _valence_limits(
        atom_classes: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return conservative maximum valences aligned with ligand atoms."""
        limits = torch.tensor(
            (4, 4, 2, 6, 5, 1, 1, 1, 1, 3, 4, 6),
            dtype=dtype,
            device=atom_classes.device,
        )
        return limits[atom_classes]

    @staticmethod
    def _nonbonded_pairs(samples: Sequence[ComplexSample]) -> torch.Tensor:
        """Create canonical intraligand nonbonded pairs disjoint from bonds."""
        device = samples[0].ligand.positions.device
        chunks: list[torch.Tensor] = []
        offset = 0
        for sample in samples:
            count = sample.ligand.positions.shape[0]
            pairs = torch.combinations(
                torch.arange(count, dtype=torch.long, device=device), r=2
            ).transpose(0, 1)
            if pairs.shape[1] and sample.ligand.halfedge_index.shape[1]:
                codes = pairs[0] * count + pairs[1]
                bond_codes = (
                    sample.ligand.halfedge_index[0] * count
                    + sample.ligand.halfedge_index[1]
                )
                pairs = pairs[:, ~torch.isin(codes, bond_codes)]
            if pairs.shape[1]:
                chunks.append(pairs + offset)
            offset += count
        return (
            torch.cat(chunks, dim=1)
            if chunks
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
