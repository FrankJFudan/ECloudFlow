"""Low-level candidate service used by the public generation pipeline."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ecloudflow.chemistry.decoder import BondDecodeProblem, ExactBondDecoder
from ecloudflow.chemistry.reconstruct import reconstruct_rdkit_molecule
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.profiles import SamplingProfile, get_profile
from ecloudflow.sampling.results import GenerationMode
from ecloudflow.sampling.solver import SamplingTrajectory

CandidateCallable = Callable[..., Any]


@dataclass
class SamplingPipeline:
    """Adapt model/sampler services to one normalized candidate interface.

    :param candidate_generator: Optional injected deterministic or learned
        service.  It receives keyword arguments ``pocket``, ``fragment``,
        ``mode``, ``profile``, ``seed``, ``generator``, and ``attempt``.
    :param model: Optional callable used when no explicit candidate generator
        is supplied.
    :param decoder: Exact graph decoder for MolecularState outputs.
    :param vocabulary: Ligand vocabulary used by reconstruction.
    :return: Candidate sampling adapter.
    :rtype: SamplingPipeline

    Production callers can inject a model-backed sampler while tests can use a
    deterministic sequence service.  The adapter never silently converts an
    invalid output into a fabricated molecule.
    """

    candidate_generator: CandidateCallable | None = None
    model: Any = None
    decoder: Any = None
    vocabulary: ChemicalVocabulary | None = None

    def __post_init__(self) -> None:
        """Fill the exact decoder lazily without importing optional tooling."""
        if self.decoder is None:
            self.decoder = ExactBondDecoder()
        if self.vocabulary is None:
            self.vocabulary = ChemicalVocabulary.default_ligand()
        if self.candidate_generator is not None and not callable(
            self.candidate_generator
        ):
            raise TypeError("candidate_generator must be callable.")

    def sample(
        self,
        *,
        pocket: Any,
        fragment: Any = None,
        fixed: Any = None,
        mode: GenerationMode = GenerationMode.DE_NOVO,
        profile: str | SamplingProfile = "balanced",
        seed: int = 2026,
        generator: torch.Generator | None = None,
        attempt: int = 1,
    ) -> Any:
        """Generate one raw candidate from an injected or learned service.

        :param pocket: Parsed pocket object or source path.
        :param fragment: Optional positioned fragment object/path.
        :param fixed: Optional exact :class:`FragmentCondition` used by the
            discrete decoder.  The public pipeline may pass a pre-built
            condition when fragment coordinates have already been parsed.
        :param mode: De novo or fragment-conditioned objective.
        :param profile: Named profile or resolved :class:`SamplingProfile`.
        :param seed: Deterministic per-attempt seed.
        :param generator: Optional caller-owned torch generator.
        :param attempt: One-based bounded attempt number.
        :return: RDKit molecule, SMILES, MolecularState, or trajectory.
        :rtype: object
        :raises RuntimeError: If no model/candidate service is configured.
        """
        resolved_profile = (
            profile if isinstance(profile, SamplingProfile) else get_profile(profile)
        )
        mode = mode if isinstance(mode, GenerationMode) else GenerationMode(mode)
        service = self.candidate_generator or self.model
        if service is None:
            raise RuntimeError("no candidate_generator or model is configured.")
        kwargs = {
            "pocket": pocket,
            "fragment": fragment,
            "mode": mode,
            "profile": resolved_profile,
            "seed": seed,
            "generator": generator,
            "attempt": attempt,
        }
        output = _invoke_service(service, kwargs)
        return self._normalize_state_output(output, fixed=fixed)

    def _normalize_state_output(self, output: Any, *, fixed: Any = None) -> Any:
        """Convert a trajectory/state to an RDKit molecule at the decoder edge."""
        if isinstance(output, dict):
            for key in ("molecule", "mol", "smiles", "state", "trajectory"):
                if key in output:
                    return self._normalize_state_output(output[key], fixed=fixed)
        if isinstance(output, SamplingTrajectory):
            output = output.final
        if isinstance(output, MolecularState):
            decoded = self.decoder.decode(
                BondDecodeProblem(
                    state=output,
                    vocabulary=self.vocabulary,
                    fixed=fixed,
                )
            )
            return reconstruct_rdkit_molecule(output, decoded, self.vocabulary)
        return output


def _invoke_service(service: CandidateCallable, kwargs: dict[str, Any]) -> Any:
    """Call services with compatible keyword subsets without hiding failures."""
    try:
        signature = inspect.signature(service)
    except (TypeError, ValueError):
        return service(**kwargs)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return service(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return service(**accepted)


__all__ = ["SamplingPipeline"]
