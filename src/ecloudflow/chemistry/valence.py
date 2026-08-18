"""Charge-conditioned valence rules adapted from CoCoGraph-style semantics.

The table follows CoCoGraph's useful distinction between vetted base rules and
valences supported by dataset occurrence counts. The semantic source is
``https://github.com/manurubo/CoCoGraph`` (MIT License, Manuel Ruiz, 2026), in
particular its valence-distribution utilities. This module is an independent
English implementation and does not modify or import the reference repository.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary

ValenceKey = tuple[str, int]
ValenceCountKey = tuple[str, int, int]


_NEUTRAL_MAXIMA = {
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 6,
    "P": 5,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "B": 3,
    "Si": 4,
    "Se": 6,
}


def _default_maximum(symbol: str, charge: int) -> int:
    """Return a conservative maximum for one supported element-charge pair.

    :param symbol: Supported ligand element symbol.
    :param charge: Formal charge in the configured vocabulary.
    :return: Non-negative maximum covalent bond-order sum.
    :rtype: int
    """
    neutral = _NEUTRAL_MAXIMA[symbol]
    if symbol in {"F", "Cl", "Br", "I"}:
        return {-2: 0, -1: 0, 0: 1, 1: 2, 2: 3}[charge]
    if symbol == "O":
        return {-2: 0, -1: 1, 0: 2, 1: 3, 2: 3}[charge]
    if symbol == "N":
        return {-2: 1, -1: 2, 0: 3, 1: 4, 2: 4}[charge]
    if symbol == "B":
        return {-2: 1, -1: 2, 0: 3, 1: 2, 2: 1}[charge]
    if symbol in {"S", "P", "Se"}:
        return max(0, neutral - abs(charge))
    return max(0, neutral - abs(charge))


@dataclass(frozen=True)
class ValenceTable:
    """Store maximum bond-order sums keyed by element and formal charge.

    :param maxima: Mapping from ``(element_symbol, formal_charge)`` to a finite,
        non-negative maximum valence.
    :return: Immutable charge-conditioned valence lookup table.
    :rtype: ValenceTable
    :raises ValueError: If keys or values are malformed.

    The table constrains only chemistry represented by a configured vocabulary.
    It does not claim universal validity for radicals, metals, or unusual
    hypervalent states.
    """

    maxima: Mapping[ValenceKey, float]

    def __post_init__(self) -> None:
        """Validate and defensively freeze all valence entries.

        :return: None.
        :rtype: None
        :raises ValueError: If a key or maximum is invalid.
        """
        copied: dict[ValenceKey, float] = {}
        for key, value in self.maxima.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or not isinstance(key[1], int)
            ):
                raise ValueError("valence keys must be (element, formal_charge) pairs.")
            maximum = float(value)
            if maximum < 0 or not torch.isfinite(torch.tensor(maximum)):
                raise ValueError("maximum valences must be finite and non-negative.")
            copied[key] = maximum
        object.__setattr__(self, "maxima", MappingProxyType(copied))

    @classmethod
    def default(cls, vocabulary: ChemicalVocabulary) -> ValenceTable:
        """Build vetted defaults for every configured ligand channel pair.

        :param vocabulary: Ligand vocabulary whose atom and charge channels
            define required lookup keys.
        :return: Complete default table aligned with the vocabulary.
        :rtype: ValenceTable
        :raises ValueError: If a pocket vocabulary or unknown ligand element is
            supplied.
        """
        if vocabulary.domain != "ligand":
            raise ValueError("valence defaults require a ligand vocabulary.")
        maxima = {
            (symbol, charge): float(_default_maximum(symbol, charge))
            for symbol in vocabulary.atom_symbols
            for charge in vocabulary.formal_charges
        }
        return cls(maxima)

    def maximum(self, symbol: str, charge: int) -> float:
        """Return the configured maximum valence for one chemical state.

        :param symbol: Ligand element symbol.
        :param charge: Signed formal charge.
        :return: Maximum allowed covalent bond-order sum.
        :rtype: float
        :raises ValueError: If the element-charge pair has no rule.
        """
        try:
            return self.maxima[(symbol, charge)]
        except KeyError as error:
            raise ValueError(
                f"no valence rule for element {symbol} with charge {charge:+d}"
            ) from error

    def with_dataset_counts(
        self,
        counts: Mapping[ValenceCountKey, int],
        *,
        minimum_count: int = 1,
    ) -> ValenceTable:
        """Extend maxima using sufficiently observed dataset valence counts.

        :param counts: Occurrence counts keyed by ``(element, charge, valence)``.
            Entries never remove vetted defaults; qualifying observed valences
            can only increase the corresponding maximum.
        :param minimum_count: Positive inclusive occurrence threshold.
        :return: New immutable table with deterministic count-derived maxima.
        :rtype: ValenceTable
        :raises ValueError: If the threshold, keys, counts, or valences are
            invalid, or an element-charge pair lacks a vetted base entry.

        This is adapted from CoCoGraph's data-derived constraint semantics. A
        count table is configuration evidence, not a universal chemistry rule.
        """
        if minimum_count < 1:
            raise ValueError("minimum_count must be at least one.")
        extended = dict(self.maxima)
        for key, count in sorted(counts.items()):
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or not isinstance(key[0], str)
                or not isinstance(key[1], int)
                or not isinstance(key[2], int)
                or key[2] < 0
            ):
                raise ValueError("dataset count keys must be (element, charge, valence).")
            if not isinstance(count, int) or count < 0:
                raise ValueError("dataset valence counts must be non-negative integers.")
            base_key = (key[0], key[1])
            if base_key not in extended:
                raise ValueError(f"dataset count has no vetted base rule: {base_key!r}")
            if count >= minimum_count:
                extended[base_key] = max(extended[base_key], float(key[2]))
        return ValenceTable(extended)

    def tensor(
        self,
        vocabulary: ChemicalVocabulary,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Materialize the small atom-by-charge valence lookup tensor.

        :param vocabulary: Vocabulary defining row and column order.
        :param dtype: Floating dtype matching trajectory logits.
        :param device: Device matching trajectory logits.
        :return: Maximum-valence tensor with shape ``[A, Q]``.
        :rtype: torch.Tensor
        :raises ValueError: If any vocabulary pair lacks a configured rule.

        This allocation is proportional to categorical channels, not node
        count; no dense molecular graph is materialized.
        """
        values = [
            [self.maximum(symbol, charge) for charge in vocabulary.formal_charges]
            for symbol in vocabulary.atom_symbols
        ]
        return torch.tensor(values, dtype=dtype, device=device)
