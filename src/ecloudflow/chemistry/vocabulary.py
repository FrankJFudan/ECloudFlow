"""Explicit, immutable chemical vocabularies for ligand and pocket atoms."""

from __future__ import annotations

from dataclasses import dataclass

LIGAND_ATOMS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se")
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
BOND_CLASSES = ("none", "single", "double", "triple")
POCKET_ATOMS = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "Na",
    "Mg",
    "K",
    "Ca",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
)


@dataclass(frozen=True)
class ChemicalVocabulary:
    """Define ordered categorical channels for one chemical graph domain.

    :param atom_symbols: Unique case-sensitive element symbols in model-channel
        order.
    :param formal_charges: Unique signed formal charges in model-channel order.
    :param bond_classes: Unique lower-case Kekule bond class names. The ligand
        model uses ``none``, ``single``, ``double``, and ``triple``.
    :param domain: Either ``"ligand"`` or ``"pocket"``. Unsupported ligand
        elements fail explicitly; the separate pocket domain may be expanded.
    :return: Immutable vocabulary with deterministic lookup methods.
    :rtype: ChemicalVocabulary
    :raises ValueError: If a channel sequence is empty, duplicated, or the
        domain is unknown.

    Aromatic is deliberately not a model bond class. Ligand targets use a
    Kekule representation and recover aromaticity only during final molecule
    sanitization.
    """

    atom_symbols: tuple[str, ...]
    formal_charges: tuple[int, ...]
    bond_classes: tuple[str, ...]
    domain: str

    def __post_init__(self) -> None:
        """Validate channel uniqueness and domain separation.

        :return: None.
        :rtype: None
        :raises ValueError: If any vocabulary invariant is invalid.
        """
        if self.domain not in {"ligand", "pocket"}:
            raise ValueError("chemical vocabulary domain must be 'ligand' or 'pocket'.")
        for name, values in (
            ("atom_symbols", self.atom_symbols),
            ("formal_charges", self.formal_charges),
            ("bond_classes", self.bond_classes),
        ):
            if not values:
                raise ValueError(f"{name} must contain at least one channel.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicate channels.")
        if any(not symbol for symbol in self.atom_symbols):
            raise ValueError("atom_symbols must contain non-empty element symbols.")
        if self.bond_classes[0] != "none":
            raise ValueError("bond_classes must place the none class first.")

    @classmethod
    def default_ligand(cls) -> ChemicalVocabulary:
        """Build the fixed model vocabulary for generated ligands.

        :return: Vocabulary with twelve supported ligand elements, five charge
            classes, and four Kekule bond classes in binding order.
        :rtype: ChemicalVocabulary
        """
        return cls(LIGAND_ATOMS, FORMAL_CHARGES, BOND_CLASSES, "ligand")

    @classmethod
    def default_pocket(
        cls, *, extra_elements: tuple[str, ...] = ()
    ) -> ChemicalVocabulary:
        """Build a separate, expandable protein-pocket atom vocabulary.

        :param extra_elements: Additional unique element symbols appended after
            the built-in protein atoms and common biological metals.
        :return: Pocket-only vocabulary; expansion never changes ligand model
            channels.
        :rtype: ChemicalVocabulary
        :raises ValueError: If an extra symbol is empty or duplicates a built-in
            or earlier extra element.
        """
        atoms = POCKET_ATOMS + tuple(extra_elements)
        return cls(atoms, FORMAL_CHARGES, BOND_CLASSES, "pocket")

    def atom_index(self, symbol: str) -> int:
        """Return the categorical index for one case-sensitive element symbol.

        :param symbol: Chemical element symbol such as ``"C"`` or ``"Cl"``.
        :return: Zero-based atom-channel index.
        :rtype: int
        :raises ValueError: If the element is unsupported by this domain. The
            ligand error explicitly identifies unsupported ligand elements.
        """
        try:
            return self.atom_symbols.index(symbol)
        except ValueError as error:
            label = "ligand" if self.domain == "ligand" else "pocket"
            raise ValueError(f"unsupported {label} element: {symbol}") from error

    def atom_symbol(self, index: int) -> str:
        """Return the element symbol for one atom-channel index.

        :param index: Zero-based channel index.
        :return: Case-sensitive element symbol.
        :rtype: str
        :raises ValueError: If ``index`` is outside the atom vocabulary.
        """
        if index < 0 or index >= len(self.atom_symbols):
            raise ValueError(f"atom index outside vocabulary: {index}")
        return self.atom_symbols[index]

    def charge_index(self, charge: int) -> int:
        """Return the categorical index for one signed formal charge.

        :param charge: Integer formal charge.
        :return: Zero-based charge-channel index.
        :rtype: int
        :raises ValueError: If the charge is not represented.
        """
        try:
            return self.formal_charges.index(charge)
        except ValueError as error:
            raise ValueError(f"unsupported formal charge: {charge}") from error

    def bond_index(self, bond_class: str) -> int:
        """Return the categorical index for one Kekule bond class.

        :param bond_class: Lower-case class name such as ``"single"``.
        :return: Zero-based bond-channel index.
        :rtype: int
        :raises ValueError: If the class is absent.
        """
        try:
            return self.bond_classes.index(bond_class)
        except ValueError as error:
            raise ValueError(f"unsupported bond class: {bond_class}") from error

    @property
    def bond_orders(self) -> tuple[float, ...]:
        """Return numeric bond order aligned with ``bond_classes``.

        :return: Bond orders where ``none`` is zero and the three covalent
            classes have orders one through three.
        :rtype: tuple[float, ...]
        :raises ValueError: If a custom class has no supported numeric order.
        """
        order_by_name = {"none": 0.0, "single": 1.0, "double": 2.0, "triple": 3.0}
        try:
            return tuple(order_by_name[name] for name in self.bond_classes)
        except KeyError as error:
            raise ValueError(f"unsupported bond class: {error.args[0]}") from error
