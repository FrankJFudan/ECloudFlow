"""Immutable provenance contracts for physical electron-field builders.

The cube parsing conventions in :mod:`ecloudflow.ecloud.xtb` were adapted at
the algorithm level from ``ecloud_utils/cubtools.py`` and
``ecloud_utils/xtb_density.py`` in the ECloudGen official repository at commit
``9c0e3b1c48ece7d5ca7b4f991f318e3b38fd2edc``. The inspected upstream snapshot
contains no license file or source-file license header, so its license is
recorded as ``NOASSERTION`` rather than inferred. The implementation here is a
new, minimal, validated parser and does not copy the upstream subprocess code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecloudflow.ecloud.pocket import PocketFieldBuilder
    from ecloudflow.ecloud.xtb import XTBRunner


@dataclass(frozen=True)
class SourceAttribution:
    """Record one externally informed implementation source.

    :param project: Human-readable upstream project name.
    :param source_url: Stable public repository URL.
    :param revision: Exact upstream revision inspected during adaptation.
    :param source_files: Relative upstream paths that informed the code.
    :param source_hashes: SHA-256 hashes of the exact inspected source files.
    :param license_identifier: SPDX identifier or ``NOASSERTION`` when the
        inspected source does not state a license.
    :param adaptation: English description of the adapted concepts.
    :return: Immutable third-party notice metadata.
    :rtype: SourceAttribution
    :raises ValueError: If any required attribution value is empty.
    """

    project: str
    source_url: str
    revision: str
    source_files: tuple[str, ...]
    source_hashes: Mapping[str, str]
    license_identifier: str
    adaptation: str

    def __post_init__(self) -> None:
        """Validate complete, non-empty attribution metadata.

        :return: None.
        :rtype: None
        :raises ValueError: If a notice field or source path is empty.
        """
        values = (
            self.project,
            self.source_url,
            self.revision,
            self.license_identifier,
            self.adaptation,
        )
        if any(not value.strip() for value in values):
            raise ValueError("attribution fields must be non-empty strings.")
        if isinstance(self.source_files, str):
            raise TypeError("source_files must be a sequence of paths.")
        source_files = tuple(self.source_files)
        if not source_files or any(not path for path in source_files):
            raise ValueError("source_files must contain non-empty paths.")
        object.__setattr__(self, "source_files", source_files)
        hashes = dict(self.source_hashes)
        if set(hashes) != set(source_files) or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise ValueError("source_hashes must hash every source file with SHA-256.")
        object.__setattr__(self, "source_hashes", MappingProxyType(hashes))


ECLOUDGEN_CUBE_ATTRIBUTION = SourceAttribution(
    project="ECloudGen official",
    source_url="https://github.com/OdinZhang/ECloudGen_official",
    revision="9c0e3b1c48ece7d5ca7b4f991f318e3b38fd2edc",
    source_files=("ecloud_utils/cubtools.py", "ecloud_utils/xtb_density.py"),
    source_hashes={
        "ecloud_utils/cubtools.py": (
            "ec2255c0b056cc7c52713dac99e32d1798fe0ce9eb3df075642fdaeafc04ef42"
        ),
        "ecloud_utils/xtb_density.py": (
            "56bca8012bc827fc331b73e8a31123535a1b1824a6132ea428b2f935c1242dd3"
        ),
    },
    license_identifier="NOASSERTION",
    adaptation=(
        "Gaussian cube axis parsing, atomic-unit conversion, and zero-filled "
        "regular-grid interpolation conventions; rewritten with strict validation."
    ),
)


@dataclass(frozen=True)
class ToolProvenance:
    """Describe one isolated external-tool attempt without sensitive data.

    :param tool: Stable tool name, normally ``"xTB"``.
    :param version: Parsed tool version or ``"unavailable"``.
    :param executable: Executable name or path supplied to the runner.
    :param command: Exact argument-list command, excluding environment values.
    :param charge: Integer molecular charge supplied to the calculation.
    :param multiplicity: Positive spin multiplicity supplied to the calculation.
    :param integrated_electron_count: Integrated cube electron count in
        electrons, when a density cube was accepted.
    :param failure_category: Stable sanitized failure category string.
    :param source_hashes: SHA-256 hashes of molecular, XYZ, cube, stdout, and
        stderr inputs or outputs that exist for this attempt.
    :return: Immutable, credential-free provenance.
    :rtype: ToolProvenance
    :raises ValueError: If identifiers, arguments, hashes, or spin data are invalid.

    Environment variables are intentionally absent: they may contain tokens or
    credentials and are not required to reproduce the molecular calculation.
    """

    tool: str
    version: str
    executable: str
    command: tuple[str, ...]
    charge: int
    multiplicity: int
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    integrated_electron_count: float | None = None
    failure_category: str = "none"

    def __post_init__(self) -> None:
        """Validate and freeze tool provenance mappings.

        :return: None.
        :rtype: None
        :raises ValueError: If provenance is incomplete or malformed.
        """
        if not self.tool or not self.version or not self.executable:
            raise ValueError("tool, version, and executable must be non-empty.")
        if isinstance(self.command, str):
            raise TypeError("command must be a sequence of arguments.")
        command = tuple(self.command)
        if not command or any(not argument for argument in command):
            raise ValueError("command must contain non-empty arguments.")
        object.__setattr__(self, "command", command)
        if not isinstance(self.charge, int) or isinstance(self.charge, bool):
            raise ValueError("charge must be an integer.")  # noqa: TRY004
        if (
            not isinstance(self.multiplicity, int)
            or isinstance(self.multiplicity, bool)
            or self.multiplicity <= 0
        ):
            raise ValueError("multiplicity must be a positive integer.")
        hashes = dict(self.source_hashes)
        if any(
            not key
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for key, value in hashes.items()
        ):
            raise ValueError("source_hashes must contain lowercase SHA-256 values.")
        object.__setattr__(self, "source_hashes", MappingProxyType(hashes))
        if self.integrated_electron_count is not None and (
            not isinstance(self.integrated_electron_count, (int, float))
            or isinstance(self.integrated_electron_count, bool)
            or self.integrated_electron_count < 0
            or self.integrated_electron_count == float("inf")
            or self.integrated_electron_count != self.integrated_electron_count
        ):
            raise ValueError(
                "integrated_electron_count must be finite and non-negative."
            )
        if not isinstance(self.failure_category, str) or not self.failure_category:
            raise ValueError("failure_category must be a non-empty string.")


@dataclass(frozen=True)
class FieldBuilderBundle:
    """Bundle compatible pocket and ligand physical-field builders.

    :param pocket_builder: Deterministic physical pocket-field builder.
    :param ligand_builder: Isolated xTB ligand-density runner.
    :return: Immutable builder pair used by complex preprocessing.
    :rtype: FieldBuilderBundle
    """

    pocket_builder: PocketFieldBuilder
    ligand_builder: XTBRunner

    @classmethod
    def default(
        cls,
        *,
        pocket_builder: PocketFieldBuilder | None = None,
        ligand_builder: XTBRunner | None = None,
    ) -> FieldBuilderBundle:
        """Construct the default physical pocket and xTB ligand builders.

        :param pocket_builder: Optional caller-configured pocket builder.
        :param ligand_builder: Optional caller-configured xTB runner.
        :return: Bundle with omitted builders replaced by safe defaults.
        :rtype: FieldBuilderBundle

        Importing concrete builders lazily avoids a provenance-module import
        cycle. Construction has no external side effects; xTB runs only when
        :meth:`ecloudflow.ecloud.xtb.XTBRunner.calculate_ligand` is called.
        """
        from ecloudflow.ecloud.pocket import PocketFieldBuilder
        from ecloudflow.ecloud.xtb import XTBRunner

        return cls(
            pocket_builder=(
                PocketFieldBuilder.default()
                if pocket_builder is None
                else pocket_builder
            ),
            ligand_builder=XTBRunner() if ligand_builder is None else ligand_builder,
        )

    @property
    def pocket(self) -> PocketFieldBuilder:
        """Return the pocket builder compatibility alias.

        :return: Configured pocket-field builder.
        :rtype: PocketFieldBuilder
        """
        return self.pocket_builder

    @property
    def ligand(self) -> XTBRunner:
        """Return the ligand builder compatibility alias.

        :return: Configured xTB ligand-density runner.
        :rtype: XTBRunner
        """
        return self.ligand_builder
