"""Read-only compatibility importer for existing DiffGui LMDB datasets."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import lmdb

from ecloudflow.config.schema import DataConfig
from ecloudflow.core.types import ComplexSample
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.exceptions import DataValidationError


class DiffGuiLMDBImporter:
    """Decode existing DiffGui records into the canonical sample contract.

    :param path: Existing LMDB directory or single-file database.
    :param converter: Optional adapter from a decoded legacy object to one
        :class:`~ecloudflow.core.types.ComplexSample`. This is required when
        a DiffGui fork does not retain the official source filename fields.
    :param decoder: Byte decoder, normally ``pickle.loads`` for DiffGui data.
    :param source_root: Read-only root used to resolve official DiffGui
        ``protein_filename`` and ``ligand_filename`` values. Reparsing these
        sources recovers formal charges and stereochemistry that the processed
        legacy tensors do not store.
    :param build_fields: Whether canonical reparsing should build Task 5
        physical fields. The default graph-only import avoids requiring xTB.
    :return: Reiterable read-only compatibility importer.
    :rtype: DiffGuiLMDBImporter
    :raises DataValidationError: If the path is absent, bytes cannot be decoded,
        or conversion does not produce a validated ``ComplexSample``.

    The LMDB environment is opened with ``readonly=True``, ``create=False``,
    and ``lock=False``. No migration marker, lock file, or converted value is
    ever written into the reference repository or database.
    """

    def __init__(
        self,
        path: str | Path,
        converter: Callable[[Any], ComplexSample] | None = None,
        decoder: Callable[[bytes], Any] = pickle.loads,
        source_root: str | Path | None = None,
        build_fields: bool = False,
    ) -> None:
        self.path = Path(path)
        self.converter = converter
        self.decoder = decoder
        self.source_root = Path(source_root) if source_root is not None else None
        self.build_fields = build_fields

    @classmethod
    def from_config(
        cls,
        config: DataConfig,
        *,
        converter: Callable[[Any], ComplexSample] | None = None,
        decoder: Callable[[bytes], Any] = pickle.loads,
    ) -> DiffGuiLMDBImporter:
        """Construct an importer from strict portable Hydra data settings.

        :param config: Frozen ``DataConfig`` containing nullable DiffGui paths.
        :param converter: Optional converter for non-official DiffGui forks.
        :param decoder: Legacy value decoder, normally ``pickle.loads``.
        :return: Read-only importer configured without hidden machine paths.
        :rtype: DiffGuiLMDBImporter
        :raises DataValidationError: If ``diffgui_lmdb`` is not configured.

        This constructor performs no filesystem access or mutation. Path
        existence and source requirements are checked lazily during iteration.
        """
        if config.diffgui_lmdb is None:
            raise DataValidationError("data.diffgui_lmdb must be configured")
        return cls(
            path=config.diffgui_lmdb,
            converter=converter,
            decoder=decoder,
            source_root=config.diffgui_source_root,
            build_fields=config.diffgui_build_fields,
        )

    def __iter__(self) -> Iterator[ComplexSample]:
        """Yield canonical samples in byte-sorted LMDB key order.

        :return: Lazy iterator of Task 6 ``ComplexSample`` instances.
        :rtype: Iterator[ComplexSample]
        :raises DataValidationError: If the database cannot be opened read-only,
            decoding fails, official sources are unavailable, or a custom
            converter violates the canonical output contract.

        A fresh read-only LMDB environment is opened for each iteration. No
        lock, migration flag, source tensor, or external repository file is
        modified. Records are revalidated during canonical construction.
        """
        yield from self.iter_samples()

    def iter_samples(self) -> Iterator[ComplexSample]:
        """Open the source read-only and lazily convert every record.

        :return: Iterator of revalidated canonical samples.
        :rtype: Iterator[ComplexSample]
        :raises DataValidationError: If opening, decoding, or conversion fails.
        """
        if not self.path.exists():
            raise DataValidationError(f"DiffGui LMDB does not exist: {self.path}")
        try:
            environment = lmdb.open(
                str(self.path),
                subdir=self.path.is_dir(),
                readonly=True,
                create=False,
                lock=False,
                readahead=False,
                meminit=False,
            )
        except lmdb.Error as error:
            raise DataValidationError(
                f"failed to open DiffGui LMDB: {error}"
            ) from error
        try:
            with (
                environment.begin(write=False, buffers=False) as transaction,
                transaction.cursor() as cursor,
            ):
                for key, value in cursor:
                    try:
                        decoded = self.decoder(bytes(value))
                        converted = self._convert(decoded, bytes(key))
                    except Exception as error:
                        key_text = bytes(key).hex()
                        raise DataValidationError(
                            f"failed to convert DiffGui record {key_text}: "
                            f"{type(error).__name__}"
                        ) from error
                    if not isinstance(converted, ComplexSample):
                        raise DataValidationError(
                            "DiffGui converter must return a ComplexSample"
                        )
                    yield converted
        finally:
            environment.close()

    def _convert(self, decoded: Any, key: bytes) -> ComplexSample:
        """Convert canonical, custom-fork, or official filename-based records."""
        if isinstance(decoded, ComplexSample):
            return decoded
        if self.converter is not None:
            return self.converter(decoded)
        if not isinstance(decoded, dict) or self.source_root is None:
            raise DataValidationError(
                "legacy DiffGui records require source_root or a custom converter"
            )
        protein_filename = decoded.get("protein_filename")
        ligand_filename = decoded.get("ligand_filename")
        if not isinstance(protein_filename, str) or not isinstance(
            ligand_filename, str
        ):
            raise DataValidationError(
                "official DiffGui record lacks protein_filename or ligand_filename"
            )
        try:
            key_text = key.decode("utf-8")
        except UnicodeDecodeError:
            key_text = key.hex()
        return build_complex_sample(
            self.source_root / protein_filename,
            self.source_root / ligand_filename,
            sample_id=f"diffgui-{key_text}",
            build_fields=self.build_fields,
        )
