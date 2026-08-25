"""Dataset preparation command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from ecloudflow.cli.common import atomic_write_json, merge_overrides
from ecloudflow.config import load_config

app = typer.Typer(help="Prepare and inspect dataset inputs.")


@app.command("import-local")
def import_local(
    source_root: Annotated[
        Path,
        typer.Option(
            "--source-root",
            help="Extracted local dataset root; this command never downloads data.",
        ),
    ],
    dataset: Annotated[
        str, typer.Option("--dataset", help="pdbbind or crossdocked.")
    ] = "pdbbind",
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o")] = None,
    index_path: Annotated[
        Path | None,
        typer.Option(
            "--index",
            help="Explicit PDBBind INDEX or CrossDocked completeset .types file.",
        ),
    ] = None,
    protein_clusters: Annotated[
        Path | None,
        typer.Option(
            "--protein-clusters",
            help="Optional two-column protein_id cluster_id mapping.",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Bounded ordered preprocessing workers."),
    ] = 1,
    rmsd_threshold: Annotated[
        float,
        typer.Option(
            "--rmsd-threshold",
            min=0.0,
            help="Inclusive CrossDocked pose RMSD cutoff in angstroms.",
        ),
    ] = 1.0,
    pocket_radius: Annotated[
        float,
        typer.Option(
            "--pocket-radius",
            min=0.1,
            help="Residue distance cutoff for generated pockets in angstroms.",
        ),
    ] = 10.0,
    train_fraction: Annotated[
        float, typer.Option("--train-fraction", min=0.0, max=1.0)
    ] = 0.8,
    val_fraction: Annotated[
        float, typer.Option("--val-fraction", min=0.0, max=1.0)
    ] = 0.1,
    sequence_identity: Annotated[
        float, typer.Option("--sequence-identity", min=0.0, max=1.0)
    ] = 0.4,
    ligand_tanimoto: Annotated[
        float, typer.Option("--ligand-tanimoto", min=0.0, max=1.0)
    ] = 0.8,
    split_seed: Annotated[int, typer.Option("--split-seed")] = 2026,
    max_pairwise_sequences: Annotated[
        int,
        typer.Option(
            "--max-pairwise-sequences",
            min=1,
            help="Require precomputed protein clusters above this raw-sequence count.",
        ),
    ] = 5000,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Deterministic smoke-test record limit."),
    ] = None,
    strict_sources: Annotated[
        bool,
        typer.Option(
            "--strict-sources",
            help="Abort before publication when any indexed source is invalid.",
        ),
    ] = False,
    no_fields: Annotated[
        bool,
        typer.Option(
            "--no-fields", help="Skip pocket and xTB ligand electron-field generation."
        ),
    ] = False,
    max_samples_per_shard: Annotated[
        int | None,
        typer.Option(
            "--max-samples-per-shard",
            min=1,
            help="Optional operational bound; byte target remains authoritative.",
        ),
    ] = None,
    overrides: Annotated[list[str] | None, typer.Option("--override", "-O")] = None,
    trailing: Annotated[
        list[str] | None, typer.Argument(help="Optional trailing key=value overrides.")
    ] = None,
) -> None:
    """Import a complete local PDBBind or CrossDocked source into shards.

    :param source_root: Existing extracted dataset root. Network access and
        dataset-license acceptance are deliberately outside this command.
    :param dataset: Supported local source family.
    :param output_dir: Content-addressed WebDataset root.
    :param index_path: Optional explicit affinity or pose index.
    :param protein_clusters: Optional scalable leakage-control cluster mapping.
    :param workers: Bounded ordered graph/field preprocessing concurrency.
    :param rmsd_threshold: CrossDocked RMSD filter; ignored for PDBBind.
    :param pocket_radius: Residue cutoff used when a pocket PDB is unavailable.
    :param train_fraction: Connected-component train target.
    :param val_fraction: Connected-component validation target.
    :param sequence_identity: Raw-sequence fallback identity threshold.
    :param ligand_tanimoto: Split-audit ligand similarity threshold.
    :param split_seed: Deterministic component allocation seed.
    :param max_pairwise_sequences: Safety bound for quadratic raw alignment.
    :param limit: Optional deterministic source prefix for smoke tests.
    :param strict_sources: Whether any invalid indexed record aborts publication.
    :param no_fields: Whether to skip physical electron-field construction.
    :param max_samples_per_shard: Optional shard sample-count bound.
    :param overrides: Hydra overrides, including target shard size.
    :param trailing: Optional trailing Hydra overrides.
    :return: None after publishing manifest and ``import-summary.json``.
    :rtype: None

    The importer validates sources before splitting, then rebuilds accepted
    records through a bounded ordered stream. PDBBind pK labels, assay type,
    censoring relation, units, and raw expression remain in sample properties.
    Missing/invalid rows are never silently replaced; non-strict runs list them
    in the summary while strict runs fail before a manifest is published.
    """
    from ecloudflow.data.importers import (
        DatasetFamily,
        LocalImportOptions,
        import_local_dataset,
    )

    family = dataset.casefold()
    if family not in {"pdbbind", "crossdocked"}:
        raise typer.BadParameter("--dataset must be pdbbind or crossdocked")
    destination = output_dir or Path("data/processed") / family
    try:
        config = load_config([*merge_overrides(overrides, trailing), f"data={family}"])
        result = import_local_dataset(
            LocalImportOptions(
                dataset=cast(DatasetFamily, family),
                source_root=source_root,
                output_dir=destination,
                index_path=index_path,
                protein_clusters=protein_clusters,
                build_fields=not no_fields,
                workers=workers,
                strict_sources=strict_sources,
                rmsd_threshold=rmsd_threshold,
                pocket_radius=pocket_radius,
                sequence_identity=sequence_identity,
                ligand_tanimoto=ligand_tanimoto,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
                split_seed=split_seed,
                max_pairwise_sequences=max_pairwise_sequences,
                limit=limit,
                target_shard_size_gb=config.data.target_shard_size_gb,
                max_samples_per_shard=max_samples_per_shard,
            )
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise typer.BadParameter(f"local dataset import failed: {error}") from error
    summary_path = destination / "import-summary.json"
    atomic_write_json(
        summary_path,
        {
            "dataset": family,
            "source_root": str(source_root.resolve()),
            "manifest": str(destination / "manifest.json"),
            **result.as_dict(),
        },
    )
    typer.echo(f"manifest: {destination / 'manifest.json'}")
    typer.echo(f"summary: {summary_path}")
    typer.echo(
        f"samples: {len(result.manifest.sample_ids)}; "
        f"issues: {len(result.issues)}; filtered: {result.filtered_count}"
    )


@app.command("prepare")
def prepare(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset family name.")
    ] = "pdbbind",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/prepared"
    ),
    overrides: Annotated[list[str] | None, typer.Option("--override", "-O")] = None,
    pocket: Annotated[
        Path | None, typer.Option("--pocket", help="Optional source pocket PDB.")
    ] = None,
    ligand: Annotated[
        Path | None, typer.Option("--ligand", help="Optional source ligand SDF.")
    ] = None,
    sample_id: Annotated[str, typer.Option("--sample-id")] = "sample-000001",
    no_fields: Annotated[
        bool,
        typer.Option("--no-fields", help="Skip optional physical-field generation."),
    ] = False,
    trailing: Annotated[
        list[str] | None, typer.Argument(help="Optional trailing key=value overrides.")
    ] = None,
) -> None:
    """Validate preparation settings and write a reproducible run manifest.

    :param dataset: Named dataset family passed to data services.
    :param output_dir: Destination for the preparation manifest.
    :param overrides: Optional Hydra configuration overrides.
    :return: None; a preparation manifest is written to ``output_dir``.
    :rtype: None

    This command intentionally does not fabricate samples. Dataset-specific
    importers consume the manifest as their explicit next-stage input.
    """
    try:
        config = load_config(
            [*merge_overrides(overrides, trailing), f"data.dataset={dataset}"]
        )
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if (pocket is None) != (ligand is None):
        raise typer.BadParameter("--pocket and --ligand must be supplied together")
    if pocket is not None and ligand is not None:
        from ecloudflow.data.parsers import build_complex_sample
        from ecloudflow.data.shards import ShardWriter

        if not pocket.is_file() or not ligand.is_file():
            raise typer.BadParameter("source pocket and ligand files must exist")
        try:
            sample = build_complex_sample(
                pocket,
                ligand,
                sample_id,
                build_fields=not no_fields,
            )
            manifest_obj = ShardWriter(
                target_shard_size_gb=config.data.target_shard_size_gb,
                max_samples_per_shard=1,
            ).write((sample,), output_dir)
        except Exception as error:
            raise typer.BadParameter(f"dataset preparation failed: {error}") from error
        typer.echo(f"manifest: {output_dir / 'manifest.json'}")
        typer.echo(f"samples: {len(manifest_obj.sample_ids)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "prepare.json"
    atomic_write_json(
        manifest,
        {
            "dataset": dataset,
            "config": config.model_dump(mode="json"),
            "status": "preflight",
            "next_step": "provide --pocket and --ligand or use a dataset importer",
        },
    )
    typer.echo(f"preparation manifest: {manifest}")
