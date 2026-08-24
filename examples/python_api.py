"""Minimal public Python API workflow for a trained ECloudFlow checkpoint."""

from ecloudflow import ECloudFlowPipeline


def main() -> None:
    """Generate and export a bounded set for one prepared pocket."""
    pipeline = ECloudFlowPipeline.from_pretrained(
        "checkpoints/ecloudflow-large.ckpt",
        map_location="cuda",
    )
    result = pipeline.generate(
        pocket="examples/3ztx_pocket.pdb",
        fragment=None,
        num_molecules=100,
        profile="balanced",
        output_dir="runs/3ztx",
    )
    docking = pipeline.dock_and_rank(
        result,
        "3ZTX",
        pocket="examples/3ztx_pocket.pdb",
        output_dir="runs/3ztx",
    )
    print(f"ranked {len(docking.ranked)} molecules")
    result.to_excel("runs/3ztx/summary.xlsx")


if __name__ == "__main__":
    main()
