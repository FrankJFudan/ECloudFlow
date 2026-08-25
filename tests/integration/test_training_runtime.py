"""End-to-end smoke test for clean shards through a genuine Lightning fit."""

from pathlib import Path

from ecloudflow.config import load_config
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.data.shards import ShardWriter
from ecloudflow.training.runtime import run_training


def test_real_runtime_trains_one_cpu_step_and_checkpoints(tmp_path: Path) -> None:
    """The production assembly must optimize one canonical complex without mocks."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/complex"
    sample = build_complex_sample(
        fixture / "toy_pocket.pdb",
        fixture / "toy_ligand.sdf",
        "runtime-smoke",
        build_fields=False,
    )
    data_root = tmp_path / "data"
    ShardWriter().write([sample], data_root)
    config = load_config(["+experiment=smoke"])
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"shard_dir": str(data_root), "partition": "all"}
            ),
            "trainer": config.trainer.model_copy(
                update={
                    "checkpoint_dir": str(tmp_path / "checkpoints"),
                    "max_steps": 1,
                    "max_epochs": 1,
                }
            ),
        }
    )

    runtime = run_training(config, tmp_path / "run")

    assert runtime.trainer.global_step == 1
    assert (tmp_path / "checkpoints/last.ckpt").is_file()
    assert (tmp_path / "checkpoints/step-00000001.ckpt").is_file()
