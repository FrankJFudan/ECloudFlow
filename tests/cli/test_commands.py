import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from typer.testing import CliRunner

from ecloudflow.cli.main import app


def test_config_show_accepts_hydra_override():
    result = CliRunner().invoke(
        app, ["config", "show", "--override", "sample.profile=fast"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["sample"]["profile"] == "fast"


def test_data_prepare_writes_resolved_manifest(tmp_path):
    result = CliRunner().invoke(
        app, ["data", "prepare", "--output-dir", str(tmp_path), "--dataset", "pdbbind"]
    )
    assert result.exit_code == 0, result.stdout
    manifest = tmp_path / "prepare.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text())["dataset"] == "pdbbind"


def test_train_dry_run_writes_launch_config(tmp_path):
    result = CliRunner().invoke(
        app, ["train", "--dry-run", "--output-dir", str(tmp_path), "--max-steps", "2"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "train-config.json").read_text())
    assert payload["dry_run"] is True
    assert payload["config"]["trainer"]["max_steps"] == 2
    assert payload["config"]["trainer"]["checkpoint_dir"].endswith(
        "checkpoints"
    )


def test_train_explicit_checkpoint_override_is_preserved(tmp_path):
    """An advanced checkpoint path override takes precedence over output-dir routing."""
    explicit = tmp_path / "explicit-checkpoints"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "run"),
            f"trainer.checkpoint_dir={explicit}",
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads((tmp_path / "run/train-config.json").read_text())
    assert payload["config"]["trainer"]["checkpoint_dir"] == str(explicit)


def test_train_dry_run_never_starts_runtime(tmp_path, monkeypatch):
    """Dry-run configuration resolution must never allocate or call fit."""
    start = Mock(side_effect=AssertionError("dry-run started training"))
    monkeypatch.setattr("ecloudflow.cli.train.run_training", start)
    result = CliRunner().invoke(
        app, ["train", "--dry-run", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    start.assert_not_called()


def test_train_command_starts_runtime_and_records_completion(tmp_path, monkeypatch):
    """A non-dry command must cross the real training application boundary."""
    trainer = SimpleNamespace(global_step=7)
    runtime = SimpleNamespace(
        trainer=trainer,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    start = Mock(return_value=runtime)
    monkeypatch.setattr("ecloudflow.cli.train.run_training", start)
    monkeypatch.delenv("RANK", raising=False)
    result = CliRunner().invoke(
        app, ["train", "--output-dir", str(tmp_path), "--max-steps", "7"]
    )
    assert result.exit_code == 0, result.stdout
    start.assert_called_once()
    passed_config, passed_output = start.call_args.args
    assert passed_config.trainer.max_steps == 7
    assert passed_output == tmp_path
    payload = json.loads((tmp_path / "train-config.json").read_text())
    assert payload["status"] == "completed"
    assert payload["global_step"] == 7


def test_train_invalid_configuration_fails_before_runtime(tmp_path, monkeypatch):
    """Strict Hydra/Pydantic errors must stop before model or Trainer creation."""
    start = Mock(side_effect=AssertionError("invalid configuration started training"))
    monkeypatch.setattr("ecloudflow.cli.train.run_training", start)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "trainer.devices=0",
        ],
    )
    assert result.exit_code != 0
    assert "greater than or equal to 1" in result.stderr
    start.assert_not_called()


def test_module_entrypoint_does_not_emit_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-m", "ecloudflow.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
