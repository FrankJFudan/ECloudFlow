import json
import subprocess
import sys

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


def test_module_entrypoint_does_not_emit_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-m", "ecloudflow.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
