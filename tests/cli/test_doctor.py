import json

from typer.testing import CliRunner

from ecloudflow.cli.main import app


def test_doctor_json_reports_required_checks():
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert {"python", "torch", "rdkit", "hydra"} <= payload.keys()


def test_sample_help_exposes_simple_count_and_fragment_options():
    result = CliRunner().invoke(app, ["sample", "--help"])
    assert result.exit_code == 0
    assert "--num-molecules" in result.stdout
    assert "--fragment" in result.stdout
    assert "--profile" in result.stdout
