import json

from typer.testing import CliRunner

from ecloudflow.cli import doctor as doctor_module
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


def test_server_doctor_requires_four_gpu_bf16_and_nccl(monkeypatch):
    """The server preset must make all three accelerator checks mandatory."""
    monkeypatch.setattr(
        doctor_module,
        "_cuda_check",
        lambda minimum: {"ok": minimum == 4, "detail": f"required={minimum}"},
    )
    monkeypatch.setattr(
        doctor_module,
        "_bf16_check",
        lambda: {"ok": True, "detail": "supported=True"},
    )
    monkeypatch.setattr(
        doctor_module,
        "_nccl_check",
        lambda: {"ok": True, "detail": "available=True"},
    )

    result = CliRunner().invoke(app, ["doctor", "--server", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["cuda"]["detail"] == "required=4"
