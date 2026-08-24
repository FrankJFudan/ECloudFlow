"""End-to-end smoke coverage for the public command-line workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from ecloudflow.cli.main import app


def test_cli_tiny_end_to_end(tmp_path: Path) -> None:
    """Generate, rank, evaluate, report, and render one local fixture run."""
    runner = CliRunner()
    pocket = Path(__file__).parents[1] / "fixtures" / "complex" / "toy_pocket.pdb"
    sampled = runner.invoke(
        app,
        [
            "sample",
            str(pocket),
            "-n",
            "3",
            "--smoke",
            "--profile",
            "fast",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert sampled.exit_code == 0, sampled.stdout
    assert (tmp_path / "samples.csv").is_file()
    resolved_config = json.loads(
        (tmp_path / "resolved-config.json").read_text(encoding="utf-8")
    )
    assert resolved_config["config"]["sample"]["profile"] == "fast"
    assert resolved_config["config"]["sample"]["num_molecules"] == 3
    assert resolved_config["request"]["docking"] == "auto"
    with (tmp_path / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["molecule_id"] for row in rows] == [
        "toy_pocket-000001",
        "toy_pocket-000002",
        "toy_pocket-000003",
    ]
    assert all(row["docking_score"] for row in rows)

    evaluated = runner.invoke(app, ["evaluate", str(tmp_path), "--profile", "smoke"])
    assert evaluated.exit_code == 0, evaluated.stdout
    payload = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["rdkit_validity"]["status"] == "success"
    reported = runner.invoke(app, ["report", str(tmp_path), "--format", "paper"])
    assert reported.exit_code == 0, reported.stdout
    assert (tmp_path / "report.html").is_file()

    molecule_view = runner.invoke(
        app,
        [
            "visualize",
            "molecule",
            str(tmp_path),
            "--id",
            "toy_pocket-000001",
        ],
    )
    assert molecule_view.exit_code == 0, molecule_view.stdout
    assert (tmp_path / "molecule_toy_pocket-000001.html").is_file()


def test_cli_config_accepts_trailing_override() -> None:
    """Bare key=value arguments remain compatible with Hydra workflows."""
    result = CliRunner().invoke(app, ["config", "show", "sample.profile=fast"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["sample"]["profile"] == "fast"
