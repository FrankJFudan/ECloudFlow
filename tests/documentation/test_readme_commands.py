import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

_COMMAND = re.compile(r"^\s*ecloudflow\s+([^\n`]+)", re.MULTILINE)
_KNOWN = {
    "doctor",
    "config",
    "data",
    "train",
    "benchmark",
    "sample",
    "evaluate",
    "report",
    "visualize",
}


def extract_ecloudflow_commands(path: Path) -> list[str]:
    commands = []
    for match in _COMMAND.finditer(path.read_text(encoding="utf-8")):
        command = " ".join(match.group(1).strip().split())
        command = command.split("\\")[0].strip()
        if command and command.split()[0] in _KNOWN:
            commands.append(command)
    return commands


def test_readme_commands_are_recognized_by_cli():
    try:
        from ecloudflow.cli.main import app
    except ModuleNotFoundError as error:
        pytest.skip(f"core runtime dependency is unavailable: {error}")
    runner = CliRunner()
    for command in extract_ecloudflow_commands(Path("README.md")):
        root = command.split()[0]
        args = [root, "--help"]
        if root == "visualize":
            args = ["visualize", "--help"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{command!r}: {result.stdout}"


def test_extractor_finds_core_workflows():
    commands = extract_ecloudflow_commands(Path("README.md"))
    roots = {command.split()[0] for command in commands}
    assert {"doctor", "sample", "evaluate", "report"}.issubset(roots)


def test_documented_benchmark_device_options_execute(tmp_path: Path) -> None:
    """The repeatable device syntax used in the README reaches the CLI."""
    from ecloudflow.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--devices",
            "1",
            "--devices",
            "2",
            "--devices",
            "4",
            "--steps",
            "1",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "scaling.json").is_file()
