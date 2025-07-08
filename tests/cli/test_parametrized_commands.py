import pytest
from typer.testing import CliRunner
from pynomaly.presentation.cli.app import app

runner = CliRunner(mix_stderr=False)

@pytest.mark.parametrize("command", [
    "alert --help",
    "benchmarking --help",
    "cost_optimization --help",
    # Add more commands and subcommands
])
def test_help_commands(command):
    result = runner.invoke(app, command.split())
    assert result.exit_code == 0

@pytest.mark.parametrize("command, args", [
    ("alert create", ["--title", "Test", "--description", "Sample", "--severity", "high"]),
    ("benchmarking comprehensive", ["--suite-name", "test"]),
    # Add more commands with happy path arguments
])
def test_commands_happy_path(command, args):
    result = runner.invoke(app, [command] + args)
    assert result.exit_code == 0

@pytest.mark.parametrize("command, args", [
    ("alert create", ["--title", "Test", "--severity", "high", "--severity", "low"]),  # Mutually exclusive
    ("benchmarking comprehensive", ["--suite-name"]),  # Missing required arg
    # Add more invalid combinations
])
def test_commands_invalid_combinations(command, args):
    result = runner.invoke(app, [command] + args)
    assert result.exit_code != 0

# Integrate pytest-snapshots if needed for capturing CLI output snapshots

