"""Functional tests for dashboard CLI commands using typer.testing.CliRunner."""

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app


@pytest.fixture
def runner():
    """Create a CliRunner instance for testing."""
    return CliRunner()


def test_help_command(runner):
    """Test that the help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Pynomaly - State-of-the-art anomaly detection CLI" in result.stdout


def test_version_command(runner):
    """Test that the version command works."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Pynomaly" in result.stdout


def test_status_command(runner):
    """Test that the status command works."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Pynomaly System Status" in result.stdout


def test_generate_config_command(runner):
    """Test that the generate-config command works."""
    result = runner.invoke(app, ["generate-config", "test", "--output", "test_config.json"])
    assert result.exit_code == 0
    assert "Test configuration generated" in result.stdout


def test_available_subcommands(runner):
    """Test that expected subcommands are available."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    
    # Check that main commands are available
    expected_commands = [
        "version",
        "settings", 
        "status",
        "generate-config",
        "auto",
        "automl",
        "config",
        "detector",
        "dataset",
        "data",
        "detect",
        "deep-learning",
        "explainability",
        "selection",
        "export",
        "server",
        "perf",
    ]
    
    for command in expected_commands:
        assert command in result.stdout


def test_command_help_output(runner):
    """Test that individual commands have help output."""
    commands_to_test = ["version", "status", "generate-config"]
    
    for command in commands_to_test:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--help" in result.stdout
        assert "Show this message and exit" in result.stdout


def test_exit_codes(runner):
    """Test that commands return appropriate exit codes."""
    # Test successful commands
    successful_commands = [
        ["--help"],
        ["version"],
        ["status"],
    ]
    
    for command in successful_commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, f"Command {command} failed with exit code {result.exit_code}"


def test_error_handling(runner):
    """Test error handling for invalid commands."""
    # Test invalid command
    result = runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code != 0
    assert "No such command" in result.stdout


def test_output_format(runner):
    """Test that output contains expected formatting."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    # Check for rich formatting elements
    assert "Pynomaly System Status" in result.stdout
    assert "Component" in result.stdout
    assert "Status" in result.stdout


def test_cli_persistence(runner):
    """Test that CLI state is maintained between commands."""
    # Test that the CLI can handle multiple sequential commands
    result1 = runner.invoke(app, ["version"])
    assert result1.exit_code == 0
    
    result2 = runner.invoke(app, ["status"])
    assert result2.exit_code == 0
    
    # Both commands should work independently
    assert "Pynomaly" in result1.stdout
    assert "System Status" in result2.stdout


def test_verbose_flag(runner):
    """Test the verbose flag functionality."""
    result = runner.invoke(app, ["--verbose", "status"])
    assert result.exit_code == 0
    # Should still produce output even with verbose flag
    assert "System Status" in result.stdout


def test_quiet_flag(runner):
    """Test the quiet flag functionality."""
    result = runner.invoke(app, ["--quiet", "status"])
    assert result.exit_code == 0
    # Should still complete successfully even with quiet flag
    assert result.stdout is not None  # Some output might still be present


def test_command_robustness(runner):
    """Test that commands handle edge cases gracefully."""
    # Test with empty arguments where applicable
    result = runner.invoke(app, ["settings"])
    assert result.exit_code == 0
    # Remove ANSI color codes for testing
    import re
    clean_output = re.sub(r'\x1b\[.*?m', '', result.stdout)
    assert "Use --show to display settings" in clean_output
    
    # Test settings with show flag
    result = runner.invoke(app, ["settings", "--show"])
    assert result.exit_code == 0
    assert "Pynomaly Settings" in result.stdout
