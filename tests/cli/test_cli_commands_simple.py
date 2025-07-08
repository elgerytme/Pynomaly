"""Simple tests to verify CLI commands work without complex configuration."""

import sys
import subprocess
import pytest


def run_command(cmd):
    """Run a command and return exit code and output."""
    import os
    # Set environment variable to suppress TensorFlow warnings
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=".",
        timeout=30,
        env=env
    )
    return result.returncode, result.stdout, result.stderr


def test_main_help():
    """Test that the main help command works."""
    exit_code, stdout, stderr = run_command("python -m pynomaly --help")
    assert exit_code == 0
    assert "Pynomaly - State-of-the-art anomaly detection CLI" in stdout
    assert "Commands" in stdout


def test_version_command():
    """Test that version command works."""
    exit_code, stdout, stderr = run_command("python -m pynomaly version")
    assert exit_code == 0
    assert "Pynomaly v" in stdout
    assert "Python" in stdout


def test_status_command():
    """Test that status command works."""
    exit_code, stdout, stderr = run_command("python -m pynomaly status")
    assert exit_code == 0
    assert "Pynomaly System Status" in stdout
    assert "Component" in stdout


def test_generate_config_command():
    """Test that generate-config command works."""
    exit_code, stdout, stderr = run_command("python -m pynomaly generate-config test --output test_config.json")
    assert exit_code == 0
    assert "Test configuration generated" in stdout


def test_subcommand_help():
    """Test that subcommands have help."""
    commands = ["detect", "auto", "dataset", "detector"]
    
    for cmd in commands:
        exit_code, stdout, stderr = run_command(f"python -m pynomaly {cmd} --help")
        assert exit_code == 0, f"Command '{cmd} --help' failed"
        assert "Usage:" in stdout
        assert "Options" in stdout


def test_invalid_command():
    """Test error handling for invalid commands."""
    exit_code, stdout, stderr = run_command("python -m pynomaly invalid-command")
    assert exit_code != 0
    assert "No such command" in stdout


def test_cli_structure():
    """Test that CLI has expected structure."""
    exit_code, stdout, stderr = run_command("python -m pynomaly --help")
    assert exit_code == 0
    
    # Check for essential commands
    expected_commands = [
        "version", "status", "generate-config", "quickstart",
        "auto", "automl", "config", "detector", "dataset", 
        "data", "detect", "tdd", "deep-learning", "explainability",
        "selection", "export", "server", "perf"
    ]
    
    for cmd in expected_commands:
        assert cmd in stdout, f"Command '{cmd}' not found in help output"


if __name__ == "__main__":
    # Run tests manually
    test_main_help()
    test_version_command()
    test_status_command()
    test_generate_config_command()
    test_subcommand_help()
    test_invalid_command()
    test_cli_structure()
    print("All tests passed!")
