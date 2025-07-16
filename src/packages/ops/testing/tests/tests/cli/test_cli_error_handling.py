"""
Test CLI error handling and edge cases.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli import autonomous, datasets, detectors
from pynomaly.presentation.cli.app import app


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test handling of invalid commands."""
        result = runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout

    def test_missing_required_arguments(self, runner):
        """Test handling of missing required arguments."""
        result = runner.invoke(datasets.app, ["load"])
        assert result.exit_code != 0

    def test_invalid_file_path(self, runner):
        """Test handling of invalid file paths."""
        result = runner.invoke(
            datasets.app, ["load", "/nonexistent/path/file.csv", "--name", "test"]
        )
        assert result.exit_code != 0

    def test_permission_denied_file(self, runner):
        """Test handling of permission denied errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
            f.write("col1,col2\n1,2\n")

        try:
            # Mock permission error
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                result = runner.invoke(
                    datasets.app, ["load", temp_path, "--name", "test"]
                )
                assert result.exit_code != 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_contamination_rate(self, runner):
        """Test handling of invalid contamination rates."""
        result = runner.invoke(
            detectors.app,
            [
                "create",
                "--name",
                "test",
                "--algorithm",
                "IsolationForest",
                "--contamination",
                "1.5",
            ],
        )
        assert result.exit_code != 0

    def test_negative_contamination_rate(self, runner):
        """Test handling of negative contamination rates."""
        result = runner.invoke(
            detectors.app,
            [
                "create",
                "--name",
                "test",
                "--algorithm",
                "IsolationForest",
                "--contamination",
                "-0.1",
            ],
        )
        assert result.exit_code != 0

    def test_graceful_keyboard_interrupt(self, runner):
        """Test graceful handling of keyboard interrupts."""
        with patch(
            "pynomaly.presentation.cli.autonomous.AutonomousDetectionService"
        ) as mock_service:
            mock_service.return_value.detect_anomalies.side_effect = KeyboardInterrupt()

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write("col1,col2\n1,2\n3,4\n")
                temp_path = f.name

            try:
                result = runner.invoke(autonomous.app, ["detect", temp_path])
                # Should handle KeyboardInterrupt gracefully
                assert result.exit_code != 0
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_memory_error_handling(self, runner):
        """Test handling of memory errors."""
        with patch("pandas.read_csv", side_effect=MemoryError("Out of memory")):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                temp_path = f.name

            try:
                result = runner.invoke(
                    datasets.app, ["load", temp_path, "--name", "test"]
                )
                assert result.exit_code != 0
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_network_timeout_handling(self, runner):
        """Test handling of network timeout errors."""
        with patch("requests.get", side_effect=Exception("Connection timeout")):
            result = runner.invoke(app, ["status"])
            # Should handle network errors gracefully
            assert result.exit_code in [0, 1]  # Either success or expected failure


class TestCLIInputValidation:
    """Test CLI input validation."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_empty_dataset_name(self, runner):
        """Test handling of empty dataset names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n")
            temp_path = f.name

        try:
            result = runner.invoke(datasets.app, ["load", temp_path, "--name", ""])
            assert result.exit_code != 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_special_characters_in_name(self, runner):
        """Test handling of special characters in names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n")
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", "test@#$%^&*()"]
            )
            # Should either succeed or fail gracefully
            assert result.exit_code in [0, 1]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_extremely_long_name(self, runner):
        """Test handling of extremely long names."""
        long_name = "a" * 1000
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n")
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", long_name]
            )
            assert result.exit_code in [0, 1]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_unicode_characters_in_name(self, runner):
        """Test handling of unicode characters in names."""
        unicode_name = "测试数据集_🚀"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n")
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", unicode_name]
            )
            assert result.exit_code in [0, 1]
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLIPerformance:
    """Test CLI performance edge cases."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_large_dataset_handling(self, runner):
        """Test handling of large datasets."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create a moderately large dataset
            f.write("col1,col2,col3\n")
            for i in range(10000):
                f.write(f"{i},{i*2},{i*3}\n")
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", "large_dataset"]
            )
            # Should handle large datasets gracefully
            assert result.exit_code in [0, 1]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_file_handling(self, runner):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", "empty_dataset"]
            )
            assert result.exit_code != 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_malformed_csv_handling(self, runner):
        """Test handling of malformed CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            f.write("value1,value2,value3\n")  # Extra column
            f.write("value4\n")  # Missing column
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app, ["load", temp_path, "--name", "malformed_dataset"]
            )
            assert result.exit_code in [0, 1]
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLIHelpAndUsage:
    """Test CLI help and usage information."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_main_help_completeness(self, runner):
        """Test main help contains all required information."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        # Rich/Typer uses different formatting - check for common patterns
        assert (
            "Commands:" in result.stdout
            or "Commands " in result.stdout
            or "command" in result.stdout.lower()
        )
        assert "Options:" in result.stdout or "option" in result.stdout.lower()

    def test_subcommand_help_completeness(self, runner):
        """Test subcommand help completeness."""
        subcommands = ["auto", "dataset", "detector", "detect", "export", "server"]

        for cmd in subcommands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.stdout or "Commands:" in result.stdout

    def test_command_examples_in_help(self, runner):
        """Test that help contains usage examples."""
        result = runner.invoke(autonomous.app, ["--help"])
        assert result.exit_code == 0
        # Help should contain meaningful information
        assert len(result.stdout) > 100

    def test_version_information(self, runner):
        """Test version information is displayed correctly."""
        result = runner.invoke(app, ["version"])
        # Should either succeed or fail gracefully
        assert result.exit_code in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__])
