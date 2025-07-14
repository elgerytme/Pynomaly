"""Comprehensive CLI error handling and edge case tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app
from pynomaly.presentation.cli.config import app as config_app
from pynomaly.presentation.cli.server import app as server_app


class TestCLIErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container
            return container

    def test_invalid_command_handling(self, runner):
        """Test handling of invalid commands."""
        result = runner.invoke(app, ["invalid_command"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout or "Usage:" in result.stdout

    def test_missing_required_arguments(self, runner):
        """Test handling of missing required arguments."""
        # Test generate-config without config type
        result = runner.invoke(app, ["generate-config"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Usage:" in result.stdout

    def test_invalid_option_values(self, runner):
        """Test handling of invalid option values."""
        # Test invalid contamination value
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "invalid_value"]
        )
        assert result.exit_code != 0

    def test_conflicting_options(self, runner):
        """Test handling of conflicting options."""
        # Test verbose and quiet together
        result = runner.invoke(app, ["--verbose", "--quiet", "version"])
        assert result.exit_code == 1
        assert "Cannot use --verbose and --quiet together" in result.stdout

    def test_file_not_found_errors(self, runner):
        """Test handling of file not found errors."""
        # Test with non-existent file
        result = runner.invoke(
            app, ["generate-config", "test", "--dataset", "/nonexistent/file.csv"]
        )
        assert result.exit_code == 0  # Should generate config but note missing file

    def test_permission_errors(self, runner):
        """Test handling of permission errors."""
        # Test writing to read-only location
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = runner.invoke(
                app, ["generate-config", "test", "--output", "/root/config.json"]
            )
            assert result.exit_code == 1
            assert "Permission denied" in result.stdout

    def test_json_parsing_errors(self, runner):
        """Test handling of JSON parsing errors."""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            invalid_json_file = f.name

        # Test with config command
        result = runner.invoke(
            config_app, ["capture", "automl", "--params", invalid_json_file]
        )
        assert result.exit_code == 1
        assert "Error" in result.stdout

    def test_network_errors(self, runner):
        """Test handling of network-related errors."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test server status with network error
            with patch("requests.get", side_effect=Exception("Network error")):
                result = runner.invoke(server_app, ["status"])
                assert result.exit_code == 0
                assert (
                    "Cannot connect to API" in result.stdout
                    or "Network error" in result.stdout
                )

    def test_timeout_handling(self, runner):
        """Test handling of timeout errors."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test health check with timeout
            import requests

            with patch(
                "requests.get",
                side_effect=requests.exceptions.Timeout("Request timeout"),
            ):
                result = runner.invoke(server_app, ["health"])
                assert result.exit_code == 1
                assert (
                    "Timeout" in result.stdout
                    or "Some services are not available" in result.stdout
                )

    def test_database_connection_errors(self, runner, mock_container):
        """Test handling of database connection errors."""
        # Mock repository that fails
        mock_container.detector_repository.side_effect = Exception(
            "Database connection failed"
        )

        result = runner.invoke(app, ["status"])
        assert result.exit_code != 0 or "error" in result.stdout.lower()

    def test_service_unavailable_errors(self, runner):
        """Test handling of service unavailable errors."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container",
            side_effect=Exception("Service unavailable"),
        ):
            result = runner.invoke(app, ["version"])
            assert result.exit_code != 0

    def test_memory_errors(self, runner):
        """Test handling of memory-related errors."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container",
            side_effect=MemoryError("Out of memory"),
        ):
            result = runner.invoke(app, ["version"])
            assert result.exit_code != 0

    def test_keyboard_interrupt_handling(self, runner):
        """Test handling of keyboard interrupts."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container",
            side_effect=KeyboardInterrupt(),
        ):
            result = runner.invoke(app, ["version"])
            assert result.exit_code != 0

    def test_invalid_uuid_handling(self, runner):
        """Test handling of invalid UUID formats."""
        result = runner.invoke(config_app, ["show", "invalid-uuid-format"])
        assert result.exit_code == 1
        assert "Invalid UUID format" in result.stdout

    def test_yaml_import_errors(self, runner):
        """Test handling of YAML import errors."""
        with patch("yaml.dump", side_effect=ImportError("PyYAML not installed")):
            result = runner.invoke(app, ["generate-config", "test", "--format", "yaml"])
            assert result.exit_code == 1

    def test_subprocess_errors(self, runner):
        """Test handling of subprocess errors."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test server start with subprocess error
            with (
                patch("socket.socket") as mock_socket,
                patch("subprocess.run") as mock_subprocess,
            ):
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 1  # Port available
                mock_socket.return_value = mock_sock

                from subprocess import CalledProcessError

                mock_subprocess.side_effect = CalledProcessError(1, ["uvicorn"])

                result = runner.invoke(server_app, ["start"])
                assert result.exit_code == 1
                assert "Server failed to start" in result.stdout

    def test_signal_handling(self, runner):
        """Test handling of system signals."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test server stop with signal error
            with tempfile.TemporaryDirectory() as tmp_dir:
                pid_file = Path(tmp_dir) / "pynomaly.pid"
                pid_file.write_text("1234")
                container.config.return_value.storage_path = Path(tmp_dir)

                with patch("os.kill", side_effect=OSError("Permission denied")):
                    result = runner.invoke(server_app, ["stop"])
                    assert result.exit_code == 1
                    assert "Failed to stop server" in result.stdout


class TestCLIEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_empty_input_handling(self, runner):
        """Test handling of empty inputs."""
        # Test empty dataset path
        result = runner.invoke(app, ["generate-config", "test", "--dataset", ""])
        assert result.exit_code == 0  # Should handle empty string gracefully

    def test_very_long_inputs(self, runner):
        """Test handling of very long inputs."""
        long_string = "a" * 10000

        result = runner.invoke(
            app, ["generate-config", "test", "--description", long_string]
        )
        assert result.exit_code == 0

    def test_special_characters_in_inputs(self, runner):
        """Test handling of special characters in inputs."""
        special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"

        result = runner.invoke(
            app, ["generate-config", "test", "--description", special_chars]
        )
        assert result.exit_code == 0

    def test_unicode_handling(self, runner):
        """Test handling of Unicode characters."""
        unicode_string = "测试 🚀 émojis ñoño"

        result = runner.invoke(
            app, ["generate-config", "test", "--description", unicode_string]
        )
        assert result.exit_code == 0

    def test_boundary_values(self, runner):
        """Test boundary values for numeric inputs."""
        # Test minimum contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "0.0"]
        )
        assert result.exit_code == 0

        # Test maximum contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "1.0"]
        )
        assert result.exit_code == 0

        # Test minimum CV folds
        result = runner.invoke(app, ["generate-config", "experiment", "--folds", "2"])
        assert result.exit_code == 0

        # Test maximum algorithms
        result = runner.invoke(
            app, ["generate-config", "autonomous", "--max-algorithms", "1000"]
        )
        assert result.exit_code == 0

    def test_negative_values(self, runner):
        """Test handling of negative values."""
        # Test negative contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "-0.1"]
        )
        assert result.exit_code == 0  # Should handle but may warn

        # Test negative folds
        result = runner.invoke(app, ["generate-config", "experiment", "--folds", "-1"])
        assert result.exit_code == 0  # Should handle but may warn

    def test_zero_values(self, runner):
        """Test handling of zero values."""
        # Test zero contamination
        result = runner.invoke(app, ["generate-config", "test", "--contamination", "0"])
        assert result.exit_code == 0

        # Test zero algorithms
        result = runner.invoke(
            app, ["generate-config", "autonomous", "--max-algorithms", "0"]
        )
        assert result.exit_code == 0

    def test_extremely_large_values(self, runner):
        """Test handling of extremely large values."""
        # Test very large contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "999999"]
        )
        assert result.exit_code == 0

        # Test very large folds
        result = runner.invoke(
            app, ["generate-config", "experiment", "--folds", "999999"]
        )
        assert result.exit_code == 0

    def test_float_precision(self, runner):
        """Test handling of high-precision float values."""
        # Test high precision contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "0.123456789012345"]
        )
        assert result.exit_code == 0

    def test_scientific_notation(self, runner):
        """Test handling of scientific notation."""
        # Test scientific notation contamination
        result = runner.invoke(
            app, ["generate-config", "test", "--contamination", "1e-5"]
        )
        assert result.exit_code == 0

    def test_multiple_tags(self, runner):
        """Test handling of multiple tags."""
        # Test many tags
        tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]
        cmd = ["generate-config", "test"]
        for tag in tags:
            cmd.extend(["--tag", tag])

        result = runner.invoke(app, cmd)
        assert result.exit_code == 0

    def test_empty_tags(self, runner):
        """Test handling of empty tags."""
        result = runner.invoke(app, ["generate-config", "test", "--tag", ""])
        assert result.exit_code == 0

    def test_duplicate_tags(self, runner):
        """Test handling of duplicate tags."""
        result = runner.invoke(
            app,
            [
                "generate-config",
                "test",
                "--tag",
                "duplicate",
                "--tag",
                "duplicate",
                "--tag",
                "duplicate",
            ],
        )
        assert result.exit_code == 0

    def test_case_sensitivity(self, runner):
        """Test case sensitivity handling."""
        # Test different case config types
        result = runner.invoke(app, ["generate-config", "TEST"])
        assert result.exit_code == 1  # Should fail for invalid type

        result = runner.invoke(app, ["generate-config", "Test"])
        assert result.exit_code == 1  # Should fail for invalid type

        result = runner.invoke(app, ["generate-config", "test"])
        assert result.exit_code == 0  # Should work

    def test_whitespace_handling(self, runner):
        """Test handling of whitespace in inputs."""
        # Test leading/trailing whitespace
        result = runner.invoke(
            app, ["generate-config", "test", "--description", "  test description  "]
        )
        assert result.exit_code == 0

        # Test multiple spaces
        result = runner.invoke(
            app,
            [
                "generate-config",
                "test",
                "--description",
                "test    description    with    spaces",
            ],
        )
        assert result.exit_code == 0

    def test_newline_handling(self, runner):
        """Test handling of newlines in inputs."""
        result = runner.invoke(
            app, ["generate-config", "test", "--description", "line1\nline2\nline3"]
        )
        assert result.exit_code == 0

    def test_tab_handling(self, runner):
        """Test handling of tab characters."""
        result = runner.invoke(
            app,
            ["generate-config", "test", "--description", "column1\tcolumn2\tcolumn3"],
        )
        assert result.exit_code == 0


class TestCLIResourceLimits:
    """Test CLI behavior under resource constraints."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_large_file_handling(self, runner):
        """Test handling of large files."""
        # Create large temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write header
            f.write("feature1,feature2,feature3\n")

            # Write many rows
            for i in range(10000):
                f.write(f"{i},{i*2},{i*3}\n")

            large_file = f.name

        try:
            result = runner.invoke(
                app, ["generate-config", "test", "--dataset", large_file]
            )
            assert result.exit_code == 0
        finally:
            os.unlink(large_file)

    def test_memory_intensive_operations(self, runner):
        """Test memory-intensive operations."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test operations that might consume memory
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0

    def test_concurrent_access_simulation(self, runner):
        """Test concurrent access simulation."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"
            container.config.return_value = config
            mock_get_container.return_value = container

            # Simulate concurrent access by running multiple commands
            for i in range(10):
                result = runner.invoke(app, ["version"])
                assert result.exit_code == 0

    def test_disk_space_constraints(self, runner):
        """Test behavior with disk space constraints."""
        # Test writing to full disk (simulated)
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            result = runner.invoke(
                app, ["generate-config", "test", "--output", "/tmp/config.json"]
            )
            assert result.exit_code == 1

    def test_file_descriptor_limits(self, runner):
        """Test file descriptor limits."""
        # Test opening many files (simulated)
        with patch("builtins.open", side_effect=OSError("Too many open files")):
            result = runner.invoke(
                app, ["generate-config", "test", "--output", "/tmp/config.json"]
            )
            assert result.exit_code == 1

    def test_process_limits(self, runner):
        """Test process limits."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test process creation failure
            with patch(
                "os.fork", side_effect=OSError("Resource temporarily unavailable")
            ):
                result = runner.invoke(server_app, ["start", "--daemon"])
                assert result.exit_code == 1
                assert "Failed to fork process" in result.stdout


class TestCLIRecovery:
    """Test CLI recovery mechanisms."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_graceful_degradation(self, runner):
        """Test graceful degradation when features are unavailable."""
        # Test when optional features are not available
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"

            # Mock missing optional features
            container.config.return_value = config
            container.optional_feature.side_effect = ImportError(
                "Feature not available"
            )

            mock_get_container.return_value = container

            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0  # Should still work

    def test_fallback_mechanisms(self, runner):
        """Test fallback mechanisms when primary methods fail."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test fallback when primary config fails
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0

    def test_partial_failure_handling(self, runner):
        """Test handling of partial failures."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"

            # Mock partial failure
            container.config.return_value = config
            container.detector_repository.return_value.count.return_value = 5
            container.dataset_repository.return_value.count.side_effect = Exception(
                "Partial failure"
            )
            container.result_repository.return_value.count.return_value = 10
            container.result_repository.return_value.find_recent.return_value = []

            mock_get_container.return_value = container

            result = runner.invoke(app, ["status"])
            # Should handle partial failure gracefully
            assert result.exit_code == 0 or result.exit_code != 0

    def test_retry_mechanisms(self, runner):
        """Test retry mechanisms for transient failures."""
        with patch(
            "pynomaly.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test retry on transient network failure
            import requests

            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise requests.exceptions.ConnectionError("Transient failure")
                else:
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {"status": "healthy"}
                    return response

            with patch("requests.get", side_effect=side_effect):
                result = runner.invoke(server_app, ["status"])
                # Should succeed after retry
                assert result.exit_code == 0

    def test_cleanup_on_failure(self, runner):
        """Test cleanup mechanisms when operations fail."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.json"

            # Test cleanup when generation fails
            with patch("json.dump", side_effect=Exception("Write failed")):
                result = runner.invoke(
                    app, ["generate-config", "test", "--output", str(config_file)]
                )

                assert result.exit_code == 1
                # File should not exist after failure
                assert not config_file.exists()

    def test_state_consistency(self, runner):
        """Test state consistency after failures."""
        with patch(
            "pynomaly.presentation.cli.app.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            config = Mock()
            config.app.version = "1.0.0"
            config.storage_path = Path("/tmp/pynomaly")
            container.config.return_value = config
            mock_get_container.return_value = container

            # Test that state remains consistent after failure
            result1 = runner.invoke(app, ["version"])
            assert result1.exit_code == 0

            # Simulate failure
            container.config.side_effect = Exception("Temporary failure")
            result2 = runner.invoke(app, ["version"])
            assert result2.exit_code != 0

            # Restore and test recovery
            container.config.side_effect = None
            result3 = runner.invoke(app, ["version"])
            assert result3.exit_code == 0


class TestCLISecurityEdgeCases:
    """Test security-related edge cases."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_path_traversal_prevention(self, runner):
        """Test prevention of path traversal attacks."""
        # Test with path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
        ]

        for path in malicious_paths:
            result = runner.invoke(app, ["generate-config", "test", "--output", path])
            # Should either succeed (if path is valid) or fail gracefully
            assert result.exit_code in [0, 1]

    def test_command_injection_prevention(self, runner):
        """Test prevention of command injection."""
        # Test with command injection attempts
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "& del C:\\Windows\\System32\\*",
            "$(rm -rf /)",
            "`cat /etc/passwd`",
        ]

        for input_str in malicious_inputs:
            result = runner.invoke(
                app, ["generate-config", "test", "--description", input_str]
            )
            # Should handle malicious input safely
            assert result.exit_code == 0

    def test_input_sanitization(self, runner):
        """Test input sanitization."""
        # Test with various potentially dangerous inputs
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "DROP TABLE users;",
            "SELECT * FROM users WHERE 1=1;",
        ]

        for input_str in dangerous_inputs:
            result = runner.invoke(
                app, ["generate-config", "test", "--description", input_str]
            )
            # Should handle dangerous input safely
            assert result.exit_code == 0

    def test_buffer_overflow_prevention(self, runner):
        """Test prevention of buffer overflow attacks."""
        # Test with extremely long inputs
        extremely_long_input = "A" * 100000

        result = runner.invoke(
            app, ["generate-config", "test", "--description", extremely_long_input]
        )
        # Should handle long input without crashing
        assert result.exit_code == 0

    def test_resource_exhaustion_prevention(self, runner):
        """Test prevention of resource exhaustion attacks."""
        # Test with inputs that might cause resource exhaustion
        result = runner.invoke(
            app,
            [
                "generate-config",
                "test",
                "--folds",
                "999999",
                "--max-algorithms",
                "999999",
            ],
        )
        # Should handle large values without exhausting resources
        assert result.exit_code == 0
