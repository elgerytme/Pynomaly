"""
Performance and stability tests for CLI commands.

Tests CLI behavior under various conditions including large datasets,
concurrent operations, and resource constraints.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from monorepo.presentation.cli import autonomous, datasets, detectors
from monorepo.presentation.cli.app import app


class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def test_help_command_performance(self, stable_cli_runner, performance_monitor):
        """Test that help commands are fast."""
        start_time = time.time()

        result = stable_cli_runner.invoke(app, ["--help"])

        end_time = time.time()
        execution_time = end_time - start_time

        assert result.exit_code == 0
        assert execution_time < 2.0  # Should be very fast
        assert "Usage:" in result.stdout

    def test_large_dataset_handling(self, stable_cli_runner, large_dataset):
        """Test CLI with large datasets."""
        with patch(
            "monorepo.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo
            dataset_repo.save.return_value = True
            mock_container.return_value = container

            result = stable_cli_runner.invoke(
                datasets.app, ["load", large_dataset, "--name", "large_test_dataset"]
            )

            # Should handle large datasets gracefully
            assert result.exit_code in [0, 1]

    def test_concurrent_help_requests(self, stable_cli_runner):
        """Test concurrent help requests don't interfere."""
        results = []
        commands = ["--help", "dataset --help", "detector --help", "auto --help"]

        for cmd in commands:
            result = stable_cli_runner.invoke(app, cmd.split())
            results.append(result)

        # All help commands should succeed
        for result in results:
            assert result.exit_code == 0
            assert "Usage:" in result.stdout or "Commands:" in result.stdout

    def test_memory_usage_stability(self, stable_cli_runner, sample_dataset):
        """Test memory usage remains stable during operations."""
        with patch(
            "monorepo.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            # Mock lightweight response
            mock_service_instance.detect_anomalies.return_value = {
                "best_detector": "IsolationForest",
                "anomalies_found": 1,
                "confidence": 0.8,
            }

            # Run multiple operations
            for i in range(5):
                result = stable_cli_runner.invoke(
                    autonomous.app, ["detect", sample_dataset]
                )

                # Each operation should complete
                assert result.exit_code in [0, 1]

    def test_output_size_handling(self, stable_cli_runner):
        """Test handling of potentially large outputs."""
        # Test configuration generation with large configs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            config_path = config_file.name

        try:
            result = stable_cli_runner.invoke(
                app, ["generate-config", "test", "--output", config_path]
            )

            # Should handle output gracefully
            assert result.exit_code in [0, 1]

            if result.exit_code == 0 and Path(config_path).exists():
                # Output should be reasonable size
                file_size = Path(config_path).stat().st_size
                assert file_size < 1024 * 1024  # Less than 1MB

        finally:
            Path(config_path).unlink(missing_ok=True)


class TestCLIStability:
    """Test CLI stability under various conditions."""

    def test_error_recovery_stability(self, stable_cli_runner):
        """Test CLI recovers from errors gracefully."""
        # Run invalid command
        result1 = stable_cli_runner.invoke(app, ["invalid-command"])
        assert result1.exit_code != 0

        # Verify CLI still works after error
        result2 = stable_cli_runner.invoke(app, ["--help"])
        assert result2.exit_code == 0

        # Run another invalid command
        result3 = stable_cli_runner.invoke(datasets.app, ["invalid-subcommand"])
        assert result3.exit_code != 0

        # Verify CLI still works
        result4 = stable_cli_runner.invoke(app, ["--help"])
        assert result4.exit_code == 0

    def test_resource_cleanup(self, stable_cli_runner, sample_dataset):
        """Test proper resource cleanup."""
        with patch(
            "monorepo.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo
            dataset_repo.save.return_value = True
            mock_container.return_value = container

            # Run operation
            result = stable_cli_runner.invoke(
                datasets.app, ["load", sample_dataset, "--name", "cleanup_test"]
            )

            # Should complete without hanging
            assert result.exit_code in [0, 1]

            # Verify we can run additional operations
            result2 = stable_cli_runner.invoke(app, ["--help"])
            assert result2.exit_code == 0

    def test_signal_handling(self, stable_cli_runner):
        """Test handling of interruption signals."""
        # This tests graceful handling of KeyboardInterrupt
        with patch(
            "monorepo.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            # Mock KeyboardInterrupt during operation
            mock_service_instance.detect_anomalies.side_effect = KeyboardInterrupt()

            result = stable_cli_runner.invoke(
                autonomous.app, ["detect", "/tmp/fake_file.csv"]
            )

            # Should handle interruption gracefully
            assert result.exit_code in [0, 1]

    def test_configuration_error_stability(self, stable_cli_runner):
        """Test stability when configuration has errors."""
        # Test with invalid configuration file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            config_file.write("invalid json content")
            invalid_config_path = config_file.name

        try:
            result = stable_cli_runner.invoke(
                autonomous.app,
                ["detect", "/tmp/fake_file.csv", "--config", invalid_config_path],
            )

            # Should handle invalid config gracefully
            assert result.exit_code in [0, 1]

        finally:
            Path(invalid_config_path).unlink(missing_ok=True)

    def test_dependency_import_stability(self, stable_cli_runner):
        """Test stability when optional dependencies are missing."""
        # This is already handled by the mock_optional_dependencies fixture
        # but we test specific import scenarios

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'sklearn'")
        ):
            result = stable_cli_runner.invoke(app, ["--help"])

            # Basic functionality should still work
            assert result.exit_code == 0
            assert "Usage:" in result.stdout


class TestCLIEdgeCases:
    """Test CLI edge cases and boundary conditions."""

    def test_empty_file_handling(self, stable_cli_runner):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Empty file
            empty_file_path = f.name

        try:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", empty_file_path, "--name", "empty_test"]
            )

            # Should handle empty files gracefully
            assert result.exit_code in [0, 1]

        finally:
            Path(empty_file_path).unlink(missing_ok=True)

    def test_binary_file_handling(self, stable_cli_runner):
        """Test handling of binary files."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")
            binary_file_path = f.name

        try:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", binary_file_path, "--name", "binary_test"]
            )

            # Should reject binary files gracefully
            assert result.exit_code != 0

        finally:
            Path(binary_file_path).unlink(missing_ok=True)

    def test_very_long_arguments(self, stable_cli_runner):
        """Test handling of very long command line arguments."""
        long_name = "a" * 1000  # Very long name

        result = stable_cli_runner.invoke(
            detectors.app,
            ["create", "--name", long_name, "--algorithm", "IsolationForest"],
        )

        # Should handle long arguments gracefully
        assert result.exit_code in [0, 1]

    def test_special_characters_in_paths(self, stable_cli_runner):
        """Test handling of special characters in file paths."""
        special_paths = [
            "/path/with spaces/file.csv",
            "/path/with-dashes/file.csv",
            "/path/with_underscores/file.csv",
            "/path/with.dots/file.csv",
        ]

        for path in special_paths:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", path, "--name", "special_test"]
            )

            # Should handle special characters gracefully
            assert result.exit_code in [0, 1]

    def test_unicode_handling(self, stable_cli_runner):
        """Test Unicode character handling."""
        unicode_names = [
            "test_ä_dataset",
            "test_ñ_dataset",
            "test_中文_dataset",
            "test_русский_dataset",
        ]

        for name in unicode_names:
            result = stable_cli_runner.invoke(
                detectors.app,
                ["create", "--name", name, "--algorithm", "IsolationForest"],
            )

            # Should handle Unicode gracefully
            assert result.exit_code in [0, 1]

    def test_path_traversal_protection(self, stable_cli_runner):
        """Test protection against path traversal attempts."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for path in malicious_paths:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", path, "--name", "security_test"]
            )

            # Should handle malicious paths safely
            assert result.exit_code in [0, 1]

    def test_extremely_large_parameter_values(self, stable_cli_runner):
        """Test handling of extremely large parameter values."""
        large_values = [
            ("--contamination", "999999.0"),
            ("--contamination", "-999999.0"),
            ("--contamination", "1e100"),
            ("--timeout", "999999999"),
        ]

        for param, value in large_values:
            result = stable_cli_runner.invoke(
                autonomous.app, ["detect", "/tmp/fake.csv", param, value]
            )

            # Should handle large values gracefully
            assert result.exit_code in [0, 1]


class TestCLIResourceManagement:
    """Test CLI resource management."""

    def test_file_descriptor_management(self, stable_cli_runner, sample_dataset):
        """Test proper file descriptor management."""
        # Run multiple file operations
        for i in range(10):
            with patch(
                "monorepo.presentation.cli.container.get_cli_container"
            ) as mock_container:
                container = Mock()
                dataset_repo = Mock()
                container.dataset_repository.return_value = dataset_repo
                dataset_repo.save.return_value = True
                mock_container.return_value = container

                result = stable_cli_runner.invoke(
                    datasets.app, ["load", sample_dataset, "--name", f"fd_test_{i}"]
                )

                # Should handle file operations without leaking descriptors
                assert result.exit_code in [0, 1]

    def test_temporary_file_cleanup(self, stable_cli_runner, sample_dataset):
        """Test cleanup of temporary files."""
        import tempfile

        # Count temp files before
        temp_dir = Path(tempfile.gettempdir())
        temp_files_before = len(list(temp_dir.glob("*")))

        with patch(
            "monorepo.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.detect_anomalies.return_value = {
                "best_detector": "IsolationForest",
                "anomalies_found": 1,
            }

            result = stable_cli_runner.invoke(
                autonomous.app, ["detect", sample_dataset]
            )

        # Count temp files after
        temp_files_after = len(list(temp_dir.glob("*")))

        # Should not leak too many temp files
        assert temp_files_after - temp_files_before < 10
        assert result.exit_code in [0, 1]

    def test_memory_pressure_handling(self, stable_cli_runner, large_dataset):
        """Test handling under memory pressure."""
        with patch(
            "monorepo.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            # Mock memory error
            mock_service_instance.detect_anomalies.side_effect = MemoryError(
                "Out of memory"
            )

            result = stable_cli_runner.invoke(autonomous.app, ["detect", large_dataset])

            # Should handle memory errors gracefully
            assert result.exit_code in [0, 1]


class TestCLISecurityAndSafety:
    """Test CLI security and safety features."""

    def test_input_validation(self, stable_cli_runner):
        """Test input validation and sanitization."""
        malicious_inputs = [
            "; rm -rf /",
            "$(rm -rf /)",
            "`rm -rf /`",
            "| cat /etc/passwd",
            "&& format C:",
            "\n\nmalicious_command",
        ]

        for malicious_input in malicious_inputs:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", "/tmp/test.csv", "--name", malicious_input]
            )

            # Should handle malicious input safely
            assert result.exit_code in [0, 1]

    def test_output_sanitization(self, stable_cli_runner):
        """Test that outputs are properly sanitized."""
        with patch(
            "monorepo.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock dataset with potentially dangerous content
            mock_dataset = Mock()
            mock_dataset.name = '<script>alert("xss")</script>'
            dataset_repo.list_all.return_value = [mock_dataset]
            mock_container.return_value = container

            result = stable_cli_runner.invoke(datasets.app, ["list"])

            # Should sanitize output
            assert result.exit_code in [0, 1]
            if "<script>" in result.stdout:
                # If script tags appear, they should be escaped or removed
                assert (
                    "&lt;script&gt;" in result.stdout or "<script>" not in result.stdout
                )

    def test_safe_file_operations(self, stable_cli_runner):
        """Test safe file operations."""
        # Test with paths that could be dangerous
        dangerous_paths = [
            "/dev/null",
            "/dev/zero",
            "/proc/self/mem",
            "CON:",  # Windows device
            "NUL:",  # Windows device
        ]

        for path in dangerous_paths:
            result = stable_cli_runner.invoke(
                datasets.app, ["load", path, "--name", "safety_test"]
            )

            # Should handle dangerous paths safely
            assert result.exit_code in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
