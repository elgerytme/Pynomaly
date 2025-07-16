"""
Comprehensive CLI error handling tests.
Tests error conditions, validation, and recovery across all CLI commands.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.domain.exceptions import (
    ConfigurationError,
    DatasetError,
    DetectorError,
    InfrastructureError,
    ValidationError,
)
from pynomaly.presentation.cli.app import app
from pynomaly.shared.error_handling.cli_errors import (
    validate_algorithm_name,
    validate_contamination_rate,
    validate_dataset_name,
    validate_file_exists,
    validate_positive_integer,
)


class TestCLIErrorHandling:
    """Test suite for CLI error handling and validation."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_services(self):
        """Mock all CLI services for error testing."""
        with (
            patch(
                "pynomaly.presentation.cli.commands.datasets.dataset_service"
            ) as mock_dataset,
            patch(
                "pynomaly.presentation.cli.commands.detector.detector_service"
            ) as mock_detector,
            patch(
                "pynomaly.presentation.cli.commands.detector.training_service"
            ) as mock_training,
            patch(
                "pynomaly.presentation.cli.commands.detect.detection_service"
            ) as mock_detection,
        ):
            services = {
                "dataset": mock_dataset,
                "detector": mock_detector,
                "training": mock_training,
                "detection": mock_detection,
            }
            yield services

    # Input Validation Tests

    def test_invalid_algorithm_names(self, runner):
        """Test validation of algorithm names."""
        invalid_algorithms = [
            "NonExistentAlgorithm",
            "invalid-algorithm",
            "",
            "123InvalidName",
            "Algorithm With Spaces",
        ]

        for algorithm in invalid_algorithms:
            result = runner.invoke(
                app, ["detector", "create", "test-detector", "--algorithm", algorithm]
            )
            assert result.exit_code == 1
            assert (
                "Unknown algorithm" in result.stdout
                or "algorithm" in result.stdout.lower()
            )

    def test_invalid_contamination_rates(self, runner):
        """Test validation of contamination rates."""
        invalid_rates = [
            "-0.1",  # Negative
            "1.5",  # > 1
            "0",  # Zero
            "1",  # Exactly 1
            "abc",  # Non-numeric
            "",  # Empty
            "0.5.5",  # Invalid format
        ]

        for rate in invalid_rates:
            result = runner.invoke(
                app,
                [
                    "detector",
                    "train",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--contamination",
                    rate,
                ],
            )
            assert result.exit_code == 1
            assert (
                "contamination" in result.stdout.lower()
                or "rate" in result.stdout.lower()
                or "between" in result.stdout.lower()
            )

    def test_invalid_file_paths(self, runner):
        """Test validation of file paths."""
        invalid_paths = [
            "/non/existent/file.csv",
            "/root/protected/file.csv",
            "",
            "file_without_extension",
            "/tmp/",  # Directory instead of file
            "relative/path.csv",
        ]

        for path in invalid_paths:
            result = runner.invoke(
                app, ["dataset", "create", "test-dataset", "--file", path]
            )
            assert result.exit_code == 1
            # Should contain file-related error message
            assert any(
                word in result.stdout.lower()
                for word in ["file", "path", "found", "exist", "directory"]
            )

    def test_invalid_dataset_names(self, runner, mock_services):
        """Test validation of dataset names."""
        invalid_names = [
            "",  # Empty
            "name/with/slashes",  # Invalid characters
            "name\\with\\backslashes",
            "name:with:colons",
            "name*with*asterisks",
            "name?with?questions",
            'name"with"quotes',
            "name<with>brackets",
            "name|with|pipes",
            "a" * 256,  # Too long
        ]

        for name in invalid_names:
            result = runner.invoke(
                app, ["dataset", "create", name, "--file", "/tmp/dummy.csv"]
            )
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower()
                for word in ["name", "invalid", "character", "empty", "long"]
            )

    def test_invalid_positive_integers(self, runner):
        """Test validation of positive integer parameters."""
        invalid_integers = [
            "-1",  # Negative
            "0",  # Zero
            "abc",  # Non-numeric
            "",  # Empty
            "1.5",  # Float
            "1e5",  # Scientific notation
        ]

        for value in invalid_integers:
            result = runner.invoke(app, ["dataset", "list", "--limit", value])
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower()
                for word in ["integer", "positive", "invalid", "number"]
            )

    def test_invalid_json_hyperparameters(self, runner):
        """Test validation of JSON hyperparameters."""
        invalid_json_strings = [
            "{invalid json}",
            '{"unclosed": "quote}',
            '{"trailing": "comma",}',
            "not json at all",
            "",
            "{",
            '{"key": }',
        ]

        for json_str in invalid_json_strings:
            result = runner.invoke(
                app,
                [
                    "detector",
                    "create",
                    "test-detector",
                    "--algorithm",
                    "IsolationForest",
                    "--hyperparameters",
                    json_str,
                ],
            )
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower()
                for word in ["json", "invalid", "parse", "format"]
            )

    def test_invalid_format_specifications(self, runner):
        """Test validation of format specifications."""
        # Test invalid export formats
        invalid_formats = [
            "xyz",  # Unsupported
            "",  # Empty
            "PDF",  # Wrong case
            "txt",  # Unsupported
            "doc",  # Unsupported
        ]

        for format_type in invalid_formats:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / f"output.{format_type}"

                result = runner.invoke(
                    app,
                    [
                        "export",
                        "results",
                        "test-detector",
                        "--dataset",
                        "test-dataset",
                        "--output",
                        str(output_file),
                        "--format",
                        format_type,
                    ],
                )
                assert result.exit_code == 1
                assert any(
                    word in result.stdout.lower()
                    for word in ["format", "unsupported", "invalid"]
                )

    # Service Error Handling Tests

    def test_dataset_service_errors(self, runner, mock_services):
        """Test handling of dataset service errors."""
        error_scenarios = [
            (DatasetError("Dataset not found"), "not found"),
            (ValidationError("Invalid dataset format"), "invalid"),
            (ConfigurationError("Storage not configured"), "configuration"),
            (InfrastructureError("Database connection failed"), "connection"),
        ]

        for exception, expected_message in error_scenarios:
            mock_services["dataset"].get_dataset.side_effect = exception

            result = runner.invoke(app, ["dataset", "show", "test-dataset"])
            assert result.exit_code == 1
            assert expected_message.lower() in result.stdout.lower()

    def test_detector_service_errors(self, runner, mock_services):
        """Test handling of detector service errors."""
        error_scenarios = [
            (DetectorError("Detector not trained"), "not trained"),
            (ValidationError("Invalid algorithm configuration"), "invalid"),
            (ConfigurationError("GPU not available"), "gpu"),
            (InfrastructureError("Model storage failed"), "storage"),
        ]

        for exception, expected_message in error_scenarios:
            mock_services["detector"].get_detector.side_effect = exception

            result = runner.invoke(app, ["detector", "show", "test-detector"])
            assert result.exit_code == 1
            assert expected_message.lower() in result.stdout.lower()

    def test_training_service_errors(self, runner, mock_services):
        """Test handling of training service errors."""
        error_scenarios = [
            (ValidationError("Insufficient training data"), "insufficient"),
            (ConfigurationError("Invalid hyperparameters"), "hyperparameters"),
            (InfrastructureError("Training timeout"), "timeout"),
            (DetectorError("Model fitting failed"), "fitting"),
        ]

        for exception, expected_message in error_scenarios:
            mock_services["training"].train_detector.side_effect = exception

            result = runner.invoke(
                app, ["detector", "train", "test-detector", "--dataset", "test-dataset"]
            )
            assert result.exit_code == 1
            assert expected_message.lower() in result.stdout.lower()

    def test_detection_service_errors(self, runner, mock_services):
        """Test handling of detection service errors."""
        error_scenarios = [
            (DetectorError("Detector not trained"), "not trained"),
            (DatasetError("Dataset format mismatch"), "format"),
            (ValidationError("Feature count mismatch"), "feature"),
            (InfrastructureError("Prediction failed"), "prediction"),
        ]

        for exception, expected_message in error_scenarios:
            mock_services["detection"].run_detection.side_effect = exception

            result = runner.invoke(
                app, ["detect", "run", "test-detector", "--dataset", "test-dataset"]
            )
            assert result.exit_code == 1
            assert expected_message.lower() in result.stdout.lower()

    # System Error Handling Tests

    def test_file_permission_errors(self, runner):
        """Test handling of file permission errors."""
        # Create a file with restricted permissions
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            restricted_file = f.name

        try:
            # Remove read permissions
            Path(restricted_file).chmod(0o000)

            result = runner.invoke(
                app, ["dataset", "create", "test-dataset", "--file", restricted_file]
            )
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower()
                for word in ["permission", "access", "denied"]
            )
        finally:
            # Restore permissions for cleanup
            Path(restricted_file).chmod(0o644)
            Path(restricted_file).unlink(missing_ok=True)

    def test_insufficient_disk_space_simulation(self, runner, mock_services):
        """Test handling of insufficient disk space."""
        mock_services["dataset"].create_dataset.side_effect = OSError(
            "No space left on device"
        )

        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            f.write(b"col1,col2\n1,2\n")
            f.flush()

            result = runner.invoke(
                app, ["dataset", "create", "test-dataset", "--file", f.name]
            )
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower() for word in ["space", "device", "storage"]
            )

    def test_memory_errors(self, runner, mock_services):
        """Test handling of memory errors."""
        mock_services["training"].train_detector.side_effect = MemoryError(
            "Not enough memory"
        )

        result = runner.invoke(
            app, ["detector", "train", "test-detector", "--dataset", "large-dataset"]
        )
        assert result.exit_code == 1
        assert any(
            word in result.stdout.lower() for word in ["memory", "available", "process"]
        )

    def test_network_errors(self, runner, mock_services):
        """Test handling of network-related errors."""
        network_errors = [
            ConnectionError("Connection refused"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable"),
        ]

        for error in network_errors:
            mock_services["dataset"].list_datasets.side_effect = error

            result = runner.invoke(app, ["dataset", "list"])
            assert result.exit_code == 1
            assert any(
                word in result.stdout.lower()
                for word in ["connection", "network", "timeout", "unreachable"]
            )

    # User Interaction Error Handling Tests

    def test_keyboard_interrupt_handling(self, runner, mock_services):
        """Test handling of keyboard interrupts (Ctrl+C)."""
        mock_services["training"].train_detector.side_effect = KeyboardInterrupt()

        result = runner.invoke(
            app, ["detector", "train", "test-detector", "--dataset", "test-dataset"]
        )
        assert result.exit_code == 1
        assert any(
            word in result.stdout.lower()
            for word in ["cancelled", "interrupted", "user"]
        )

    def test_invalid_confirmation_inputs(self, runner, mock_services):
        """Test handling of invalid confirmation inputs."""
        mock_services["detector"].delete_detector.return_value = None

        # Test with invalid input (should be cancelled)
        result = runner.invoke(
            app, ["detector", "delete", "test-detector"], input="maybe\n"
        )

        # Should default to cancellation for invalid input
        assert result.exit_code == 0
        assert any(word in result.stdout.lower() for word in ["cancelled", "aborted"])

    def test_empty_input_handling(self, runner):
        """Test handling of empty required inputs."""
        # Test commands that require arguments
        empty_input_commands = [
            ["dataset", "show", ""],
            ["detector", "create", ""],
            ["detect", "run", ""],
            ["export", "results", ""],
        ]

        for cmd in empty_input_commands:
            result = runner.invoke(app, cmd)
            assert result.exit_code == 1 or result.exit_code == 2  # Argument error

    # Edge Case Error Handling Tests

    def test_unicode_and_special_characters(self, runner):
        """Test handling of unicode and special characters in inputs."""
        special_names = [
            "测试数据集",  # Chinese characters
            "датасет",  # Cyrillic
            "🚀dataset",  # Emoji
            "café_dataset",  # Accented characters
            "data\nset",  # Newline
            "data\tset",  # Tab
            "data\0set",  # Null character
        ]

        for name in special_names:
            result = runner.invoke(
                app, ["dataset", "create", name, "--file", "/tmp/dummy.csv"]
            )
            # Should handle gracefully, either success or proper error message
            assert result.exit_code in [0, 1]
            if result.exit_code == 1:
                # Should have a clear error message
                assert len(result.stdout.strip()) > 0

    def test_extremely_long_inputs(self, runner):
        """Test handling of extremely long inputs."""
        long_string = "a" * 10000

        result = runner.invoke(
            app, ["dataset", "create", long_string, "--file", "/tmp/dummy.csv"]
        )
        assert result.exit_code == 1
        assert any(
            word in result.stdout.lower()
            for word in ["long", "length", "limit", "invalid"]
        )

    def test_binary_data_in_text_fields(self, runner):
        """Test handling of binary data in text fields."""
        binary_data = b"\x00\x01\x02\xff".decode("latin1")

        result = runner.invoke(
            app, ["dataset", "create", binary_data, "--file", "/tmp/dummy.csv"]
        )
        assert result.exit_code == 1
        # Should handle binary data gracefully

    # Recovery and Graceful Degradation Tests

    def test_partial_failure_recovery(self, runner, mock_services):
        """Test recovery from partial failures."""
        # Simulate a scenario where some operations succeed and others fail
        mock_services["dataset"].create_dataset.return_value = Mock(id="test-dataset")
        mock_services["detector"].create_detector.side_effect = DetectorError(
            "Creation failed"
        )

        # First operation should succeed
        result1 = runner.invoke(
            app, ["dataset", "create", "test-dataset", "--file", "/tmp/dummy.csv"]
        )
        # This would succeed if we had a proper file

        # Second operation should fail gracefully
        result2 = runner.invoke(
            app,
            ["detector", "create", "test-detector", "--algorithm", "IsolationForest"],
        )
        assert result2.exit_code == 1
        assert "Creation failed" in result2.stdout

    def test_service_unavailable_fallback(self, runner, mock_services):
        """Test fallback behavior when services are unavailable."""
        # Simulate service unavailable
        mock_services["dataset"].list_datasets.side_effect = InfrastructureError(
            "Service unavailable"
        )

        result = runner.invoke(app, ["dataset", "list"])
        assert result.exit_code == 1
        assert any(
            word in result.stdout.lower()
            for word in ["service", "unavailable", "connection"]
        )

    # Validation Function Tests

    def test_validate_algorithm_name_function(self):
        """Test the validate_algorithm_name function directly."""
        # Valid algorithms
        valid_algorithms = ["IsolationForest", "LOF", "OneClassSVM"]
        for algo in valid_algorithms:
            result = validate_algorithm_name(algo)
            assert result == algo

        # Invalid algorithms
        with pytest.raises(ValidationError):
            validate_algorithm_name("InvalidAlgorithm")

        with pytest.raises(ValidationError):
            validate_algorithm_name("")

    def test_validate_contamination_rate_function(self):
        """Test the validate_contamination_rate function directly."""
        # Valid rates
        valid_rates = [0.1, 0.05, 0.5, "0.2"]
        for rate in valid_rates:
            result = validate_contamination_rate(rate)
            assert 0.0 < result < 1.0

        # Invalid rates
        invalid_rates = [-0.1, 1.5, 0, 1, "abc", ""]
        for rate in invalid_rates:
            with pytest.raises(ValidationError):
                validate_contamination_rate(rate)

    def test_validate_file_exists_function(self):
        """Test the validate_file_exists function directly."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_file = f.name

        try:
            # Valid file
            result = validate_file_exists(temp_file)
            assert result == Path(temp_file)

            # Invalid file
            with pytest.raises(ValidationError):
                validate_file_exists("/non/existent/file.txt")
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_validate_dataset_name_function(self):
        """Test the validate_dataset_name function directly."""
        # Valid names
        valid_names = ["dataset1", "my_dataset", "Dataset-2024"]
        for name in valid_names:
            result = validate_dataset_name(name)
            assert result == name.strip()

        # Invalid names
        invalid_names = ["", "  ", "name/with/slash", "name:with:colon"]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                validate_dataset_name(name)

    def test_validate_positive_integer_function(self):
        """Test the validate_positive_integer function directly."""
        # Valid integers
        valid_integers = [1, 10, 100, "5", "50"]
        for value in valid_integers:
            result = validate_positive_integer(value, "test_param")
            assert result > 0
            assert isinstance(result, int)

        # Invalid integers
        invalid_integers = [0, -1, "abc", "", 1.5]
        for value in invalid_integers:
            with pytest.raises(ValidationError):
                validate_positive_integer(value, "test_param")

    # Comprehensive Error Message Tests

    def test_error_message_clarity(self, runner):
        """Test that error messages are clear and helpful."""
        # Test various error scenarios and check message quality
        test_cases = [
            {
                "command": ["detector", "create", "", "--algorithm", "IsolationForest"],
                "expected_words": ["name", "empty", "required"],
            },
            {
                "command": ["dataset", "create", "test", "--file", "/non/existent"],
                "expected_words": ["file", "not", "found", "exist"],
            },
            {
                "command": ["detector", "train", "test", "--contamination", "2.0"],
                "expected_words": ["contamination", "between", "0", "1"],
            },
        ]

        for test_case in test_cases:
            result = runner.invoke(app, test_case["command"])
            assert result.exit_code == 1

            # Check that error message contains expected words
            stdout_lower = result.stdout.lower()
            found_words = [
                word for word in test_case["expected_words"] if word in stdout_lower
            ]
            assert (
                len(found_words) >= len(test_case["expected_words"]) // 2
            ), f"Error message not clear enough: {result.stdout}"

    def test_error_message_formatting(self, runner):
        """Test that error messages are properly formatted."""
        result = runner.invoke(
            app, ["detector", "create", "test", "--algorithm", "NonExistentAlgorithm"]
        )

        assert result.exit_code == 1
        # Error message should be non-empty and readable
        assert len(result.stdout.strip()) > 0
        # Should not contain raw exception traces in user-facing output
        assert "Traceback" not in result.stdout
        assert "Exception" not in result.stdout
