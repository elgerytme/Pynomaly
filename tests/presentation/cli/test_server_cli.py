"""
Server CLI Testing Suite
Comprehensive tests for server management CLI commands.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from typer.testing import CliRunner

from pynomaly.presentation.cli.server import app


class TestServerCLI:
    """Test suite for server CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_uvicorn(self):
        """Mock uvicorn server."""
        with patch("uvicorn.run") as mock_run:
            yield mock_run

    @pytest.fixture
    def mock_requests(self):
        """Mock requests for API calls."""
        with patch("requests.get") as mock_get:
            with patch("requests.post") as mock_post:
                yield {"get": mock_get, "post": mock_post}

    # Basic Command Tests

    def test_server_help(self, runner):
        """Test server CLI help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Manage API server" in result.stdout
        assert "Commands:" in result.stdout
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout

    def test_start_help(self, runner):
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start the API server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--workers" in result.stdout

    def test_status_help(self, runner):
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "Check server status" in result.stdout

    # Server Start Tests

    def test_start_server_default(self, runner, mock_uvicorn):
        """Test starting server with default settings."""

        # Mock the server start in a way that doesn't block
        def mock_run_server(*args, **kwargs):
            # Simulate server startup without actually starting
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()

        # Check that uvicorn.run was called with expected parameters
        args, kwargs = mock_uvicorn.call_args
        assert "pynomaly.presentation.api:app" in args or "app" in kwargs
        assert kwargs.get("host", "localhost") == "localhost"
        assert kwargs.get("port", 8000) == 8000

    def test_start_server_custom_host_port(self, runner, mock_uvicorn):
        """Test starting server with custom host and port."""

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["start", "--host", "0.0.0.0", "--port", "9000"])

        assert result.exit_code == 0

        args, kwargs = mock_uvicorn.call_args
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 9000

    def test_start_server_with_workers(self, runner, mock_uvicorn):
        """Test starting server with multiple workers."""

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["start", "--workers", "4"])

        assert result.exit_code == 0

        args, kwargs = mock_uvicorn.call_args
        assert kwargs.get("workers") == 4

    def test_start_server_development_mode(self, runner, mock_uvicorn):
        """Test starting server in development mode."""

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["start", "--dev"])

        assert result.exit_code == 0

        args, kwargs = mock_uvicorn.call_args
        assert kwargs.get("reload") is True
        assert kwargs.get("debug") is True

    def test_start_server_production_mode(self, runner, mock_uvicorn):
        """Test starting server in production mode."""

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["start", "--production"])

        assert result.exit_code == 0

        args, kwargs = mock_uvicorn.call_args
        assert kwargs.get("reload") is False
        assert kwargs.get("workers", 1) > 1  # Should use multiple workers

    def test_start_server_with_ssl(self, runner, mock_uvicorn):
        """Test starting server with SSL configuration."""

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        # Create temporary SSL files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pem", delete=False
        ) as cert_file:
            cert_path = cert_file.name
            cert_file.write("dummy cert content")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".key", delete=False
        ) as key_file:
            key_path = key_file.name
            key_file.write("dummy key content")

        try:
            result = runner.invoke(
                app, ["start", "--ssl-cert", cert_path, "--ssl-key", key_path]
            )

            assert result.exit_code == 0

            args, kwargs = mock_uvicorn.call_args
            assert kwargs.get("ssl_certfile") == cert_path
            assert kwargs.get("ssl_keyfile") == key_path

        finally:
            Path(cert_path).unlink(missing_ok=True)
            Path(key_path).unlink(missing_ok=True)

    def test_start_server_with_config_file(self, runner, mock_uvicorn):
        """Test starting server with configuration file."""
        config = {
            "host": "127.0.0.1",
            "port": 8080,
            "workers": 2,
            "log_level": "info",
            "access_log": True,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            json.dump(config, config_file)
            config_path = config_file.name

        try:

            def mock_run_server(*args, **kwargs):
                pass

            mock_uvicorn.side_effect = mock_run_server

            result = runner.invoke(app, ["start", "--config", config_path])

            assert result.exit_code == 0

            args, kwargs = mock_uvicorn.call_args
            assert kwargs.get("host") == "127.0.0.1"
            assert kwargs.get("port") == 8080
            assert kwargs.get("workers") == 2

        finally:
            Path(config_path).unlink(missing_ok=True)

    # Server Status Tests

    def test_status_server_running(self, runner, mock_requests):
        """Test server status when server is running."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600,
            "database": "connected",
            "memory_usage": "45%",
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Server Status" in result.stdout
        assert "✅ Server is running" in result.stdout
        assert "Version: 1.0.0" in result.stdout
        assert "Uptime:" in result.stdout

    def test_status_server_not_running(self, runner, mock_requests):
        """Test server status when server is not running."""
        # Mock connection error
        mock_requests["get"].side_effect = requests.ConnectionError(
            "Connection refused"
        )

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "❌ Server is not running" in result.stdout

    def test_status_server_unhealthy(self, runner, mock_requests):
        """Test server status when server is unhealthy."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {
            "status": "unhealthy",
            "errors": ["Database connection failed"],
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "⚠️ Server is unhealthy" in result.stdout
        assert "Database connection failed" in result.stdout

    def test_status_detailed(self, runner, mock_requests):
        """Test detailed server status."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 7200,
            "database": "connected",
            "memory_usage": "55%",
            "cpu_usage": "23%",
            "active_connections": 15,
            "requests_per_minute": 45,
            "disk_usage": "67%",
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["status", "--detailed"])

        assert result.exit_code == 0
        assert "Memory Usage: 55%" in result.stdout
        assert "CPU Usage: 23%" in result.stdout
        assert "Active Connections: 15" in result.stdout

    def test_status_custom_url(self, runner, mock_requests):
        """Test server status with custom URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["status", "--url", "http://custom-server:9000"])

        assert result.exit_code == 0
        mock_requests["get"].assert_called_with(
            "http://custom-server:9000/health", timeout=5
        )

    # Server Stop Tests

    def test_stop_server(self, runner, mock_requests):
        """Test stopping the server."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Server stopping"}
        mock_requests["post"].return_value = mock_response

        result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "Stopping server" in result.stdout

    def test_stop_server_not_running(self, runner, mock_requests):
        """Test stopping server when it's not running."""
        mock_requests["post"].side_effect = requests.ConnectionError(
            "Connection refused"
        )

        result = runner.invoke(app, ["stop"])

        assert result.exit_code == 0
        assert "Server is not running" in result.stdout

    def test_stop_server_force(self, runner):
        """Test force stopping the server."""
        with patch("os.system") as mock_system:
            mock_system.return_value = 0

            result = runner.invoke(app, ["stop", "--force"])

            assert result.exit_code == 0
            assert "Force stopping server" in result.stdout

    # Server Restart Tests

    def test_restart_server(self, runner, mock_uvicorn, mock_requests):
        """Test restarting the server."""
        # Mock stop request
        stop_response = Mock()
        stop_response.status_code = 200
        mock_requests["post"].return_value = stop_response

        # Mock server start
        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        result = runner.invoke(app, ["restart"])

        assert result.exit_code == 0
        assert "Restarting server" in result.stdout

    def test_restart_server_with_delay(self, runner, mock_uvicorn, mock_requests):
        """Test restarting server with delay."""
        stop_response = Mock()
        stop_response.status_code = 200
        mock_requests["post"].return_value = stop_response

        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        with patch("time.sleep") as mock_sleep:
            result = runner.invoke(app, ["restart", "--delay", "3"])

            assert result.exit_code == 0
            mock_sleep.assert_called_with(3)

    # Health Check Tests

    def test_health_check(self, runner, mock_requests):
        """Test health check endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "checks": {"database": "ok", "redis": "ok", "disk_space": "ok"},
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "Health Check Results" in result.stdout
        assert "Database: ✅ ok" in result.stdout
        assert "Redis: ✅ ok" in result.stdout

    def test_health_check_failed(self, runner, mock_requests):
        """Test health check with failures."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.json.return_value = {
            "status": "unhealthy",
            "checks": {"database": "failed", "redis": "ok", "disk_space": "warning"},
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Health Check Results" in result.stdout
        assert "Database: ❌ failed" in result.stdout
        assert "Disk Space: ⚠️ warning" in result.stdout

    # Log Management Tests

    def test_logs_view(self, runner):
        """Test viewing server logs."""
        log_content = """
        2024-01-15 10:00:00 INFO: Server started on localhost:8000
        2024-01-15 10:01:00 INFO: GET /api/v1/detectors/ 200
        2024-01-15 10:02:00 ERROR: POST /api/v1/detect/ 500 - Database error
        2024-01-15 10:03:00 INFO: GET /health 200
        """

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=log_content)):
                result = runner.invoke(app, ["logs"])

                assert result.exit_code == 0
                assert "Server started" in result.stdout
                assert "Database error" in result.stdout

    def test_logs_tail(self, runner):
        """Test tailing server logs."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                mock_process = Mock()
                mock_popen.return_value = mock_process
                mock_process.poll.return_value = None
                mock_process.stdout.readline.side_effect = [
                    b"2024-01-15 10:00:00 INFO: New log entry\n",
                    b"",  # End of output
                ]

                # Use timeout to prevent infinite loop
                result = runner.invoke(app, ["logs", "--tail", "--timeout", "1"])

                assert result.exit_code == 0

    def test_logs_filter_level(self, runner):
        """Test filtering logs by level."""
        log_content = """
        2024-01-15 10:00:00 INFO: Server started
        2024-01-15 10:01:00 DEBUG: Connection established
        2024-01-15 10:02:00 ERROR: Database error
        2024-01-15 10:03:00 WARNING: High memory usage
        """

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=log_content)):
                result = runner.invoke(app, ["logs", "--level", "ERROR"])

                assert result.exit_code == 0
                assert "Database error" in result.stdout
                assert "Server started" not in result.stdout

    # Configuration Management Tests

    def test_config_show(self, runner):
        """Test showing server configuration."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            config = Mock()
            config.api_host = "localhost"
            config.api_port = 8000
            config.debug = False
            config.workers = 1
            config.log_level = "info"
            container.config.return_value = config
            mock_container.return_value = container

            result = runner.invoke(app, ["config", "show"])

            assert result.exit_code == 0
            assert "Server Configuration" in result.stdout
            assert "Host: localhost" in result.stdout
            assert "Port: 8000" in result.stdout

    def test_config_validate(self, runner):
        """Test validating server configuration."""
        config = {"host": "localhost", "port": 8000, "workers": 4, "log_level": "info"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            json.dump(config, config_file)
            config_path = config_file.name

        try:
            result = runner.invoke(app, ["config", "validate", config_path])

            assert result.exit_code == 0
            assert "Configuration is valid" in result.stdout

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_config_validate_invalid(self, runner):
        """Test validating invalid server configuration."""
        invalid_config = {
            "host": "localhost",
            "port": "invalid_port",  # Should be integer
            "workers": -1,  # Should be positive
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            json.dump(invalid_config, config_file)
            config_path = config_file.name

        try:
            result = runner.invoke(app, ["config", "validate", config_path])

            assert result.exit_code == 1
            assert "Configuration errors:" in result.stdout

        finally:
            Path(config_path).unlink(missing_ok=True)

    # Performance Monitoring Tests

    def test_metrics(self, runner, mock_requests):
        """Test viewing server metrics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requests_per_minute": 45,
            "average_response_time": 0.12,
            "error_rate": 0.02,
            "active_connections": 15,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 23,
        }
        mock_requests["get"].return_value = mock_response

        result = runner.invoke(app, ["metrics"])

        assert result.exit_code == 0
        assert "Server Metrics" in result.stdout
        assert "Requests/min: 45" in result.stdout
        assert "Response time: 0.12s" in result.stdout
        assert "Error rate: 2.0%" in result.stdout

    def test_metrics_watch(self, runner, mock_requests):
        """Test watching server metrics."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requests_per_minute": 45,
            "average_response_time": 0.12,
        }
        mock_requests["get"].return_value = mock_response

        with patch("time.sleep") as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

            result = runner.invoke(app, ["metrics", "--watch", "--interval", "2"])

            # Should exit gracefully on KeyboardInterrupt
            assert result.exit_code == 0

    # Error Handling Tests

    def test_start_server_port_in_use(self, runner, mock_uvicorn):
        """Test starting server when port is already in use."""
        mock_uvicorn.side_effect = OSError("Address already in use")

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 1
        assert "Address already in use" in result.stdout

    def test_start_server_permission_denied(self, runner, mock_uvicorn):
        """Test starting server with permission denied error."""
        mock_uvicorn.side_effect = PermissionError("Permission denied")

        result = runner.invoke(
            app,
            [
                "start",
                "--port",
                "80",  # Privileged port
            ],
        )

        assert result.exit_code == 1
        assert "Permission denied" in result.stdout

    def test_status_timeout(self, runner, mock_requests):
        """Test server status with timeout."""
        mock_requests["get"].side_effect = requests.Timeout("Request timeout")

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "timeout" in result.stdout.lower()

    # Integration Tests

    def test_complete_server_lifecycle(self, runner, mock_uvicorn, mock_requests):
        """Test complete server lifecycle."""

        # Mock server start
        def mock_run_server(*args, **kwargs):
            pass

        mock_uvicorn.side_effect = mock_run_server

        # Mock status responses
        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "healthy"}

        stop_response = Mock()
        stop_response.status_code = 200
        stop_response.json.return_value = {"message": "Server stopping"}

        mock_requests["get"].return_value = status_response
        mock_requests["post"].return_value = stop_response

        # 1. Start server
        start_result = runner.invoke(app, ["start"])
        assert start_result.exit_code == 0

        # 2. Check status
        status_result = runner.invoke(app, ["status"])
        assert status_result.exit_code == 0
        assert "running" in status_result.stdout

        # 3. Stop server
        stop_result = runner.invoke(app, ["stop"])
        assert stop_result.exit_code == 0


def mock_open(read_data=""):
    """Helper function to create a mock open function."""
    from unittest.mock import mock_open as _mock_open

    return _mock_open(read_data=read_data)
