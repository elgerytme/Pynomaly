"""Comprehensive tests for server management CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from typer.testing import CliRunner

from monorepo.presentation.cli.server import app


class TestServerCommands:
    """Test suite for server management CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container with server settings."""
        with patch(
            "monorepo.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()

            # Mock settings
            settings = Mock()
            settings.api_host = "127.0.0.1"
            settings.api_port = 8000
            settings.storage_path = Path("/tmp/pynomaly")
            settings.log_path = Path("/tmp/pynomaly/logs")
            settings.temp_path = Path("/tmp/pynomaly/temp")
            settings.model_path = Path("/tmp/pynomaly/models")
            settings.cors_origins = ["*"]
            settings.rate_limit_requests = 1000
            settings.max_workers = 4
            settings.batch_size = 100
            settings.cache_ttl_seconds = 3600
            settings.gpu_enabled = True
            settings.max_dataset_size_mb = 1000
            settings.default_contamination_rate = 0.1
            settings.debug = False
            settings.environment = "development"

            container.config.return_value = settings
            mock_get_container.return_value = container
            return container

    @patch("socket.socket")
    @patch("subprocess.run")
    def test_start_server_success(
        self, mock_subprocess, mock_socket, runner, mock_container
    ):
        """Test successful server start."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1  # Connection refused (port available)
        mock_socket.return_value = mock_sock

        # Mock subprocess success
        mock_subprocess.return_value = Mock()

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 0
        assert "Starting Pynomaly API server" in result.stdout
        assert "Host: 127.0.0.1" in result.stdout
        assert "Port: 8000" in result.stdout
        assert "API docs: http://127.0.0.1:8000/docs" in result.stdout
        assert "Health check: http://127.0.0.1:8000/health" in result.stdout

        # Verify subprocess was called with correct command
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "uvicorn" in args
        assert "monorepo.presentation.api.app:app" in args
        assert "--host=127.0.0.1" in args
        assert "--port=8000" in args
        assert "--log-level=info" in args

    @patch("socket.socket")
    def test_start_server_port_in_use(self, mock_socket, runner, mock_container):
        """Test server start when port is already in use."""
        # Mock socket check (port in use)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0  # Connection successful (port in use)
        mock_socket.return_value = mock_sock

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 1
        assert "Port 8000 is already in use" in result.stdout

    @patch("socket.socket")
    @patch("subprocess.run")
    def test_start_server_with_options(
        self, mock_subprocess, mock_socket, runner, mock_container
    ):
        """Test server start with custom options."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        result = runner.invoke(
            app,
            [
                "start",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--reload",
                "--workers",
                "2",
                "--log-level",
                "debug",
            ],
        )

        assert result.exit_code == 0
        assert "Host: 0.0.0.0" in result.stdout
        assert "Port: 8080" in result.stdout
        assert "Reload: True" in result.stdout

        # Verify subprocess was called with correct options
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "--host=0.0.0.0" in args
        assert "--port=8080" in args
        assert "--reload" in args
        assert "--log-level=debug" in args

    @patch("socket.socket")
    @patch("subprocess.run")
    def test_start_server_keyboard_interrupt(
        self, mock_subprocess, mock_socket, runner, mock_container
    ):
        """Test server start with keyboard interrupt."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        # Mock subprocess keyboard interrupt
        mock_subprocess.side_effect = KeyboardInterrupt()

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 0
        assert "Server stopped" in result.stdout

    @patch("socket.socket")
    @patch("subprocess.run")
    def test_start_server_subprocess_error(
        self, mock_subprocess, mock_socket, runner, mock_container
    ):
        """Test server start with subprocess error."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        # Mock subprocess error
        from subprocess import CalledProcessError

        mock_subprocess.side_effect = CalledProcessError(1, ["uvicorn"])

        result = runner.invoke(app, ["start"])

        assert result.exit_code == 1
        assert "Server failed to start" in result.stdout

    @patch("socket.socket")
    @patch("subprocess.run")
    @patch("os.fork")
    def test_start_server_daemon_mode(
        self, mock_fork, mock_subprocess, mock_socket, runner, mock_container
    ):
        """Test server start in daemon mode."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        # Mock fork - parent process
        mock_fork.return_value = 1234

        # Mock PID file
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            result = runner.invoke(app, ["start", "--daemon"])

            assert result.exit_code == 0
            assert "Server started as daemon (PID: 1234)" in result.stdout
            assert pid_file.exists()
            assert pid_file.read_text() == "1234"

    @patch("socket.socket")
    @patch("os.fork")
    def test_start_server_daemon_fork_error(
        self, mock_fork, mock_socket, runner, mock_container
    ):
        """Test server start in daemon mode with fork error."""
        # Mock socket check (port available)
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        # Mock fork error
        mock_fork.side_effect = OSError("Fork failed")

        result = runner.invoke(app, ["start", "--daemon"])

        assert result.exit_code == 1
        assert "Failed to fork process: Fork failed" in result.stdout

    def test_stop_server_no_pid_file(self, runner, mock_container):
        """Test stop server when no PID file exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            result = runner.invoke(app, ["stop"])

            assert result.exit_code == 0
            assert "No server PID file found" in result.stdout

    @patch("os.kill")
    def test_stop_server_success(self, mock_kill, runner, mock_container):
        """Test successful server stop."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("1234")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            # Mock process exists check
            mock_kill.return_value = None

            result = runner.invoke(app, ["stop"])

            assert result.exit_code == 0
            assert "Sent stop signal to server (PID: 1234)" in result.stdout
            assert not pid_file.exists()

            # Verify kill was called twice (check + stop)
            assert mock_kill.call_count == 2

    @patch("os.kill")
    def test_stop_server_force(self, mock_kill, runner, mock_container):
        """Test force stop server."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("1234")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            result = runner.invoke(app, ["stop", "--force"])

            assert result.exit_code == 0
            assert "Forcefully killed server (PID: 1234)" in result.stdout

            # Verify SIGKILL was used
            import signal

            mock_kill.assert_called_with(1234, signal.SIGKILL)

    @patch("os.kill")
    def test_stop_server_process_not_found(self, mock_kill, runner, mock_container):
        """Test stop server when process doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("1234")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            # Mock process not found
            mock_kill.side_effect = ProcessLookupError()

            result = runner.invoke(app, ["stop"])

            assert result.exit_code == 0
            assert "Server process not found" in result.stdout
            assert not pid_file.exists()

    @patch("os.kill")
    def test_stop_server_invalid_pid_file(self, mock_kill, runner, mock_container):
        """Test stop server with invalid PID file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("invalid_pid")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            result = runner.invoke(app, ["stop"])

            assert result.exit_code == 1
            assert "Failed to stop server" in result.stdout

    @patch("os.kill")
    def test_status_server_running(self, mock_kill, runner, mock_container):
        """Test server status when server is running."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("1234")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            # Mock process exists
            mock_kill.return_value = None

            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "status": "healthy",
                    "version": "1.0.0",
                    "uptime_seconds": 3600,
                }
                mock_get.return_value = mock_response

                result = runner.invoke(app, ["status"])

                assert result.exit_code == 0
                assert "Server is running (PID: 1234)" in result.stdout
                assert "API is accessible" in result.stdout
                assert "Status: healthy" in result.stdout
                assert "Version: 1.0.0" in result.stdout
                assert "Uptime: 3600 seconds" in result.stdout

    @patch("os.kill")
    def test_status_server_not_running(self, mock_kill, runner, mock_container):
        """Test server status when server is not running."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            with patch("requests.get") as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()

                result = runner.invoke(app, ["status"])

                assert result.exit_code == 0
                assert "Server is not running as daemon" in result.stdout
                assert "Cannot connect to API" in result.stdout

    def test_logs_no_file(self, runner, mock_container):
        """Test logs command when no log file exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs"])

            assert result.exit_code == 0
            assert "No log file found" in result.stdout

    def test_logs_show_last_lines(self, runner, mock_container):
        """Test logs command showing last N lines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text(
                "INFO: Server started\nWARNING: High memory usage\nERROR: Database connection failed\nINFO: Request processed\n"
            )
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs", "--lines", "2"])

            assert result.exit_code == 0
            assert "ERROR: Database connection failed" in result.stdout
            assert "INFO: Request processed" in result.stdout
            assert "Showing last 2 lines" in result.stdout

    def test_logs_show_errors_only(self, runner, mock_container):
        """Test logs command showing only errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text(
                "INFO: Server started\nERROR: Database connection failed\nWARNING: High memory usage\nERROR: Authentication failed\n"
            )
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs", "--error"])

            assert result.exit_code == 0
            assert "ERROR: Database connection failed" in result.stdout
            assert "ERROR: Authentication failed" in result.stdout
            assert "INFO: Server started" not in result.stdout

    @patch("subprocess.run")
    def test_logs_follow_success(self, mock_subprocess, runner, mock_container):
        """Test logs command with follow option."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text("INFO: Server started\n")
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs", "--follow"])

            assert result.exit_code == 0
            assert "Following" in result.stdout

            # Verify tail command was called
            mock_subprocess.assert_called_once()
            args = mock_subprocess.call_args[0][0]
            assert "tail" in args
            assert "-f" in args

    @patch("subprocess.run")
    def test_logs_follow_keyboard_interrupt(
        self, mock_subprocess, runner, mock_container
    ):
        """Test logs command with follow interrupted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text("INFO: Server started\n")
            mock_container.config.return_value.log_path = Path(tmp_dir)

            # Mock keyboard interrupt
            mock_subprocess.side_effect = KeyboardInterrupt()

            result = runner.invoke(app, ["logs", "--follow"])

            assert result.exit_code == 0
            assert "Stopped following logs" in result.stdout

    def test_show_server_config(self, runner, mock_container):
        """Test show server configuration command."""
        result = runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "Server Configuration:" in result.stdout
        assert "API Settings:" in result.stdout
        assert "Host: 127.0.0.1" in result.stdout
        assert "Port: 8000" in result.stdout
        assert "Storage Settings:" in result.stdout
        assert "Performance Settings:" in result.stdout
        assert "Max Workers: 4" in result.stdout
        assert "GPU Enabled: True" in result.stdout
        assert "Environment:" in result.stdout
        assert "Debug Mode: False" in result.stdout

    @patch("requests.get")
    def test_health_check_all_healthy(self, mock_get, runner, mock_container):
        """Test health check when all endpoints are healthy."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "API Health Check:" in result.stdout
        assert "Health Check: OK" in result.stdout
        assert "Detectors API: OK" in result.stdout
        assert "API Documentation: OK" in result.stdout
        assert "All systems operational!" in result.stdout

    @patch("requests.get")
    def test_health_check_some_unhealthy(self, mock_get, runner, mock_container):
        """Test health check when some endpoints are unhealthy."""

        def mock_response_func(url, **kwargs):
            if "/health" in url:
                response = Mock()
                response.status_code = 200
                return response
            else:
                raise requests.exceptions.ConnectionError()

        mock_get.side_effect = mock_response_func

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Health Check: OK" in result.stdout
        assert "Connection failed" in result.stdout
        assert "Some services are not available" in result.stdout

    @patch("requests.get")
    def test_health_check_connection_error(self, mock_get, runner, mock_container):
        """Test health check with connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Connection failed" in result.stdout
        assert "Some services are not available" in result.stdout

    @patch("requests.get")
    def test_health_check_non_200_status(self, mock_get, runner, mock_container):
        """Test health check with non-200 status codes."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Status 503" in result.stdout
        assert "Some services are not available" in result.stdout


class TestServerCommandsHelp:
    """Test help functionality for server commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_server_help(self, runner):
        """Test server command help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout
        assert "logs" in result.stdout
        assert "config" in result.stdout
        assert "health" in result.stdout

    def test_start_help(self, runner):
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start the API server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--reload" in result.stdout
        assert "--workers" in result.stdout
        assert "--daemon" in result.stdout

    def test_stop_help(self, runner):
        """Test stop command help."""
        result = runner.invoke(app, ["stop", "--help"])

        assert result.exit_code == 0
        assert "Stop the API server" in result.stdout
        assert "--force" in result.stdout

    def test_status_help(self, runner):
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "Check server status" in result.stdout

    def test_logs_help(self, runner):
        """Test logs command help."""
        result = runner.invoke(app, ["logs", "--help"])

        assert result.exit_code == 0
        assert "Show server logs" in result.stdout
        assert "--lines" in result.stdout
        assert "--follow" in result.stdout
        assert "--error" in result.stdout

    def test_config_help(self, runner):
        """Test config command help."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "Show server configuration" in result.stdout

    def test_health_help(self, runner):
        """Test health command help."""
        result = runner.invoke(app, ["health", "--help"])

        assert result.exit_code == 0
        assert "Check API health" in result.stdout


class TestServerCommandsEdgeCases:
    """Test edge cases and error conditions for server commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container with server settings."""
        with patch(
            "monorepo.presentation.cli.server.get_cli_container"
        ) as mock_get_container:
            container = Mock()
            settings = Mock()
            settings.api_host = "127.0.0.1"
            settings.api_port = 8000
            settings.storage_path = Path("/tmp/pynomaly")
            settings.log_path = Path("/tmp/pynomaly/logs")
            container.config.return_value = settings
            mock_get_container.return_value = container
            return container

    def test_logs_file_read_error(self, runner, mock_container):
        """Test logs command with file read error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text("test log")
            mock_container.config.return_value.log_path = Path(tmp_dir)

            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                result = runner.invoke(app, ["logs"])

                assert result.exit_code == 0
                assert "Failed to read logs: Permission denied" in result.stdout

    def test_logs_empty_file(self, runner, mock_container):
        """Test logs command with empty log file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text("")
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs"])

            assert result.exit_code == 0
            assert "No matching log entries found" in result.stdout

    def test_logs_error_filter_no_matches(self, runner, mock_container):
        """Test logs command with error filter but no error lines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "server.log"
            log_file.write_text("INFO: Server started\nINFO: Request processed\n")
            mock_container.config.return_value.log_path = Path(tmp_dir)

            result = runner.invoke(app, ["logs", "--error"])

            assert result.exit_code == 0
            assert "No matching log entries found" in result.stdout

    @patch("requests.get")
    def test_health_check_timeout(self, mock_get, runner, mock_container):
        """Test health check with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Timeout" in result.stdout

    @patch("requests.get")
    def test_health_check_request_exception(self, mock_get, runner, mock_container):
        """Test health check with general request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "RequestException" in result.stdout

    def test_status_invalid_pid_file(self, runner, mock_container):
        """Test server status with invalid PID file content."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("not_a_number")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            result = runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "Error reading PID file" in result.stdout

    @patch("os.kill")
    def test_status_process_check_exception(self, mock_kill, runner, mock_container):
        """Test server status with process check exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pid_file = Path(tmp_dir) / "monorepo.pid"
            pid_file.write_text("1234")
            mock_container.config.return_value.storage_path = Path(tmp_dir)

            # Mock general exception during process check
            mock_kill.side_effect = OSError("Permission denied")

            result = runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert (
                "Error reading PID file" in result.stdout
                or "Cannot connect to API" in result.stdout
            )
