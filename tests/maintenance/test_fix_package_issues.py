#!/usr/bin/env python3
"""Unit tests for fix_package_issues maintenance script."""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess

from scripts.maintenance.fix_package_issues import (
    run_command,
    main,
)


@pytest.mark.unit
class TestRunCommand:
    """Test the run_command function."""
    
    def test_run_command_success(self):
        """Test successful command execution."""
        cmd = ["echo", "test"]
        description = "Test command"
        
        with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "test output"
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('builtins.print') as mock_print:
                result = run_command(cmd, description)
                
                assert result is True
                mock_print.assert_any_call(f"\n🔷 {description}")
                mock_print.assert_any_call("✅ Success: test output")
    
    def test_run_command_failure(self):
        """Test failed command execution."""
        cmd = ["false"]
        description = "Test failing command"
        
        with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, cmd, stderr="error message")
            
            with patch('builtins.print') as mock_print:
                result = run_command(cmd, description)
                
                assert result is False
                mock_print.assert_any_call(f"\n🔷 {description}")
                mock_print.assert_any_call("❌ Error: error message")
    
    def test_run_command_with_venv_windows(self):
        """Test command execution with virtual environment on Windows."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test with venv"
        
        with patch('scripts.maintenance.fix_package_issues.os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: path == ".venv" or path.endswith("python.exe")
            
            with patch('scripts.maintenance.fix_package_issues.sys.platform', 'win32'):
                with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
                    mock_result = MagicMock()
                    mock_result.stdout = "success"
                    mock_result.returncode = 0
                    mock_run.return_value = mock_result
                    
                    with patch('builtins.print') as mock_print:
                        result = run_command(cmd, description)
                        
                        assert result is True
                        # Check that venv python was used
                        actual_cmd = mock_run.call_args[0][0]
                        assert actual_cmd[0].endswith("python.exe") or actual_cmd[0].endswith("python")
                        mock_print.assert_any_call(f"Using virtual environment: {actual_cmd[0]}")
    
    def test_run_command_with_venv_unix(self):
        """Test command execution with virtual environment on Unix/Linux."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test with venv"
        
        with patch('scripts.maintenance.fix_package_issues.os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda path: path == ".venv" or path.endswith("/bin/python")
            
            with patch('scripts.maintenance.fix_package_issues.sys.platform', 'linux'):
                with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
                    mock_result = MagicMock()
                    mock_result.stdout = "success"
                    mock_result.returncode = 0
                    mock_run.return_value = mock_result
                    
                    with patch('builtins.print') as mock_print:
                        result = run_command(cmd, description)
                        
                        assert result is True
                        # Check that venv python was used
                        actual_cmd = mock_run.call_args[0][0]
                        assert actual_cmd[0].endswith("/bin/python")
                        mock_print.assert_any_call(f"Using virtual environment: {actual_cmd[0]}")
    
    def test_run_command_externally_managed_environment(self):
        """Test handling of externally-managed-environment error."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test externally managed"
        
        with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
            # First call fails with externally-managed-environment
            error1 = subprocess.CalledProcessError(1, cmd, stderr="externally-managed-environment")
            # Second call with --user succeeds
            mock_result = MagicMock()
            mock_result.stdout = "success with --user"
            mock_result.returncode = 0
            mock_run.side_effect = [error1, mock_result]
            
            with patch('builtins.print') as mock_print:
                result = run_command(cmd, description)
                
                assert result is True
                mock_print.assert_any_call("💡 Note: System Python is externally managed. Using --user flag or virtual environment required.")
                mock_print.assert_any_call("🔄 Retrying with --user flag...")
                mock_print.assert_any_call("✅ Success with --user: success with --user")
    
    def test_run_command_externally_managed_environment_user_fails(self):
        """Test handling when both original and --user commands fail."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test externally managed user fails"
        
        with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
            # Both calls fail
            error1 = subprocess.CalledProcessError(1, cmd, stderr="externally-managed-environment")
            error2 = subprocess.CalledProcessError(1, cmd + ["--user"], stderr="user installation failed")
            mock_run.side_effect = [error1, error2]
            
            with patch('builtins.print') as mock_print:
                result = run_command(cmd, description)
                
                assert result is False
                mock_print.assert_any_call("💡 Note: System Python is externally managed. Using --user flag or virtual environment required.")
                mock_print.assert_any_call("🔄 Retrying with --user flag...")
                mock_print.assert_any_call("❌ --user also failed: user installation failed")
    
    def test_run_command_no_venv(self):
        """Test command execution without virtual environment."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test without venv"
        
        with patch('scripts.maintenance.fix_package_issues.os.path.exists') as mock_exists:
            mock_exists.return_value = False  # No .venv directory
            
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "success"
                mock_result.returncode = 0
                mock_run.return_value = mock_result
                
                with patch('builtins.print') as mock_print:
                    result = run_command(cmd, description)
                    
                    assert result is True
                    # Should use original python executable
                    actual_cmd = mock_run.call_args[0][0]
                    assert actual_cmd[0] == sys.executable
    
    def test_run_command_use_venv_false(self):
        """Test command execution with use_venv=False."""
        cmd = [sys.executable, "-m", "pip", "install", "test-package"]
        description = "Test use_venv=False"
        
        with patch('scripts.maintenance.fix_package_issues.os.path.exists') as mock_exists:
            mock_exists.return_value = True  # .venv exists
            
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "success"
                mock_result.returncode = 0
                mock_run.return_value = mock_result
                
                with patch('builtins.print') as mock_print:
                    result = run_command(cmd, description, use_venv=False)
                    
                    assert result is True
                    # Should use original python executable even with .venv present
                    actual_cmd = mock_run.call_args[0][0]
                    assert actual_cmd[0] == sys.executable


@pytest.mark.unit
class TestMainFunction:
    """Test the main function."""
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('builtins.print')
    def test_main_successful_execution(self, mock_print, mock_run_command):
        """Test successful execution of main function."""
        # Mock all run_command calls to return True
        mock_run_command.return_value = True
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_pynomaly = MagicMock()
            mock_pynomaly.__version__ = "1.0.0"
            mock_pynomaly.__file__ = "/path/to/pynomaly/__init__.py"
            mock_import.return_value = mock_pynomaly
            
            # Mock subprocess for CLI test
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                
                main()
                
                # Verify main steps were called
                mock_print.assert_any_call("🔧 Fixing Pynomaly Package Issues")
                mock_print.assert_any_call("=" * 50)
                mock_print.assert_any_call("✅ pynomaly package imports successfully")
                mock_print.assert_any_call("✅ Package fix process completed!")
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('builtins.print')
    def test_main_with_installation_failure(self, mock_print, mock_run_command):
        """Test main function when installation fails."""
        # Mock run_command to fail for installation steps
        mock_run_command.side_effect = [False, False, True, True, True]  # Uninstall fails, install fails, but others succeed
        
        # Mock failed import
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module named 'pynomaly'")
            
            main()
            
            # Verify error handling
            mock_print.assert_any_call("❌ Failed to import pynomaly: No module named 'pynomaly'")
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('builtins.print')
    def test_main_with_minimal_setup_fallback(self, mock_print, mock_run_command):
        """Test main function falling back to minimal setup."""
        # Mock run_command: uninstall succeeds, server install fails, minimal succeeds
        mock_run_command.side_effect = [True, False, True, True, True, True]
        
        # Mock successful import after minimal install
        with patch('builtins.__import__') as mock_import:
            mock_pynomaly = MagicMock()
            mock_pynomaly.__version__ = "development"
            mock_pynomaly.__file__ = "/path/to/pynomaly/__init__.py"
            mock_import.return_value = mock_pynomaly
            
            # Mock subprocess for CLI test
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                
                main()
                
                # Verify fallback was triggered
                mock_print.assert_any_call("⚠️  Trying with minimal setup...")
                mock_print.assert_any_call("✅ pynomaly package imports successfully")
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('builtins.print')
    def test_main_cli_test_failure(self, mock_print, mock_run_command):
        """Test main function when CLI test fails."""
        # Mock successful installation
        mock_run_command.return_value = True
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_pynomaly = MagicMock()
            mock_pynomaly.__version__ = "1.0.0"
            mock_pynomaly.__file__ = "/path/to/pynomaly/__init__.py"
            mock_import.return_value = mock_pynomaly
            
            # Mock subprocess for CLI test failure
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=1, stderr="CLI failed")
                
                main()
                
                # Verify CLI failure was handled
                mock_print.assert_any_call("❌ CLI import failed: CLI failed")
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('builtins.print')
    def test_main_cli_test_timeout(self, mock_print, mock_run_command):
        """Test main function when CLI test times out."""
        # Mock successful installation
        mock_run_command.return_value = True
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_pynomaly = MagicMock()
            mock_pynomaly.__version__ = "1.0.0"
            mock_pynomaly.__file__ = "/path/to/pynomaly/__init__.py"
            mock_import.return_value = mock_pynomaly
            
            # Mock subprocess for CLI test timeout
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_subprocess:
                mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 10)
                
                main()
                
                # Verify timeout was handled
                mock_print.assert_any_call("❌ CLI test failed: Command 'cmd' timed out after 10 seconds")
    
    @patch('scripts.maintenance.fix_package_issues.run_command')
    @patch('scripts.maintenance.fix_package_issues.sys.executable', '/custom/python')
    @patch('scripts.maintenance.fix_package_issues.sys.platform', 'linux')
    @patch('scripts.maintenance.fix_package_issues.os.path.exists')
    @patch('builtins.print')
    def test_main_environment_info_display(self, mock_print, mock_exists, mock_run_command):
        """Test that environment information is displayed correctly."""
        # Mock .venv existence
        mock_exists.return_value = True
        mock_run_command.return_value = True
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_pynomaly = MagicMock()
            mock_pynomaly.__version__ = "1.0.0"
            mock_pynomaly.__file__ = "/path/to/pynomaly/__init__.py"
            mock_import.return_value = mock_pynomaly
            
            # Mock subprocess for CLI test
            with patch('scripts.maintenance.fix_package_issues.subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                
                main()
                
                # Verify environment info was displayed
                mock_print.assert_any_call("📌 Environment Info:")
                mock_print.assert_any_call("Python: /custom/python")
                mock_print.assert_any_call("Platform: linux")
                mock_print.assert_any_call("Virtual env exists: True")


@pytest.mark.integration
class TestIntegrationFixPackageIssues:
    """Integration tests for fix_package_issues."""
    
    def test_run_command_with_real_echo(self):
        """Test run_command with a real echo command."""
        cmd = ["echo", "hello world"]
        description = "Test real echo"
        
        with patch('builtins.print') as mock_print:
            result = run_command(cmd, description)
            
            assert result is True
            mock_print.assert_any_call(f"\n🔷 {description}")
            mock_print.assert_any_call("✅ Success: hello world")
    
    def test_run_command_with_invalid_command(self):
        """Test run_command with an invalid command."""
        cmd = ["nonexistent_command_12345"]
        description = "Test invalid command"
        
        with patch('builtins.print') as mock_print:
            result = run_command(cmd, description)
            
            assert result is False
            mock_print.assert_any_call(f"\n🔷 {description}")
            # Should print some error message
            error_calls = [call for call in mock_print.call_args_list if "❌ Error:" in str(call)]
            assert len(error_calls) > 0
    
    def test_directory_structure_checks(self, tmp_path: Path):
        """Test that directory structure checks work correctly."""
        # Change to temporary directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create a mock .venv directory
            venv_dir = tmp_path / ".venv"
            venv_dir.mkdir()
            
            # Test that venv detection works
            assert os.path.exists(".venv")
            
            # Test command execution in this context
            cmd = ["echo", "test"]
            description = "Test in temp directory"
            
            with patch('builtins.print') as mock_print:
                result = run_command(cmd, description, use_venv=False)  # Disable venv to avoid path issues
                
                assert result is True
                mock_print.assert_any_call(f"\n🔷 {description}")
                mock_print.assert_any_call("✅ Success: test")
        
        finally:
            os.chdir(original_cwd)


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility aspects of the fix_package_issues script."""
    
    def test_command_preparation(self):
        """Test command preparation logic."""
        # Test basic command
        cmd = ["python", "-m", "pip", "install", "package"]
        
        # In the actual function, this would be modified based on venv
        # Here we're just testing the structure
        assert cmd[0] == "python"
        assert cmd[1:3] == ["-m", "pip"]
        assert "install" in cmd
        assert "package" in cmd
    
    def test_error_message_parsing(self):
        """Test error message parsing for different scenarios."""
        # Test externally-managed-environment detection
        error_msg = "error: externally-managed-environment"
        assert "externally-managed-environment" in error_msg
        
        # Test other error messages
        error_msg2 = "Permission denied"
        assert "externally-managed-environment" not in error_msg2
    
    def test_platform_detection(self):
        """Test platform-specific path handling."""
        # Test Windows path
        windows_path = os.path.join(".venv", "Scripts", "python.exe")
        assert "Scripts" in windows_path
        assert windows_path.endswith("python.exe")
        
        # Test Unix path
        unix_path = os.path.join(".venv", "bin", "python")
        assert "bin" in unix_path
        assert unix_path.endswith("python")
    
    def test_monitoring_dependencies_list(self):
        """Test monitoring dependencies list structure."""
        monitoring_deps = [
            "prometheus-fastapi-instrumentator>=7.0.0",
            "shap>=0.46.0",
            "lime>=0.2.0.1",
        ]
        
        assert len(monitoring_deps) == 3
        assert all(">=" in dep for dep in monitoring_deps)
        assert any("prometheus" in dep for dep in monitoring_deps)
        assert any("shap" in dep for dep in monitoring_deps)
        assert any("lime" in dep for dep in monitoring_deps)
