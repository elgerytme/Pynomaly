#!/usr/bin/env python3
"""
Failing unit tests to reproduce current CLI failures and trace root cause.

This test file is designed to capture stack traces for commands that were failing 
in the integration report with AttributeError: 'Group' object has no attribute 'registered_commands'.
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from pynomaly.presentation.cli.app import app


class TestCLIFailuresReproduction:
    """Test suite to reproduce and diagnose CLI failures."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_help_command_failure(self, runner):
        """Test that reproduces help command failure - EXPECTED TO FAIL."""
        result = runner.invoke(app, ["--help"])
        
        # This should currently fail with AttributeError
        # We're documenting the current bad behavior
        if result.exit_code != 0:
            print(f"❌ Help command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output: {result.stderr}")
        
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Help command should work but failed: {result.exception}"

    def test_version_command_failure(self, runner):
        """Test that reproduces version command failure - EXPECTED TO FAIL.""" 
        result = runner.invoke(app, ["version"])
        
        # This should currently fail with AttributeError
        # We're documenting the current bad behavior
        if result.exit_code != 0:
            print(f"❌ Version command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output: {result.stderr}")
                
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Version command should work but failed: {result.exception}"

    def test_detector_list_failure(self, runner):
        """Test that reproduces detector list command failure - EXPECTED TO FAIL."""
        result = runner.invoke(app, ["detector", "list"])
        
        # This should currently fail with AttributeError
        if result.exit_code != 0:
            print(f"❌ Detector list command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Detector list command should work but failed: {result.exception}"

    def test_dataset_show_failure(self, runner):
        """Test dataset show command as info command does not exist."""
        result = runner.invoke(app, ["dataset", "show"])
        
        # This is now testing the correct command based on available options
        if result.exit_code != 0:
            print(f"❌ Dataset show command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            
        # This should actually work in a healthy app state
        assert result.exit_code == 0, "Dataset show command should not fail"

    def test_auto_help_failure(self, runner):
        """Test that reproduces auto help command failure - EXPECTED TO FAIL."""
        result = runner.invoke(app, ["auto", "--help"])
        
        # This should currently fail with AttributeError
        if result.exit_code != 0:
            print(f"❌ Auto help command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Auto help command should work but failed: {result.exception}"

    def test_export_formats_failure(self, runner):
        """Test that reproduces export formats command failure - EXPECTED TO FAIL."""
        result = runner.invoke(app, ["export", "list-formats"])
        
        # This should currently fail with AttributeError
        if result.exit_code != 0:
            print(f"❌ Export formats command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Export formats command should work but failed: {result.exception}"

    def test_server_help_failure(self, runner):
        """Test that reproduces server help command failure - EXPECTED TO FAIL."""
        result = runner.invoke(app, ["server", "--help"])
        
        # This should currently fail with AttributeError
        if result.exit_code != 0:
            print(f"❌ Server help command failed with exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
            
        # Assert the current bad behavior
        assert result.exit_code == 0, f"Server help command should work but failed: {result.exception}"

    def test_typer_registered_commands_attribute_error(self, runner):
        """Test to check if registered_commands attribute exists on Typer app."""
        import typer
        
        # Test the main app
        from pynomaly.presentation.cli.app import app
        
        print(f"Main app type: {type(app)}")
        print(f"Main app attributes: {[attr for attr in dir(app) if not attr.startswith('_')]}")
        
        if hasattr(app, 'registered_commands'):
            print(f"✅ Main app has registered_commands: {app.registered_commands}")
        else:
            print("❌ Main app missing registered_commands attribute")
            
        # Test if this is a Typer version issue
        print(f"Typer version: {typer.__version__}")
        
        # Create a minimal Typer app to test
        test_app = typer.Typer()
        
        @test_app.command()
        def test_cmd():
            """Test command."""
            pass
            
        print(f"Test app type: {type(test_app)}")
        print(f"Test app has registered_commands: {hasattr(test_app, 'registered_commands')}")
        
        if hasattr(test_app, 'registered_commands'):
            print(f"Test app registered_commands: {test_app.registered_commands}")

    @patch('pynomaly.presentation.cli.app.get_cli_container')
    def test_version_command_with_mocked_container(self, mock_container, runner):
        """Test version command with mocked container to isolate the issue."""
        # Mock the container to avoid dependency issues
        mock_container_instance = MagicMock()
        mock_config = MagicMock()
        mock_config.app.version = "0.1.0"
        mock_config.storage_path = "test_storage"
        mock_container_instance.config.return_value = mock_config
        mock_container.return_value = mock_container_instance
        
        result = runner.invoke(app, ["version"])
        
        if result.exit_code != 0:
            print(f"❌ Mocked version command failed: {result.exception}")
            import traceback
            if result.exception:
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        
        # This should work with mocked dependencies
        assert result.exit_code == 0, f"Mocked version command failed: {result.exception}"

    def test_check_for_group_object_registered_commands_error(self):
        """Test to manually check for Group object registered_commands error."""
        import typer
        from typer.main import get_group_from_info
        
        # Try to reproduce the specific error from the stack trace
        try:
            # This attempts to trigger the same code path that failed
            app_instance = typer.Typer()
            
            # Try to access the method that was failing in the stack trace
            # This might reveal the internal state causing the issue
            if hasattr(app_instance, 'registered_commands'):
                print(f"✅ App has registered_commands: {type(app_instance.registered_commands)}")
            else:
                print("❌ App missing registered_commands")
                
            # Check if there are any Group objects that don't have registered_commands
            for attr_name in dir(app_instance):
                attr_value = getattr(app_instance, attr_name)
                if hasattr(attr_value, '__class__') and 'Group' in str(type(attr_value)):
                    print(f"Found Group-like object: {attr_name} = {type(attr_value)}")
                    if not hasattr(attr_value, 'registered_commands'):
                        print(f"❌ {attr_name} missing registered_commands attribute!")
                        
        except Exception as e:
            print(f"Error in Group object check: {e}")
            import traceback
            traceback.print_exc()

    def test_trace_typer_internal_call_stack(self, runner):
        """Test to trace the internal Typer call stack that leads to the error."""
        # This test is designed to capture detailed stack traces
        
        # Enable detailed exception information
        import sys
        old_excepthook = sys.excepthook
        
        def detailed_excepthook(exc_type, exc_value, exc_traceback):
            import traceback
            print("=== DETAILED EXCEPTION TRACE ===")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("=== END TRACE ===")
        
        sys.excepthook = detailed_excepthook
        
        try:
            # Try various commands that might trigger the error
            commands_to_test = [
                ["--help"],
                ["version"],
                ["detector", "--help"],
                ["dataset", "--help"],
                ["auto", "--help"],
                ["export", "--help"],
                ["server", "--help"]
            ]
            
            for cmd in commands_to_test:
                print(f"\n--- Testing command: {' '.join(cmd)} ---")
                try:
                    result = runner.invoke(app, cmd)
                    if result.exit_code != 0:
                        print(f"Command failed with exit code: {result.exit_code}")
                        if result.exception:
                            print(f"Exception: {result.exception}")
                            import traceback
                            traceback.print_exception(
                                type(result.exception), 
                                result.exception, 
                                result.exception.__traceback__
                            )
                    else:
                        print(f"✅ Command succeeded")
                except Exception as e:
                    print(f"Test execution error: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            sys.excepthook = old_excepthook


# Test to check Typer version compatibility
def test_typer_version_compatibility():
    """Test Typer version and check for known compatibility issues."""
    import typer
    
    print(f"Typer version: {typer.__version__}")
    
    # Check if we're using a version that has the registered_commands issue
    from packaging import version
    typer_version = version.parse(typer.__version__)
    
    # Known problematic versions (this is hypothetical - adjust based on findings)
    if typer_version >= version.parse("0.15.0"):
        print("⚠️  Using Typer >= 0.15.0 - checking for registered_commands compatibility")
        
        # Test basic Typer functionality
        test_app = typer.Typer()
        
        @test_app.command()
        def dummy():
            pass
            
        if hasattr(test_app, 'registered_commands'):
            print("✅ Typer registered_commands attribute exists")
        else:
            print("❌ Typer registered_commands attribute missing - this may cause issues")
    
    print("Typer version check complete.")


if __name__ == "__main__":
    # Run individual tests for debugging
    runner = CliRunner()
    test_instance = TestCLIFailuresReproduction()
    
    print("🧪 Running CLI failure reproduction tests")
    print("=" * 50)
    
    # Run each test and capture results
    test_methods = [
        test_instance.test_help_command_failure,
        test_instance.test_version_command_failure,
        test_instance.test_detector_list_failure,
        test_instance.test_dataset_show_failure,
        test_instance.test_auto_help_failure,
        test_instance.test_export_formats_failure,
        test_instance.test_server_help_failure,
        test_instance.test_typer_registered_commands_attribute_error,
        test_instance.test_check_for_group_object_registered_commands_error,
    ]
    
    for test_method in test_methods:
        try:
            print(f"\n--- Running {test_method.__name__} ---")
            if 'runner' in test_method.__code__.co_varnames:
                test_method(runner)
            else:
                test_method()
            print("✅ Test completed")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 CLI failure reproduction tests complete")
