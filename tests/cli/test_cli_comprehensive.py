"""Comprehensive CLI tests using typer.testing and pexpect."""
import io
import sys
import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import os
import tempfile
import subprocess

# Mock CLI app for testing
app = typer.Typer()

@app.command()
def greet(
    name: str,
    greeting: str = typer.Option("Hello", "--greeting", "-g", help="Greeting to use"),
    count: int = typer.Option(1, "--count", "-c", help="Number of times to greet"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Greet someone with customizable options."""
    if verbose:
        typer.echo(f"Greeting '{name}' {count} times with '{greeting}'")
    
    for i in range(count):
        typer.echo(f"{greeting} {name}!")

@app.command()
def interactive():
    """Interactive command that prompts for input."""
    name = typer.prompt("What's your name?")
    age = typer.prompt("What's your age?", type=int)
    typer.echo(f"Hello {name}, you are {age} years old!")

@app.command()
def config(
    set_key: str = typer.Option(None, "--set", help="Set a configuration key"),
    get_key: str = typer.Option(None, "--get", help="Get a configuration key"),
    list_all: bool = typer.Option(False, "--list", help="List all configuration"),
):
    """Configuration command with multiple options."""
    if set_key:
        key, value = set_key.split("=", 1)
        typer.echo(f"Set {key} = {value}")
    elif get_key:
        typer.echo(f"Getting value for {get_key}")
    elif list_all:
        typer.echo("Listing all configuration...")
    else:
        typer.echo("No action specified. Use --set, --get, or --list")

@app.command()
def failing_command():
    """Command that always fails."""
    typer.echo("This command will fail", err=True)
    raise typer.Exit(1)

@app.command()
def file_processor(
    input_file: typer.FileText = typer.Option(..., "--input", "-i", help="Input file"),
    output_file: typer.FileTextWrite = typer.Option(..., "--output", "-o", help="Output file"),
):
    """Process files with input and output options."""
    content = input_file.read()
    processed = content.upper()
    output_file.write(processed)
    typer.echo(f"Processed {len(content)} characters")

runner = CliRunner()

class TestCommandFlags:
    """Test various command flags and options."""
    
    def test_basic_command_no_flags(self):
        """Test basic command without any flags."""
        result = runner.invoke(app, ["greet", "World"])
        assert result.exit_code == 0
        assert "Hello World!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_command_with_short_flags(self):
        """Test command with short flags."""
        result = runner.invoke(app, ["greet", "Alice", "-g", "Hi", "-c", "2", "-v"])
        assert result.exit_code == 0
        assert "Greeting 'Alice' 2 times with 'Hi'" in result.stdout
        assert "Hi Alice!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_command_with_long_flags(self):
        """Test command with long flags."""
        result = runner.invoke(app, ["greet", "Bob", "--greeting", "Howdy", "--count", "3", "--verbose"])
        assert result.exit_code == 0
        assert "Greeting 'Bob' 3 times with 'Howdy'" in result.stdout
        assert result.stdout.count("Howdy Bob!") == 3
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_mixed_flags(self):
        """Test mixing short and long flags."""
        result = runner.invoke(app, ["greet", "Charlie", "-g", "Hey", "--count", "1", "-v"])
        assert result.exit_code == 0
        assert "Hey Charlie!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")

class TestInteractiveFallbacks:
    """Test interactive command fallbacks."""
    
    def test_interactive_input(self):
        """Test interactive command with input."""
        result = runner.invoke(app, ["interactive"], input="John\n25\n")
        assert result.exit_code == 0
        assert "What's your name?" in result.stdout
        assert "What's your age?" in result.stdout
        assert "Hello John, you are 25 years old!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_interactive_invalid_input(self):
        """Test interactive command with invalid input."""
        result = runner.invoke(app, ["interactive"], input="Jane\ninvalid_age\n30\n")
        assert result.exit_code == 0
        assert "What's your age?" in result.stdout
        # Should eventually succeed with valid input
        assert "Hello Jane, you are 30 years old!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_interactive_eof(self):
        """Test interactive command with EOF."""
        result = runner.invoke(app, ["interactive"], input="")
        assert result.exit_code != 0  # Should fail with EOF
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")

class TestFailurePaths:
    """Test various failure scenarios."""
    
    def test_missing_required_argument(self):
        """Test command fails with missing required argument."""
        result = runner.invoke(app, ["greet"])
        assert result.exit_code != 0
        assert "Error" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_invalid_option_value(self):
        """Test command fails with invalid option value."""
        result = runner.invoke(app, ["greet", "Alice", "--count", "invalid"])
        assert result.exit_code != 0
        assert "Error" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_unknown_command(self):
        """Test unknown command failure."""
        result = runner.invoke(app, ["unknown_command"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_unknown_option(self):
        """Test unknown option failure."""
        result = runner.invoke(app, ["greet", "Alice", "--unknown-option"])
        assert result.exit_code != 0
        assert "No such option" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_failing_command(self):
        """Test command that explicitly fails."""
        result = runner.invoke(app, ["failing_command"])
        assert result.exit_code == 1
        assert "This command will fail" in result.stderr
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")

class TestOutputCapture:
    """Test stdout/stderr capture and exit codes."""
    
    def test_stdout_capture(self):
        """Test capturing stdout."""
        result = runner.invoke(app, ["greet", "Test", "-v"])
        assert result.exit_code == 0
        assert len(result.stdout) > 0
        assert "Hello Test!" in result.stdout
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_stderr_capture(self):
        """Test capturing stderr."""
        result = runner.invoke(app, ["failing_command"])
        assert result.exit_code == 1
        assert len(result.stderr) > 0
        assert "This command will fail" in result.stderr
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_exit_code_success(self):
        """Test successful command exit code."""
        result = runner.invoke(app, ["greet", "Success"])
        assert result.exit_code == 0
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")
    
    def test_exit_code_failure(self):
        """Test failed command exit code."""
        result = runner.invoke(app, ["failing_command"])
        assert result.exit_code == 1
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"EXIT CODE: {result.exit_code}")

class TestSubprocessIntegration:
    """Test CLI execution using subprocess for cross-platform compatibility."""
    
    def test_subprocess_cli_execution(self):
        """Test CLI execution using subprocess."""
        # Create a simple script that uses our CLI
        script_content = f"""
import sys
import os
sys.path.insert(0, '{os.getcwd()}')
from tests.cli.test_cli_comprehensive import app

if __name__ == "__main__":
    app()
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_file.write(script_content)
            script_file.flush()
            
            try:
                # Test successful command
                result = subprocess.run(
                    [sys.executable, script_file.name, "greet", "subprocess"],
                    capture_output=True,
                    text=True
                )
                assert result.returncode == 0
                assert "Hello subprocess!" in result.stdout
                print(f"SUBPROCESS STDOUT: {result.stdout}")
                print(f"SUBPROCESS STDERR: {result.stderr}")
                print(f"SUBPROCESS EXIT CODE: {result.returncode}")
                
                # Test failing command  
                result = subprocess.run(
                    [sys.executable, script_file.name, "failing_command"],
                    capture_output=True,
                    text=True
                )
                assert result.returncode == 1
                assert "This command will fail" in result.stderr
                print(f"SUBPROCESS STDOUT: {result.stdout}")
                print(f"SUBPROCESS STDERR: {result.stderr}")
                print(f"SUBPROCESS EXIT CODE: {result.returncode}")
                
            finally:
                os.unlink(script_file.name)

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "-s"])
