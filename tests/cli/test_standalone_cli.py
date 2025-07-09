#!/usr/bin/env python3
"""Standalone CLI tests using typer.testing - no conftest dependencies."""

import typer
from typer.testing import CliRunner
import subprocess
import sys
import os
import tempfile
import json
from datetime import datetime

# Create a comprehensive CLI application for testing
app = typer.Typer(help="PyNomaly CLI for anomaly detection")

# CLI Test Results Storage
test_results = []

def record_test_result(test_name, command, stdout, stderr, exit_code, success=True):
    """Record test results for later analysis."""
    result = {
        "test_name": test_name,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    test_results.append(result)
    print(f"\n--- {test_name} ---")
    print(f"Command: {command}")
    print(f"Exit Code: {exit_code}")
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")
    print(f"Success: {success}")
    print("-" * 50)

@app.command()
def detect(
    input_file: str = typer.Option(..., "--input", "-i", help="Input data file"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
):
    """Run anomaly detection on input data."""
    if verbose:
        typer.echo(f"Running {algorithm} on {input_file}")
        typer.echo(f"Contamination rate: {contamination}")
        typer.echo(f"Output format: {output_format}")
    
    # Simulate detection process
    if not os.path.exists(input_file):
        typer.echo(f"Error: Input file {input_file} not found", err=True)
        raise typer.Exit(1)
    
    # Mock detection results
    results = {
        "algorithm": algorithm,
        "contamination": contamination,
        "anomalies_detected": 5,
        "total_samples": 100,
        "score": 0.95
    }
    
    if output_format == "json":
        typer.echo(json.dumps(results, indent=2))
    else:
        typer.echo(f"Detected {results['anomalies_detected']} anomalies out of {results['total_samples']} samples")

@app.command()
def train(
    data_file: str = typer.Option(..., "--data", "-d", help="Training data file"),
    model_name: str = typer.Option("default", "--model", "-m", help="Model name"),
    save_path: str = typer.Option("./models", "--save", "-s", help="Save path for model"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
):
    """Train an anomaly detection model."""
    typer.echo(f"Training model {model_name} on {data_file}")
    typer.echo(f"Epochs: {epochs}")
    typer.echo(f"Save path: {save_path}")
    
    if not os.path.exists(data_file):
        typer.echo(f"Error: Training data file {data_file} not found", err=True)
        raise typer.Exit(1)
    
    # Simulate training
    typer.echo("Training completed successfully")
    typer.echo(f"Model saved to {save_path}/{model_name}.pkl")

@app.command()
def interactive():
    """Interactive mode for anomaly detection."""
    typer.echo("Welcome to PyNomaly Interactive Mode")
    
    data_file = typer.prompt("Enter data file path")
    algorithm = typer.prompt("Select algorithm", default="isolation_forest")
    contamination = typer.prompt("Contamination rate", default=0.1, type=float)
    
    typer.echo(f"Running {algorithm} on {data_file} with contamination={contamination}")
    
    # Mock results
    typer.echo("Detection completed!")
    typer.echo("Found 3 anomalies out of 50 samples")

@app.command()
def config(
    set_value: str = typer.Option(None, "--set", help="Set configuration value (key=value)"),
    get_key: str = typer.Option(None, "--get", help="Get configuration value"),
    list_all: bool = typer.Option(False, "--list", help="List all configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration"),
):
    """Manage configuration settings."""
    if set_value:
        key, value = set_value.split("=", 1)
        typer.echo(f"Configuration set: {key} = {value}")
    elif get_key:
        # Mock getting configuration
        typer.echo(f"Configuration value for {get_key}: default_value")
    elif list_all:
        typer.echo("Current configuration:")
        typer.echo("  algorithm: isolation_forest")
        typer.echo("  contamination: 0.1")
        typer.echo("  output_format: json")
    elif reset:
        typer.echo("Configuration reset to defaults")
    else:
        typer.echo("No action specified. Use --set, --get, --list, or --reset")

@app.command()
def failing_command():
    """A command that always fails for testing error handling."""
    typer.echo("This command will fail", err=True)
    typer.echo("Simulating an error condition", err=True)
    raise typer.Exit(1)

def run_cli_tests():
    """Run comprehensive CLI tests."""
    runner = CliRunner(mix_stderr=False)
    
    # Test 1: Basic command with required options
    print("=== Running CLI Test Suite ===")
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_file.write("data,value\n1,2\n3,4\n")
        temp_file.flush()
        temp_data_file = temp_file.name
    
    try:
        # Test 1: Basic detect command
        result = runner.invoke(app, ["detect", "--input", temp_data_file, "--algorithm", "isolation_forest"])
        record_test_result(
            "Basic detect command",
            f"detect --input {temp_data_file} --algorithm isolation_forest",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 2: Detect with verbose flag
        result = runner.invoke(app, ["detect", "--input", temp_data_file, "--verbose"])
        record_test_result(
            "Detect with verbose flag",
            f"detect --input {temp_data_file} --verbose",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 3: Detect with all options
        result = runner.invoke(app, ["detect", "--input", temp_data_file, "--algorithm", "lof", "--format", "text", "--verbose", "--contamination", "0.2"])
        record_test_result(
            "Detect with all options",
            f"detect --input {temp_data_file} --algorithm lof --format text --verbose --contamination 0.2",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 4: Train command
        result = runner.invoke(app, ["train", "--data", temp_data_file, "--model", "test_model", "--epochs", "50"])
        record_test_result(
            "Train command",
            f"train --data {temp_data_file} --model test_model --epochs 50",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 5: Interactive command with input
        result = runner.invoke(app, ["interactive"], input=f"{temp_data_file}\\nisolation_forest\\n0.15\\n")
        record_test_result(
            "Interactive command",
            "interactive",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 6: Config commands
        result = runner.invoke(app, ["config", "--set", "algorithm=lof"])
        record_test_result(
            "Config set command",
            "config --set algorithm=lof",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        result = runner.invoke(app, ["config", "--get", "algorithm"])
        record_test_result(
            "Config get command",
            "config --get algorithm",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        result = runner.invoke(app, ["config", "--list"])
        record_test_result(
            "Config list command",
            "config --list",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 0
        )
        
        # Test 7: Failing command
        result = runner.invoke(app, ["failing_command"])
        record_test_result(
            "Failing command",
            "failing_command",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code == 1  # Expected to fail
        )
        
        # Test 8: Missing required argument
        result = runner.invoke(app, ["detect"])
        record_test_result(
            "Missing required argument",
            "detect",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code != 0  # Expected to fail
        )
        
        # Test 9: Invalid option value
        result = runner.invoke(app, ["detect", "--input", temp_data_file, "--contamination", "invalid"])
        record_test_result(
            "Invalid option value",
            f"detect --input {temp_data_file} --contamination invalid",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code != 0  # Expected to fail
        )
        
        # Test 10: Non-existent file
        result = runner.invoke(app, ["detect", "--input", "non_existent_file.csv"])
        record_test_result(
            "Non-existent file",
            "detect --input non_existent_file.csv",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code != 0  # Expected to fail
        )
        
        # Test 11: Unknown command
        result = runner.invoke(app, ["unknown_command"])
        record_test_result(
            "Unknown command",
            "unknown_command",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code != 0  # Expected to fail
        )
        
        # Test 12: Unknown option
        result = runner.invoke(app, ["detect", "--input", temp_data_file, "--unknown-option"])
        record_test_result(
            "Unknown option",
            f"detect --input {temp_data_file} --unknown-option",
            result.stdout,
            getattr(result, 'stderr', ''),
            result.exit_code,
            result.exit_code != 0  # Expected to fail
        )
        
    finally:
        # Clean up temporary file
        os.unlink(temp_data_file)
    
    # Test summary
    print("\\n=== TEST SUMMARY ===")
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    # Save results to file
    with open("tests/cli/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\\nDetailed results saved to tests/cli/test_results.json")
    
    return passed_tests == total_tests

def test_subprocess_execution():
    """Test CLI execution via subprocess."""
    print("\\n=== TESTING SUBPROCESS EXECUTION ===")
    
    # Create a script that uses our CLI
    current_dir = os.getcwd().replace('\\', '/')
    script_content = f'''
import sys
import os
sys.path.insert(0, r"{current_dir}")
from tests.cli.test_standalone_cli import app

if __name__ == "__main__":
    app()
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_file.write(script_content)
        script_file.flush()
        
        try:
            # Create test data file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
                data_file.write("data,value\\n1,2\\n3,4\\n")
                data_file.flush()
                
                try:
                    # Test successful command via subprocess
                    result = subprocess.run(
                        [sys.executable, script_file.name, "detect", "--input", data_file.name],
                        capture_output=True,
                        text=True
                    )
                    
                    print(f"Subprocess test - Success command:")
                    print(f"  Exit code: {result.returncode}")
                    print(f"  STDOUT: {result.stdout}")
                    print(f"  STDERR: {result.stderr}")
                    
                    # Test failing command via subprocess
                    result = subprocess.run(
                        [sys.executable, script_file.name, "failing_command"],
                        capture_output=True,
                        text=True
                    )
                    
                    print(f"\\nSubprocess test - Failing command:")
                    print(f"  Exit code: {result.returncode}")
                    print(f"  STDOUT: {result.stdout}")
                    print(f"  STDERR: {result.stderr}")
                    
                finally:
                    os.unlink(data_file.name)
        finally:
            os.unlink(script_file.name)

if __name__ == "__main__":
    success = run_cli_tests()
    test_subprocess_execution()
    
    if success:
        print("\\n✅ All CLI tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some CLI tests failed!")
        sys.exit(1)
