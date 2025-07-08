# tests/cli/test_end_to_end.py
import subprocess
import pytest
import os

def run_cli_command(command):
    result = subprocess.run(
        command, shell=True, text=True, capture_output=True
    )
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    return result.stdout

@pytest.fixture
def sample_csv_file(tmp_path):
    filepath = tmp_path / "sample_data.csv"
    with open(filepath, "w") as f:
        f.write("feature1,feature2\n")
        f.write("0.1,0.2\n")  # normal data
        f.write("10,10\n")    # anomaly
    return str(filepath)

def test_cli_flow(sample_csv_file):
    # Load CSV
    run_cli_command(f"pynomaly dataset load {sample_csv_file} --name test_data")
    
    # Create detector
    run_cli_command("pynomaly detector create --name test_detector --algorithm IsolationForest")

    # Train detector
    run_cli_command("pynomaly detect train test_detector test_data")

    # Run detection
    detection_output = run_cli_command("pynomaly detect run test_detector test_data --output result.csv")
    
    assert "Anomalies found: 1" in detection_output

    # Check output file exists
    assert os.path.exists("result.csv")

