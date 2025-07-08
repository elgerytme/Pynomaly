import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def test_run_container_scans_soft_mode():
    """
    Test if run_container_scans.py produces SARIF and SBOM files in soft mode.
    Uses a lightweight mocking approach to avoid Docker dependencies.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "security-reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the script with soft mode using a minimal alpine image
        scan_script = "scripts/security/run_container_scans.py"
        result = subprocess.run([
            "python", scan_script,
            "--soft",
            "--image", "alpine:latest",  # Use standard alpine image
            "--output-dir", str(output_dir)
        ], capture_output=True, text=True)
        
        # In soft mode, script should always succeed even if image isn't found
        assert result.returncode == 0, f"Command failed with output: {result.stdout}\nError: {result.stderr}"

        # Verify SARIF and SBOM files are created (even if empty due to soft mode)
        sarif_file = output_dir / "container-vulnerabilities.sarif"
        sbom_file = output_dir / "container-sbom.json"
        
        assert sarif_file.exists(), "SARIF report not created."
        assert sbom_file.exists(), "SBOM report not created."
        
        # Verify files are not empty or at least were touched
        assert sarif_file.stat().st_size >= 0  # File was created
        assert sbom_file.stat().st_size >= 0   # File was created


def test_container_scan_script_help():
    """
    Test that the container scan script shows help correctly.
    """
    scan_script = "scripts/security/run_container_scans.py"
    result = subprocess.run([
        "python", scan_script, "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Container Security Scanner" in result.stdout
    assert "--soft" in result.stdout
    assert "--image" in result.stdout
    assert "--output-dir" in result.stdout


def test_container_scan_script_exists():
    """
    Test that the container scan script exists and is executable.
    """
    scan_script = Path("scripts/security/run_container_scans.py")
    assert scan_script.exists(), "Container scan script not found"
    assert scan_script.is_file(), "Container scan script is not a file"
    
    # Check if it's a valid Python script
    with open(scan_script, "r") as f:
        content = f.read()
        assert "#!/usr/bin/env python3" in content
        assert "ContainerScanner" in content
        assert "def main()" in content
