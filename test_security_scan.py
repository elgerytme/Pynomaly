#!/usr/bin/env python3
"""
Test script to run security scan locally and ensure non-zero exit on high severity vulnerabilities.
This script runs bandit, safety, and pip-audit and fails with non-zero exit code on high severity issues.
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path


def run_bandit(source_path="pynomaly"):
    """Run bandit security scan on source code."""
    print("Running bandit security scan...")
    
    # Create a temporary file for bandit output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    try:
        # Run bandit with JSON output
        result = subprocess.run([
            'bandit', '-r', source_path, '-f', 'json', '-o', output_file
        ], capture_output=True, text=True)
        
        # Read the JSON output
        with open(output_file, 'r') as f:
            bandit_report = json.load(f)
        
        # Check for high severity issues
        high_severity_issues = []
        for result in bandit_report.get('results', []):
            if result.get('issue_severity') == 'HIGH':
                high_severity_issues.append(result)
        
        print(f"Bandit scan completed. Found {len(high_severity_issues)} high severity issues.")
        
        # Clean up temp file
        os.unlink(output_file)
        
        return len(high_severity_issues), bandit_report
        
    except subprocess.CalledProcessError as e:
        print(f"Bandit scan failed: {e}")
        return 0, {}
    except Exception as e:
        print(f"Error running bandit: {e}")
        return 0, {}


def run_safety():
    """Run safety check for known vulnerabilities."""
    print("Running safety check...")
    
    try:
        result = subprocess.run(['safety', 'check'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Safety check found vulnerabilities:")
            print(result.stdout)
            return True
        else:
            print("Safety check passed - no known vulnerabilities found.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Safety check failed: {e}")
        return False
    except Exception as e:
        print(f"Error running safety: {e}")
        return False


def run_pip_audit():
    """Run pip-audit for vulnerability scanning."""
    print("Running pip-audit...")
    
    try:
        result = subprocess.run(['pip-audit'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"pip-audit found vulnerabilities:")
            print(result.stdout)
            return True
        else:
            print("pip-audit passed - no vulnerabilities found.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"pip-audit failed: {e}")
        return False
    except Exception as e:
        print(f"Error running pip-audit: {e}")
        return False


def create_vulnerable_requirements():
    """Create a test requirements file with known vulnerable packages."""
    vulnerable_packages = [
        "django==2.0.0",  # Known CVE
        "requests==2.18.0",  # Known CVE
        "pillow==5.0.0"  # Known CVE
    ]
    
    with open("test_requirements.txt", "w") as f:
        for pkg in vulnerable_packages:
            f.write(f"{pkg}\n")
    
    print("Created test_requirements.txt with vulnerable packages for testing.")


def main():
    """Main function to run security scans."""
    print("Starting security scan validation...")
    
    # Check if source directory exists
    if not os.path.exists("pynomaly"):
        print("Source directory 'pynomaly' not found. Creating sample for testing...")
        os.makedirs("pynomaly", exist_ok=True)
        
        # Create a sample file with potential security issues for testing
        sample_code = '''
import os
import subprocess

# Potential security issues for testing
password = "hardcoded_password"
sql_query = "SELECT * FROM users WHERE name = '%s'" % user_input
os.system(user_command)
subprocess.call(shell_command, shell=True)
'''
        with open("pynomaly/test_file.py", "w") as f:
            f.write(sample_code)
    
    # Create vulnerable requirements for testing
    create_vulnerable_requirements()
    
    # Run security scans
    high_severity_count, bandit_report = run_bandit()
    safety_vulnerabilities = run_safety()
    pip_audit_vulnerabilities = run_pip_audit()
    
    # Determine exit code
    exit_code = 0
    
    if high_severity_count > 0:
        print(f"FAIL: Found {high_severity_count} high severity security issues in code.")
        exit_code = 1
    
    if safety_vulnerabilities:
        print("FAIL: Found vulnerabilities in dependencies (safety).")
        exit_code = 1
    
    if pip_audit_vulnerabilities:
        print("FAIL: Found vulnerabilities in dependencies (pip-audit).")
        exit_code = 1
    
    if exit_code == 0:
        print("SUCCESS: No high severity security issues found.")
    else:
        print("FAIL: Security scan detected high severity issues.")
    
    # Clean up test files
    if os.path.exists("test_requirements.txt"):
        os.remove("test_requirements.txt")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
