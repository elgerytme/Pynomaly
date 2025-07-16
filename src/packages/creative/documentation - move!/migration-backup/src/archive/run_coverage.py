#!/usr/bin/env python3
"""Run coverage analysis on the comprehensive test."""

import subprocess
import sys
import os

# Set the working directory to the project root
os.chdir('/mnt/c/Users/andre/Pynomaly')

# Run coverage on the comprehensive test
result = subprocess.run([
    sys.executable, '-m', 'pytest', 
    '--cov=src/pynomaly', 
    '--cov-report=term-missing',
    'test_comprehensive_functionality.py',
    '-v'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"Return code: {result.returncode}")