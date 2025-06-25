#!/usr/bin/env python3
"""Test script to validate pyproject.toml setup"""

import sys
import subprocess

def test_setup():
    print("🔍 Testing pyproject.toml setup validation...")
    
    # Test basic install in dry-run mode
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", "--dry-run"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ pyproject.toml is valid for pip installation")
            return True
        else:
            print("❌ pip dry-run failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)