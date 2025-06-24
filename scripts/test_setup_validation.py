#!/usr/bin/env python3
"""Test script to validate setup.py functionality."""

import os
import sys
import subprocess
import tempfile
import shutil

def test_setup_py_syntax():
    """Test if setup.py has valid syntax."""
    try:
        with open('setup.py', 'r') as f:
            content = f.read()
        
        # Try to compile the setup.py content
        compile(content, 'setup.py', 'exec')
        print("[PASS] setup.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] setup.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] setup.py error: {e}")
        return False

def test_setup_py_metadata():
    """Test if setup.py can extract metadata."""
    try:
        # Test basic metadata extraction
        result = subprocess.run([sys.executable, 'setup.py', '--name'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[PASS] Package name: {result.stdout.strip()}")
        else:
            print(f"[FAIL] Failed to get package name: {result.stderr}")
            return False
        
        result = subprocess.run([sys.executable, 'setup.py', '--version'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"[PASS] Package version: {result.stdout.strip()}")
        else:
            print(f"[FAIL] Failed to get package version: {result.stderr}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        print("[FAIL] setup.py metadata extraction timed out")
        return False
    except Exception as e:
        print(f"[FAIL] setup.py metadata error: {e}")
        return False

def test_setup_py_check():
    """Test if setup.py passes check command."""
    try:
        result = subprocess.run([sys.executable, 'setup.py', 'check'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("[PASS] setup.py check passed")
            return True
        else:
            print(f"[FAIL] setup.py check failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] setup.py check timed out")
        return False
    except Exception as e:
        print(f"[FAIL] setup.py check error: {e}")
        return False

def test_setup_py_build_dry_run():
    """Test if setup.py can do a dry run build."""
    try:
        result = subprocess.run([sys.executable, 'setup.py', 'build', '--dry-run'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("[PASS] setup.py build dry run passed")
            return True
        else:
            print(f"[FAIL] setup.py build dry run failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] setup.py build dry run timed out")
        return False
    except Exception as e:
        print(f"[FAIL] setup.py build dry run error: {e}")
        return False

def main():
    """Run all setup.py tests."""
    print("Testing setup.py functionality...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)
    
    tests = [
        test_setup_py_syntax,
        test_setup_py_metadata,
        test_setup_py_check,
        test_setup_py_build_dry_run,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All setup.py tests passed!")
        return 0
    else:
        print("[ERROR] Some setup.py tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())