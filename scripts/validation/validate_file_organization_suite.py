#!/usr/bin/env python3
"""
Validation script for the file organization test suite.

This script validates that all test components are properly set up and working
for the comprehensive file organization test suite covering:
- Correct detection for each category
- Safe moves/deletes in dry-run mode  
- Actual filesystem changes in tmpfs with --fix
- Integration into existing CI matrix (Python 3.8-3.12)
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_command(cmd: List[str], cwd: str = None) -> Dict[str, Any]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or str(project_root),
            encoding='utf-8',
            errors='replace'
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def check_test_files_exist() -> bool:
    """Check that all required test files exist."""
    required_files = [
        'tests/integration/test_file_organization.py',
        'pytest.ini',
        '.github/workflows/tests.yml'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required test files exist")
    return True

def validate_pytest_config() -> bool:
    """Validate pytest configuration."""
    pytest_ini = project_root / 'pytest.ini'
    
    if not pytest_ini.exists():
        print("❌ pytest.ini file not found")
        return False
    
    content = pytest_ini.read_text(encoding='utf-8')
    required_markers = [
        'integration',
        'file_organization',
        'slow'
    ]
    
    missing_markers = []
    for marker in required_markers:
        if marker not in content:
            missing_markers.append(marker)
    
    if missing_markers:
        print(f"❌ Missing pytest markers: {missing_markers}")
        return False
    
    print("✅ pytest.ini configuration is valid")
    return True

def test_imports() -> bool:
    """Test that all required imports work."""
    try:
        # Test core module imports
        from scripts.analysis.organize_files import FileOrganizer
        from scripts.analysis.detect_stray_files import detect_stray_files
        
        # Test pytest imports
        import pytest
        import tempfile
        
        print("✅ All required imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_specific_tests() -> bool:
    """Run specific file organization tests."""
    test_commands = [
        # Test stray files detection
        ['python', '-m', 'pytest', 'tests/integration/test_file_organization.py::test_detect_stray_files', '-v'],
        
        # Test file organizer dry run
        ['python', '-m', 'pytest', 'tests/integration/test_file_organization.py::test_file_organizer_dry_run', '-v'],
        
        # Test file organizer fix mode
        ['python', '-m', 'pytest', 'tests/integration/test_file_organization.py::test_file_organizer_fix', '-v']
    ]
    
    all_passed = True
    for cmd in test_commands:
        print(f"\nRunning: {' '.join(cmd)}")
        result = run_command(cmd)
        
        if result['success']:
            print(f"✅ Test passed")
        else:
            print(f"❌ Test failed")
            print(f"STDOUT: {result['stdout']}")
            print(f"STDERR: {result['stderr']}")
            all_passed = False
    
    return all_passed

def validate_ci_integration() -> bool:
    """Validate CI workflow includes file organization tests."""
    workflow_file = project_root / '.github/workflows/tests.yml'
    
    if not workflow_file.exists():
        print("❌ CI workflow file not found")
        return False
    
    content = workflow_file.read_text(encoding='utf-8')
    
    # Check for Python version matrix
    python_versions = ['3.8', '3.9', '3.10', '3.11', '3.12']
    missing_versions = []
    
    for version in python_versions:
        if version not in content:
            missing_versions.append(version)
    
    if missing_versions:
        print(f"❌ Missing Python versions in CI: {missing_versions}")
        return False
    
    # Check for pytest integration test execution
    if 'pytest tests/integration' not in content and 'pytest' not in content:
        print("❌ CI workflow doesn't include pytest integration tests")
        return False
    
    print("✅ CI workflow properly configured")
    return True

def run_full_test_suite() -> bool:
    """Run the complete file organization test suite."""
    print("\nRunning full file organization test suite...")
    
    # Run all file organization tests
    cmd = ['python', '-m', 'pytest', 'tests/integration/test_file_organization.py', '-v', '--tb=short']
    result = run_command(cmd)
    
    if result['success']:
        print("✅ Full test suite passed")
        return True
    else:
        print("❌ Full test suite failed")
        print(f"STDOUT: {result['stdout']}")
        print(f"STDERR: {result['stderr']}")
        return False

def main():
    """Main validation function."""
    print("🔍 Validating File Organization Test Suite")
    print("=" * 50)
    
    validations = [
        ("Check test files exist", check_test_files_exist),
        ("Validate pytest config", validate_pytest_config),
        ("Test imports", test_imports),
        ("Validate CI integration", validate_ci_integration),
        ("Run specific tests", run_specific_tests),
        ("Run full test suite", run_full_test_suite)
    ]
    
    all_passed = True
    results = {}
    
    for name, validation_func in validations:
        print(f"\n📋 {name}...")
        try:
            success = validation_func()
            results[name] = success
            if not success:
                all_passed = False
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("File organization test suite is ready for use.")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED!")
        print("Please address the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
