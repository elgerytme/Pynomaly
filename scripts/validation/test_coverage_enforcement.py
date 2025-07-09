#!/usr/bin/env python3
"""
Test script for coverage enforcement validation.
This script validates that the coverage threshold enforcement works correctly.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

def run_command(command: str, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    return result

def parse_coverage_json(coverage_file: Path) -> Dict[str, Any]:
    """Parse coverage JSON file and extract coverage data."""
    try:
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        return coverage_data
    except Exception as e:
        print(f"Error parsing coverage file: {e}")
        return {}

def calculate_package_coverage(coverage_data: Dict[str, Any], package_prefix: str) -> float:
    """Calculate coverage for a specific package."""
    if 'files' not in coverage_data:
        return 0.0
    
    files = coverage_data['files']
    matching_files = [
        file_data for file_path, file_data in files.items()
        if file_path.startswith(package_prefix)
    ]
    
    if not matching_files:
        return 0.0
    
    total_lines = sum(file_data['summary']['num_statements'] for file_data in matching_files)
    covered_lines = sum(file_data['summary']['covered_lines'] for file_data in matching_files)
    
    return (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

def test_coverage_enforcement():
    """Test coverage enforcement functionality."""
    project_root = Path(__file__).parent.parent.parent
    
    print("🔍 Testing Coverage Enforcement")
    print("=" * 50)
    
    # Define coverage thresholds
    thresholds = {
        'overall': 95,
        'domain': 98,
        'application': 95,
        'infrastructure': 90,
        'presentation': 85
    }
    
    # Run tests with coverage
    print("\n📊 Running tests with coverage...")
    test_result = run_command(
        "hatch run test:run tests/domain/ tests/application/ -v --tb=short "
        "--cov=src/pynomaly --cov-report=json:coverage.json --cov-report=term",
        cwd=project_root
    )
    
    if test_result.returncode != 0:
        print("❌ Tests failed - cannot validate coverage")
        return False
    
    # Check if coverage.json exists
    coverage_file = project_root / "coverage.json"
    if not coverage_file.exists():
        print("❌ Coverage JSON file not found")
        return False
    
    # Parse coverage data
    coverage_data = parse_coverage_json(coverage_file)
    if not coverage_data:
        print("❌ Failed to parse coverage data")
        return False
    
    # Extract overall coverage
    overall_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
    print(f"\n📈 Overall coverage: {overall_coverage:.1f}%")
    
    # Check overall threshold
    overall_passed = overall_coverage >= thresholds['overall']
    print(f"{'✅' if overall_passed else '❌'} Overall threshold: {overall_coverage:.1f}% >= {thresholds['overall']}%")
    
    # Check per-package coverage
    package_results = {}
    packages = {
        'domain': 'src/pynomaly/domain',
        'application': 'src/pynomaly/application',
        'infrastructure': 'src/pynomaly/infrastructure',
        'presentation': 'src/pynomaly/presentation'
    }
    
    all_passed = overall_passed
    
    for package_name, package_path in packages.items():
        coverage = calculate_package_coverage(coverage_data, package_path)
        threshold = thresholds[package_name]
        passed = coverage >= threshold
        
        package_results[package_name] = {
            'coverage': coverage,
            'threshold': threshold,
            'passed': passed
        }
        
        print(f"{'✅' if passed else '❌'} {package_name.capitalize()} coverage: {coverage:.1f}% >= {threshold}%")
        
        if not passed:
            all_passed = False
    
    # Summary
    print("\n🎯 Coverage Enforcement Summary")
    print("=" * 35)
    
    if all_passed:
        print("✅ All coverage thresholds met!")
        print("🚀 CI/CD pipeline would pass")
    else:
        print("❌ Some coverage thresholds not met")
        print("🛑 CI/CD pipeline would fail")
        
        # Show failing packages
        print("\n💡 Failing packages:")
        for package_name, result in package_results.items():
            if not result['passed']:
                gap = result['threshold'] - result['coverage']
                print(f"   • {package_name.capitalize()}: {result['coverage']:.1f}% (need {gap:.1f}% more)")
    
    return all_passed

def main():
    """Main function."""
    try:
        success = test_coverage_enforcement()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
