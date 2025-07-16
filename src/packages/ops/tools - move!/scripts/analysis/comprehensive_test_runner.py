#!/usr/bin/env python3
"""
Comprehensive Test Runner for Pynomaly
Tests all packages and applications systematically
"""

import asyncio
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Test results storage
test_results = {
    "start_time": datetime.now().isoformat(),
    "tests": {},
    "summary": {"passed": 0, "failed": 0, "skipped": 0}
}

def log_test_result(test_name: str, status: str, message: str = "", details: Any = None):
    """Log test result"""
    test_results["tests"][test_name] = {
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    
    if status == "PASS":
        test_results["summary"]["passed"] += 1
        print(f"✅ {test_name}: {message}")
    elif status == "FAIL":
        test_results["summary"]["failed"] += 1
        print(f"❌ {test_name}: {message}")
    elif status == "SKIP":
        test_results["summary"]["skipped"] += 1
        print(f"⏭️  {test_name}: {message}")

def test_core_package():
    """Test core Pynomaly package"""
    try:
        # Test basic import
        sys.path.insert(0, 'src')
        import monorepo
        log_test_result("core_package_import", "PASS", f"Version {monorepo.__version__}")
        
        # Test core functionality
        from monorepo.domain.entities import Detector, Dataset
        from monorepo.domain.value_objects import ContaminationRate
        
        detector = monorepo.create_detector("test_detector", "IsolationForest", 0.1)
        log_test_result("core_detector_creation", "PASS", f"Created detector: {detector.name}")
        
        # Test data loading
        import pandas as pd
        import numpy as np
        
        data = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
        dataset = monorepo.load_dataset(data, "test_dataset")
        log_test_result("core_dataset_loading", "PASS", f"Loaded dataset: {dataset.name}")
        
        return True
        
    except Exception as e:
        log_test_result("core_package", "FAIL", str(e), traceback.format_exc())
        return False

def test_cli_application():
    """Test CLI application"""
    try:
        # Test CLI import
        sys.path.insert(0, 'src')
        from monorepo.presentation.cli.app import app
        log_test_result("cli_import", "PASS", "CLI app imported successfully")
        
        # Test CLI help command
        result = subprocess.run([
            sys.executable, "-m", "monorepo", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            log_test_result("cli_help", "PASS", "CLI help command works")
        else:
            log_test_result("cli_help", "FAIL", f"Exit code: {result.returncode}")
            
        # Test CLI version command
        result = subprocess.run([
            sys.executable, "-m", "monorepo", "version"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            log_test_result("cli_version", "PASS", f"Version: {result.stdout.strip()}")
        else:
            log_test_result("cli_version", "FAIL", f"Exit code: {result.returncode}")
            
        return True
        
    except Exception as e:
        log_test_result("cli_application", "FAIL", str(e), traceback.format_exc())
        return False

def test_web_api():
    """Test Web API application"""
    try:
        # Test API import
        sys.path.insert(0, 'src')
        from monorepo.presentation.api.app import app
        log_test_result("api_import", "PASS", "API app imported successfully")
        
        # Test API structure
        from monorepo.presentation.api.dependencies import get_container
        container = get_container()
        log_test_result("api_container", "PASS", "DI container working")
        
        return True
        
    except Exception as e:
        log_test_result("web_api", "FAIL", str(e), traceback.format_exc())
        return False

def test_web_ui():
    """Test Web UI application"""
    try:
        # Test Web UI import
        sys.path.insert(0, 'src')
        from monorepo.presentation.web.app import main
        log_test_result("web_ui_import", "PASS", "Web UI imported successfully")
        
        return True
        
    except Exception as e:
        log_test_result("web_ui", "FAIL", str(e), traceback.format_exc())
        return False

def test_packages():
    """Test individual packages"""
    packages = [
        ("data_profiling", "src/packages/data_profiling"),
        ("data_quality", "src/packages/data_quality"),
        ("data_science", "src/packages/data_science"),
        ("data_transformation", "src/packages/data_transformation"),
        ("mlops", "src/packages/mlops"),
        ("mathematics", "src/packages/mathematics"),
        ("enterprise", "src/packages/enterprise"),
        ("python_sdk", "src/packages/python_sdk"),
    ]
    
    for package_name, package_path in packages:
        try:
            package_dir = Path(package_path)
            if package_dir.exists():
                init_file = package_dir / "__init__.py"
                if init_file.exists():
                    log_test_result(f"package_{package_name}", "PASS", f"Package structure valid")
                else:
                    log_test_result(f"package_{package_name}", "SKIP", "No __init__.py found")
            else:
                log_test_result(f"package_{package_name}", "SKIP", "Package directory not found")
                
        except Exception as e:
            log_test_result(f"package_{package_name}", "FAIL", str(e))

def test_dependencies():
    """Test dependency installation and compatibility"""
    try:
        # Run dependency validation
        result = subprocess.run([
            sys.executable, "scripts/validate_dependencies.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            log_test_result("dependencies", "PASS", "All dependencies validated")
        else:
            log_test_result("dependencies", "FAIL", f"Validation failed: {result.stderr}")
            
    except Exception as e:
        log_test_result("dependencies", "FAIL", str(e))

def test_health_checks():
    """Test health check functionality"""
    try:
        # Test health check CLI
        result = subprocess.run([
            sys.executable, "-m", "monorepo", "health", "system"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            log_test_result("health_checks", "PASS", "Health checks working")
        else:
            log_test_result("health_checks", "FAIL", f"Health check failed: {result.stderr}")
            
    except Exception as e:
        log_test_result("health_checks", "FAIL", str(e))

def generate_report():
    """Generate comprehensive test report"""
    test_results["end_time"] = datetime.now().isoformat()
    
    # Save JSON report
    with open("environments/test_environments/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Generate markdown report
    report = f"""# Pynomaly Comprehensive Test Report

**Generated:** {test_results['end_time']}

## Summary
- ✅ **Passed:** {test_results['summary']['passed']}
- ❌ **Failed:** {test_results['summary']['failed']}
- ⏭️ **Skipped:** {test_results['summary']['skipped']}

## Test Results

"""
    
    for test_name, result in test_results["tests"].items():
        status_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(result["status"], "❓")
        report += f"### {status_icon} {test_name}\n"
        report += f"**Status:** {result['status']}\n"
        report += f"**Message:** {result['message']}\n"
        if result.get('details'):
            report += f"**Details:** {result['details'][:500]}...\n"
        report += f"**Time:** {result['timestamp']}\n\n"
    
    with open("environments/test_environments/COMPREHENSIVE_TEST_REPORT.md", "w") as f:
        f.write(report)
    
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Passed: {test_results['summary']['passed']}")
    print(f"   ❌ Failed: {test_results['summary']['failed']}")
    print(f"   ⏭️ Skipped: {test_results['summary']['skipped']}")
    print(f"\n📝 Reports saved:")
    print(f"   - environments/test_environments/test_results.json")
    print(f"   - environments/test_environments/COMPREHENSIVE_TEST_REPORT.md")

def main():
    """Run comprehensive test suite"""
    print("🚀 Starting Comprehensive Pynomaly Testing")
    print("=" * 50)
    
    # Run all tests
    test_core_package()
    test_cli_application()
    test_web_api()
    test_web_ui()
    test_packages()
    test_dependencies()
    test_health_checks()
    
    # Generate report
    generate_report()
    
    # Exit with appropriate code
    if test_results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()