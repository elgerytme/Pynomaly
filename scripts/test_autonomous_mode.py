#!/usr/bin/env python3
"""
Cross-platform test script for Pynomaly autonomous mode functionality.
Tests autonomous detection with sample data in both bash and PowerShell environments.
"""

import os
import sys
import subprocess
import tempfile
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any
import platform

def create_test_environment() -> Path:
    """Create a clean test environment with sample data."""
    # Create temporary directory for test
    test_dir = Path(tempfile.mkdtemp(prefix="pynomaly_autonomous_test_"))
    
    # Create sample data directory
    data_dir = test_dir / "test_data"
    data_dir.mkdir()
    
    # Generate sample CSV data (simple format for cross-platform compatibility)
    import csv
    import random
    import math
    
    # Set seed for reproducible results
    random.seed(42)
    
    # Create tabular data with known anomalies
    csv_file = data_dir / "sample_data.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature1', 'feature2', 'feature3', 'feature4', 'target'])
        
        # Normal data points (first 950 rows)
        for i in range(950):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            x3 = random.gauss(0, 1)
            x4 = random.gauss(0, 1)
            target = 0  # Normal
            writer.writerow([x1, x2, x3, x4, target])
        
        # Anomalous data points (last 50 rows)
        for i in range(50):
            x1 = random.uniform(-5, 5)
            x2 = random.uniform(-5, 5) 
            x3 = random.uniform(-5, 5)
            x4 = random.uniform(-5, 5)
            target = 1  # Anomaly
            writer.writerow([x1, x2, x3, x4, target])
    
    # Create a simple JSON data file
    json_file = data_dir / "sample_data.json"
    json_data = []
    for i in range(100):
        is_anomaly = i >= 90  # Last 10 are anomalies
        record = {
            "id": i,
            "value1": random.gauss(50, 10) if not is_anomaly else random.gauss(100, 20),
            "value2": random.gauss(25, 5) if not is_anomaly else random.gauss(0, 10),
            "category": random.choice(['A', 'B', 'C']),
            "is_anomaly": is_anomaly
        }
        json_data.append(record)
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"[INFO] Test environment created at: {test_dir}")
    print(f"[INFO] Sample data files:")
    print(f"  - CSV: {csv_file}")
    print(f"  - JSON: {json_file}")
    
    return test_dir

def test_autonomous_basic_functionality(test_env: Path, python_cmd: str) -> Dict[str, Any]:
    """Test basic autonomous mode functionality."""
    results = {
        "test_name": "autonomous_basic_functionality",
        "python_cmd": python_cmd,
        "platform": platform.system(),
        "tests": [],
        "overall_success": True
    }
    
    data_dir = test_env / "test_data"
    csv_file = data_dir / "sample_data.csv"
    
    # Test 1: Check if pynomaly module can be imported
    test_result = {
        "name": "import_test",
        "description": "Test if pynomaly can be imported",
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }
    
    start_time = time.time()
    try:
        cmd = [python_cmd, "-c", "import pynomaly; print('Pynomaly imported successfully')"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(test_env))
        test_result["success"] = result.returncode == 0
        test_result["output"] = result.stdout
        test_result["error"] = result.stderr
    except subprocess.TimeoutExpired:
        test_result["error"] = "Import test timed out"
    except Exception as e:
        test_result["error"] = str(e)
    
    test_result["duration"] = time.time() - start_time
    results["tests"].append(test_result)
    
    if not test_result["success"]:
        results["overall_success"] = False
        return results
    
    # Test 2: Test autonomous service import
    test_result = {
        "name": "autonomous_service_import",
        "description": "Test if autonomous service can be imported",
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }
    
    start_time = time.time()
    try:
        import_cmd = """
try:
    from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
    print('Autonomous service imported successfully')
except ImportError as e:
    print(f'Import failed: {e}')
    exit(1)
"""
        cmd = [python_cmd, "-c", import_cmd]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(test_env))
        test_result["success"] = result.returncode == 0
        test_result["output"] = result.stdout
        test_result["error"] = result.stderr
    except subprocess.TimeoutExpired:
        test_result["error"] = "Autonomous service import test timed out"
    except Exception as e:
        test_result["error"] = str(e)
    
    test_result["duration"] = time.time() - start_time
    results["tests"].append(test_result)
    
    if not test_result["success"]:
        results["overall_success"] = False
        return results
    
    # Test 3: Test basic data loading
    test_result = {
        "name": "data_loading_test",
        "description": "Test basic data loading functionality",
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }
    
    start_time = time.time()
    try:
        data_loading_script = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Test basic pandas functionality
try:
    df = pd.read_csv('{csv_file}')
    print(f'Data loaded successfully: {{df.shape[0]}} rows, {{df.shape[1]}} columns')
    print(f'Features: {{list(df.columns)}}')
    print('Data loading test passed')
except Exception as e:
    print(f'Data loading failed: {{e}}')
    exit(1)
"""
        cmd = [python_cmd, "-c", data_loading_script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(test_env))
        test_result["success"] = result.returncode == 0
        test_result["output"] = result.stdout
        test_result["error"] = result.stderr
    except subprocess.TimeoutExpired:
        test_result["error"] = "Data loading test timed out"
    except Exception as e:
        test_result["error"] = str(e)
    
    test_result["duration"] = time.time() - start_time
    results["tests"].append(test_result)
    
    if not test_result["success"]:
        results["overall_success"] = False
    
    return results

def test_autonomous_detection(test_env: Path, python_cmd: str) -> Dict[str, Any]:
    """Test autonomous detection functionality."""
    results = {
        "test_name": "autonomous_detection",
        "python_cmd": python_cmd,
        "platform": platform.system(),
        "tests": [],
        "overall_success": True
    }
    
    data_dir = test_env / "test_data"
    csv_file = data_dir / "sample_data.csv"
    
    # Test autonomous detection with sample data
    test_result = {
        "name": "autonomous_detection_basic",
        "description": "Test basic autonomous detection functionality",
        "success": False,
        "output": "",
        "error": "",
        "duration": 0
    }
    
    start_time = time.time()
    try:
        # Create a simple autonomous detection script
        detection_script = f"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

try:
    from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
    from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
    from pynomaly.infrastructure.repositories.in_memory_repositories import (
        InMemoryDetectorRepository,
        InMemoryDetectionResultRepository
    )
    
    async def test_detection():
        # Setup service
        data_loaders = {{"csv": CSVLoader()}}
        
        autonomous_service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryDetectionResultRepository(),
            data_loaders=data_loaders
        )
        
        # Quick config for testing
        config = AutonomousConfig(
            max_algorithms=2,
            auto_tune_hyperparams=False,
            verbose=True
        )
        
        # Load data first
        data_path = "{csv_file}"
        
        try:
            # Test data loading
            dataset = await autonomous_service._auto_load_data(str(data_path), config)
            print(f"Data loaded: {{dataset.name}}, shape: {{dataset.data.shape}}")
            
            # Test data profiling
            profile = await autonomous_service._profile_data(dataset, config)
            print(f"Data profiled: {{profile.n_samples}} samples, {{profile.n_features}} features")
            print(f"Complexity score: {{profile.complexity_score:.3f}}")
            
            # Test algorithm recommendation
            recommendations = await autonomous_service._recommend_algorithms(profile, config)
            print(f"Got {{len(recommendations)}} algorithm recommendations")
            
            if recommendations:
                print(f"Top recommendation: {{recommendations[0].algorithm}} (confidence: {{recommendations[0].confidence:.1%}})")
            
            print("Autonomous detection test completed successfully")
            return True
            
        except Exception as e:
            print(f"Detection failed: {{e}}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test
    success = asyncio.run(test_detection())
    sys.exit(0 if success else 1)
    
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write script to file and execute
        script_file = test_env / "detection_test.py"
        with open(script_file, 'w') as f:
            f.write(detection_script)
        
        cmd = [python_cmd, str(script_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(test_env))
        test_result["success"] = result.returncode == 0
        test_result["output"] = result.stdout
        test_result["error"] = result.stderr
        
    except subprocess.TimeoutExpired:
        test_result["error"] = "Autonomous detection test timed out"
    except Exception as e:
        test_result["error"] = str(e)
    
    test_result["duration"] = time.time() - start_time
    results["tests"].append(test_result)
    
    if not test_result["success"]:
        results["overall_success"] = False
    
    return results

def run_environment_test(env_name: str, python_cmd: str, shell_prefix: List[str] = None) -> Dict[str, Any]:
    """Run autonomous mode tests in a specific environment."""
    print(f"\n{'='*50}")
    print(f"Testing Pynomaly Autonomous Mode in {env_name}")
    print(f"Python command: {python_cmd}")
    print(f"Platform: {platform.system()}")
    if shell_prefix:
        print(f"Shell prefix: {' '.join(shell_prefix)}")
    print(f"{'='*50}")
    
    # Create test environment
    test_env = None
    try:
        test_env = create_test_environment()
        
        # Adjust python command for shell prefix
        if shell_prefix:
            full_python_cmd = shell_prefix + [python_cmd]
        else:
            full_python_cmd = python_cmd
        
        # Run tests
        results = {
            "environment": env_name,
            "python_cmd": python_cmd,
            "shell_prefix": shell_prefix,
            "platform": platform.system(),
            "test_env": str(test_env),
            "tests": [],
            "overall_success": True,
            "start_time": time.time()
        }
        
        # Test 1: Basic functionality
        print(f"\n[TEST] Running basic functionality tests...")
        basic_results = test_autonomous_basic_functionality(test_env, ' '.join(full_python_cmd) if isinstance(full_python_cmd, list) else full_python_cmd)
        results["tests"].append(basic_results)
        
        if not basic_results["overall_success"]:
            print(f"[FAIL] Basic functionality tests failed in {env_name}")
            results["overall_success"] = False
        else:
            print(f"[PASS] Basic functionality tests passed in {env_name}")
        
        # Test 2: Autonomous detection
        if basic_results["overall_success"]:
            print(f"\n[TEST] Running autonomous detection tests...")
            detection_results = test_autonomous_detection(test_env, ' '.join(full_python_cmd) if isinstance(full_python_cmd, list) else full_python_cmd)
            results["tests"].append(detection_results)
            
            if not detection_results["overall_success"]:
                print(f"[FAIL] Autonomous detection tests failed in {env_name}")
                results["overall_success"] = False
            else:
                print(f"[PASS] Autonomous detection tests passed in {env_name}")
        else:
            print(f"[SKIP] Skipping autonomous detection tests due to basic functionality failures")
        
        results["end_time"] = time.time()
        results["total_duration"] = results["end_time"] - results["start_time"]
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Test environment setup failed: {e}")
        return {
            "environment": env_name,
            "error": str(e),
            "overall_success": False
        }
    
    finally:
        # Cleanup test environment
        if test_env and test_env.exists():
            try:
                shutil.rmtree(test_env)
                print(f"[INFO] Cleaned up test environment: {test_env}")
            except Exception as e:
                print(f"[WARN] Failed to cleanup test environment: {e}")

def generate_test_report(results: List[Dict[str, Any]]) -> None:
    """Generate a comprehensive test report."""
    print(f"\n{'='*60}")
    print("PYNOMALY AUTONOMOUS MODE TEST REPORT")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("overall_success", False))
    
    print(f"Total Environments Tested: {total_tests}")
    print(f"Successful Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    for result in results:
        env_name = result.get("environment", "Unknown")
        success = result.get("overall_success", False)
        status = "[PASS]" if success else "[FAIL]"
        
        print(f"\n{status} {env_name}")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue
        
        print(f"  Platform: {result.get('platform', 'Unknown')}")
        print(f"  Python: {result.get('python_cmd', 'Unknown')}")
        
        if "total_duration" in result:
            print(f"  Duration: {result['total_duration']:.2f}s")
        
        # Show test details
        for test_group in result.get("tests", []):
            test_name = test_group.get("test_name", "Unknown")
            test_success = test_group.get("overall_success", False)
            test_status = "[PASS]" if test_success else "[FAIL]"
            print(f"    {test_status} {test_name}")
            
            # Show individual test results
            for test in test_group.get("tests", []):
                name = test.get("name", "Unknown")
                success = test.get("success", False)
                duration = test.get("duration", 0)
                status = "✓" if success else "✗"
                print(f"      {status} {name} ({duration:.2f}s)")
                
                if not success and test.get("error"):
                    error_lines = test["error"].split('\n')[:3]  # Show first 3 lines
                    for line in error_lines:
                        if line.strip():
                            print(f"        ERROR: {line.strip()}")
    
    # Save detailed report to file
    report_file = Path("autonomous_mode_test_report.json")
    try:
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[INFO] Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n[WARN] Failed to save detailed report: {e}")

def main():
    """Main test function."""
    print("Pynomaly Autonomous Mode Cross-Platform Test Suite")
    print("This test suite validates autonomous anomaly detection functionality")
    print("across different Python environments and platforms.")
    
    all_results = []
    
    # Test in Bash environment (Linux/WSL)
    try:
        bash_results = run_environment_test(
            env_name="Bash (Linux/WSL)",
            python_cmd="python3"
        )
        all_results.append(bash_results)
    except Exception as e:
        print(f"[ERROR] Bash environment test failed: {e}")
        all_results.append({
            "environment": "Bash (Linux/WSL)",
            "error": str(e),
            "overall_success": False
        })
    
    # Test in PowerShell environment (Windows)
    try:
        powershell_results = run_environment_test(
            env_name="PowerShell (Windows)",
            python_cmd="python",
            shell_prefix=["powershell.exe", "-Command"]
        )
        all_results.append(powershell_results)
    except Exception as e:
        print(f"[ERROR] PowerShell environment test failed: {e}")
        all_results.append({
            "environment": "PowerShell (Windows)",
            "error": str(e),
            "overall_success": False
        })
    
    # Generate comprehensive report
    generate_test_report(all_results)
    
    # Return appropriate exit code
    success_count = sum(1 for r in all_results if r.get("overall_success", False))
    if success_count == len(all_results):
        print(f"\n[SUCCESS] All autonomous mode tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {success_count}/{len(all_results)} environments passed")
        return 1

if __name__ == "__main__":
    sys.exit(main())