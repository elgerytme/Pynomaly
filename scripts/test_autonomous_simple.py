#!/usr/bin/env python3
"""
Simplified autonomous mode test that works without full installation.
Tests core autonomous functionality with sample data.
"""

import os
import sys
import tempfile
import json
import time
import csv
import random
import subprocess
from pathlib import Path

# Add source directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def create_sample_data():
    """Create simple sample data for testing."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
    
    # Create CSV with normal and anomalous data
    csv_file = temp_dir / "test_data.csv"
    
    # Set seed for reproducible results
    random.seed(42)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'x3', 'x4'])
        
        # Normal data (first 80 rows)
        for _ in range(80):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            x3 = random.gauss(0, 1)
            x4 = random.gauss(0, 1)
            writer.writerow([x1, x2, x3, x4])
        
        # Anomalous data (last 20 rows)
        for _ in range(20):
            x1 = random.uniform(-3, 3)
            x2 = random.uniform(-3, 3)
            x3 = random.uniform(-3, 3)
            x4 = random.uniform(-3, 3)
            writer.writerow([x1, x2, x3, x4])
    
    return temp_dir, csv_file

def test_basic_imports():
    """Test if basic components can be imported."""
    print("[TEST] Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ pandas and numpy available")
        
        # Test if we can import pynomaly components
        from pynomaly.domain.entities.dataset import Dataset
        print("✓ pynomaly.domain.entities.dataset imported")
        
        from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
        print("✓ pynomaly.domain.value_objects imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_data_loading(csv_file):
    """Test basic data loading functionality."""
    print(f"[TEST] Testing data loading from {csv_file}...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Basic validation
        if df.shape[0] == 100 and df.shape[1] == 4:
            print("✓ Data shape is correct")
        else:
            print(f"✗ Unexpected data shape: {df.shape}")
            return False
        
        # Check for numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 4:
            print("✓ All columns are numeric")
        else:
            print(f"✗ Expected 4 numeric columns, got {len(numeric_cols)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_autonomous_components():
    """Test autonomous service components."""
    print("[TEST] Testing autonomous service components...")
    
    try:
        # Test autonomous service import
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
        print("✓ Autonomous service imported")
        
        # Test data loaders
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        print("✓ CSV loader imported")
        
        # Test repositories
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryDetectionResultRepository
        )
        print("✓ In-memory repositories imported")
        
        # Create basic config
        config = AutonomousConfig(
            max_algorithms=2,
            auto_tune_hyperparams=False,
            verbose=True
        )
        print("✓ Autonomous config created")
        
        return True
        
    except ImportError as e:
        print(f"✗ Component import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

def test_basic_detection_pipeline(csv_file):
    """Test basic detection pipeline functionality."""
    print(f"[TEST] Testing basic detection pipeline...")
    
    try:
        import asyncio
        import pandas as pd
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryDetectionResultRepository
        )
        
        async def run_detection():
            # Setup service
            data_loaders = {"csv": CSVLoader()}
            
            autonomous_service = AutonomousDetectionService(
                detector_repository=InMemoryDetectorRepository(),
                result_repository=InMemoryDetectionResultRepository(),
                data_loaders=data_loaders
            )
            
            # Simple config
            config = AutonomousConfig(
                max_algorithms=1,  # Just one algorithm for testing
                auto_tune_hyperparams=False,
                verbose=True
            )
            
            try:
                # Test data loading
                dataset = await autonomous_service._auto_load_data(str(csv_file), config)
                print(f"✓ Data loaded: {dataset.name}, shape: {dataset.data.shape}")
                
                # Test data profiling
                profile = await autonomous_service._profile_data(dataset, config)
                print(f"✓ Data profiled: {profile.n_samples} samples, {profile.n_features} features")
                
                # Test algorithm recommendation
                recommendations = await autonomous_service._recommend_algorithms(profile, config)
                print(f"✓ Algorithm recommendations: {len(recommendations)} algorithms")
                
                if recommendations:
                    top_rec = recommendations[0]
                    print(f"  Top recommendation: {top_rec.algorithm} (confidence: {top_rec.confidence:.1%})")
                
                return True
                
            except Exception as e:
                print(f"✗ Detection pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Run async function
        result = asyncio.run(run_detection())
        return result
        
    except Exception as e:
        print(f"✗ Detection pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment(env_name, python_cmd):
    """Test autonomous mode in a specific environment."""
    print(f"\n{'='*50}")
    print(f"Testing Autonomous Mode in {env_name}")
    print(f"Python: {python_cmd}")
    print(f"{'='*50}")
    
    results = {
        "environment": env_name,
        "python_cmd": python_cmd,
        "tests": [],
        "success": True
    }
    
    # Create test script for the environment
    test_script = f'''
import sys
from pathlib import Path

# Add source to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import test functions from this module
sys.path.insert(0, str(script_dir))

try:
    from test_autonomous_simple import test_basic_imports, test_data_loading, test_autonomous_components
    
    # Run tests
    print("Running autonomous mode tests...")
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("Basic imports failed")
        sys.exit(1)
    
    # Test 2: Create sample data and test loading
    import tempfile
    import csv
    import random
    from pathlib import Path
    
    temp_dir = Path(tempfile.mkdtemp(prefix="test_"))
    csv_file = temp_dir / "data.csv"
    
    random.seed(42)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'x3', 'x4'])
        for _ in range(100):
            writer.writerow([random.gauss(0, 1) for _ in range(4)])
    
    if not test_data_loading(csv_file):
        print("Data loading failed")
        sys.exit(1)
    
    # Test 3: Autonomous components
    if not test_autonomous_components():
        print("Autonomous components failed")
        sys.exit(1)
    
    print("All tests passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"Test failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    script_file = Path("temp_test_script.py")
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        # Run the test script
        if env_name == "PowerShell":
            cmd = ["powershell.exe", "-Command", python_cmd, str(script_file)]
        else:
            cmd = [python_cmd, str(script_file)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✓ {env_name} tests passed")
            results["output"] = result.stdout
        else:
            print(f"✗ {env_name} tests failed")
            results["success"] = False
            results["error"] = result.stderr
            results["output"] = result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"✗ {env_name} tests timed out")
        results["success"] = False
        results["error"] = "Test timed out"
    except Exception as e:
        print(f"✗ {env_name} tests failed with exception: {e}")
        results["success"] = False
        results["error"] = str(e)
    finally:
        # Cleanup
        if script_file.exists():
            script_file.unlink()
    
    return results

def main():
    """Main test function."""
    print("Pynomaly Autonomous Mode Test (Simplified)")
    print("Testing core autonomous functionality...")
    
    # First, run local tests
    print(f"\n{'='*50}")
    print("Testing Local Environment")
    print(f"{'='*50}")
    
    temp_dir, csv_file = create_sample_data()
    
    try:
        success = True
        
        # Run all tests locally first
        if not test_basic_imports():
            success = False
        
        if success and not test_data_loading(csv_file):
            success = False
        
        if success and not test_autonomous_components():
            success = False
        
        if success and not test_basic_detection_pipeline(csv_file):
            success = False
        
        if success:
            print("✓ All local tests passed!")
        else:
            print("✗ Some local tests failed")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    # Now test different environments
    results = []
    
    # Test Bash
    bash_result = test_environment("Bash", "python3")
    results.append(bash_result)
    
    # Test PowerShell (if available)
    try:
        ps_result = test_environment("PowerShell", "python")
        results.append(ps_result)
    except Exception as e:
        print(f"PowerShell test skipped: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Environments tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['environment']}")
        if not result["success"] and "error" in result:
            print(f"  Error: {result['error']}")
    
    if success and passed == total:
        print("\n✓ All autonomous mode tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())