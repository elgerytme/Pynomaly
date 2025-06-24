#!/usr/bin/env python3
"""
Working autonomous mode test that validates core functionality.
Tests basic autonomous detection capabilities with sample data.
"""

import os
import sys
import tempfile
import csv
import random
import json
import subprocess
import platform
from pathlib import Path

# Add source directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def create_test_data():
    """Create test dataset with known anomalies."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_autonomous_"))
    csv_file = temp_dir / "test_anomalies.csv"
    
    random.seed(42)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature1', 'feature2', 'feature3', 'feature4'])
        
        # Normal data (80% of samples)
        for _ in range(80):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1) 
            x3 = random.gauss(0, 1)
            x4 = random.gauss(0, 1)
            writer.writerow([x1, x2, x3, x4])
        
        # Anomalous data (20% of samples)
        for _ in range(20):
            x1 = random.uniform(-4, 4)
            x2 = random.uniform(-4, 4)
            x3 = random.uniform(-4, 4) 
            x4 = random.uniform(-4, 4)
            writer.writerow([x1, x2, x3, x4])
    
    return temp_dir, csv_file

def test_basic_functionality():
    """Test that basic components can be imported and used."""
    print("[TEST] Basic autonomous functionality...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        print("  ✓ pandas/numpy available")
        
        # Test pynomaly domain
        from pynomaly.domain.entities.dataset import Dataset
        from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
        print("  ✓ pynomaly domain components imported")
        
        # Test autonomous service
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
        print("  ✓ autonomous service imported")
        
        # Test data loaders
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        print("  ✓ data loaders imported")
        
        # Test repositories (with correct names)
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository  # Correct name
        )
        print("  ✓ repositories imported")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

def test_data_processing(csv_file):
    """Test basic data loading and processing."""
    print(f"[TEST] Data processing with {csv_file.name}...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate data
        if df.shape == (100, 4):
            print("  ✓ Data shape correct")
        else:
            print(f"  ✗ Expected (100, 4), got {df.shape}")
            return False
        
        # Check for numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 4:
            print("  ✓ All columns numeric")
        else:
            print(f"  ✗ Expected 4 numeric columns, got {len(numeric_cols)}")
            return False
        
        # Basic stats
        print(f"  ✓ Data range: [{df.min().min():.2f}, {df.max().max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data processing error: {e}")
        return False

def test_autonomous_service(csv_file):
    """Test autonomous service functionality."""
    print("[TEST] Autonomous service functionality...")
    
    try:
        import asyncio
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        async def test_service():
            # Setup
            data_loaders = {"csv": CSVLoader()}
            
            service = AutonomousDetectionService(
                detector_repository=InMemoryDetectorRepository(),
                result_repository=InMemoryResultRepository(),
                data_loaders=data_loaders
            )
            
            # Basic config
            config = AutonomousConfig(
                max_algorithms=1,
                auto_tune_hyperparams=False,
                verbose=True
            )
            
            # Test data loading
            dataset = await service._auto_load_data(str(csv_file), config)
            print(f"  ✓ Dataset loaded: {dataset.name}")
            print(f"  ✓ Shape: {dataset.data.shape}")
            
            # Test data profiling
            profile = await service._profile_data(dataset, config)
            print(f"  ✓ Data profiled: {profile.n_samples} samples, {profile.n_features} features")
            print(f"  ✓ Complexity: {profile.complexity_score:.3f}")
            
            # Test algorithm recommendation
            recommendations = await service._recommend_algorithms(profile, config)
            print(f"  ✓ Recommendations: {len(recommendations)} algorithms")
            
            if recommendations:
                top = recommendations[0]
                print(f"  ✓ Top algorithm: {top.algorithm} ({top.confidence:.1%})")
            
            return True
        
        # Run test
        result = asyncio.run(test_service())
        return result
        
    except Exception as e:
        print(f"  ✗ Autonomous service error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_cross_platform_test():
    """Run tests in both bash and PowerShell environments."""
    print("\n" + "="*60)
    print("CROSS-PLATFORM AUTONOMOUS MODE TEST")
    print("="*60)
    
    # Create test data
    temp_dir, csv_file = create_test_data()
    
    try:
        # Local tests first
        print(f"\n[LOCAL] Testing in current environment...")
        print(f"Python: {sys.version}")
        print(f"Platform: {platform.system()}")
        
        local_success = True
        
        if not test_basic_functionality():
            local_success = False
        
        if local_success and not test_data_processing(csv_file):
            local_success = False
        
        if local_success and not test_autonomous_service(csv_file):
            local_success = False
        
        if local_success:
            print("  ✓ All local tests passed!")
        else:
            print("  ✗ Local tests failed")
        
        # Now test different environments
        results = {"local": local_success}
        
        # Test bash environment
        print(f"\n[BASH] Testing with python3...")
        bash_result = test_external_environment("python3", csv_file)
        results["bash"] = bash_result
        
        # Test PowerShell environment  
        print(f"\n[POWERSHELL] Testing with python...")
        ps_result = test_external_environment("python", csv_file, use_powershell=True)
        results["powershell"] = ps_result
        
        # Summary
        print(f"\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for env, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} {env.upper()}")
        
        print(f"\nOverall: {passed}/{total} environments passed")
        
        if passed == total:
            print("🎉 All autonomous mode tests successful!")
            return True
        else:
            print("⚠️ Some tests failed")
            return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n[CLEANUP] Removed test directory: {temp_dir}")

def test_external_environment(python_cmd, csv_file, use_powershell=False):
    """Test autonomous mode in external environment."""
    
    # Create a minimal test script
    test_script = f'''
import sys
from pathlib import Path

# Add source path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    # Test basic imports
    import pandas as pd
    import numpy as np
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
    
    print("✓ Basic imports successful")
    
    # Test data loading
    df = pd.read_csv("{csv_file}")
    print(f"✓ Data loaded: {{df.shape}}")
    
    # Test configuration
    config = AutonomousConfig(max_algorithms=1, verbose=False)
    print("✓ Config created")
    
    print("✓ All tests passed")
    
except Exception as e:
    print(f"✗ Test failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    script_file = Path("temp_autonomous_test.py")
    
    try:
        with open(script_file, 'w') as f:
            f.write(test_script)
        
        # Choose command based on environment
        if use_powershell:
            cmd = ["powershell.exe", "-Command", python_cmd, str(script_file)]
        else:
            cmd = [python_cmd, str(script_file)]
        
        # Run test
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"  ✓ External test passed")
            print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ✗ External test failed")
            print(f"  Error: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timed out")
        return False
    except Exception as e:
        print(f"  ✗ Test error: {e}")
        return False
    finally:
        if script_file.exists():
            script_file.unlink()

def create_demo_report(csv_file):
    """Create a demonstration of autonomous mode capabilities."""
    print("\n" + "="*60)
    print("AUTONOMOUS MODE DEMONSTRATION")
    print("="*60)
    
    try:
        import asyncio
        import pandas as pd
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        async def demo():
            print("\n[DEMO] Setting up autonomous detection service...")
            
            # Setup service
            data_loaders = {"csv": CSVLoader()}
            service = AutonomousDetectionService(
                detector_repository=InMemoryDetectorRepository(),
                result_repository=InMemoryResultRepository(),
                data_loaders=data_loaders
            )
            
            # Demo different configurations
            configs = {
                "Quick": AutonomousConfig(
                    max_algorithms=1,
                    auto_tune_hyperparams=False,
                    verbose=True
                ),
                "Comprehensive": AutonomousConfig(
                    max_algorithms=3,
                    auto_tune_hyperparams=True,
                    confidence_threshold=0.8,
                    verbose=True
                )
            }
            
            for config_name, config in configs.items():
                print(f"\n[DEMO] {config_name} Mode:")
                
                # Load and profile data
                dataset = await service._auto_load_data(str(csv_file), config)
                profile = await service._profile_data(dataset, config)
                recommendations = await service._recommend_algorithms(profile, config)
                
                print(f"  📊 Dataset: {dataset.name}")
                print(f"  📈 Shape: {dataset.data.shape}")
                print(f"  🧮 Complexity: {profile.complexity_score:.3f}")
                print(f"  🎯 Contamination: {profile.recommended_contamination:.1%}")
                print(f"  🤖 Algorithms: {len(recommendations)}")
                
                if recommendations:
                    for i, rec in enumerate(recommendations[:3], 1):
                        print(f"    {i}. {rec.algorithm} ({rec.confidence:.1%})")
        
        asyncio.run(demo())
        
        # Show CLI usage examples
        print(f"\n[DEMO] CLI Usage Examples:")
        print(f"  # Quick detection:")
        print(f"  pynomaly auto quick {csv_file}")
        print(f"  ")
        print(f"  # Comprehensive detection:")
        print(f"  pynomaly auto detect {csv_file} --output results.csv")
        print(f"  ")
        print(f"  # Data profiling:")
        print(f"  pynomaly auto profile {csv_file} --verbose")
        
        return True
        
    except Exception as e:
        print(f"[DEMO] Failed: {e}")
        return False

def main():
    """Main test function."""
    print("Pynomaly Autonomous Mode Test Suite")
    print("Testing autonomous anomaly detection functionality")
    
    # Create test data
    temp_dir, csv_file = create_test_data()
    
    try:
        # Run comprehensive tests
        success = run_cross_platform_test()
        
        # Run demonstration if tests passed
        if success:
            create_demo_report(csv_file)
        
        return 0 if success else 1
        
    finally:
        # Final cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())