#!/usr/bin/env python3
"""
Health check script for Pynomaly Detection container.
Returns 0 for healthy, 1 for unhealthy.
"""

import sys
import time
import requests
import subprocess
from pathlib import Path

def check_basic_import():
    """Check if basic package import works."""
    try:
        import pynomaly_detection
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_phase2_availability():
    """Check if Phase 2 components are available."""
    try:
        from pynomaly_detection import check_phase2_availability
        availability = check_phase2_availability()
        all_available = all(availability.values())
        
        if not all_available:
            missing = [k for k, v in availability.items() if not v]
            print(f"❌ Missing Phase 2 components: {missing}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Phase 2 check failed: {e}")
        return False

def check_memory_usage():
    """Check if memory usage is within limits."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        
        # Get memory limit from environment
        import os
        memory_limit_gb = float(os.getenv('PYNOMALY_MEMORY_LIMIT_GB', '4'))
        
        if used_gb > memory_limit_gb * 0.9:  # 90% of limit
            print(f"❌ High memory usage: {used_gb:.1f}GB (limit: {memory_limit_gb}GB)")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Memory check failed: {e}")
        return True  # Don't fail health check for memory monitoring issues

def check_api_server():
    """Check if API server is responding."""
    try:
        import os
        port = os.getenv('PYNOMALY_PORT', '8080')
        
        # Try to connect to health endpoint
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ API server unhealthy: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        # API server might not be running (worker mode, etc.)
        return True
    except Exception as e:
        print(f"⚠️  API check failed: {e}")
        return True

def check_disk_space():
    """Check if sufficient disk space is available."""
    try:
        import shutil
        
        # Check data directory
        data_dir = Path('/app/data')
        if data_dir.exists():
            total, used, free = shutil.disk_usage(data_dir)
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                print(f"❌ Low disk space: {free_gb:.1f}GB free")
                return False
        
        return True
    except Exception as e:
        print(f"⚠️  Disk space check failed: {e}")
        return True

def check_critical_files():
    """Check if critical files are present."""
    critical_files = [
        '/app/src/pynomaly_detection/__init__.py',
        '/app/entrypoint.sh'
    ]
    
    for file_path in critical_files:
        if not Path(file_path).exists():
            print(f"❌ Missing critical file: {file_path}")
            return False
    
    return True

def run_quick_detection_test():
    """Run a quick detection test to verify functionality."""
    try:
        import numpy as np
        from pynomaly_detection import AnomalyDetector
        
        # Generate small test data
        np.random.seed(42)
        test_data = np.random.randn(100, 5)
        
        # Quick detection test
        detector = AnomalyDetector()
        
        start_time = time.time()
        predictions = detector.detect(test_data, contamination=0.1)
        end_time = time.time()
        
        # Verify results
        if not hasattr(predictions, '__len__') or len(predictions) != 100:
            print(f"❌ Detection test failed: invalid predictions shape")
            return False
        
        detection_time = end_time - start_time
        if detection_time > 5.0:  # Should be very fast for small data
            print(f"❌ Detection too slow: {detection_time:.2f}s")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    """Main health check function."""
    print("🏥 Pynomaly Detection Health Check")
    print("=" * 40)
    
    checks = [
        ("Basic Import", check_basic_import),
        ("Phase 2 Components", check_phase2_availability),
        ("Memory Usage", check_memory_usage),
        ("Disk Space", check_disk_space),
        ("Critical Files", check_critical_files),
        ("API Server", check_api_server),
        ("Detection Test", run_quick_detection_test)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"Checking {check_name}...", end=" ")
        
        try:
            if check_func():
                print("✅ PASS")
            else:
                print("❌ FAIL")
                all_passed = False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    print("=" * 40)
    
    if all_passed:
        print("🎉 All health checks passed!")
        return 0
    else:
        print("💥 Some health checks failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)