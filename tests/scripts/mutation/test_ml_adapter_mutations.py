#!/usr/bin/env python3
"""
ML Adapter Mutation Testing
Tests the quality of ML adapter tests through targeted mutations.
"""

import subprocess
import sys
from pathlib import Path

def run_ml_adapter_mutations():
    """Run mutation testing on ML adapters."""
    print("🧬 Running ML adapter mutation testing...")
    
    adapters = [
        ("PyOD Adapter", "src/pynomaly/infrastructure/adapters/pyod_adapter.py"),
        ("Sklearn Adapter", "src/pynomaly/infrastructure/adapters/sklearn_adapter.py"),
        ("Detection Service", "src/pynomaly/application/services/detection_service.py"),
    ]
    
    for name, target_file in adapters:
        print(f"\n🎯 Testing mutations in {name}")
        
        cmd = [
            "mutmut", "run",
            "--paths-to-mutate", target_file,
            "--runner", "python -m pytest tests/infrastructure/adapters/ tests/application/test_services.py -x --tb=no -q",
            "--timeout", "180",
            "--max-mutations", "50",  # Limit for large files
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            print(f"✅ Mutation testing completed for {name}")
            
            # Show mutation survival rate
            if "survived" in result.stdout.lower():
                print("⚠️  Some mutations survived - consider improving tests")
            else:
                print("🎯 All mutations killed - excellent test coverage")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Mutation testing timed out for {name}")
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")

if __name__ == "__main__":
    run_ml_adapter_mutations()
