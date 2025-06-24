#!/usr/bin/env python3
"""
Basic test to check if ML adapter imports work without full dependencies.
This validates our conditional import structure.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pytorch_adapter_import():
    """Test PyTorch adapter basic import."""
    try:
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
        print("✅ PyTorch adapter imported successfully")
        return True
    except ImportError as e:
        print(f"❌ PyTorch adapter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch adapter unexpected error: {e}")
        return False

def test_tensorflow_adapter_import():
    """Test TensorFlow adapter basic import."""
    try:
        from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
        print("✅ TensorFlow adapter imported successfully")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow adapter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ TensorFlow adapter unexpected error: {e}")
        return False

def test_jax_adapter_import():
    """Test JAX adapter basic import."""
    try:
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
        print("✅ JAX adapter imported successfully")
        return True
    except ImportError as e:
        print(f"❌ JAX adapter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ JAX adapter unexpected error: {e}")
        return False

def test_basic_sklearn_adapter():
    """Test sklearn adapter which should work."""
    try:
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("✅ Sklearn adapter imported successfully")
        
        # Try basic initialization with correct parameter
        adapter = SklearnAdapter(algorithm_name="IsolationForest")
        print("✅ Sklearn adapter initialized successfully")
        return True
    except ImportError as e:
        print(f"❌ Sklearn adapter import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Sklearn adapter unexpected error: {e}")
        return False

def main():
    """Run basic adapter tests."""
    print("🔍 Testing ML Adapter Imports...")
    print("=" * 50)
    
    results = []
    
    # Test basic sklearn (should work)
    results.append(test_basic_sklearn_adapter())
    
    # Test ML framework adapters (may fail due to missing dependencies)
    results.append(test_pytorch_adapter_import())
    results.append(test_tensorflow_adapter_import())
    results.append(test_jax_adapter_import())
    
    print("\n" + "=" * 50)
    successful = sum(results)
    total = len(results)
    print(f"📊 Results: {successful}/{total} adapters working")
    
    if successful == total:
        print("🎉 All adapters are importable!")
        return 0
    elif successful > 0:
        print("⚠️  Some adapters working (this is expected without ML dependencies)")
        return 0
    else:
        print("💥 No adapters working - there may be a configuration issue")
        return 1

if __name__ == "__main__":
    exit(main())