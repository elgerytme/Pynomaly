#!/usr/bin/env python3
"""Test script for new dependency extras."""

import subprocess
import sys
import time
import venv
from pathlib import Path

def create_test_environment(env_name: str):
    """Create a test virtual environment."""
    env_path = Path(f"test_env_{env_name}")
    if env_path.exists():
        import shutil
        shutil.rmtree(env_path)
    
    venv.create(env_path, with_pip=True)
    return env_path

def run_in_env(env_path: Path, command: list):
    """Run a command in the virtual environment."""
    if sys.platform == "win32":
        python_path = env_path / "Scripts" / "python"
        pip_path = env_path / "Scripts" / "pip"
    else:
        python_path = env_path / "bin" / "python"
        pip_path = env_path / "bin" / "pip"
    
    if command[0] == "python":
        command[0] = str(python_path)
    elif command[0] == "pip":
        command[0] = str(pip_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result

def test_extras():
    """Test all extras installations."""
    extras_to_test = [
        "minimal",
        "automl", 
        "deep-cpu",
        "explainability",
        "automl-advanced"
    ]
    
    results = {}
    
    for extra in extras_to_test:
        print(f"\n🧪 Testing extra: {extra}")
        print("=" * 50)
        
        # Create test environment
        env_path = create_test_environment(extra)
        
        try:
            # Build package
            print("📦 Building package...")
            build_result = subprocess.run(["hatch", "build"], 
                                        capture_output=True, text=True)
            if build_result.returncode != 0:
                print(f"❌ Build failed: {build_result.stderr}")
                results[extra] = False
                continue
            
            # Install with extra
            print(f"⬇️ Installing pynomaly[{extra}]...")
            install_result = run_in_env(env_path, ["pip", "install", f"dist/*.whl[{extra}]"])
            
            if install_result.returncode != 0:
                print(f"❌ Installation failed: {install_result.stderr}")
                results[extra] = False
                continue
            
            # Test import
            print("🔍 Testing import...")
            import_result = run_in_env(env_path, [
                "python", "-c", 
                "import pynomaly; from pynomaly.utils.dependency_stubs import print_dependency_status; print_dependency_status()"
            ])
            
            if import_result.returncode == 0:
                print("✅ Success!")
                print(import_result.stdout)
                results[extra] = True
            else:
                print(f"❌ Import failed: {import_result.stderr}")
                results[extra] = False
        
        except Exception as e:
            print(f"❌ Error testing {extra}: {e}")
            results[extra] = False
        
        finally:
            # Cleanup
            import shutil
            if env_path.exists():
                shutil.rmtree(env_path)
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 50)
    for extra, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{extra:20} {status}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    test_extras()
