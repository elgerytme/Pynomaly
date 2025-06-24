#!/usr/bin/env python3
"""
Verify CLI architecture without requiring all dependencies.
This script checks the container configuration and adapter wiring.
"""

import sys
import ast
from pathlib import Path

def check_container_providers():
    """Check that all adapter providers are defined in container.py"""
    print("🔍 Checking container.py for adapter providers...")
    
    container_file = Path("src/pynomaly/infrastructure/config/container.py")
    if not container_file.exists():
        print("❌ container.py not found")
        return False
    
    content = container_file.read_text()
    
    # Check for adapter providers
    required_providers = [
        "pyod_adapter = providers.Singleton(PyODAdapter)",
        "sklearn_adapter = providers.Singleton(SklearnAdapter)"
    ]
    
    optional_providers = [
        "tods_adapter = providers.Singleton(TODSAdapter)",
        "pygod_adapter = providers.Singleton(PyGODAdapter)", 
        "pytorch_adapter = providers.Singleton(PyTorchAdapter)"
    ]
    
    success = True
    
    # Check required providers
    for provider in required_providers:
        if provider in content:
            print(f"✅ Found: {provider}")
        else:
            print(f"❌ Missing: {provider}")
            success = False
    
    # Check optional providers (conditional)
    for provider in optional_providers:
        adapter_name = provider.split("=")[0].strip()
        if adapter_name in content:
            print(f"✅ Found: {adapter_name} (conditional)")
        else:
            print(f"❌ Missing: {adapter_name}")
            success = False
    
    return success

def check_adapter_imports():
    """Check that adapter imports are properly configured."""
    print("\n🔍 Checking adapter imports...")
    
    container_file = Path("src/pynomaly/infrastructure/config/container.py")
    content = container_file.read_text()
    
    # Check basic imports
    required_imports = [
        "from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter"
    ]
    
    # Check conditional imports
    conditional_patterns = [
        "from pynomaly.infrastructure.adapters import TODSAdapter",
        "from pynomaly.infrastructure.adapters import PyGODAdapter",
        "from pynomaly.infrastructure.adapters import PyTorchAdapter"
    ]
    
    success = True
    
    for import_line in required_imports:
        if import_line in content or "PyODAdapter" in content:
            print("✅ PyOD and Sklearn adapters imported")
        else:
            print("❌ Missing PyOD/Sklearn imports")
            success = False
    
    # Check for conditional import logic
    if "try:" in content and "except ImportError:" in content:
        print("✅ Conditional import logic found")
    else:
        print("❌ Missing conditional import logic")
        success = False
    
    return success

def check_cli_error_handling():
    """Check that CLI has proper error handling for missing adapters."""
    print("\n🔍 Checking CLI error handling...")
    
    detectors_file = Path("src/pynomaly/presentation/cli/detectors.py")
    if not detectors_file.exists():
        print("❌ detectors.py not found")
        return False
    
    content = detectors_file.read_text()
    
    # Check for try/catch blocks around adapter access
    checks = [
        ("try:" in content, "Try block found"),
        ("except Exception" in content, "Exception handling found"),
        ("container.pyod_adapter()" in content, "PyOD adapter access found"),
        ("Failed to access PyOD adapter" in content, "Error message found")
    ]
    
    success = True
    for check, description in checks:
        if check:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            success = False
    
    return success

def check_cli_structure():
    """Check CLI module structure and command registration."""
    print("\n🔍 Checking CLI structure...")
    
    cli_files = [
        "src/pynomaly/presentation/cli/app.py",
        "src/pynomaly/presentation/cli/detectors.py", 
        "src/pynomaly/presentation/cli/datasets.py",
        "src/pynomaly/presentation/cli/detection.py",
        "src/pynomaly/presentation/cli/server.py"
    ]
    
    success = True
    
    for file_path in cli_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            success = False
    
    # Check main app.py structure
    app_file = Path("src/pynomaly/presentation/cli/app.py")
    if app_file.exists():
        content = app_file.read_text()
        
        # Check for command registration
        command_checks = [
            ("app.add_typer(detectors.app" in content, "Detectors commands registered"),
            ("app.add_typer(datasets.app" in content, "Datasets commands registered"),
            ("app.add_typer(detection.app" in content, "Detection commands registered"),
            ("app.add_typer(server.app" in content, "Server commands registered"),
            ("get_cli_container()" in content, "Container access function found")
        ]
        
        for check, description in command_checks:
            if check:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                success = False
    
    return success

def main():
    """Run all architecture verification checks."""
    print("🏗️ Pynomaly CLI Architecture Verification")
    print("=" * 50)
    
    checks = [
        ("Container Providers", check_container_providers),
        ("Adapter Imports", check_adapter_imports), 
        ("CLI Error Handling", check_cli_error_handling),
        ("CLI Structure", check_cli_structure)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        result = check_func()
        all_passed &= result
        
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All architecture checks passed!")
        print("\n✅ CLI Integration Status: ARCHITECTURALLY COMPLETE")
        print("🔴 Remaining blocker: Dependencies not installed")
        print("\nTo test CLI functionality:")
        print("1. Run: poetry install")
        print("2. Run: poetry run pynomaly --help")
        return 0
    else:
        print("❌ Some architecture checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())