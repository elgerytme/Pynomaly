#!/usr/bin/env python3
"""Dependency validation script for Pynomaly package.

This script validates that all required dependencies are installed and compatible.
It can be run to diagnose installation issues and provide helpful guidance.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pkg_resources


class DependencyValidator:
    """Validates package dependencies and provides installation guidance."""
    
    def __init__(self):
        self.required_core = [
            ("pyod", "2.0.5"),
            ("numpy", "1.26.0"),
            ("pandas", "2.2.3"),
            ("polars", "1.19.0"),
            ("pydantic", "2.10.4"),
            ("structlog", "24.4.0"),
            ("dependency_injector", "4.42.0"),
            ("networkx", "3.0"),
            ("yaml", "6.0"),
            ("pydantic_settings", "2.8.0"),
            ("cryptography", "45.0.0"),
            ("email_validator", "2.2.0"),
        ]
        
        self.optional_groups = {
            "api": [
                ("fastapi", "0.115.0"),
                ("uvicorn", "0.34.0"),
                ("httpx", "0.28.1"),
                ("requests", "2.32.3"),
                ("multipart", "0.0.20"),
                ("jinja2", "3.1.5"),
                ("aiofiles", "24.1.0"),
                ("itsdangerous", "2.2.0"),
                ("jwt", "2.10.1"),
                ("passlib", "1.7.4"),
                ("sqlalchemy", "2.0.36"),
                ("prometheus_client", "0.21.1"),
                ("psutil", "6.1.1"),
            ],
            "cli": [
                ("typer", "0.15.1"),
                ("rich", "13.9.4"),
                ("shellingham", "1.3.0"),
                ("click", "8.0.0"),
            ],
            "ml": [
                ("sklearn", "1.6.0"),
                ("scipy", "1.15.0"),
            ]
        }
        
        self.issues = []
        self.successes = []
    
    def validate_import(self, module_name: str, min_version: Optional[str] = None) -> Tuple[bool, str]:
        """Validate that a module can be imported and meets version requirements."""
        try:
            module = importlib.import_module(module_name)
            
            if min_version:
                # Try to get version
                version = getattr(module, '__version__', None)
                if not version:
                    # Try alternative version attributes
                    version = getattr(module, 'version', None)
                    if hasattr(module, 'VERSION'):
                        version = getattr(module, 'VERSION')
                
                if version:
                    try:
                        if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                            return False, f"Version {version} < required {min_version}"
                    except Exception:
                        return True, f"Import successful, version check skipped: {version}"
                else:
                    return True, "Import successful, version unknown"
            
            return True, "Import successful"
            
        except ImportError as e:
            return False, f"Import failed: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def validate_core_dependencies(self) -> None:
        """Validate all core dependencies."""
        print("🔍 Validating core dependencies...")
        print("=" * 50)
        
        for module_name, min_version in self.required_core:
            success, message = self.validate_import(module_name, min_version)
            
            if success:
                print(f"✅ {module_name:<20} {message}")
                self.successes.append(f"{module_name}: {message}")
            else:
                print(f"❌ {module_name:<20} {message}")
                self.issues.append(f"{module_name}: {message}")
        
        print()
    
    def validate_optional_group(self, group_name: str) -> None:
        """Validate an optional dependency group."""
        if group_name not in self.optional_groups:
            print(f"❌ Unknown group: {group_name}")
            return
        
        print(f"🔍 Validating optional group: {group_name}")
        print("=" * 50)
        
        group_issues = []
        group_successes = []
        
        for module_name, min_version in self.optional_groups[group_name]:
            success, message = self.validate_import(module_name, min_version)
            
            if success:
                print(f"✅ {module_name:<20} {message}")
                group_successes.append(f"{module_name}: {message}")
            else:
                print(f"❌ {module_name:<20} {message}")
                group_issues.append(f"{module_name}: {message}")
        
        if group_issues:
            print(f"\n⚠️  {group_name} group has {len(group_issues)} issues:")
            for issue in group_issues:
                print(f"   - {issue}")
            print(f"\n💡 Install with: pip install pynomaly[{group_name}]")
        else:
            print(f"\n✅ {group_name} group fully functional!")
        
        print()
    
    def test_pynomaly_imports(self) -> None:
        """Test core Pynomaly package imports."""
        print("🔍 Testing Pynomaly package imports...")
        print("=" * 50)
        
        # Test main package
        success, message = self.validate_import("pynomaly")
        if success:
            print(f"✅ {'pynomaly':<20} {message}")
        else:
            print(f"❌ {'pynomaly':<20} {message}")
            self.issues.append(f"pynomaly: {message}")
            return
        
        # Test key submodules
        submodules = [
            "pynomaly.domain.entities",
            "pynomaly.domain.value_objects",
            "pynomaly.presentation.cli",
        ]
        
        for module in submodules:
            success, message = self.validate_import(module)
            if success:
                print(f"✅ {module:<30} Import successful")
            else:
                print(f"⚠️  {module:<30} {message}")
        
        print()
    
    def test_basic_functionality(self) -> None:
        """Test basic Pynomaly functionality."""
        print("🔍 Testing basic functionality...")
        print("=" * 50)
        
        try:
            # Test PyOD integration
            from pyod.models.iforest import IForest
            import numpy as np
            
            data = np.random.randn(100, 3)
            detector = IForest()
            detector.fit(data)
            scores = detector.decision_function(data)
            
            print(f"✅ {'PyOD integration':<20} IForest working with {len(scores)} samples")
            
        except Exception as e:
            print(f"❌ {'PyOD integration':<20} Error: {e}")
            self.issues.append(f"PyOD integration: {e}")
        
        try:
            # Test pandas
            import pandas as pd
            df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            print(f"✅ {'Pandas integration':<20} DataFrame operations working")
            
        except Exception as e:
            print(f"❌ {'Pandas integration':<20} Error: {e}")
            self.issues.append(f"Pandas integration: {e}")
        
        print()
    
    def generate_report(self) -> None:
        """Generate a comprehensive validation report."""
        print("📊 VALIDATION REPORT")
        print("=" * 50)
        
        total_checks = len(self.successes) + len(self.issues)
        success_rate = len(self.successes) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"Success Rate: {success_rate:.1f}% ({len(self.successes)}/{total_checks})")
        print(f"Issues Found: {len(self.issues)}")
        print()
        
        if self.issues:
            print("❌ ISSUES TO RESOLVE:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            print()
            
            print("💡 RECOMMENDED ACTIONS:")
            if any("Import failed" in issue for issue in self.issues):
                print("   1. Install missing dependencies:")
                print("      pip install pynomaly[server]  # For full functionality")
                print("      pip install pynomaly[api]     # For API only")
                print("      pip install pynomaly[cli]     # For CLI only")
            
            if any("Version" in issue for issue in self.issues):
                print("   2. Upgrade outdated packages:")
                print("      pip install --upgrade pynomaly")
            
            print("   3. Check your Python version (>=3.11 required)")
            print("   4. Consider using a fresh virtual environment")
        else:
            print("🎉 ALL CHECKS PASSED!")
            print("   Your Pynomaly installation is ready to use.")
        
        print()
    
    def run_validation(self, groups: Optional[List[str]] = None) -> bool:
        """Run complete validation process."""
        print("🚀 Pynomaly Dependency Validation")
        print("=" * 50)
        print(f"Python: {sys.version}")
        print(f"Platform: {sys.platform}")
        print()
        
        # Validate core dependencies
        self.validate_core_dependencies()
        
        # Validate optional groups
        if groups:
            for group in groups:
                self.validate_optional_group(group)
        
        # Test Pynomaly imports
        self.test_pynomaly_imports()
        
        # Test basic functionality
        self.test_basic_functionality()
        
        # Generate report
        self.generate_report()
        
        return len(self.issues) == 0


def main():
    """Main entry point for dependency validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Pynomaly dependencies and installation"
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=["api", "cli", "ml", "all"],
        help="Optional dependency groups to validate"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to install missing dependencies"
    )
    
    args = parser.parse_args()
    
    validator = DependencyValidator()
    
    groups = args.groups
    if groups and "all" in groups:
        groups = ["api", "cli", "ml"]
    
    success = validator.run_validation(groups)
    
    if not success and args.fix:
        print("🔧 Attempting to fix issues...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pynomaly[server]"
            ])
            print("✅ Fix attempt completed. Run validation again to verify.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fix attempt failed: {e}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()