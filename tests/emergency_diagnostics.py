#!/usr/bin/env python3
"""
Emergency Diagnostics Script
Phase 1: Environment Validation and Root Cause Analysis
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
import importlib
import traceback

class EmergencyDiagnostics:
    """Comprehensive system diagnostics to identify root causes"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": {},
            "poetry": {},
            "imports": {},
            "entry_points": {},
            "dependencies": {},
            "recommendations": []
        }
        self.project_root = Path(__file__).parent.parent
    
    def run_all_diagnostics(self):
        """Execute all diagnostic phases"""
        print("🚨 EMERGENCY DIAGNOSTICS - Phase 1")
        print("=" * 50)
        
        self.check_environment()
        self.check_poetry_setup()
        self.check_imports()
        self.check_entry_points()
        self.check_dependencies()
        self.analyze_results()
        
        return self.results
    
    def check_environment(self):
        """Validate basic environment setup"""
        print("\n🔍 Environment Validation...")
        
        env_info = {}
        
        # Python version
        env_info["python_version"] = sys.version
        env_info["python_executable"] = sys.executable
        env_info["python_path"] = sys.path
        
        # Working directory
        env_info["working_directory"] = str(self.project_root)
        env_info["directory_exists"] = self.project_root.exists()
        
        # Virtual environment detection
        env_info["virtual_env"] = sys.prefix != sys.base_prefix
        env_info["prefix"] = sys.prefix
        env_info["base_prefix"] = sys.base_prefix
        
        self.results["environment"] = env_info
        print(f"✅ Python: {sys.version.split()[0]}")
        print(f"✅ Virtual Env: {env_info['virtual_env']}")
        print(f"✅ Working Dir: {env_info['working_directory']}")
    
    def check_poetry_setup(self):
        """Validate Poetry configuration and installation"""
        print("\n🔍 Poetry Setup Validation...")
        
        poetry_info = {}
        
        try:
            # Poetry version
            result = subprocess.run(
                ["poetry", "--version"], 
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            poetry_info["version_check"] = {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }
            
            # Poetry env info
            result = subprocess.run(
                ["poetry", "env", "info"], 
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            poetry_info["env_info"] = {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }
            
            # Poetry show (installed packages)
            result = subprocess.run(
                ["poetry", "show"], 
                capture_output=True, text=True, timeout=60,
                cwd=self.project_root
            )
            poetry_info["packages"] = {
                "success": result.returncode == 0,
                "package_count": len(result.stdout.strip().split('\n')) if result.returncode == 0 else 0,
                "error": result.stderr.strip()
            }
            
            # Check if pynomaly is installed
            result = subprocess.run(
                ["poetry", "show", "pynomaly"], 
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            poetry_info["pynomaly_installed"] = {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }
            
        except Exception as e:
            poetry_info["error"] = str(e)
        
        self.results["poetry"] = poetry_info
        
        if poetry_info.get("version_check", {}).get("success"):
            print(f"✅ Poetry: {poetry_info['version_check']['output']}")
        else:
            print("❌ Poetry: Not accessible or failed")
        
        if poetry_info.get("packages", {}).get("success"):
            print(f"✅ Packages: {poetry_info['packages']['package_count']} installed")
        else:
            print("❌ Packages: Cannot list installed packages")
    
    def check_imports(self):
        """Test critical module imports"""
        print("\n🔍 Module Import Testing...")
        
        import_tests = [
            "pynomaly",
            "pynomaly.presentation",
            "pynomaly.presentation.cli",
            "pynomaly.presentation.api",
            "pynomaly.infrastructure.config.container",
            "pynomaly.domain.entities",
            "pynomaly.application.services"
        ]
        
        import_results = {}
        
        for module in import_tests:
            try:
                importlib.import_module(module)
                import_results[module] = {"success": True, "error": None}
                print(f"✅ {module}")
            except Exception as e:
                import_results[module] = {"success": False, "error": str(e)}
                print(f"❌ {module}: {str(e)}")
        
        self.results["imports"] = import_results
    
    def check_entry_points(self):
        """Validate CLI entry point registration"""
        print("\n🔍 Entry Point Validation...")
        
        entry_point_info = {}
        
        # Check pyproject.toml for console scripts
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path) as f:
                    content = f.read()
                    entry_point_info["pyproject_exists"] = True
                    entry_point_info["has_console_scripts"] = "console_scripts" in content
                    entry_point_info["has_pynomaly_entry"] = "pynomaly" in content
            except Exception as e:
                entry_point_info["pyproject_error"] = str(e)
        else:
            entry_point_info["pyproject_exists"] = False
        
        # Test CLI command directly
        try:
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "from pynomaly.presentation.cli import main; main()"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            entry_point_info["direct_cli_test"] = {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip()
            }
        except Exception as e:
            entry_point_info["direct_cli_test"] = {"success": False, "error": str(e)}
        
        # Test pynomaly command
        try:
            result = subprocess.run(
                ["poetry", "run", "pynomaly", "--help"],
                capture_output=True, text=True, timeout=30,
                cwd=self.project_root
            )
            entry_point_info["pynomaly_command"] = {
                "success": result.returncode == 0,
                "output_length": len(result.stdout),
                "error": result.stderr.strip()
            }
        except Exception as e:
            entry_point_info["pynomaly_command"] = {"success": False, "error": str(e)}
        
        self.results["entry_points"] = entry_point_info
        
        if entry_point_info.get("pyproject_exists"):
            print("✅ pyproject.toml exists")
        else:
            print("❌ pyproject.toml missing")
        
        if entry_point_info.get("pynomaly_command", {}).get("success"):
            print("✅ pynomaly command accessible")
        else:
            print("❌ pynomaly command failed")
    
    def check_dependencies(self):
        """Check critical dependencies"""
        print("\n🔍 Dependency Validation...")
        
        critical_deps = [
            "fastapi", "uvicorn", "click", "typer", 
            "pandas", "numpy", "scikit-learn",
            "pydantic", "dependency-injector"
        ]
        
        dep_results = {}
        
        for dep in critical_deps:
            try:
                importlib.import_module(dep.replace("-", "_"))
                dep_results[dep] = {"success": True, "error": None}
                print(f"✅ {dep}")
            except Exception as e:
                dep_results[dep] = {"success": False, "error": str(e)}
                print(f"❌ {dep}: {str(e)}")
        
        self.results["dependencies"] = dep_results
    
    def analyze_results(self):
        """Analyze results and provide recommendations"""
        print("\n🔍 Analysis and Recommendations...")
        
        recommendations = []
        
        # Check Poetry issues
        if not self.results["poetry"].get("version_check", {}).get("success"):
            recommendations.append("CRITICAL: Poetry not accessible - reinstall Poetry")
        
        if not self.results["poetry"].get("packages", {}).get("success"):
            recommendations.append("CRITICAL: Cannot list packages - run 'poetry install'")
        
        # Check import issues
        failed_imports = [
            module for module, result in self.results["imports"].items() 
            if not result["success"]
        ]
        if failed_imports:
            recommendations.append(f"CRITICAL: Module imports failing - {len(failed_imports)} modules")
        
        # Check entry point issues
        if not self.results["entry_points"].get("pynomaly_command", {}).get("success"):
            recommendations.append("CRITICAL: CLI entry point broken - fix console_scripts in pyproject.toml")
        
        # Check dependency issues
        failed_deps = [
            dep for dep, result in self.results["dependencies"].items() 
            if not result["success"]
        ]
        if failed_deps:
            recommendations.append(f"HIGH: Missing dependencies - {len(failed_deps)} packages")
        
        self.results["recommendations"] = recommendations
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def save_results(self, filename="emergency_diagnostics.json"):
        """Save diagnostic results"""
        results_file = self.project_root / "tests" / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Diagnostic results saved: {results_file}")
        return results_file

def main():
    """Main diagnostic execution"""
    diagnostics = EmergencyDiagnostics()
    results = diagnostics.run_all_diagnostics()
    results_file = diagnostics.save_results()
    
    # Print summary
    print("\n" + "=" * 50)
    print("🚨 EMERGENCY DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    issues_found = len(results["recommendations"])
    
    if issues_found == 0:
        print("✅ No critical issues detected")
        return True
    else:
        print(f"❌ {issues_found} critical issues detected")
        print("\nIMMEDIATE ACTIONS REQUIRED:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📁 Full diagnostic report: {results_file}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)