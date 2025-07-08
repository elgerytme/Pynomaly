#!/usr/bin/env python3
"""
Validate Hatch environment configuration for CI/CD pipeline.

This script checks that all required Hatch environments are properly configured
and that the necessary scripts are available for the enhanced CI/CD pipeline.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def check_hatch_installed() -> bool:
    """Check if Hatch is installed."""
    print("🔍 Checking Hatch installation...")
    exit_code, stdout, stderr = run_command(["hatch", "--version"])
    
    if exit_code == 0:
        print(f"✅ Hatch is installed: {stdout.strip()}")
        return True
    else:
        print(f"❌ Hatch is not installed or not in PATH: {stderr}")
        return False


def check_project_config() -> bool:
    """Check if pyproject.toml exists."""
    print("🔍 Checking project configuration...")
    pyproject_path = Path("pyproject.toml")
    
    if pyproject_path.exists():
        print("✅ pyproject.toml found")
        return True
    else:
        print("❌ pyproject.toml not found")
        return False


def check_hatch_environments() -> bool:
    """Check if required Hatch environments are configured."""
    print("🔍 Checking Hatch environments...")
    
    required_envs = ["lint", "test", "docs", "prod"]
    exit_code, stdout, stderr = run_command(["hatch", "env", "show"])
    
    if exit_code != 0:
        print(f"❌ Failed to get Hatch environments: {stderr}")
        return False
    
    print("Available environments:")
    print(stdout)
    
    missing_envs = []
    for env in required_envs:
        if env not in stdout:
            missing_envs.append(env)
    
    if missing_envs:
        print(f"❌ Missing required environments: {missing_envs}")
        return False
    else:
        print("✅ All required environments are configured")
        return True


def check_environment_scripts() -> bool:
    """Check if required scripts are configured in environments."""
    print("🔍 Checking environment scripts...")
    
    required_scripts = {
        "lint": ["all"],
        "test": ["run-cov"],
        "docs": ["build"],
        "prod": ["serve-api"]
    }
    
    all_scripts_ok = True
    
    for env, scripts in required_scripts.items():
        print(f"  Checking {env} environment...")
        
        for script in scripts:
            exit_code, stdout, stderr = run_command([
                "hatch", "run", f"{env}:{script}", "--help"
            ])
            
            if exit_code == 0:
                print(f"    ✅ {env}:{script} is available")
            else:
                print(f"    ❌ {env}:{script} is not available or has issues")
                all_scripts_ok = False
    
    return all_scripts_ok


def check_dependencies() -> bool:
    """Check if environment dependencies are installed."""
    print("🔍 Checking environment dependencies...")
    
    environments = ["lint", "test", "docs"]
    all_deps_ok = True
    
    for env in environments:
        print(f"  Checking {env} environment dependencies...")
        exit_code, stdout, stderr = run_command([
            "hatch", "run", f"{env}:python", "-m", "pip", "list"
        ])
        
        if exit_code == 0:
            print(f"    ✅ {env} environment dependencies are installed")
        else:
            print(f"    ❌ {env} environment has dependency issues: {stderr}")
            all_deps_ok = False
    
    return all_deps_ok


def check_api_startup() -> bool:
    """Test if the API can start in the prod environment."""
    print("🔍 Testing API startup (prod environment)...")
    
    # This is a quick test - we'll try to import the app
    exit_code, stdout, stderr = run_command([
        "hatch", "run", "prod:python", "-c", 
        "from pynomaly.presentation.api.app import app; print('API app can be imported')"
    ])
    
    if exit_code == 0:
        print("✅ API app can be imported in prod environment")
        return True
    else:
        print(f"❌ API app cannot be imported in prod environment: {stderr}")
        return False


def generate_report(results: Dict[str, bool]) -> None:
    """Generate a validation report."""
    print("\n" + "="*50)
    print("🚀 HATCH ENVIRONMENT VALIDATION REPORT")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Overall Status: {passed}/{total} checks passed")
    print()
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check}")
    
    print("\n" + "="*50)
    
    if all(results.values()):
        print("🎉 All checks passed! Your Hatch environment is ready for CI/CD.")
        print("\nYou can now:")
        print("- Run the enhanced CI/CD pipeline")
        print("- Use hatch run lint:all")
        print("- Use hatch run test:run-cov")
        print("- Use hatch run docs:build")
        print("- Use hatch run prod:serve-api")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        
        if not results["Hatch Installation"]:
            print("- Install Hatch: pip install hatch")
        
        if not results["Project Configuration"]:
            print("- Ensure you're in the project root directory")
        
        if not results["Hatch Environments"]:
            print("- Configure missing environments in pyproject.toml")
        
        if not results["Environment Scripts"]:
            print("- Add missing scripts to environment configuration")
        
        if not results["Dependencies"]:
            print("- Install environment dependencies: hatch env create <env-name>")
        
        if not results["API Startup"]:
            print("- Check prod environment configuration")
            print("- Ensure API dependencies are installed")
        
        return False


def main() -> int:
    """Main validation function."""
    print("🚀 Validating Hatch Environment Configuration for CI/CD Pipeline")
    print("="*60)
    print()
    
    # Run all checks
    checks = {
        "Hatch Installation": check_hatch_installed(),
        "Project Configuration": check_project_config(),
        "Hatch Environments": check_hatch_environments(),
        "Environment Scripts": check_environment_scripts(),
        "Dependencies": check_dependencies(),
        "API Startup": check_api_startup(),
    }
    
    # Generate report
    all_passed = generate_report(checks)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
