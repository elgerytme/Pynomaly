#!/usr/bin/env python3
"""Fix package installation and environment issues"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔷 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("✅ Success:", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error:", e.stderr.strip() if e.stderr else str(e))
        return False

def main():
    print("🔧 Fixing Pynomaly Package Issues")
    print("=" * 50)
    
    # Step 1: Uninstall the wrong pynomaly package
    print("\n📌 Step 1: Remove conflicting pynomaly package")
    success = run_command([
        sys.executable, "-m", "pip", "uninstall", "pynomaly", "-y"
    ], "Uninstalling existing pynomaly package")
    
    # Step 2: Install in development mode with server extras
    print("\n📌 Step 2: Install current package in development mode")
    success = run_command([
        sys.executable, "-m", "pip", "install", "-e", ".[server]"
    ], "Installing package with server extras")
    
    if not success:
        print("\n⚠️  Trying with minimal setup...")
        success = run_command([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], "Installing minimal package")
    
    # Step 3: Install missing monitoring dependencies
    print("\n📌 Step 3: Install missing monitoring dependencies")
    monitoring_deps = [
        "prometheus-fastapi-instrumentator>=7.0.0",
        "shap>=0.46.0",
        "lime>=0.2.0.1"
    ]
    
    for dep in monitoring_deps:
        run_command([
            sys.executable, "-m", "pip", "install", dep
        ], f"Installing {dep}")
    
    # Step 4: Verify installation
    print("\n📌 Step 4: Verify installation")
    
    # Test imports
    try:
        import pynomaly
        print("✅ pynomaly package imports successfully")
        print(f"   Version: {getattr(pynomaly, '__version__', 'development')}")
    except ImportError as e:
        print(f"❌ Failed to import pynomaly: {e}")
    
    # Test CLI availability
    try:
        result = subprocess.run([
            sys.executable, "-c", "from pynomaly.presentation.cli.app import app; print('CLI available')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI module imports successfully")
        else:
            print(f"❌ CLI import failed: {result.stderr}")
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
    
    # Step 5: Show next steps
    print("\n📌 Next Steps:")
    print("1. Try running: python -m pynomaly.presentation.cli.app --help")
    print("2. Or create a console script with: pip install -e .[cli]") 
    print("3. For API server: python scripts/run_api.py")
    print("4. For full setup: pip install -e .[production]")
    
    print("\n✅ Package fix process completed!")

if __name__ == "__main__":
    main()