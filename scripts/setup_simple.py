#!/usr/bin/env python
"""Simple setup script to run Pynomaly without Poetry"""

import subprocess
import sys
import os
import shutil

def run_command(cmd, allow_failure=False):
    """Run a shell command and print output"""
    print(f"\n🔷 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"⚠️  {result.stderr}", file=sys.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        if not allow_failure:
            sys.exit(1)
    
    return result

def main():
    print("=" * 60)
    print("🔍 Pynomaly Simple Setup (without Poetry)")
    print("=" * 60)
    
    # Check Python version
    print("\n📌 Checking Python version...")
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, you have {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment if it doesn't exist or if it's corrupted
    venv_broken = False
    if os.path.exists('.venv'):
        # Check if virtual environment is functional
        if sys.platform == "win32":
            python_path = os.path.join(".venv", "Scripts", "python.exe")
            if not os.path.exists(python_path):
                python_path = os.path.join(".venv", "Scripts", "python")
        else:
            python_path = os.path.join(".venv", "bin", "python")
        
        if not os.path.exists(python_path):
            venv_broken = True
        else:
            # Test if venv python works
            test_result = subprocess.run([python_path, "--version"], capture_output=True, text=True)
            if test_result.returncode != 0:
                venv_broken = True
    
    if not os.path.exists('.venv') or venv_broken:
        if venv_broken:
            print("\n⚠️  Virtual environment appears corrupted. Recreating...")
            shutil.rmtree(".venv", ignore_errors=True)
        else:
            print("\n📌 Creating virtual environment...")
        
        try:
            run_command([sys.executable, "-m", "venv", ".venv"])
            print("✅ Virtual environment created")
        except SystemExit:
            print("\n⚠️  Standard venv creation failed. Trying alternatives...")
            try:
                run_command([sys.executable, "-m", "venv", ".venv", "--system-site-packages"])
                print("✅ Virtual environment created with system site packages")
            except SystemExit:
                print("\n❌ Virtual environment creation failed.")
                print("💡 For WSL/Ubuntu/Debian: sudo apt install python3.12-venv python3-pip")
                print("💡 For CentOS/RHEL: sudo yum install python3-venv python3-pip")
                print("💡 For macOS: brew install python@3.12")
                print("\n⚠️  Continuing without virtual environment (not recommended for development)")
                return
    else:
        print("\n✅ Virtual environment already exists and functional")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
        python_path = os.path.join(".venv", "Scripts", "python.exe")
        
        # Check if files exist
        if not os.path.exists(python_path):
            # Try without .exe extension
            python_path = os.path.join(".venv", "Scripts", "python")
            pip_path = os.path.join(".venv", "Scripts", "pip")
    else:
        pip_path = os.path.join(".venv", "bin", "pip")
        python_path = os.path.join(".venv", "bin", "python")
    
    # Ensure virtual environment python exists
    if not os.path.exists(python_path):
        print(f"❌ Virtual environment python not found at {python_path}")
        print("⚠️  Using system python instead (not recommended)")
        python_path = sys.executable
        pip_path = "pip"
    
    # Upgrade pip and ensure it's available
    print("\n📌 Upgrading pip...")
    pip_result = run_command([python_path, "-m", "pip", "install", "--upgrade", "pip"], allow_failure=True)
    if pip_result.returncode != 0:
        print("\n⚠️  pip upgrade failed. Trying to bootstrap pip...")
        # Try to install pip first
        import urllib.request
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as f:
                urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', f.name)
                run_command([python_path, f.name])
                os.unlink(f.name)
            print("✅ pip bootstrapped successfully")
        except Exception as e:
            print(f"❌ Failed to bootstrap pip: {e}")
            print("💡 Try: python -m ensurepip --upgrade")
            return
    
    # Install dependencies
    print("\n📌 Installing dependencies from requirements.txt...")
    if os.path.exists("requirements.txt"):
        run_command([python_path, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("⚠️  requirements.txt not found. Installing core dependencies directly...")
        core_deps = ["pyod>=2.0.5", "numpy>=1.26.0,<2.2.0", "pandas>=2.2.3", "polars>=1.19.0", 
                     "pydantic>=2.10.4", "structlog>=24.4.0", "dependency-injector>=4.42.0"]
        for dep in core_deps:
            run_command([python_path, "-m", "pip", "install", dep])
    
    # Install package in development mode
    print("\n📌 Installing Pynomaly in development mode...")
    run_command([python_path, "-m", "pip", "install", "-e", "."])
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    
    print("\n📝 How to run Pynomaly:\n")
    
    # Activation instructions
    if sys.platform == "win32":
        print("1. Activate the virtual environment:")
        print("   .venv\\Scripts\\activate")
    else:
        print("1. Activate the virtual environment:")
        print("   source .venv/bin/activate")
    
    print("\n2. Run the CLI directly:")
    print("   pynomaly --help")
    print("   # or")
    print("   python -m pynomaly.presentation.cli.app --help")
    print("   # or")
    print("   python scripts/cli.py --help")
    
    print("\n3. Start the API server:")
    print("   pynomaly server start")
    print("   # or")
    print("   python -m uvicorn pynomaly.presentation.api.app:app --reload")
    print("   # or")
    print("   python -m pynomaly.presentation.cli.app server start")
    
    print("\n4. Access the web UI:")
    print("   http://localhost:8000")
    
    print("\n📖 Example commands:")
    print("   # List available algorithms")
    print("   pynomaly detector algorithms")
    print("\n   # Create a detector")
    print("   pynomaly detector create --name 'My Detector' --algorithm IsolationForest")
    print("\n   # Start the server")
    print("   pynomaly server start")

if __name__ == "__main__":
    main()