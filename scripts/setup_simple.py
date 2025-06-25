#!/usr/bin/env python
"""Simple setup script to run Pynomaly without Poetry"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"\n🔷 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"⚠️  {result.stderr}", file=sys.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
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
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('.venv'):
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
    else:
        print("\n✅ Virtual environment already exists")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
        python_path = os.path.join(".venv", "Scripts", "python.exe")
        
        # Check if files exist
        if not os.path.exists(python_path):
            print(f"⚠️  Python not found at {python_path}")
            # Try without .exe extension
            python_path = os.path.join(".venv", "Scripts", "python")
            pip_path = os.path.join(".venv", "Scripts", "pip")
            
        if not os.path.exists(python_path):
            print("❌ Virtual environment appears broken. Recreating...")
            import shutil
            shutil.rmtree(".venv", ignore_errors=True)
            run_command([sys.executable, "-m", "venv", ".venv"])
            
    else:
        pip_path = os.path.join(".venv", "bin", "pip")
        python_path = os.path.join(".venv", "bin", "python")
    
    # Upgrade pip
    print("\n📌 Upgrading pip...")
    run_command([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("\n📌 Installing dependencies from requirements.txt...")
    run_command([pip_path, "install", "-r", "requirements.txt"])
    
    # Install package in development mode
    print("\n📌 Installing Pynomaly in development mode...")
    run_command([pip_path, "install", "-e", "."])
    
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