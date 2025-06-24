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
        run_command([sys.executable, "-m", "venv", ".venv"])
        print("✅ Virtual environment created")
    else:
        print("\n✅ Virtual environment already exists")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
        python_path = os.path.join(".venv", "Scripts", "python.exe")
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
    print("   python -m pynomaly.presentation.cli.app --help")
    print("   # or")
    print("   python cli.py --help")
    
    print("\n3. Start the API server:")
    print("   python -m uvicorn pynomaly.presentation.api.app:app --reload")
    print("   # or")
    print("   python -m pynomaly.presentation.cli.app server start")
    
    print("\n4. Access the web UI:")
    print("   http://localhost:8000")
    
    print("\n📖 Example commands:")
    print("   # List available algorithms")
    print("   python cli.py detector algorithms")
    print("\n   # Create a detector")
    print("   python cli.py detector create --name 'My Detector' --algorithm IsolationForest")
    print("\n   # Start the server")
    print("   python cli.py server start")

if __name__ == "__main__":
    main()