#!/usr/bin/env python3
"""Simple setup script to run Pynomaly without Poetry"""

import os
import shutil
import subprocess
import sys


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

    # Check for clean flag
    clean_install = len(sys.argv) > 1 and sys.argv[1] == "--clean"
    if clean_install:
        print("\n🧹 Clean install requested - removing existing .venv")
        if os.path.exists(".venv"):
            shutil.rmtree(".venv", ignore_errors=True)
            print("✅ Removed existing virtual environment")

    # Check Python version
    print("\n📌 Checking Python version...")
    print(f"✅ Python {sys.version.split()[0]} detected")

    # Create virtual environment if it doesn't exist or if it's corrupted
    venv_broken = False
    if os.path.exists(".venv"):
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
            test_result = subprocess.run(
                [python_path, "--version"], capture_output=True, text=True
            )
            if test_result.returncode != 0:
                venv_broken = True

    if not os.path.exists(".venv") or venv_broken:
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
                run_command(
                    [sys.executable, "-m", "venv", ".venv", "--system-site-packages"]
                )
                print("✅ Virtual environment created with system site packages")
            except SystemExit:
                print("\n❌ Virtual environment creation failed.")
                print(
                    "💡 For WSL/Ubuntu/Debian: sudo apt install python3.12-venv python3-pip"
                )
                print("💡 For CentOS/RHEL: sudo yum install python3-venv python3-pip")
                print("💡 For macOS: brew install python@3.12")
                print(
                    "\n⚠️  Continuing with system Python (not recommended for development)"
                )
                # Continue with system Python
                python_path = sys.executable
                "pip3" if shutil.which("pip3") else "pip"
                # Clean up broken venv
                if os.path.exists(".venv"):
                    shutil.rmtree(".venv", ignore_errors=True)
    else:
        print("\n✅ Virtual environment already exists and functional")

    # Determine pip path based on OS (only if venv exists)
    if os.path.exists(".venv"):
        if sys.platform == "win32":
            os.path.join(".venv", "Scripts", "pip.exe")
            python_path = os.path.join(".venv", "Scripts", "python.exe")

            # Check if files exist
            if not os.path.exists(python_path):
                # Try without .exe extension
                python_path = os.path.join(".venv", "Scripts", "python")
                os.path.join(".venv", "Scripts", "pip")
        else:
            os.path.join(".venv", "bin", "pip")
            python_path = os.path.join(".venv", "bin", "python")

        # Ensure virtual environment python exists
        if not os.path.exists(python_path):
            print(f"❌ Virtual environment python not found at {python_path}")
            print("⚠️  Using system python instead (not recommended)")
            python_path = sys.executable
            "pip3" if shutil.which("pip3") else "pip"
    else:
        # No virtual environment, use system Python
        python_path = sys.executable
        "pip3" if shutil.which("pip3") else "pip"

    # Check if we're using system Python and it's externally managed (PEP 668)
    if python_path == sys.executable and not os.path.exists(".venv"):
        print("\n⚠️  Using system Python - checking if externally managed (PEP 668)...")
        test_pip = run_command(
            [python_path, "-c", "import pip; print('pip available')"],
            allow_failure=True,
        )
        if test_pip.returncode != 0:
            print("❌ System Python doesn't have pip or is externally managed")
            print("💡 Recommended solutions:")
            print("   1. Install python3-venv: sudo apt install python3-venv")
            print("   2. Use pipx: pipx install pynomaly")
            print("   3. Use conda/mamba virtual environment")
            print("   4. Override with --break-system-packages (not recommended)")
            print("\n🛑 Cannot proceed without proper package management")
            return

    # Upgrade pip and ensure it's available
    print("\n📌 Upgrading pip...")
    pip_result = run_command(
        [python_path, "-m", "pip", "install", "--upgrade", "pip"], allow_failure=True
    )
    if pip_result.returncode != 0:
        if "externally-managed-environment" in pip_result.stderr:
            print("❌ Python environment is externally managed (PEP 668)")
            print("💡 Solutions:")
            print("   1. Create virtual environment: python3 -m venv .venv")
            print("   2. Install python3-venv: sudo apt install python3-venv")
            print("   3. Use pipx: pipx install pynomaly")
            print("   4. Override (risky): add --break-system-packages")
            return

        print("\n⚠️  pip not available. Trying to install pip...")
        # Try ensurepip first (simpler and more reliable)
        ensurepip_result = run_command(
            [python_path, "-m", "ensurepip", "--upgrade"], allow_failure=True
        )
        if ensurepip_result.returncode != 0:
            print("⚠️  ensurepip failed. Virtual environment may need to be recreated.")
            print("💡 Try manually: python -m ensurepip --upgrade")
            print(
                "💡 Or recreate virtual environment with: rm -rf .venv && python -m venv .venv"
            )
            return
        else:
            print("✅ pip installed via ensurepip")

    # Install dependencies
    print("\n📌 Installing dependencies from requirements.txt...")
    if os.path.exists("requirements.txt"):
        run_command([python_path, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("⚠️  requirements.txt not found. Installing core dependencies directly...")
        core_deps = [
            "pyod>=2.0.5",
            "numpy>=1.26.0,<2.2.0",
            "pandas>=2.2.3",
            "polars>=1.19.0",
            "pydantic>=2.10.4",
            "structlog>=24.4.0",
            "dependency-injector>=4.42.0",
        ]
        for dep in core_deps:
            run_command([python_path, "-m", "pip", "install", dep])

    # Install package in development mode
    print("\n📌 Installing Pynomaly in development mode...")
    run_command([python_path, "-m", "pip", "install", "-e", "."])

    # Verify installation
    print("\n📌 Verifying installation...")
    try:
        verify_result = run_command(
            [
                python_path,
                "-c",
                "import pynomaly; from pynomaly.domain.entities import Dataset; print('✅ Core imports successful')",
            ],
            allow_failure=True,
        )
        if verify_result.returncode != 0:
            print("⚠️  Core imports failed, but package was installed")
    except Exception:
        print("⚠️  Verification failed, but continuing...")

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

    print("\n2. Test basic functionality:")
    print(
        f"   {python_path} -c \"import pynomaly; print('Pynomaly imported successfully')\""
    )

    print("\n3. Run the CLI (requires CLI extras):")
    print("   # Install CLI support first:")
    print(f'   {python_path} -m pip install -e ".[cli]"')
    print("   # Then run CLI:")
    print("   python -m pynomaly.presentation.cli.app --help")

    print("\n4. Start the API server (requires API extras):")
    print("   # Install API support first:")
    print(f'   {python_path} -m pip install -e ".[api]"')
    print("   # Then start server:")
    print("   python -m uvicorn pynomaly.presentation.api.app:app --reload")

    print("\n5. Quick server setup (includes API + CLI):")
    print(f'   {python_path} -m pip install -e ".[server]"')

    print("\n6. Access the web UI:")
    print("   http://localhost:8000 (after starting the API server)")

    print("\n📖 Quick test commands:")
    print("   # Test core functionality")
    print(
        f"   {python_path} -c \"import pyod; print('PyOD version:', pyod.__version__)\""
    )
    print(
        f"   {python_path} -c \"import pandas as pd; print('Pandas version:', pd.__version__)\""
    )
    print(
        f"   {python_path} -c \"from pynomaly.domain.entities import Dataset; print('Core entities loaded')\""
    )

    print("\n🔧 For development extras:")
    print(f'   {python_path} -m pip install -e ".[all]"  # Install everything')


if __name__ == "__main__":
    main()
