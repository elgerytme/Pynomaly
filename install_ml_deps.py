#!/usr/bin/env python3
"""
Install minimal ML dependencies for testing Phase 2 adapters.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Install ML dependencies for Phase 2 testing."""
    print("🚀 Installing ML Dependencies for Phase 2 Testing")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ pyproject.toml not found. Please run from project root.")
        return 1
    
    commands = [
        ("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch CPU"),
        ("pip3 install tensorflow-cpu", "Installing TensorFlow CPU"),
        ("pip3 install jax[cpu]", "Installing JAX CPU"),
        ("pip3 install optax", "Installing Optax for JAX"),
        ("pip3 install pytest pytest-mock", "Installing test dependencies"),
    ]
    
    success_count = 0
    for cmd, desc in commands:
        if run_command(cmd, desc):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Results: {success_count}/{len(commands)} successful")
    
    if success_count == len(commands):
        print("🎉 All dependencies installed successfully!")
        print("\n🧪 Running basic adapter tests...")
        
        # Test the adapters now
        if run_command("python3 test_adapter_basic.py", "Testing adapters"):
            print("✅ Adapters are working!")
        else:
            print("⚠️  Some adapter tests failed")
        
        return 0
    else:
        print("⚠️  Some installations failed - tests may not work completely")
        return 1

if __name__ == "__main__":
    exit(main())