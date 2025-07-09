#!/usr/bin/env python3
"""
Install pre-commit hooks for structure enforcement.

This script ensures that all contributors have the necessary pre-commit hooks
installed to enforce FILE_ORGANIZATION_STANDARDS.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    """Main function to install pre-commit hooks."""
    print("🔧 Installing Pre-commit Hooks for Structure Enforcement")
    print("=" * 60)
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent.parent
    print(f"📁 Project root: {project_root}")
    
    # Check if pre-commit is installed
    print("\n1. Checking pre-commit installation...")
    success, stdout, stderr = run_command("pre-commit --version")
    
    if not success:
        print("❌ pre-commit not found. Installing...")
        success, stdout, stderr = run_command("pip install pre-commit")
        if not success:
            print(f"❌ Failed to install pre-commit: {stderr}")
            return 1
        print("✅ pre-commit installed successfully")
    else:
        print(f"✅ pre-commit found: {stdout.strip()}")
    
    # Install pre-commit hooks
    print("\n2. Installing pre-commit hooks...")
    success, stdout, stderr = run_command("pre-commit install", cwd=project_root)
    
    if not success:
        print(f"❌ Failed to install pre-commit hooks: {stderr}")
        return 1
    
    print("✅ Pre-commit hooks installed successfully")
    print(stdout)
    
    # Test the hook installation
    print("\n3. Testing hook installation...")
    success, stdout, stderr = run_command("pre-commit run --all-files", cwd=project_root)
    
    if success:
        print("✅ All pre-commit hooks passed!")
    else:
        print("⚠️  Some pre-commit hooks failed (this is expected if there are violations)")
        print("The structure enforcement system is working correctly.")
    
    print("\n" + "=" * 60)
    print("🎉 PRE-COMMIT HOOKS SETUP COMPLETE")
    print("=" * 60)
    print("📋 What happens now:")
    print("  • Structure validation runs automatically before each commit")
    print("  • Commits that violate FILE_ORGANIZATION_STANDARDS will be blocked")
    print("  • Use 'python scripts/analysis/organize_files.py --execute' to fix violations")
    print("  • Run 'pre-commit run --all-files' to manually check all files")
    print("  • See docs/developer-guides/contributing/FILE_ORGANIZATION_STANDARDS.md for details")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
