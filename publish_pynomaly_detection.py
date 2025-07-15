#!/usr/bin/env python3
"""
Script to publish pynomaly-detection package to TestPyPI and PyPI
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"Error output: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Exception running command {cmd}: {e}")
        return False

def main():
    # Change to pynomaly-detection directory
    package_dir = Path("pynomaly-detection")
    if not package_dir.exists():
        print("pynomaly-detection directory not found!")
        sys.exit(1)
    
    print("📦 Building pynomaly-detection package...")
    
    # Activate virtual environment and build
    if not run_command("source venv/bin/activate && python -m build", cwd=package_dir):
        print("❌ Build failed!")
        sys.exit(1)
    
    print("✅ Package built successfully!")
    
    # Check dist files
    dist_dir = package_dir / "dist"
    dist_files = list(dist_dir.glob("*"))
    if not dist_files:
        print("❌ No distribution files found!")
        sys.exit(1)
    
    print(f"📋 Distribution files:")
    for file in dist_files:
        print(f"  - {file.name}")
    
    # Validate package
    print("\n🔍 Validating package...")
    if not run_command("source venv/bin/activate && twine check dist/*", cwd=package_dir):
        print("❌ Package validation failed!")
        sys.exit(1)
    
    print("✅ Package validation passed!")
    
    # Ask user which repository to upload to
    choice = input("\nChoose upload target:\n1. TestPyPI (recommended for testing)\n2. PyPI (production)\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        repo_url = "https://test.pypi.org/legacy/"
        repo_name = "TestPyPI"
        token_var = "TEST_PYPI_API_TOKEN"
    elif choice == "2":
        repo_url = "https://pypi.org/legacy/"
        repo_name = "PyPI" 
        token_var = "PYPI_API_TOKEN"
    else:
        print("❌ Invalid choice!")
        sys.exit(1)
    
    # Check for API token
    api_token = os.environ.get(token_var)
    if not api_token:
        print(f"❌ {token_var} environment variable not set!")
        print(f"Please set your {repo_name} API token:")
        print(f"export {token_var}=your_token_here")
        sys.exit(1)
    
    print(f"\n🚀 Uploading to {repo_name}...")
    
    # Upload to repository
    upload_cmd = f"source venv/bin/activate && TWINE_USERNAME=__token__ TWINE_PASSWORD={api_token} twine upload --repository-url {repo_url} dist/* --non-interactive"
    
    if not run_command(upload_cmd, cwd=package_dir):
        print(f"❌ Upload to {repo_name} failed!")
        sys.exit(1)
    
    print(f"🎉 Successfully uploaded pynomaly-detection to {repo_name}!")
    
    if choice == "1":
        print("\n📋 TestPyPI Information:")
        print("📦 Package URL: https://test.pypi.org/project/pynomaly-detection/")
        print("📥 Install command: pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pynomaly-detection")
    else:
        print("\n📋 PyPI Information:")
        print("📦 Package URL: https://pypi.org/project/pynomaly-detection/")
        print("📥 Install command: pip install pynomaly-detection")

if __name__ == "__main__":
    main()