#!/usr/bin/env python3
"""
Setup script for GitHub Issues to TODO.md sync system

This script sets up the automated sync system including:
- Installing dependencies
- Creating configuration files
- Setting up GitHub Actions workflow
- Testing the sync functionality
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str = None) -> bool:
    """Run a command and return success status"""
    if description:
        print(f"📝 {description}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_requirements():
    """Check if required tools are available"""
    print("🔍 Checking requirements...")
    
    # Check Python
    if not run_command("python3 --version", "Checking Python"):
        print("❌ Python 3 is required")
        return False
    
    # Check pip
    if not run_command("pip3 --version", "Checking pip"):
        print("❌ pip is required")
        return False
    
    # Check git
    if not run_command("git --version", "Checking git"):
        print("❌ git is required")
        return False
    
    # Check GitHub CLI
    if not run_command("gh --version", "Checking GitHub CLI"):
        print("⚠️  GitHub CLI not found. Please install it for easier authentication.")
        print("   Visit: https://cli.github.com/")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(f"pip3 install -r {requirements_file}", "Installing Python packages")

def setup_github_token():
    """Set up GitHub token authentication"""
    print("🔐 Setting up GitHub authentication...")
    
    # Try to get token from gh CLI first
    try:
        result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GitHub CLI is authenticated")
            return True
    except FileNotFoundError:
        pass
    
    # Check environment variable
    if os.getenv('GITHUB_TOKEN'):
        print("✅ GITHUB_TOKEN environment variable is set")
        return True
    
    print("⚠️  GitHub authentication not found.")
    print("   Please run: gh auth login")
    print("   Or set GITHUB_TOKEN environment variable")
    return False

def test_sync():
    """Test the sync functionality"""
    print("🧪 Testing sync functionality...")
    
    sync_script = Path(__file__).parent / "manual_sync.py"
    if not sync_script.exists():
        print("❌ Sync script not found")
        return False
    
    # Test dry run
    return run_command(f"python3 {sync_script} --dry-run", "Running sync dry run")

def create_example_webhook():
    """Create an example webhook configuration"""
    print("📋 Creating webhook configuration example...")
    
    webhook_config = Path(__file__).parent / "webhook_config.example.json"
    
    config = {
        "url": "https://your-domain.com/webhook",
        "secret": "your-webhook-secret",
        "events": [
            "issues",
            "issue_comment"
        ],
        "active": True
    }
    
    try:
        import json
        with open(webhook_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Example webhook config created at {webhook_config}")
        return True
    except Exception as e:
        print(f"❌ Error creating webhook config: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up GitHub Issues to TODO.md sync system")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Setup GitHub authentication
    if not setup_github_token():
        print("⚠️  GitHub authentication setup incomplete")
        print("   The sync system will still work but may be rate-limited")
    
    # Create example webhook config
    create_example_webhook()
    
    # Test sync
    if not test_sync():
        print("❌ Sync test failed")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run manual sync: python3 scripts/automation/manual_sync.py")
    print("2. Set up webhook (optional): Configure GitHub webhook to point to your server")
    print("3. Monitor GitHub Actions: Check .github/workflows/issue-sync.yml")
    print("\n📚 Documentation: See scripts/automation/README.md for more details")

if __name__ == "__main__":
    main()