#!/usr/bin/env python3
"""
Developer Workflow Helper

Provides developer-friendly commands and helpers for maintaining
repository organization during development.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import argparse
import json
from datetime import datetime


class DeveloperHelper:
    """Helper for developer workflow enforcement."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.validation_script = repo_root / 'scripts' / 'validation' / 'validate_organization.py'
        self.cleanup_script = repo_root / 'scripts' / 'cleanup' / 'auto_organize.py'
        self.monitor_script = repo_root / 'scripts' / 'monitoring' / 'organization_monitor.py'

    def check_before_commit(self) -> bool:
        """Check repository organization before commit."""
        print("🔍 Pre-commit organization check...")
        
        if not self.validation_script.exists():
            print("⚠️  Validation script not found, skipping check")
            return True
        
        try:
            result = subprocess.run([
                'python3', str(self.validation_script)
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                print("✅ Repository organization is healthy")
                return True
            else:
                print("❌ Repository organization issues found:")
                print(result.stdout)
                print("\n💡 Suggestions:")
                print("1. Run: python3 scripts/cleanup/auto_organize.py")
                print("2. Fix issues manually")
                print("3. Use git commit --no-verify to skip (not recommended)")
                return False
                
        except Exception as e:
            print(f"❌ Failed to run organization check: {e}")
            return False

    def quick_organize(self, execute: bool = False) -> bool:
        """Quick organization of repository."""
        print("🔧 Quick repository organization...")
        
        if not self.cleanup_script.exists():
            print("❌ Cleanup script not found")
            return False
        
        try:
            cmd = ['python3', str(self.cleanup_script)]
            if execute:
                cmd.append('--execute')
            
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                if execute:
                    print("✅ Repository organized successfully")
                else:
                    print("✅ Organization plan generated (use --execute to apply)")
                return True
            else:
                print("❌ Organization failed")
                return False
                
        except Exception as e:
            print(f"❌ Failed to organize repository: {e}")
            return False

    def health_status(self) -> dict:
        """Get current repository health status."""
        if not self.monitor_script.exists():
            return {"error": "Monitor script not found"}
        
        try:
            result = subprocess.run([
                'python3', str(self.monitor_script), '--status'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                # Parse output to extract health score
                lines = result.stdout.strip().split('\n')
                health_line = [line for line in lines if 'Health:' in line]
                if health_line:
                    # Extract score from line like "📊 Repository Organization Health: 95.0/100"
                    health_text = health_line[0]
                    try:
                        score = float(health_text.split(':')[1].split('/')[0].strip())
                        return {
                            "health_score": score,
                            "status": "healthy" if score >= 90 else "needs_attention" if score >= 70 else "critical",
                            "output": result.stdout
                        }
                    except:
                        pass
                
                return {"output": result.stdout}
            else:
                return {"error": "Failed to get health status"}
                
        except Exception as e:
            return {"error": f"Failed to get health status: {e}"}

    def suggest_file_location(self, file_path: str) -> Optional[str]:
        """Suggest correct location for a file."""
        filename = os.path.basename(file_path)
        
        # Configuration files
        config_patterns = {
            r'\.buckconfig.*': 'scripts/config/buck/',
            r'\.dockerignore': 'scripts/config/docker/',
            r'\.env.*': 'scripts/config/env/',
            r'\.eslintrc.*': 'scripts/config/linting/',
            r'\.pre-commit.*': 'scripts/config/git/',
            r'\.python-version-dev': 'scripts/config/python/',
            r'\.secrets\.baseline': 'scripts/config/security/',
        }
        
        # Documentation files
        doc_patterns = {
            r'.*_GUIDE\.md$': 'docs/guides/',
            r'.*_PLAN\.md$': 'docs/plans/',
            r'.*_SUMMARY\.md$': 'docs/summaries/',
            r'.*_REPORT\.md$': 'docs/reports/',
            r'MIGRATION.*\.md$': 'docs/migration/',
            r'DOMAIN.*\.md$': 'docs/architecture/',
        }
        
        # Script files
        script_patterns = {
            r'validate_.*\.py$': 'scripts/validation/',
            r'test_.*\.py$': 'scripts/testing/',
            r'fix_.*\.py$': 'scripts/maintenance/',
            r'setup_.*\.py$': 'scripts/setup/',
            r'deploy_.*\.py$': 'scripts/deployment/',
            r'run_.*\.py$': 'scripts/development/',
        }
        
        # Check each pattern group
        import re
        
        for pattern, location in config_patterns.items():
            if re.match(pattern, filename):
                return location
        
        for pattern, location in doc_patterns.items():
            if re.match(pattern, filename):
                return location
        
        for pattern, location in script_patterns.items():
            if re.match(pattern, filename):
                return location
        
        # General rules
        if filename.endswith('.py') and filename not in ['setup.py']:
            return 'src/'
        
        if filename.endswith('.md') and filename not in ['README.md', 'CHANGELOG.md']:
            return 'docs/'
        
        return None

    def interactive_organization(self):
        """Interactive organization helper."""
        print("🎯 Interactive Repository Organization Helper")
        print("=" * 50)
        
        # Check current status
        print("\n1. Checking current status...")
        status = self.health_status()
        
        if "health_score" in status:
            score = status["health_score"]
            print(f"   Health Score: {score:.1f}/100")
            
            if score >= 90:
                print("   ✅ Repository is well organized!")
                return
            elif score >= 70:
                print("   ⚠️  Repository needs some attention")
            else:
                print("   🔴 Repository needs significant organization")
        
        print("\n2. Available actions:")
        print("   a) Quick organize (dry run)")
        print("   b) Quick organize (execute)")
        print("   c) Full validation report")
        print("   d) Suggest file locations")
        print("   e) Exit")
        
        while True:
            choice = input("\nSelect action (a-e): ").lower().strip()
            
            if choice == 'a':
                print("\n🔍 Running organization analysis...")
                self.quick_organize(execute=False)
                
            elif choice == 'b':
                print("\n⚠️  This will modify files. Continue? (y/N): ", end="")
                confirm = input().lower().strip()
                if confirm == 'y':
                    print("\n🔧 Organizing repository...")
                    self.quick_organize(execute=True)
                else:
                    print("❌ Cancelled")
                
            elif choice == 'c':
                print("\n📊 Generating validation report...")
                try:
                    subprocess.run([
                        'python3', str(self.validation_script)
                    ], cwd=self.repo_root)
                except Exception as e:
                    print(f"❌ Failed to run validation: {e}")
                
            elif choice == 'd':
                print("\n📁 File Location Suggestions")
                print("Enter file paths (one per line, empty line to finish):")
                
                while True:
                    file_path = input("File path: ").strip()
                    if not file_path:
                        break
                    
                    suggestion = self.suggest_file_location(file_path)
                    if suggestion:
                        print(f"   💡 Suggested location: {suggestion}")
                    else:
                        print(f"   ❓ No specific suggestion for {file_path}")
                
            elif choice == 'e':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select a-e.")

    def setup_development_environment(self):
        """Set up development environment with organization tools."""
        print("🚀 Setting up development environment...")
        
        # Install git hooks
        hooks_script = self.repo_root / 'scripts' / 'git-hooks' / 'install-hooks.sh'
        if hooks_script.exists():
            print("📌 Installing git hooks...")
            try:
                subprocess.run(['bash', str(hooks_script)], cwd=self.repo_root)
                print("✅ Git hooks installed")
            except Exception as e:
                print(f"❌ Failed to install git hooks: {e}")
        
        # Create development shortcuts
        shortcuts_dir = self.repo_root / '.dev-shortcuts'
        shortcuts_dir.mkdir(exist_ok=True)
        
        # Create shortcut scripts
        shortcuts = {
            'check': 'python3 scripts/validation/validate_organization.py',
            'organize': 'python3 scripts/cleanup/auto_organize.py',
            'health': 'python3 scripts/monitoring/organization_monitor.py --status',
            'help': 'python3 scripts/workflow/developer_helper.py --interactive'
        }
        
        for name, command in shortcuts.items():
            shortcut_file = shortcuts_dir / f'{name}.sh'
            with open(shortcut_file, 'w') as f:
                f.write(f'#!/bin/bash\ncd "$(dirname "$0")/.." && {command} "$@"\n')
            os.chmod(shortcut_file, 0o755)
        
        print("✅ Development shortcuts created in .dev-shortcuts/")
        print("\nQuick commands:")
        for name, command in shortcuts.items():
            print(f"  .dev-shortcuts/{name}.sh")
        
        print("\n🎯 Development environment ready!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Developer workflow helper")
    parser.add_argument('--check', action='store_true', help='Check repository before commit')
    parser.add_argument('--organize', action='store_true', help='Quick organize repository')
    parser.add_argument('--execute', action='store_true', help='Execute organization changes')
    parser.add_argument('--status', action='store_true', help='Show health status')
    parser.add_argument('--suggest', type=str, help='Suggest location for file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--setup', action='store_true', help='Setup development environment')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.parent
    helper = DeveloperHelper(repo_root)
    
    if args.check:
        success = helper.check_before_commit()
        sys.exit(0 if success else 1)
        
    elif args.organize:
        success = helper.quick_organize(execute=args.execute)
        sys.exit(0 if success else 1)
        
    elif args.status:
        status = helper.health_status()
        if "error" in status:
            print(f"❌ {status['error']}")
            sys.exit(1)
        else:
            print(status.get("output", "No status available"))
            
    elif args.suggest:
        suggestion = helper.suggest_file_location(args.suggest)
        if suggestion:
            print(f"💡 Suggested location for '{args.suggest}': {suggestion}")
        else:
            print(f"❓ No specific suggestion for '{args.suggest}'")
            
    elif args.interactive:
        helper.interactive_organization()
        
    elif args.setup:
        helper.setup_development_environment()
        
    else:
        print("🎯 Repository Organization Developer Helper")
        print("\nUsage:")
        print("  --check         Check repository before commit")
        print("  --organize      Quick organize repository")
        print("  --status        Show health status")
        print("  --suggest FILE  Suggest location for file")
        print("  --interactive   Interactive mode")
        print("  --setup         Setup development environment")


if __name__ == "__main__":
    main()