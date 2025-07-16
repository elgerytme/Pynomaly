#!/usr/bin/env python3
"""
Repository Organization System Installer

Installs and configures the complete repository organization system.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List
import argparse


class OrganizationSystemInstaller:
    """Installs the repository organization system."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.install_log: List[str] = []

    def log(self, message: str):
        """Log installation step."""
        print(message)
        self.install_log.append(message)

    def check_prerequisites(self) -> bool:
        """Check system prerequisites."""
        self.log("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log("❌ Python 3.8+ required")
            return False
        
        self.log(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Check if we're in a git repository
        if not (self.repo_root / '.git').exists():
            self.log("❌ Not in a git repository")
            return False
        
        self.log("✅ Git repository detected")
        
        # Check for required directories
        required_dirs = ['scripts', 'src', 'docs']
        for dir_name in required_dirs:
            if not (self.repo_root / dir_name).exists():
                self.log(f"❌ Required directory missing: {dir_name}")
                return False
        
        self.log("✅ Required directories exist")
        return True

    def make_scripts_executable(self):
        """Make all Python scripts executable."""
        self.log("🔧 Making scripts executable...")
        
        script_files = [
            'scripts/validation/validate_organization.py',
            'scripts/cleanup/auto_organize.py',
            'scripts/monitoring/organization_monitor.py',
            'scripts/workflow/developer_helper.py',
            'scripts/git-hooks/install-hooks.sh',
        ]
        
        for script_file in script_files:
            script_path = self.repo_root / script_file
            if script_path.exists():
                os.chmod(script_path, 0o755)
                self.log(f"✅ Made executable: {script_file}")
            else:
                self.log(f"⚠️  Script not found: {script_file}")

    def install_git_hooks(self):
        """Install git hooks."""
        self.log("📌 Installing git hooks...")
        
        hooks_installer = self.repo_root / 'scripts' / 'git-hooks' / 'install-hooks.sh'
        
        if not hooks_installer.exists():
            self.log("❌ Git hooks installer not found")
            return False
        
        try:
            result = subprocess.run(
                ['bash', str(hooks_installer)],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("✅ Git hooks installed successfully")
                return True
            else:
                self.log(f"❌ Git hooks installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Failed to install git hooks: {e}")
            return False

    def create_development_shortcuts(self):
        """Create development shortcuts."""
        self.log("🚀 Creating development shortcuts...")
        
        shortcuts_dir = self.repo_root / '.dev-shortcuts'
        shortcuts_dir.mkdir(exist_ok=True)
        
        shortcuts = {
            'check': 'scripts/validation/validate_organization.py',
            'organize': 'scripts/cleanup/auto_organize.py',
            'organize-execute': 'scripts/cleanup/auto_organize.py --execute',
            'health': 'scripts/monitoring/organization_monitor.py --status',
            'monitor': 'scripts/monitoring/organization_monitor.py --start',
            'helper': 'scripts/workflow/developer_helper.py --interactive',
        }
        
        # Create shell scripts
        for name, command in shortcuts.items():
            script_content = f'''#!/bin/bash
# {name.title()} - Repository organization shortcut
cd "$(dirname "$0")/.."
python3 {command} "$@"
'''
            
            script_file = shortcuts_dir / f'{name}.sh'
            with open(script_file, 'w') as f:
                f.write(script_content)
            os.chmod(script_file, 0o755)
        
        # Create batch files for Windows
        for name, command in shortcuts.items():
            batch_content = f'''@echo off
REM {name.title()} - Repository organization shortcut
cd /d "%~dp0.."
python {command} %*
'''
            
            batch_file = shortcuts_dir / f'{name}.bat'
            with open(batch_file, 'w') as f:
                f.write(batch_content)
        
        self.log(f"✅ Created {len(shortcuts)} development shortcuts")

    def create_vscode_tasks(self):
        """Create VS Code tasks for organization tools."""
        self.log("📝 Creating VS Code tasks...")
        
        vscode_dir = self.repo_root / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        tasks_file = vscode_dir / 'tasks.json'
        
        tasks_config = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Organization: Validate",
                    "type": "shell",
                    "command": "python3",
                    "args": ["scripts/validation/validate_organization.py"],
                    "group": "build",
                    "presentation": {"echo": True, "reveal": "always"},
                    "problemMatcher": []
                },
                {
                    "label": "Organization: Auto-organize (dry run)",
                    "type": "shell",
                    "command": "python3",
                    "args": ["scripts/cleanup/auto_organize.py"],
                    "group": "build",
                    "presentation": {"echo": True, "reveal": "always"},
                    "problemMatcher": []
                },
                {
                    "label": "Organization: Auto-organize (execute)",
                    "type": "shell",
                    "command": "python3",
                    "args": ["scripts/cleanup/auto_organize.py", "--execute"],
                    "group": "build",
                    "presentation": {"echo": True, "reveal": "always"},
                    "problemMatcher": []
                },
                {
                    "label": "Organization: Health status",
                    "type": "shell",
                    "command": "python3",
                    "args": ["scripts/monitoring/organization_monitor.py", "--status"],
                    "group": "test",
                    "presentation": {"echo": True, "reveal": "always"},
                    "problemMatcher": []
                },
                {
                    "label": "Organization: Interactive helper",
                    "type": "shell",
                    "command": "python3",
                    "args": ["scripts/workflow/developer_helper.py", "--interactive"],
                    "group": "build",
                    "presentation": {"echo": True, "reveal": "always"},
                    "problemMatcher": []
                }
            ]
        }
        
        import json
        with open(tasks_file, 'w') as f:
            json.dump(tasks_config, f, indent=2)
        
        self.log("✅ VS Code tasks created")

    def run_initial_validation(self):
        """Run initial validation to check installation."""
        self.log("🔍 Running initial validation...")
        
        validation_script = self.repo_root / 'scripts' / 'validation' / 'validate_organization.py'
        
        if not validation_script.exists():
            self.log("❌ Validation script not found")
            return False
        
        try:
            result = subprocess.run(
                ['python3', str(validation_script)],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("✅ Initial validation passed")
                return True
            else:
                self.log("⚠️  Initial validation found issues (this is normal for first run)")
                self.log("   Run 'python3 scripts/cleanup/auto_organize.py' to fix common issues")
                return True  # Not a failure for installation
                
        except Exception as e:
            self.log(f"❌ Failed to run initial validation: {e}")
            return False

    def generate_readme_section(self) -> str:
        """Generate README section for organization system."""
        return """
## 🔧 Repository Organization System

This repository uses an automated organization system to maintain clean structure and consistent file placement.

### Quick Commands

```bash
# Check repository organization
python3 scripts/validation/validate_organization.py

# Auto-organize repository
python3 scripts/cleanup/auto_organize.py --execute

# Check health status  
python3 scripts/monitoring/organization_monitor.py --status

# Interactive helper
python3 scripts/workflow/developer_helper.py --interactive
```

### Development Shortcuts

Use the shortcuts in `.dev-shortcuts/` for quick access:

```bash
# Unix/Linux/Mac
.dev-shortcuts/check.sh          # Validate organization
.dev-shortcuts/organize.sh       # Auto-organize (dry run)
.dev-shortcuts/organize-execute.sh # Auto-organize (execute)
.dev-shortcuts/health.sh         # Health status
.dev-shortcuts/helper.sh         # Interactive helper

# Windows
.dev-shortcuts/check.bat
.dev-shortcuts/organize.bat
# ... etc
```

### Git Hooks

The system includes git hooks that automatically:
- Validate organization before commits
- Check for temporary files and large files
- Provide health status after commits
- Block pushes with critical organization issues

To disable hooks temporarily: `git commit --no-verify`

### VS Code Integration

Organization tasks are available in VS Code:
- `Ctrl+Shift+P` → "Tasks: Run Task" → "Organization: ..."

### Monitoring

The system provides continuous monitoring with:
- Daily health reports
- Automatic organization suggestions
- GitHub Actions integration
- Developer workflow helpers

### Rules

Repository organization rules are defined in `.project-rules/REPOSITORY_ORGANIZATION_RULES.md`

Only these directories are allowed in the repository root:
- `.claude`, `.github`, `.hypothesis`, `.project-rules`, `.ruff_cache`, `.storybook`, `.vscode`
- `docs`, `pkg`, `scripts`, `src`

All other files must be organized into appropriate directories.
"""

    def install(self) -> bool:
        """Run complete installation."""
        self.log("🚀 Installing Repository Organization System")
        self.log("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Make scripts executable
        self.make_scripts_executable()
        
        # Install git hooks
        if not self.install_git_hooks():
            self.log("⚠️  Git hooks installation failed, continuing...")
        
        # Create development shortcuts
        self.create_development_shortcuts()
        
        # Create VS Code tasks
        self.create_vscode_tasks()
        
        # Run initial validation
        self.run_initial_validation()
        
        self.log("")
        self.log("🎉 Repository Organization System installed successfully!")
        self.log("")
        self.log("📝 Next steps:")
        self.log("1. Run: python3 scripts/cleanup/auto_organize.py --execute")
        self.log("2. Add organization section to README.md")
        self.log("3. Commit the organization system")
        self.log("4. Share with your team!")
        self.log("")
        self.log("📄 Installation log:")
        for log_entry in self.install_log:
            if not log_entry.startswith(('🚀', '=')):
                self.log(f"   {log_entry}")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Install repository organization system")
    parser.add_argument('--readme', action='store_true', help='Show README section to add')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.parent
    installer = OrganizationSystemInstaller(repo_root)
    
    if args.readme:
        print(installer.generate_readme_section())
        return
    
    success = installer.install()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()