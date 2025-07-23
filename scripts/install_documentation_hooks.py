#!/usr/bin/env python3
"""
Install documentation domain boundary validation hooks for Git.

This script sets up pre-commit hooks to validate documentation domain boundaries
and prevent commits that introduce documentation domain leakage.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    """Install documentation validation hooks."""
    # Get repository root
    repo_root = Path(__file__).parent.parent
    git_hooks_dir = repo_root / '.git' / 'hooks'
    scripts_hooks_dir = repo_root / 'scripts' / 'git-hooks'
    
    if not git_hooks_dir.exists():
        print("Error: This is not a Git repository or .git/hooks directory not found")
        sys.exit(1)
    
    print("Installing documentation domain boundary validation hooks...")
    
    # Install pre-commit hook
    pre_commit_hook = git_hooks_dir / 'pre-commit'
    docs_validation_script = scripts_hooks_dir / 'pre-commit-docs-validation'
    
    if not docs_validation_script.exists():
        print(f"Error: Documentation validation script not found: {docs_validation_script}")
        sys.exit(1)
    
    # Create or update pre-commit hook
    hook_content = create_pre_commit_hook_content(str(docs_validation_script))
    
    if pre_commit_hook.exists():
        # Check if our hook is already installed
        existing_content = pre_commit_hook.read_text()
        if 'pre-commit-docs-validation' in existing_content:
            print("Documentation validation hook is already installed")
            return
        
        # Backup existing hook
        backup_path = pre_commit_hook.with_suffix('.backup')
        print(f"Backing up existing pre-commit hook to: {backup_path}")
        shutil.copy2(pre_commit_hook, backup_path)
        
        # Append our hook to existing hook
        hook_content = existing_content.rstrip() + '\n\n' + hook_content
    
    # Write the hook
    pre_commit_hook.write_text(hook_content)
    
    # Make it executable
    os.chmod(pre_commit_hook, 0o755)
    
    print(f"✅ Documentation validation hook installed: {pre_commit_hook}")
    
    # Test the hook
    print("Testing documentation validation hook...")
    try:
        result = subprocess.run([str(docs_validation_script)], 
                              capture_output=True, text=True, cwd=repo_root)
        if result.returncode == 0:
            print("✅ Documentation validation hook test passed")
        else:
            print("⚠️  Documentation validation hook test completed with warnings:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
    except Exception as e:
        print(f"⚠️  Could not test documentation validation hook: {e}")
    
    print("\nDocumentation domain boundary validation is now active!")
    print("This will check documentation files for domain boundary violations on each commit.")
    print("\nTo temporarily bypass validation (not recommended):")
    print("  git commit --no-verify")
    print("\nFor more information:")
    print("  docs/rules/DOCUMENTATION_DOMAIN_BOUNDARY_RULES.md")


def create_pre_commit_hook_content(docs_validation_script_path: str) -> str:
    """Create the pre-commit hook content."""
    return f"""#!/bin/bash
#
# Pre-commit hook with documentation domain boundary validation
#

# Run documentation domain boundary validation
echo "Running documentation domain boundary validation..."
if ! "{docs_validation_script_path}"; then
    echo "Documentation domain boundary validation failed"
    exit 1
fi

echo "All pre-commit checks passed"
exit 0
"""


if __name__ == '__main__':
    main()