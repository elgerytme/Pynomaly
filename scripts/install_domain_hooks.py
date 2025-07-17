#!/usr/bin/env python3
"""
Install Domain Boundary Pre-commit Hooks

This script installs Git hooks to enforce domain boundary compliance.
"""

import os
import stat
import sys
from pathlib import Path

def create_pre_commit_hook():
    """Create pre-commit hook for domain boundary validation"""
    
    hook_content = '''#!/bin/bash
# Pre-commit hook for domain boundary validation
# This hook runs domain boundary validation before allowing commits

echo "🔍 Running domain boundary validation..."

# Run domain boundary validator
python3 scripts/domain_boundary_validator.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ COMMIT BLOCKED: Domain boundary violations detected!"
    echo ""
    echo "Please fix the violations before committing:"
    echo "1. Review the violation report above"
    echo "2. Move domain-specific code to appropriate packages"
    echo "3. Use generic abstractions in the software package"
    echo "4. Update configuration files to be domain-agnostic"
    echo ""
    echo "For help, see: DOMAIN_BOUNDARY_RULES.md"
    echo "For detailed plan, see: DOMAIN_COMPLIANCE_PLAN.md"
    echo ""
    exit 1
fi

echo "✅ Domain boundary validation passed!"
exit 0
'''
    
    # Create .git/hooks directory if it doesn't exist
    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    # Write pre-commit hook
    pre_commit_path = hooks_dir / 'pre-commit'
    with open(pre_commit_path, 'w') as f:
        f.write(hook_content)
    
    # Make executable
    st = os.stat(pre_commit_path)
    os.chmod(pre_commit_path, st.st_mode | stat.S_IEXEC)
    
    print(f"✅ Pre-commit hook installed at {pre_commit_path}")

def create_pre_push_hook():
    """Create pre-push hook for additional validation"""
    
    hook_content = '''#!/bin/bash
# Pre-push hook for domain boundary validation
# This hook runs comprehensive validation before pushing

echo "🔍 Running comprehensive domain boundary validation..."

# Run domain boundary validator with full report
python3 scripts/domain_boundary_validator.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ PUSH BLOCKED: Domain boundary violations detected!"
    echo ""
    echo "Please fix all violations before pushing:"
    echo "1. Review the detailed violation report"
    echo "2. Ensure 100% compliance in the software package"
    echo "3. Run tests to verify functionality"
    echo ""
    echo "For help, see: DOMAIN_BOUNDARY_RULES.md"
    echo ""
    exit 1
fi

echo "✅ Domain boundary validation passed!"
exit 0
'''
    
    # Create .git/hooks directory if it doesn't exist
    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    # Write pre-push hook
    pre_push_path = hooks_dir / 'pre-push'
    with open(pre_push_path, 'w') as f:
        f.write(hook_content)
    
    # Make executable
    st = os.stat(pre_push_path)
    os.chmod(pre_push_path, st.st_mode | stat.S_IEXEC)
    
    print(f"✅ Pre-push hook installed at {pre_push_path}")

def create_commit_msg_hook():
    """Create commit message hook for domain boundary tracking"""
    
    hook_content = '''#!/bin/bash
# Commit message hook for domain boundary compliance
# This hook adds domain boundary compliance status to commit messages

commit_msg_file="$1"

# Check if this is a domain boundary fix
if grep -q -i "domain\\|boundary\\|violation" "$commit_msg_file"; then
    echo "🔍 Domain boundary related commit detected"
    
    # Run validation to get current status
    python3 scripts/domain_boundary_validator.py > /tmp/domain_status.txt 2>&1
    
    # Extract violation count
    violations=$(grep "Total violations:" /tmp/domain_status.txt | awk '{print $3}')
    
    if [ ! -z "$violations" ]; then
        echo "" >> "$commit_msg_file"
        echo "Domain-Boundary-Status: $violations violations remaining" >> "$commit_msg_file"
        echo "Validated-By: domain_boundary_validator.py" >> "$commit_msg_file"
    fi
    
    # Clean up
    rm -f /tmp/domain_status.txt
fi

exit 0
'''
    
    # Create .git/hooks directory if it doesn't exist
    hooks_dir = Path('.git/hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    # Write commit-msg hook
    commit_msg_path = hooks_dir / 'commit-msg'
    with open(commit_msg_path, 'w') as f:
        f.write(hook_content)
    
    # Make executable
    st = os.stat(commit_msg_path)
    os.chmod(commit_msg_path, st.st_mode | stat.S_IEXEC)
    
    print(f"✅ Commit message hook installed at {commit_msg_path}")

def install_hooks():
    """Install all domain boundary hooks"""
    print("Installing Domain Boundary Git Hooks")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not Path('.git').exists():
        print("❌ Not in a git repository. Please run from the repository root.")
        sys.exit(1)
    
    # Install hooks
    create_pre_commit_hook()
    create_pre_push_hook()
    create_commit_msg_hook()
    
    print("\n🎉 All domain boundary hooks installed successfully!")
    print("\nNext steps:")
    print("1. Test the hooks with a test commit")
    print("2. Share installation script with team members")
    print("3. Add hook installation to onboarding documentation")
    print("4. Consider adding hooks to CI/CD pipeline")
    
    print("\nTo test the hooks:")
    print("  git add .")
    print("  git commit -m 'test: domain boundary hook test'")

def uninstall_hooks():
    """Uninstall domain boundary hooks"""
    print("Uninstalling Domain Boundary Git Hooks")
    print("=" * 50)
    
    hooks_dir = Path('.git/hooks')
    hooks_to_remove = ['pre-commit', 'pre-push', 'commit-msg']
    
    for hook_name in hooks_to_remove:
        hook_path = hooks_dir / hook_name
        if hook_path.exists():
            # Check if it's our hook
            with open(hook_path, 'r') as f:
                content = f.read()
                if 'domain boundary' in content.lower():
                    hook_path.unlink()
                    print(f"✅ Removed {hook_name} hook")
                else:
                    print(f"⚠️  {hook_name} hook exists but not domain boundary hook")
        else:
            print(f"ℹ️  {hook_name} hook not found")
    
    print("\n🎉 Domain boundary hooks uninstalled successfully!")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'uninstall':
        uninstall_hooks()
    else:
        install_hooks()

if __name__ == "__main__":
    main()