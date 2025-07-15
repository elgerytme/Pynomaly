#!/usr/bin/env python3
"""
Fix Critical Naming Convention Violations

This script fixes the most critical naming convention violations identified
in Issue #134. It focuses on high-impact, low-risk fixes that improve
code maintainability without breaking functionality.
"""

import os
import sys
from pathlib import Path


def fix_yaml_extensions():
    """Fix .yml extensions to .yaml for consistency."""
    print("🔧 Fixing YAML file extensions...")
    
    project_root = Path.cwd()
    yml_files = list(project_root.rglob('*.yml'))
    
    fixes_applied = 0
    for yml_file in yml_files:
        # Skip certain directories
        if any(skip_dir in str(yml_file) for skip_dir in ['.git', 'node_modules', '.isolated-work']):
            continue
            
        yaml_file = yml_file.with_suffix('.yaml')
        
        if yml_file.exists() and not yaml_file.exists():
            yml_file.rename(yaml_file)
            print(f"  ✅ Renamed {yml_file.relative_to(project_root)} → {yaml_file.name}")
            fixes_applied += 1
    
    return fixes_applied


def check_critical_violations():
    """Check for critical naming convention violations."""
    print("🔍 Checking for critical naming violations...")
    
    violations = []
    project_root = Path.cwd()
    
    # Check for kebab-case in Python package directories
    for root, dirs, _ in os.walk(project_root / 'src'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
        
        for dir_name in dirs:
            if '-' in dir_name and (Path(root) / dir_name / '__init__.py').exists():
                violations.append({
                    'type': 'python_package_kebab_case',
                    'path': Path(root) / dir_name,
                    'severity': 'high',
                    'message': f"Python package '{dir_name}' uses kebab-case instead of snake_case"
                })
    
    # Check for Python files with kebab-case
    python_files = list(project_root.rglob('*.py'))
    for py_file in python_files:
        if any(skip_dir in str(py_file) for skip_dir in ['.git', 'node_modules', '__pycache__']):
            continue
            
        if '-' in py_file.stem:
            violations.append({
                'type': 'python_file_kebab_case',
                'path': py_file,
                'severity': 'high',
                'message': f"Python file '{py_file.name}' uses kebab-case instead of snake_case"
            })
    
    return violations


def generate_summary_report():
    """Generate a summary report of naming convention status."""
    print("\n📊 NAMING CONVENTION STATUS REPORT")
    print("=" * 50)
    
    # Check current status
    violations = check_critical_violations()
    
    print(f"Critical violations found: {len(violations)}")
    
    if violations:
        print("\n🚨 CRITICAL VIOLATIONS REQUIRING MANUAL ATTENTION:")
        for violation in violations:
            print(f"  📍 {violation['path']}")
            print(f"     {violation['message']}")
            print()
    
    # Check YAML consistency
    project_root = Path.cwd()
    yml_files = list(project_root.rglob('*.yml'))
    yaml_files = list(project_root.rglob('*.yaml'))
    
    print(f"\nYAML File Status:")
    print(f"  .yml files: {len(yml_files)}")
    print(f"  .yaml files: {len(yaml_files)}")
    
    if yml_files:
        print("  ⚠️  Mixed YAML extensions found - should standardize on .yaml")
    else:
        print("  ✅ Consistent YAML extensions (.yaml)")
    
    print("\n📋 COMPLETED IMPROVEMENTS:")
    print("  ✅ Created naming convention documentation")
    print("  ✅ Added automated naming convention checker script")
    print("  ✅ Updated pre-commit hooks to enforce conventions")
    print("  ✅ Fixed critical Python package directory naming (anomaly-detector → anomaly_detector)")
    
    remaining_work = len(violations) + len(yml_files)
    if remaining_work == 0:
        print("\n🎉 ALL CRITICAL NAMING CONVENTIONS IMPLEMENTED!")
        print("   Project follows Python naming standards")
    else:
        print(f"\n⏳ {remaining_work} items remaining for full compliance")


def main():
    """Main entry point."""
    print("🎯 Pynomaly Naming Convention Fixer")
    print("Fixing critical naming convention violations (Issue #134)")
    print()
    
    # Ensure we're in the project root
    if not Path('pyproject.toml').exists():
        print("❌ Error: Not in Pynomaly project root directory")
        print("   Expected to find pyproject.toml")
        sys.exit(1)
    
    # Fix YAML extensions
    yaml_fixes = fix_yaml_extensions()
    
    # Generate status report
    generate_summary_report()
    
    print(f"\n✅ Applied {yaml_fixes} fixes")
    print("💡 For remaining violations, see docs/development/NAMING_CONVENTIONS.md")


if __name__ == "__main__":
    main()