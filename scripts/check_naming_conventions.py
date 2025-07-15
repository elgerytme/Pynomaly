#!/usr/bin/env python3
"""
Naming Convention Checker and Fixer for Pynomaly Project

This script analyzes the project for naming convention violations and optionally fixes them.
It enforces the naming conventions defined in docs/development/NAMING_CONVENTIONS.md.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import subprocess


class NamingConventionChecker:
    """Checks and fixes naming convention violations in the Pynomaly project."""

    def __init__(self, project_root: Path, fix_mode: bool = False):
        self.project_root = project_root
        self.fix_mode = fix_mode
        self.violations: List[Dict[str, str]] = []
        self.fixes_applied: List[Dict[str, str]] = []
        
        # Define patterns and rules
        self.python_package_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        self.python_file_pattern = re.compile(r'^[a-z][a-z0-9_]*\.py$')
        self.test_file_pattern = re.compile(r'^test_[a-z][a-z0-9_]*\.py$')
        self.class_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.function_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        self.constant_pattern = re.compile(r'^[A-Z][A-Z0-9_]*$')
        
        # Files and directories to skip
        self.skip_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules', 
            '.venv', 'venv', '.env', 'build', 'dist', '.isolated-work'
        }
        self.skip_files = {
            '__init__.py', '__main__.py', '.gitignore', '.gitkeep'
        }

    def check_all_violations(self) -> bool:
        """Check all naming convention violations. Returns True if violations found."""
        print("🔍 Checking naming conventions...")
        
        self.check_directory_naming()
        self.check_file_naming()
        self.check_yaml_extensions()
        self.check_dockerfile_naming()
        self.check_python_code_naming()
        
        if self.violations:
            self.report_violations()
            return True
        else:
            print("✅ No naming convention violations found!")
            return False

    def check_directory_naming(self):
        """Check directory naming conventions."""
        print("  📁 Checking directory naming...")
        
        for root, dirs, _ in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                relative_path = dir_path.relative_to(self.project_root)
                
                # Check if this is a Python package directory
                if self._is_python_package_dir(dir_path):
                    if '-' in dir_name:
                        violation = {
                            'type': 'directory_naming',
                            'path': str(relative_path),
                            'issue': f"Python package directory '{dir_name}' uses kebab-case",
                            'suggestion': dir_name.replace('-', '_'),
                            'severity': 'high'
                        }
                        self.violations.append(violation)

    def check_file_naming(self):
        """Check file naming conventions."""
        print("  📄 Checking file naming...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file_name in files:
                if file_name in self.skip_files:
                    continue
                    
                file_path = Path(root) / file_name
                relative_path = file_path.relative_to(self.project_root)
                
                # Check Python files
                if file_name.endswith('.py'):
                    self._check_python_file_naming(relative_path, file_name)
                
                # Check test files
                if 'test' in str(relative_path).lower() and file_name.endswith('.py'):
                    self._check_test_file_naming(relative_path, file_name)

    def check_yaml_extensions(self):
        """Check YAML file extensions for consistency."""
        print("  📝 Checking YAML file extensions...")
        
        yml_files = list(self.project_root.rglob('*.yml'))
        
        for yml_file in yml_files:
            # Skip node_modules and other irrelevant directories
            if any(skip_dir in str(yml_file) for skip_dir in self.skip_dirs):
                continue
                
            relative_path = yml_file.relative_to(self.project_root)
            yaml_equivalent = yml_file.with_suffix('.yaml')
            
            violation = {
                'type': 'yaml_extension',
                'path': str(relative_path),
                'issue': "Uses .yml extension instead of .yaml",
                'suggestion': str(relative_path).replace('.yml', '.yaml'),
                'severity': 'medium'
            }
            self.violations.append(violation)

    def check_dockerfile_naming(self):
        """Check Dockerfile naming conventions."""
        print("  🐳 Checking Dockerfile naming...")
        
        dockerfiles = list(self.project_root.rglob('Dockerfile*'))
        
        for dockerfile in dockerfiles:
            relative_path = dockerfile.relative_to(self.project_root)
            file_name = dockerfile.name
            
            # Check for kebab-case suffixes in Dockerfiles
            if '.' in file_name and file_name.startswith('Dockerfile.'):
                suffix = file_name.split('.', 1)[1]
                if '-' in suffix:
                    violation = {
                        'type': 'dockerfile_naming',
                        'path': str(relative_path),
                        'issue': f"Dockerfile suffix '{suffix}' uses kebab-case",
                        'suggestion': file_name.replace('-', '_'),
                        'severity': 'medium'
                    }
                    self.violations.append(violation)

    def check_python_code_naming(self):
        """Check Python code naming conventions."""
        print("  🐍 Checking Python code naming...")
        
        python_files = list(self.project_root.rglob('*.py'))
        
        for python_file in python_files:
            # Skip certain directories
            if any(skip_dir in str(python_file) for skip_dir in self.skip_dirs):
                continue
                
            try:
                self._check_python_file_content(python_file)
            except Exception as e:
                print(f"    ⚠️  Warning: Could not analyze {python_file}: {e}")

    def _is_python_package_dir(self, dir_path: Path) -> bool:
        """Check if directory is a Python package."""
        # Check if it's under src/ or contains __init__.py
        return (
            'src' in str(dir_path) or 
            (dir_path / '__init__.py').exists() or
            any(f.suffix == '.py' for f in dir_path.iterdir() if f.is_file())
        )

    def _check_python_file_naming(self, relative_path: Path, file_name: str):
        """Check Python file naming conventions."""
        if '-' in file_name:
            violation = {
                'type': 'python_file_naming',
                'path': str(relative_path),
                'issue': f"Python file '{file_name}' uses kebab-case",
                'suggestion': file_name.replace('-', '_'),
                'severity': 'high'
            }
            self.violations.append(violation)

    def _check_test_file_naming(self, relative_path: Path, file_name: str):
        """Check test file naming conventions."""
        # Test files should start with test_ or end with _test.py
        # Prefer test_ prefix (pytest convention)
        if file_name.endswith('.py') and 'test' in file_name.lower():
            if not file_name.startswith('test_') and not file_name.endswith('_test.py'):
                # Check if it's in a test directory and contains test code
                if 'test' in str(relative_path).lower():
                    violation = {
                        'type': 'test_file_naming',
                        'path': str(relative_path),
                        'issue': f"Test file '{file_name}' doesn't follow test_*.py pattern",
                        'suggestion': f"test_{file_name}" if not file_name.startswith('test') else file_name,
                        'severity': 'medium'
                    }
                    self.violations.append(violation)

    def _check_python_file_content(self, file_path: Path):
        """Check naming conventions in Python file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            relative_path = file_path.relative_to(self.project_root)
            
            # Look for class definitions
            class_matches = re.finditer(r'^class\s+(\w+)', content, re.MULTILINE)
            for match in class_matches:
                class_name = match.group(1)
                if not self.class_pattern.match(class_name):
                    violation = {
                        'type': 'class_naming',
                        'path': str(relative_path),
                        'issue': f"Class '{class_name}' doesn't follow PascalCase",
                        'suggestion': self._to_pascal_case(class_name),
                        'severity': 'high'
                    }
                    self.violations.append(violation)
                    
        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            pass

    def _to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        return ''.join(word.capitalize() for word in re.split(r'[_\-]', name))

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Convert PascalCase/camelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def apply_fixes(self):
        """Apply fixes for naming convention violations."""
        if not self.fix_mode:
            print("⚠️  Fix mode not enabled. Use --fix to apply changes.")
            return
            
        print("🔧 Applying fixes...")
        
        # Group violations by type for efficient processing
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation['type']
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
        
        # Apply fixes by type
        for vtype, violations in violations_by_type.items():
            if vtype == 'yaml_extension':
                self._fix_yaml_extensions(violations)
            elif vtype == 'directory_naming':
                self._fix_directory_naming(violations)
            elif vtype == 'dockerfile_naming':
                self._fix_dockerfile_naming(violations)
            elif vtype == 'python_file_naming':
                self._fix_python_file_naming(violations)

    def _fix_yaml_extensions(self, violations: List[Dict]):
        """Fix YAML file extensions."""
        for violation in violations:
            old_path = self.project_root / violation['path']
            new_path = old_path.with_suffix('.yaml')
            
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                self.fixes_applied.append({
                    'type': 'yaml_extension',
                    'old': str(violation['path']),
                    'new': str(new_path.relative_to(self.project_root))
                })
                print(f"  ✅ Renamed {violation['path']} → {new_path.name}")

    def _fix_directory_naming(self, violations: List[Dict]):
        """Fix directory naming violations."""
        # Sort by depth (deepest first) to avoid conflicts
        violations.sort(key=lambda v: v['path'].count('/'), reverse=True)
        
        for violation in violations:
            old_path = self.project_root / violation['path']
            new_name = violation['suggestion']
            new_path = old_path.parent / new_name
            
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                self.fixes_applied.append({
                    'type': 'directory_naming',
                    'old': str(violation['path']),
                    'new': str(new_path.relative_to(self.project_root))
                })
                print(f"  ✅ Renamed directory {old_path.name} → {new_name}")

    def _fix_dockerfile_naming(self, violations: List[Dict]):
        """Fix Dockerfile naming violations."""
        for violation in violations:
            old_path = self.project_root / violation['path']
            new_name = violation['suggestion']
            new_path = old_path.parent / new_name
            
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                self.fixes_applied.append({
                    'type': 'dockerfile_naming',
                    'old': str(violation['path']),
                    'new': str(new_path.relative_to(self.project_root))
                })
                print(f"  ✅ Renamed {old_path.name} → {new_name}")

    def _fix_python_file_naming(self, violations: List[Dict]):
        """Fix Python file naming violations."""
        for violation in violations:
            old_path = self.project_root / violation['path']
            new_name = violation['suggestion']
            new_path = old_path.parent / new_name
            
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                self.fixes_applied.append({
                    'type': 'python_file_naming',
                    'old': str(violation['path']),
                    'new': str(new_path.relative_to(self.project_root))
                })
                print(f"  ✅ Renamed {old_path.name} → {new_name}")

    def report_violations(self):
        """Report all found violations."""
        print(f"\n❌ Found {len(self.violations)} naming convention violations:")
        
        # Group by severity
        high_severity = [v for v in self.violations if v.get('severity') == 'high']
        medium_severity = [v for v in self.violations if v.get('severity') == 'medium']
        low_severity = [v for v in self.violations if v.get('severity') == 'low']
        
        if high_severity:
            print(f"\n🚨 HIGH SEVERITY ({len(high_severity)} violations):")
            for violation in high_severity:
                print(f"  📍 {violation['path']}")
                print(f"     Issue: {violation['issue']}")
                print(f"     Suggestion: {violation['suggestion']}")
                print()
        
        if medium_severity:
            print(f"\n⚠️  MEDIUM SEVERITY ({len(medium_severity)} violations):")
            for violation in medium_severity:
                print(f"  📍 {violation['path']}")
                print(f"     Issue: {violation['issue']}")
                print(f"     Suggestion: {violation['suggestion']}")
                print()
        
        if low_severity:
            print(f"\n💡 LOW SEVERITY ({len(low_severity)} violations):")
            for violation in low_severity:
                print(f"  📍 {violation['path']}")
                print(f"     Issue: {violation['issue']}")
                print(f"     Suggestion: {violation['suggestion']}")
                print()

    def generate_summary_report(self):
        """Generate a summary report."""
        print("\n📊 NAMING CONVENTION ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total violations found: {len(self.violations)}")
        print(f"Fixes applied: {len(self.fixes_applied)}")
        
        # Breakdown by type
        violation_types = {}
        for violation in self.violations:
            vtype = violation['type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        if violation_types:
            print("\nViolations by type:")
            for vtype, count in sorted(violation_types.items()):
                print(f"  {vtype}: {count}")
        
        if self.fixes_applied:
            print(f"\n✅ Applied {len(self.fixes_applied)} fixes successfully")
        
        remaining_violations = len(self.violations) - len(self.fixes_applied)
        if remaining_violations > 0:
            print(f"\n⚠️  {remaining_violations} violations require manual attention")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check and fix naming convention violations in Pynomaly project"
    )
    parser.add_argument(
        '--fix', 
        action='store_true', 
        help="Apply fixes automatically (use with caution)"
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    if not (args.project_root / 'pyproject.toml').exists():
        print("❌ Error: Not in Pynomaly project root directory")
        print("   Expected to find pyproject.toml")
        sys.exit(1)
    
    checker = NamingConventionChecker(args.project_root, fix_mode=args.fix)
    
    # Check for violations
    has_violations = checker.check_all_violations()
    
    if has_violations:
        if args.fix:
            checker.apply_fixes()
        else:
            print("\n💡 To fix these violations automatically, run:")
            print("   python scripts/check_naming_conventions.py --fix")
    
    # Generate summary
    checker.generate_summary_report()
    
    # Return appropriate exit code
    remaining_violations = len(checker.violations) - len(checker.fixes_applied)
    sys.exit(remaining_violations if remaining_violations > 0 else 0)


if __name__ == "__main__":
    main()