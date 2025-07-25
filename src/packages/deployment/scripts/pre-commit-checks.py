#!/usr/bin/env python3
"""
Pre-commit hooks for maintaining code quality and domain boundaries.

This script provides various pre-commit checks including:
- Domain boundary violation detection
- Import statement validation
- Code quality checks
- Security scanning
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

# Import the boundary violation detector
sys.path.insert(0, os.path.dirname(__file__))
from boundary_violation_check import BoundaryViolationDetector, ReportFormatter


class PreCommitChecker:
    """Pre-commit check orchestrator."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.checks_passed = True
        self.results = {}
    
    def run_all_checks(self, staged_files: List[str] = None) -> bool:
        """Run all pre-commit checks."""
        print("🔍 Running pre-commit checks...")
        print("=" * 50)
        
        # Get staged files if not provided
        if staged_files is None:
            staged_files = self._get_staged_files()
        
        # Filter for Python files
        python_files = [f for f in staged_files if f.endswith('.py')]
        
        if not python_files:
            print("ℹ️  No Python files to check")
            return True
        
        print(f"📁 Checking {len(python_files)} Python files...")
        print()
        
        # Run individual checks
        self._check_boundary_violations(python_files)
        self._check_import_standards(python_files)
        self._check_security_issues(python_files)
        self._check_code_quality(python_files)
        
        # Summary
        self._print_summary()
        
        return self.checks_passed
    
    def _get_staged_files(self) -> List[str]:
        """Get list of staged files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except subprocess.CalledProcessError:
            return []
    
    def _check_boundary_violations(self, files: List[str]) -> None:
        """Check for domain boundary violations."""
        print("🏗️  Checking domain boundaries...")
        
        try:
            # Create temporary directory with only staged files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_packages = Path(temp_dir) / "packages"
                temp_packages.mkdir(parents=True)
                
                # Copy staged files to temp directory
                for file_path in files:
                    if "packages" in file_path:
                        src_path = self.repo_root / file_path
                        if src_path.exists():
                            # Recreate directory structure
                            rel_path = Path(file_path)
                            temp_file = Path(temp_dir) / rel_path
                            temp_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Copy file
                            temp_file.write_text(src_path.read_text())
                
                # Run boundary check
                detector = BoundaryViolationDetector()
                report = detector.scan_directory(str(temp_packages))
                
                self.results['boundary_violations'] = {
                    'total_violations': report.violation_count,
                    'critical_violations': len(report.critical_violations),
                    'high_violations': len(report.high_violations)
                }
                
                if report.violation_count > 0:
                    print(f"   ❌ Found {report.violation_count} boundary violations")
                    
                    # Show critical and high violations
                    critical_and_high = report.critical_violations + report.high_violations
                    for violation in critical_and_high[:5]:  # Show first 5
                        print(f"      • {violation.violation_type}: {violation.import_statement}")
                        for location in violation.locations[:1]:  # Show first location
                            print(f"        📍 {location.file_path}:{location.line_number}")
                    
                    if len(critical_and_high) > 5:
                        print(f"      ... and {len(critical_and_high) - 5} more")
                    
                    self.checks_passed = False
                else:
                    print("   ✅ No boundary violations found")
        
        except Exception as e:
            print(f"   ⚠️  Boundary check failed: {e}")
            self.results['boundary_violations'] = {'error': str(e)}
        
        print()
    
    def _check_import_standards(self, files: List[str]) -> None:
        """Check import statement standards."""
        print("📦 Checking import standards...")
        
        issues = []
        
        for file_path in files:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Check for discouraged import patterns
                    if line.startswith('from') and 'import *' in line:
                        issues.append(f"{file_path}:{i} - Wildcard import discouraged: {line}")
                    
                    # Check for relative imports going up too many levels
                    if line.startswith('from ...') and line.count('.') > 3:
                        issues.append(f"{file_path}:{i} - Deep relative import: {line}")
                    
                    # Check for direct package imports without using shared interfaces
                    if 'from src.packages.' in line and 'shared' not in line:
                        issues.append(f"{file_path}:{i} - Direct package import: {line}")
            
            except Exception:
                continue
        
        self.results['import_standards'] = {'issues': len(issues)}
        
        if issues:
            print(f"   ❌ Found {len(issues)} import issues")
            for issue in issues[:3]:  # Show first 3
                print(f"      • {issue}")
            if len(issues) > 3:
                print(f"      ... and {len(issues) - 3} more")
            self.checks_passed = False
        else:
            print("   ✅ Import standards compliance")
        
        print()
    
    def _check_security_issues(self, files: List[str]) -> None:
        """Check for security issues."""
        print("🔒 Checking security issues...")
        
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'eval\s*\(', 'Use of eval()'),
            (r'exec\s*\(', 'Use of exec()'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'Shell injection risk'),
        ]
        
        issues = []
        
        for file_path in files:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                import re
                for i, line in enumerate(lines, 1):
                    for pattern, description in security_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append(f"{file_path}:{i} - {description}: {line.strip()}")
            
            except Exception:
                continue
        
        self.results['security_issues'] = {'issues': len(issues)}
        
        if issues:
            print(f"   ❌ Found {len(issues)} security issues")
            for issue in issues[:3]:  # Show first 3
                print(f"      • {issue}")
            if len(issues) > 3:
                print(f"      ... and {len(issues) - 3} more")
            self.checks_passed = False
        else:
            print("   ✅ No security issues found")
        
        print()
    
    def _check_code_quality(self, files: List[str]) -> None:
        """Check code quality metrics."""
        print("📊 Checking code quality...")
        
        issues = []
        
        for file_path in files:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check file length
                if len(lines) > 500:
                    issues.append(f"{file_path} - File too long: {len(lines)} lines")
                
                # Check for very long lines
                long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120]
                if long_lines:
                    issues.append(f"{file_path} - {len(long_lines)} lines exceed 120 characters")
                
                # Check for TODO/FIXME comments
                todo_lines = [i for i, line in enumerate(lines, 1) 
                             if 'TODO' in line.upper() or 'FIXME' in line.upper()]
                if len(todo_lines) > 5:
                    issues.append(f"{file_path} - {len(todo_lines)} TODO/FIXME comments")
            
            except Exception:
                continue
        
        self.results['code_quality'] = {'issues': len(issues)}
        
        if issues:
            print(f"   ⚠️  Found {len(issues)} code quality issues")
            for issue in issues[:3]:  # Show first 3
                print(f"      • {issue}")
            if len(issues) > 3:
                print(f"      ... and {len(issues) - 3} more")
            # Don't fail on code quality issues, just warn
        else:
            print("   ✅ Code quality checks passed")
        
        print()
    
    def _print_summary(self) -> None:
        """Print summary of all checks."""
        print("📋 Pre-commit Check Summary")
        print("=" * 30)
        
        if self.checks_passed:
            print("✅ All critical checks passed!")
            print("🚀 Ready to commit")
        else:
            print("❌ Some checks failed!")
            print("🛑 Please fix issues before committing")
        
        print()
        
        # Print detailed results
        for check_name, results in self.results.items():
            if 'error' in results:
                print(f"   {check_name}: Error - {results['error']}")
            elif 'issues' in results:
                count = results['issues']
                status = "✅" if count == 0 else "❌" if count > 0 else "⚠️"
                print(f"   {check_name}: {status} {count} issues")
            else:
                # For boundary violations
                total = results.get('total_violations', 0)
                critical = results.get('critical_violations', 0)
                high = results.get('high_violations', 0)
                
                if total == 0:
                    print(f"   {check_name}: ✅ No violations")
                else:
                    print(f"   {check_name}: ❌ {total} violations ({critical} critical, {high} high)")


def install_pre_commit_hook():
    """Install the pre-commit hook."""
    repo_root = Path(__file__).parent.parent.parent.parent
    hooks_dir = repo_root / ".git" / "hooks"
    hook_file = hooks_dir / "pre-commit"
    
    if not hooks_dir.exists():
        print("❌ Not a git repository or hooks directory not found")
        return False
    
    hook_content = f"""#!/bin/bash
# Pre-commit hook for domain boundary and code quality checks

echo "Running pre-commit checks..."

# Run the Python pre-commit checker
python "{__file__}" --staged

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Pre-commit checks failed!"
    echo "Please fix the issues above before committing."
    echo ""
    echo "To bypass these checks (not recommended):"
    echo "  git commit --no-verify"
    echo ""
    exit 1
fi

echo "✅ Pre-commit checks passed!"
exit 0
"""
    
    try:
        hook_file.write_text(hook_content)
        hook_file.chmod(0o755)  # Make executable
        print(f"✅ Pre-commit hook installed at {hook_file}")
        print("   Commit checks will now run automatically")
        return True
    except Exception as e:
        print(f"❌ Failed to install pre-commit hook: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pre-commit checks for monorepo")
    parser.add_argument("--staged", action="store_true", 
                       help="Check only staged files")
    parser.add_argument("--install", action="store_true",
                       help="Install pre-commit hook")
    parser.add_argument("--files", nargs="*",
                       help="Specific files to check")
    
    args = parser.parse_args()
    
    if args.install:
        success = install_pre_commit_hook()
        sys.exit(0 if success else 1)
    
    # Determine repository root
    repo_root = Path(__file__).parent.parent.parent.parent
    
    # Run checks
    checker = PreCommitChecker(str(repo_root))
    
    if args.files:
        success = checker.run_all_checks(args.files)
    elif args.staged:
        success = checker.run_all_checks()
    else:
        # Check all Python files in packages
        packages_dir = repo_root / "src" / "packages"
        python_files = []
        if packages_dir.exists():
            python_files = [str(p.relative_to(repo_root)) 
                          for p in packages_dir.rglob("*.py")]
        success = checker.run_all_checks(python_files)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()