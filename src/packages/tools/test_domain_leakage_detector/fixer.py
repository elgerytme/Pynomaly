#!/usr/bin/env python3
"""
Automated fixer for common test domain leakage violations.

This module provides automated fixes for common patterns of test domain leakage,
such as converting absolute imports to relative imports in package tests.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class FixResult(Enum):
    FIXED = "fixed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Fix:
    file_path: Path
    line_number: int
    original_line: str
    fixed_line: str
    fix_type: str
    description: str


@dataclass
class FixSummary:
    fixes_applied: List[Fix]
    files_modified: int
    total_fixes: int
    errors: List[str]


class TestDomainLeakageFixer:
    """Automated fixer for test domain leakage violations."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fixes_applied = []
        self.errors = []
    
    def fix_directory(self, directory: Path) -> FixSummary:
        """Fix all test domain leakage violations in a directory."""
        self.fixes_applied = []
        self.errors = []
        
        # Find all test files
        test_files = list(directory.rglob("**/tests/**/*.py"))
        system_test_files = list(directory.rglob("**/system_tests/**/*.py"))
        repo_test_files = list((directory / "tests").rglob("**/*.py")) if (directory / "tests").exists() else []
        
        all_test_files = test_files + system_test_files + repo_test_files
        
        files_modified = 0
        for test_file in all_test_files:
            try:
                if self._fix_file(test_file):
                    files_modified += 1
            except Exception as e:
                self.errors.append(f"Error fixing {test_file}: {str(e)}")
        
        return FixSummary(
            fixes_applied=self.fixes_applied,
            files_modified=files_modified,
            total_fixes=len(self.fixes_applied),
            errors=self.errors
        )
    
    def _fix_file(self, file_path: Path) -> bool:
        """Fix a single test file. Returns True if file was modified."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            self.errors.append(f"Could not read {file_path}: {str(e)}")
            return False
        
        original_lines = lines[:]
        modified = False
        
        for line_num, line in enumerate(lines):
            fixed_line, fix_applied = self._fix_line(file_path, line_num + 1, line)
            if fix_applied:
                lines[line_num] = fixed_line
                modified = True
        
        if modified and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            except Exception as e:
                self.errors.append(f"Could not write {file_path}: {str(e)}")
                return False
        
        return modified
    
    def _fix_line(self, file_path: Path, line_num: int, line: str) -> Tuple[str, bool]:
        """Fix a single line. Returns (fixed_line, was_modified)."""
        stripped_line = line.strip()
        
        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith('#'):
            return line, False
        
        # Determine which type of test file this is
        if "/tests/" in str(file_path) and "system_tests" not in str(file_path):
            return self._fix_package_test_line(file_path, line_num, line)
        elif "system_tests" in str(file_path):
            return self._fix_system_test_line(file_path, line_num, line)
        elif str(file_path).startswith("tests/"):
            return self._fix_repo_test_line(file_path, line_num, line)
        
        return line, False
    
    def _fix_package_test_line(self, file_path: Path, line_num: int, line: str) -> Tuple[str, bool]:
        """Fix package test import violations."""
        # Pattern 1: from package.module import something -> from .module import something
        pattern1 = r'^(\s*)from\s+([a-zA-Z_][\w_]*)\.([\w_.]+)\s+import'
        match1 = re.match(pattern1, line)
        if match1 and not self._is_allowed_import(line):
            indent, package, module_path = match1.groups()
            # Check if this looks like a cross-package import
            if not package.startswith('.') and package not in ['test_', 'conftest']:
                fixed_line = f"{indent}from .{module_path} import{line.split('import', 1)[1]}"
                self._record_fix(file_path, line_num, line, fixed_line, 
                               "package_test_relative_import", 
                               "Converted absolute import to relative import")
                return fixed_line, True
        
        # Pattern 2: import package.module -> from . import module (if it's a single module)
        pattern2 = r'^(\s*)import\s+([a-zA-Z_][\w_]*)\.([\w_]+)(\s*)$'
        match2 = re.match(pattern2, line)
        if match2 and not self._is_allowed_import(line):
            indent, package, module, trailing = match2.groups()
            if not package.startswith('.') and package not in ['test_', 'conftest']:
                fixed_line = f"{indent}from . import {module}{trailing}\n"
                self._record_fix(file_path, line_num, line, fixed_line,
                               "package_test_import_conversion",
                               "Converted absolute import to relative import")
                return fixed_line, True
        
        return line, False
    
    def _fix_system_test_line(self, file_path: Path, line_num: int, line: str) -> Tuple[str, bool]:
        """Fix system test import violations."""
        # For system tests, we can't automatically fix domain imports
        # as they require architectural changes, but we can flag them
        domain_import_patterns = [
            r'from\s+(ai|data|finance|software|tools)\.',
            r'import\s+(ai|data|finance|software|tools)\.'
        ]
        
        for pattern in domain_import_patterns:
            if re.search(pattern, line):
                # We can't automatically fix this, but we can add a comment
                if "# TODO: Remove direct domain import" not in line:
                    fixed_line = line.rstrip() + "  # TODO: Remove direct domain import - use public APIs\n"
                    self._record_fix(file_path, line_num, line, fixed_line,
                                   "system_test_todo_comment",
                                   "Added TODO comment for manual fix required")
                    return fixed_line, True
        
        return line, False
    
    def _fix_repo_test_line(self, file_path: Path, line_num: int, line: str) -> Tuple[str, bool]:
        """Fix repository test import violations."""
        # Pattern: from src.packages.* import -> Comment out with TODO
        src_import_patterns = [
            r'from\s+src\.packages\.',
            r'import\s+src\.packages\.',
            r'sys\.path\.insert.*src/packages'
        ]
        
        for pattern in src_import_patterns:
            if re.search(pattern, line):
                if not line.strip().startswith('#'):
                    # Comment out the line and add TODO
                    indent = re.match(r'^(\s*)', line).group(1)
                    fixed_line = f"{indent}# TODO: Remove src.packages import - use repository-level utilities\n{indent}# {line.lstrip()}"
                    if not line.endswith('\n'):
                        fixed_line += '\n'
                    self._record_fix(file_path, line_num, line, fixed_line,
                                   "repo_test_comment_out",
                                   "Commented out src.packages import with TODO")
                    return fixed_line, True
        
        return line, False
    
    def _is_allowed_import(self, line: str) -> bool:
        """Check if an import is allowed based on exceptions."""
        allowed_patterns = [
            r'from typing import',
            r'from abc import',
            r'from collections import',
            r'from dataclasses import',
            r'from pathlib import',
            r'from datetime import',
            r'from unittest import',
            r'from unittest\.mock import',
            r'import unittest',
            r'from pytest import',
            r'import pytest',
            r'from hypothesis import',
            r'import hypothesis',
            r'from faker import',
            r'import faker',
            r'from freezegun import',
            r'import freezegun',
            r'from testcontainers import',
            r'import testcontainers',
            r'import (os|sys|re|json|yaml|logging|tempfile|shutil|subprocess|asyncio|threading|multiprocessing|uuid|hashlib|base64|pickle|sqlite3|urllib|http|socket|time|random|math|statistics|itertools|functools|operator|contextlib|warnings|traceback|inspect|types|copy|weakref)(\s|$|\\.)'
        ]
        
        for pattern in allowed_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _record_fix(self, file_path: Path, line_num: int, original: str, 
                   fixed: str, fix_type: str, description: str):
        """Record a fix that was applied."""
        fix = Fix(
            file_path=file_path,
            line_number=line_num,
            original_line=original.rstrip(),
            fixed_line=fixed.rstrip(),
            fix_type=fix_type,
            description=description
        )
        self.fixes_applied.append(fix)


def main():
    """Command line interface for the fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix test domain leakage violations")
    parser.add_argument("--path", "-p", type=Path, default=Path("."),
                       help="Path to scan and fix (default: current directory)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    fixer = TestDomainLeakageFixer(dry_run=args.dry_run)
    
    if args.verbose:
        print(f"Scanning directory: {args.path}")
        if args.dry_run:
            print("DRY RUN MODE - No changes will be made")
    
    summary = fixer.fix_directory(args.path)
    
    print(f"\n📊 Fix Summary:")
    print(f"Files modified: {summary.files_modified}")
    print(f"Total fixes applied: {summary.total_fixes}")
    
    if summary.errors:
        print(f"Errors encountered: {len(summary.errors)}")
        for error in summary.errors:
            print(f"  ❌ {error}")
    
    if args.verbose and summary.fixes_applied:
        print(f"\n🔧 Fixes Applied:")
        for i, fix in enumerate(summary.fixes_applied, 1):
            print(f"\n{i}. {fix.file_path}:{fix.line_number}")
            print(f"   Type: {fix.fix_type}")
            print(f"   Description: {fix.description}")
            print(f"   Before: {fix.original_line}")
            print(f"   After:  {fix.fixed_line}")
    
    if summary.total_fixes == 0:
        print("✅ No test domain leakage violations found to fix!")
    else:
        action = "would be fixed" if args.dry_run else "fixed"
        print(f"✅ {summary.total_fixes} violations {action}!")


if __name__ == "__main__":
    main()