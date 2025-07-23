#!/usr/bin/env python3
"""CLI interface for test domain leakage detector."""

import click
import sys
from pathlib import Path
from typing import Optional
import yaml
import re
from dataclasses import dataclass, field
from enum import Enum


class ViolationType(Enum):
    PACKAGE_CROSS_IMPORT = "package_cross_import"
    SYSTEM_TEST_PACKAGE_IMPORT = "system_test_package_import"
    REPO_TEST_PACKAGE_IMPORT = "repo_test_package_import"


class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TestViolation:
    file_path: Path
    line_number: int
    line_content: str
    violation_type: ViolationType
    severity: Severity
    message: str
    import_statement: str
    suggested_fix: Optional[str] = None


@dataclass
class ScanResult:
    violations: list[TestViolation] = field(default_factory=list)
    files_scanned: int = 0
    test_files_found: int = 0


class TestDomainLeakageDetector:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".domain-boundaries.yaml")
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from .domain-boundaries.yaml"""
        if not self.config_path.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration for test domain leakage detection"""
        return {
            'testing': {
                'rules': [
                    {
                        'name': 'no_cross_package_imports_in_package_tests',
                        'description': 'Package tests must not import from other packages',
                        'severity': 'critical',
                        'scope': 'src/packages/*/tests/**/*.py'
                    }
                ]
            }
        }

    def scan_directory(self, directory: Path) -> ScanResult:
        """Scan directory for test domain leakage violations"""
        result = ScanResult()
        
        # Find all test files
        test_files = list(directory.rglob("**/tests/**/*.py"))
        system_test_files = list(directory.rglob("**/system_tests/**/*.py"))
        repo_test_files = list((directory / "tests").rglob("**/*.py")) if (directory / "tests").exists() else []
        
        all_test_files = test_files + system_test_files + repo_test_files
        result.test_files_found = len(all_test_files)
        
        for test_file in all_test_files:
            result.files_scanned += 1
            violations = self._scan_test_file(test_file, directory)
            result.violations.extend(violations)
            
        return result

    def _scan_test_file(self, file_path: Path, root_path: Path) -> list[TestViolation]:
        """Scan a single test file for domain leakage violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return violations

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check for different types of violations
            violations.extend(self._check_package_test_violations(
                file_path, line_num, line, root_path))
            violations.extend(self._check_system_test_violations(
                file_path, line_num, line, root_path))
            violations.extend(self._check_repo_test_violations(
                file_path, line_num, line, root_path))
                
        return violations

    def _check_package_test_violations(self, file_path: Path, line_num: int, 
                                     line: str, root_path: Path) -> list[TestViolation]:
        """Check for cross-package imports in package tests"""
        violations = []
        
        # Only check files in package test directories
        if "/tests/" not in str(file_path) or "system_tests" in str(file_path):
            return violations
            
        # Check for absolute imports from other packages
        patterns = [
            r'from\s+(?!\.{1,2}|test_|conftest)([a-zA-Z_][\w_]*)\.\S*\s+import',
            r'import\s+(?!test_|conftest)([a-zA-Z_][\w_]*)\.\S*'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match and not self._is_allowed_import(line):
                violations.append(TestViolation(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line,
                    violation_type=ViolationType.PACKAGE_CROSS_IMPORT,
                    severity=Severity.CRITICAL,
                    message="Package tests must only import from their own package - use relative imports",
                    import_statement=match.group(0),
                    suggested_fix=self._suggest_relative_import_fix(match.group(0))
                ))
                
        return violations

    def _check_system_test_violations(self, file_path: Path, line_num: int,
                                    line: str, root_path: Path) -> list[TestViolation]:
        """Check for package imports in system tests"""
        violations = []
        
        if "system_tests" not in str(file_path):
            return violations
            
        # Check for domain package imports
        patterns = [
            r'from\s+(ai|data|finance|software|tools)\.\S*\s+import',
            r'import\s+(ai|data|finance|software|tools)\.\S*'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                violations.append(TestViolation(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line,
                    violation_type=ViolationType.SYSTEM_TEST_PACKAGE_IMPORT,
                    severity=Severity.CRITICAL,
                    message="System tests must not import directly from domain packages - use public APIs or test fixtures",
                    import_statement=match.group(0)
                ))
                
        return violations

    def _check_repo_test_violations(self, file_path: Path, line_num: int,
                                  line: str, root_path: Path) -> list[TestViolation]:
        """Check for package imports in repository-level tests"""
        violations = []
        
        # Check if this is a repository-level test (in tests/ at root)
        relative_path = file_path.relative_to(root_path)
        if not str(relative_path).startswith("tests/"):
            return violations
            
        # Check for src.packages imports or sys.path manipulation
        patterns = [
            r'from\s+src\.packages\.\S*\s+import',
            r'import\s+src\.packages\.\S*',
            r'sys\.path\.insert.*src/packages'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                violations.append(TestViolation(
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line,
                    violation_type=ViolationType.REPO_TEST_PACKAGE_IMPORT,
                    severity=Severity.CRITICAL,
                    message="Repository tests must not import from src.packages - use repository-level utilities only",
                    import_statement=match.group(0)
                ))
                
        return violations

    def _is_allowed_import(self, line: str) -> bool:
        """Check if an import is allowed based on exceptions"""
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

    def _suggest_relative_import_fix(self, import_statement: str) -> str:
        """Suggest a relative import fix"""
        if import_statement.startswith('from '):
            return import_statement.replace('from ', 'from .', 1)
        return f"Use relative import: from .{import_statement.split('.')[0]} import ..."


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Test Domain Leakage Detector - Prevent test domain leakage."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config


@cli.command()
@click.option('--path', '-p', type=click.Path(exists=True), default='.', 
              help='Path to scan (default: current directory)')
@click.option('--format', '-f', type=click.Choice(['console', 'json']), 
              default='console', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file (for json)')
@click.option('--strict', is_flag=True, help='Exit with error code on violations')
@click.option('--show-fixes', is_flag=True, help='Show suggested fixes')
@click.pass_context
def scan(ctx: click.Context, path: str, format: str, output: Optional[str], 
         strict: bool, show_fixes: bool) -> None:
    """Scan for test domain leakage violations."""
    verbose = ctx.obj['verbose']
    config_path = ctx.obj['config']
    
    # Initialize detector
    detector = TestDomainLeakageDetector(
        Path(config_path) if config_path else None
    )
    
    # Scan directory
    scan_path = Path(path).resolve()
    if verbose:
        click.echo(f"Scanning: {scan_path}")
        
    result = detector.scan_directory(scan_path)
    
    if verbose:
        click.echo(f"Files scanned: {result.files_scanned}")
        click.echo(f"Test files found: {result.test_files_found}")
        click.echo(f"Violations found: {len(result.violations)}")
    
    # Output results
    if format == 'console':
        _print_console_report(result, show_fixes)
    elif format == 'json':
        _print_json_report(result, output)
    
    # Exit code for CI/CD
    if strict and result.violations:
        critical_violations = [v for v in result.violations if v.severity == Severity.CRITICAL]
        if critical_violations:
            sys.exit(1)


def _print_console_report(result: ScanResult, show_fixes: bool) -> None:
    """Print console report"""
    if not result.violations:
        click.echo("✅ No test domain leakage violations found!")
        return
    
    click.echo(f"❌ Found {len(result.violations)} test domain leakage violations:")
    click.echo()
    
    # Group by severity
    by_severity = {}
    for violation in result.violations:
        severity = violation.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(violation)
    
    for severity in ['critical', 'warning', 'info']:
        violations = by_severity.get(severity, [])
        if not violations:
            continue
            
        severity_color = {'critical': 'red', 'warning': 'yellow', 'info': 'blue'}
        click.echo(click.style(f"{severity.upper()}: {len(violations)} violations", 
                              fg=severity_color[severity], bold=True))
        
        for violation in violations:
            click.echo(f"  📁 {violation.file_path}:{violation.line_number}")
            click.echo(f"     {violation.message}")
            click.echo(f"     Import: {violation.import_statement}")
            if show_fixes and violation.suggested_fix:
                click.echo(f"     💡 Suggested fix: {violation.suggested_fix}")
            click.echo()


def _print_json_report(result: ScanResult, output: Optional[str]) -> None:
    """Print JSON report"""
    import json
    
    report = {
        'summary': {
            'files_scanned': result.files_scanned,
            'test_files_found': result.test_files_found,
            'violations_found': len(result.violations)
        },
        'violations': [
            {
                'file_path': str(v.file_path),
                'line_number': v.line_number,
                'line_content': v.line_content,
                'violation_type': v.violation_type.value,
                'severity': v.severity.value,
                'message': v.message,
                'import_statement': v.import_statement,
                'suggested_fix': v.suggested_fix
            }
            for v in result.violations
        ]
    }
    
    json_output = json.dumps(report, indent=2)
    
    if output:
        with open(output, 'w') as f:
            f.write(json_output)
        click.echo(f"Report saved to: {output}")
    else:
        click.echo(json_output)


@cli.command()
@click.option('--path', '-p', type=click.Path(exists=True), default='.', 
              help='Path to fix (default: current directory)')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def fix(ctx: click.Context, path: str, dry_run: bool, verbose: bool) -> None:
    """Fix common test domain leakage violations automatically."""
    from .fixer import TestDomainLeakageFixer
    
    if verbose or ctx.obj['verbose']:
        click.echo(f"Fixing test domain leakage violations in: {path}")
        if dry_run:
            click.echo("DRY RUN MODE - No changes will be made")
    
    # Initialize fixer
    fixer = TestDomainLeakageFixer(dry_run=dry_run)
    
    # Fix violations
    fix_path = Path(path).resolve()
    summary = fixer.fix_directory(fix_path)
    
    # Report results
    click.echo(f"\n📊 Fix Summary:")
    click.echo(f"Files modified: {summary.files_modified}")
    click.echo(f"Total fixes applied: {summary.total_fixes}")
    
    if summary.errors:
        click.echo(f"Errors encountered: {len(summary.errors)}")
        for error in summary.errors:
            click.echo(f"  ❌ {error}")
    
    if (verbose or ctx.obj['verbose']) and summary.fixes_applied:
        click.echo(f"\n🔧 Fixes Applied:")
        for i, fix in enumerate(summary.fixes_applied, 1):
            click.echo(f"\n{i}. {fix.file_path}:{fix.line_number}")
            click.echo(f"   Type: {fix.fix_type}")
            click.echo(f"   Description: {fix.description}")
            click.echo(f"   Before: {fix.original_line}")
            click.echo(f"   After:  {fix.fixed_line}")
    
    if summary.total_fixes == 0:
        click.echo("✅ No test domain leakage violations found to fix!")
    else:
        action = "would be fixed" if dry_run else "fixed"
        click.echo(f"✅ {summary.total_fixes} violations {action}!")


if __name__ == '__main__':
    cli()