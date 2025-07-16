#!/usr/bin/env python3
"""
Repository Organization Validator

This script validates that the repository follows the organization rules
defined in .project-rules/REPOSITORY_ORGANIZATION_RULES.md
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: str  # 'error', 'warning', 'info'
    category: str
    message: str
    file_path: Optional[str] = None
    suggested_action: Optional[str] = None


class RepositoryValidator:
    """Validates repository organization against defined rules."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results: List[ValidationResult] = []
        
        # Allowed directories in root
        self.allowed_root_dirs = {
            '.claude', '.github', '.hypothesis', '.project-rules', 
            '.ruff_cache', '.storybook', '.vscode',
            'docs', 'pkg', 'scripts', 'src'
        }
        
        # Allowed files in root
        self.allowed_root_files = {
            'README.md', 'CHANGELOG.md', 'LICENSE', 'pyproject.toml',
            '.gitignore', '.python-version'
        }
        
        # Prohibited patterns in root
        self.prohibited_root_patterns = [
            r'^test_.*',
            r'^.*_test\..*',
            r'^temp.*',
            r'^tmp.*',
            r'^scratch.*',
            r'^debug.*',
            r'^backup.*',
            r'^\.env.*',
            r'^env.*',
            r'^venv.*',
            r'^build.*',
            r'^dist.*',
            r'^.*\.egg-info$',
            r'^.*\.bak$',
            r'^.*\.backup$',
            r'^.*\.tmp$',
            r'^.*\.temp$',
        ]
        
        # Configuration file patterns that should be in scripts/config/
        self.config_patterns = [
            r'^\..*rc$',
            r'^\..*rc\..*',
            r'^\..*config.*',
            r'^.*\.config$',
            r'^.*\.conf$',
            r'^\.pre-commit.*',
            r'^\.buck.*',
            r'^\.docker.*',
            r'^\.eslint.*',
            r'^\.prettier.*',
            r'^\.style.*',
            r'^\.mutmut.*',
            r'^\.percy.*',
        ]

    def validate(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        self.results = []
        
        self._validate_root_directory()
        self._validate_directory_structure()
        self._validate_file_naming()
        self._validate_temporary_files()
        self._validate_configuration_placement()
        self._validate_documentation_structure()
        self._validate_source_organization()
        
        # Determine if validation passed
        has_errors = any(r.level == 'error' for r in self.results)
        return not has_errors, self.results

    def _validate_root_directory(self):
        """Validate root directory contents."""
        root_items = set(os.listdir(self.repo_root))
        
        # Check for prohibited files/directories
        for item in root_items:
            if os.path.isfile(self.repo_root / item):
                if item not in self.allowed_root_files:
                    # Check if it matches prohibited patterns
                    for pattern in self.prohibited_root_patterns:
                        if re.match(pattern, item):
                            self.results.append(ValidationResult(
                                level='error',
                                category='root_organization',
                                message=f"Prohibited file in root: {item}",
                                file_path=str(item),
                                suggested_action=f"Move {item} to appropriate directory or remove"
                            ))
                            break
                    else:
                        # Check if it's a configuration file
                        for pattern in self.config_patterns:
                            if re.match(pattern, item):
                                self.results.append(ValidationResult(
                                    level='error',
                                    category='config_placement',
                                    message=f"Configuration file in root: {item}",
                                    file_path=str(item),
                                    suggested_action=f"Move {item} to scripts/config/"
                                ))
                                break
                        else:
                            self.results.append(ValidationResult(
                                level='warning',
                                category='root_organization',
                                message=f"Unexpected file in root: {item}",
                                file_path=str(item),
                                suggested_action=f"Verify {item} belongs in root or move to appropriate location"
                            ))
            
            elif os.path.isdir(self.repo_root / item):
                if item not in self.allowed_root_dirs:
                    # Check for prohibited patterns
                    for pattern in self.prohibited_root_patterns:
                        if re.match(pattern, item):
                            self.results.append(ValidationResult(
                                level='error',
                                category='root_organization',
                                message=f"Prohibited directory in root: {item}/",
                                file_path=str(item),
                                suggested_action=f"Move {item}/ to appropriate location or remove"
                            ))
                            break
                    else:
                        self.results.append(ValidationResult(
                            level='error',
                            category='root_organization',
                            message=f"Unauthorized directory in root: {item}/",
                            file_path=str(item),
                            suggested_action=f"Move {item}/ to src/, scripts/, or docs/ as appropriate"
                        ))

    def _validate_directory_structure(self):
        """Validate overall directory structure."""
        required_dirs = ['docs', 'pkg', 'scripts', 'src']
        
        for required_dir in required_dirs:
            if not (self.repo_root / required_dir).exists():
                self.results.append(ValidationResult(
                    level='error',
                    category='structure',
                    message=f"Required directory missing: {required_dir}/",
                    suggested_action=f"Create {required_dir}/ directory"
                ))

    def _validate_file_naming(self):
        """Validate file naming conventions."""
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            for filename in files:
                # Skip hidden files and known exceptions
                if filename.startswith('.') and filename not in self.allowed_root_files:
                    continue
                
                # Check for spaces in filenames
                if ' ' in filename:
                    self.results.append(ValidationResult(
                        level='warning',
                        category='naming',
                        message=f"File contains spaces: {relative_root / filename}",
                        file_path=str(relative_root / filename),
                        suggested_action="Replace spaces with underscores or hyphens"
                    ))
                
                # Check for uppercase in Python files
                if filename.endswith('.py') and any(c.isupper() for c in filename[:-3]):
                    self.results.append(ValidationResult(
                        level='warning',
                        category='naming',
                        message=f"Python file should use snake_case: {relative_root / filename}",
                        file_path=str(relative_root / filename),
                        suggested_action="Rename to use snake_case"
                    ))

    def _validate_temporary_files(self):
        """Find and flag temporary files in inappropriate locations."""
        temp_patterns = [r'.*\.tmp$', r'.*\.temp$', r'.*\.bak$', r'.*\.backup$', r'^temp_.*', r'^tmp_.*']
        
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip allowed temporary locations
            if any(part in ['temp', 'tmp', 'cache', '.cache'] for part in relative_root.parts):
                continue
                
            for filename in files:
                for pattern in temp_patterns:
                    if re.match(pattern, filename):
                        self.results.append(ValidationResult(
                            level='warning',
                            category='temporary_files',
                            message=f"Temporary file in organized directory: {relative_root / filename}",
                            file_path=str(relative_root / filename),
                            suggested_action="Move to temp/ directory or remove"
                        ))
                        break

    def _validate_configuration_placement(self):
        """Validate configuration files are in the right place."""
        config_dir = self.repo_root / 'scripts' / 'config'
        
        if not config_dir.exists():
            self.results.append(ValidationResult(
                level='error',
                category='config_placement',
                message="Configuration directory missing: scripts/config/",
                suggested_action="Create scripts/config/ directory structure"
            ))
            return
        
        # Check for configuration files in wrong locations
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip the config directory itself
            if 'scripts/config' in str(relative_root):
                continue
                
            for filename in files:
                for pattern in self.config_patterns:
                    if re.match(pattern, filename):
                        self.results.append(ValidationResult(
                            level='error',
                            category='config_placement',
                            message=f"Configuration file outside scripts/config/: {relative_root / filename}",
                            file_path=str(relative_root / filename),
                            suggested_action=f"Move to scripts/config/ subdirectory"
                        ))
                        break

    def _validate_documentation_structure(self):
        """Validate documentation organization."""
        docs_dir = self.repo_root / 'docs'
        if not docs_dir.exists():
            return
            
        # Check for documentation files outside docs/
        doc_patterns = [r'.*_GUIDE\.md$', r'.*_PLAN\.md$', r'.*_SUMMARY\.md$', r'.*_REPORT\.md$']
        
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip docs directory
            if relative_root.parts and relative_root.parts[0] == 'docs':
                continue
                
            for filename in files:
                if filename in ['README.md', 'CHANGELOG.md', 'LICENSE']:
                    continue
                    
                for pattern in doc_patterns:
                    if re.match(pattern, filename):
                        self.results.append(ValidationResult(
                            level='warning',
                            category='documentation',
                            message=f"Documentation file outside docs/: {relative_root / filename}",
                            file_path=str(relative_root / filename),
                            suggested_action="Move to docs/ directory"
                        ))
                        break

    def _validate_source_organization(self):
        """Validate source code organization."""
        src_dir = self.repo_root / 'src'
        if not src_dir.exists():
            return
            
        # Check for source files outside src/
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.repo_root)
            
            # Skip src and scripts directories
            if (relative_root.parts and 
                relative_root.parts[0] in ['src', 'scripts', '.git', '.github', '.vscode', 
                                         '.project-rules', '.hypothesis', '.ruff_cache', 
                                         '.storybook', '.claude']):
                continue
                
            for filename in files:
                if filename.endswith('.py') and not filename.startswith('.'):
                    self.results.append(ValidationResult(
                        level='warning',
                        category='source_organization',
                        message=f"Python file outside src/: {relative_root / filename}",
                        file_path=str(relative_root / filename),
                        suggested_action="Move to src/ directory"
                    ))

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        report_lines = [
            "# Repository Organization Validation Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Repository**: {self.repo_root}",
            "",
        ]
        
        # Summary
        error_count = sum(1 for r in self.results if r.level == 'error')
        warning_count = sum(1 for r in self.results if r.level == 'warning')
        info_count = sum(1 for r in self.results if r.level == 'info')
        
        report_lines.extend([
            "## Summary",
            f"- **Errors**: {error_count}",
            f"- **Warnings**: {warning_count}",
            f"- **Info**: {info_count}",
            f"- **Total Issues**: {len(self.results)}",
            "",
        ])
        
        if error_count == 0:
            report_lines.append("✅ **Validation PASSED** - No critical errors found")
        else:
            report_lines.append("❌ **Validation FAILED** - Critical errors must be fixed")
        
        report_lines.append("")
        
        # Group results by category
        by_category = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)
        
        # Generate sections for each category
        for category, results in sorted(by_category.items()):
            report_lines.extend([
                f"## {category.replace('_', ' ').title()}",
                ""
            ])
            
            for result in results:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[result.level]
                report_lines.append(f"{icon} **{result.level.upper()}**: {result.message}")
                
                if result.file_path:
                    report_lines.append(f"   - File: `{result.file_path}`")
                
                if result.suggested_action:
                    report_lines.append(f"   - Suggested: {result.suggested_action}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent.parent
    validator = RepositoryValidator(repo_root)
    
    print("🔍 Validating repository organization...")
    passed, results = validator.validate()
    
    report = validator.generate_report()
    
    # Write report to file
    report_file = repo_root / 'scripts' / 'validation' / 'organization_report.md'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report)
    
    print(f"📄 Report written to: {report_file}")
    
    # Print summary
    error_count = sum(1 for r in results if r.level == 'error')
    warning_count = sum(1 for r in results if r.level == 'warning')
    
    if passed:
        print(f"✅ Validation PASSED ({warning_count} warnings)")
        sys.exit(0)
    else:
        print(f"❌ Validation FAILED ({error_count} errors, {warning_count} warnings)")
        print("\nCritical issues found:")
        for result in results:
            if result.level == 'error':
                print(f"  - {result.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()