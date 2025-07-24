#!/usr/bin/env python3
"""
Comprehensive Package Validation Script

This script validates all packages in the Monorepo monorepo to ensure they achieve 100% compliance
with the required structure, domain-driven design patterns, and entry points.

Requirements checked:
1. Required structure (src, pyproject.toml, BUCK, build, deploy, scripts, examples, docs, tests)
2. DDD layers (application, domain, infrastructure, presentation)
3. Entry points (cli.py, server.py, worker.py)
4. pyproject.toml parsing and validation
"""

import os
import sys
import json
import toml
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationResult:
    """Represents the validation result for a single check."""
    name: str
    passed: bool
    details: str = ""
    score: float = 0.0


@dataclass
class PackageValidation:
    """Represents the complete validation results for a package."""
    package_name: str
    package_path: Path
    overall_score: float = 0.0
    structure_score: float = 0.0
    ddd_score: float = 0.0
    entry_points_score: float = 0.0
    pyproject_score: float = 0.0
    results: List[ValidationResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class PackageValidator:
    """Validates package compliance with monorepo standards."""
    
    # Required top-level directories/files
    REQUIRED_STRUCTURE = {
        'src',
        'pyproject.toml',
        'BUCK',
        'build',
        'deploy',
        'scripts',
        'examples',
        'docs',
        'tests'
    }
    
    # DDD layers that should exist in src/<package_name>/
    DDD_LAYERS = {
        'application',
        'domain',
        'infrastructure', 
        'presentation'
    }
    
    # Required entry points in src/<package_name>/
    ENTRY_POINTS = {
        'cli.py',
        'server.py',
        'worker.py'
    }
    
    # Required pyproject.toml sections
    REQUIRED_PYPROJECT_SECTIONS = {
        'project',
        'build-system', 
        'tool.hatch.build.targets.wheel',
        'tool.pytest.ini_options'
    }

    def __init__(self, packages_root: Path):
        """Initialize validator with packages root directory."""
        self.packages_root = packages_root
        self.validation_results: List[PackageValidation] = []
    
    def discover_packages(self) -> List[Path]:
        """Discover all packages in the monorepo."""
        packages = []
        
        # Walk through all directories in packages root
        for category_dir in self.packages_root.iterdir():
            if not category_dir.is_dir():
                continue
                
            # Skip system directories
            if category_dir.name.startswith('.') or category_dir.name in ['__pycache__', 'temp']:
                continue
            
            # Check if this is a category directory (contains subdirectories with packages)
            if any(subdir.is_dir() and (subdir / 'pyproject.toml').exists() 
                   for subdir in category_dir.iterdir()):
                # This is a category directory
                for package_dir in category_dir.iterdir():
                    if package_dir.is_dir() and (package_dir / 'pyproject.toml').exists():
                        packages.append(package_dir)
            elif (category_dir / 'pyproject.toml').exists():
                # This is a direct package directory
                packages.append(category_dir)
        
        return sorted(packages)
    
    def validate_required_structure(self, package_path: Path) -> List[ValidationResult]:
        """Validate that package has required directory structure."""
        results = []
        
        for item in self.REQUIRED_STRUCTURE:
            item_path = package_path / item
            if item_path.exists():
                results.append(ValidationResult(
                    name=f"Structure: {item}",
                    passed=True,
                    details=f"Found {item}",
                    score=1.0
                ))
            else:
                results.append(ValidationResult(
                    name=f"Structure: {item}",
                    passed=False,
                    details=f"Missing {item}",
                    score=0.0
                ))
        
        return results
    
    def validate_ddd_layers(self, package_path: Path, package_name: str) -> List[ValidationResult]:
        """Validate DDD layer structure in src/<package_name>/."""
        results = []
        src_package_dir = package_path / 'src' / package_name
        
        if not src_package_dir.exists():
            results.append(ValidationResult(
                name="DDD: Source package directory",
                passed=False,
                details=f"Missing src/{package_name} directory",
                score=0.0
            ))
            return results
        
        results.append(ValidationResult(
            name="DDD: Source package directory",
            passed=True,
            details=f"Found src/{package_name} directory",
            score=1.0
        ))
        
        for layer in self.DDD_LAYERS:
            layer_path = src_package_dir / layer
            if layer_path.exists() and layer_path.is_dir():
                # Check if layer has __init__.py
                init_file = layer_path / '__init__.py'
                if init_file.exists():
                    results.append(ValidationResult(
                        name=f"DDD: {layer} layer",
                        passed=True,
                        details=f"Found {layer} layer with __init__.py",
                        score=1.0
                    ))
                else:
                    results.append(ValidationResult(
                        name=f"DDD: {layer} layer",
                        passed=False,
                        details=f"Found {layer} layer but missing __init__.py",
                        score=0.5
                    ))
            else:
                results.append(ValidationResult(
                    name=f"DDD: {layer} layer",
                    passed=False,
                    details=f"Missing {layer} layer",
                    score=0.0
                ))
        
        return results
    
    def validate_entry_points(self, package_path: Path, package_name: str) -> List[ValidationResult]:
        """Validate required entry points in src/<package_name>/."""
        results = []
        src_package_dir = package_path / 'src' / package_name
        
        if not src_package_dir.exists():
            for entry_point in self.ENTRY_POINTS:
                results.append(ValidationResult(
                    name=f"Entry Point: {entry_point}",
                    passed=False,
                    details=f"Cannot check {entry_point} - src/{package_name} missing",
                    score=0.0
                ))
            return results
        
        for entry_point in self.ENTRY_POINTS:
            entry_point_path = src_package_dir / entry_point
            if entry_point_path.exists():
                results.append(ValidationResult(
                    name=f"Entry Point: {entry_point}",
                    passed=True,
                    details=f"Found {entry_point}",
                    score=1.0
                ))
            else:
                results.append(ValidationResult(
                    name=f"Entry Point: {entry_point}",
                    passed=False,
                    details=f"Missing {entry_point}",
                    score=0.0
                ))
        
        return results
    
    def validate_pyproject_toml(self, package_path: Path) -> List[ValidationResult]:
        """Validate pyproject.toml structure and content."""
        results = []
        pyproject_path = package_path / 'pyproject.toml'
        
        if not pyproject_path.exists():
            results.append(ValidationResult(
                name="pyproject.toml: File exists",
                passed=False,
                details="Missing pyproject.toml file",
                score=0.0
            ))
            return results
        
        try:
            pyproject_data = toml.load(pyproject_path)
            
            results.append(ValidationResult(
                name="pyproject.toml: File exists",
                passed=True,
                details="Found and parsed pyproject.toml",
                score=1.0
            ))
            
            # Check required sections
            for section in self.REQUIRED_PYPROJECT_SECTIONS:
                if self._has_nested_key(pyproject_data, section):
                    results.append(ValidationResult(
                        name=f"pyproject.toml: {section}",
                        passed=True,
                        details=f"Found [{section}] section",
                        score=1.0
                    ))
                else:
                    results.append(ValidationResult(
                        name=f"pyproject.toml: {section}",
                        passed=False,
                        details=f"Missing [{section}] section",
                        score=0.0
                    ))
            
            # Validate project metadata
            if 'project' in pyproject_data:
                project = pyproject_data['project']
                required_fields = ['name', 'description']
                
                # Check for version (static) or dynamic versioning
                has_version = 'version' in project or 'dynamic' in project
                if has_version:
                    results.append(ValidationResult(
                        name="pyproject.toml: project.version",
                        passed=True,
                        details="Found version field or dynamic versioning",
                        score=1.0
                    ))
                else:
                    results.append(ValidationResult(
                        name="pyproject.toml: project.version",
                        passed=False,
                        details="Missing version field and dynamic versioning",
                        score=0.0
                    ))
                
                for field in required_fields:
                    if field in project:
                        results.append(ValidationResult(
                            name=f"pyproject.toml: project.{field}",
                            passed=True,
                            details=f"Found project.{field}",
                            score=1.0
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"pyproject.toml: project.{field}",
                            passed=False,
                            details=f"Missing project.{field}",
                            score=0.0
                        ))
        
        except Exception as e:
            results.append(ValidationResult(
                name="pyproject.toml: Parse error",
                passed=False,
                details=f"Failed to parse pyproject.toml: {e}",
                score=0.0
            ))
        
        return results
    
    def _has_nested_key(self, data: Dict, key_path: str) -> bool:
        """Check if nested key exists in data."""
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        return True
    
    def validate_package(self, package_path: Path) -> PackageValidation:
        """Validate a single package comprehensively."""
        package_name = package_path.name
        validation = PackageValidation(
            package_name=package_name,
            package_path=package_path
        )
        
        # Validate required structure
        structure_results = self.validate_required_structure(package_path)
        validation.results.extend(structure_results)
        
        # Validate DDD layers
        ddd_results = self.validate_ddd_layers(package_path, package_name)
        validation.results.extend(ddd_results)
        
        # Validate entry points
        entry_point_results = self.validate_entry_points(package_path, package_name)
        validation.results.extend(entry_point_results)
        
        # Validate pyproject.toml
        pyproject_results = self.validate_pyproject_toml(package_path)
        validation.results.extend(pyproject_results)
        
        # Calculate scores
        validation.structure_score = self._calculate_category_score(structure_results)
        validation.ddd_score = self._calculate_category_score(ddd_results)
        validation.entry_points_score = self._calculate_category_score(entry_point_results)
        validation.pyproject_score = self._calculate_category_score(pyproject_results)
        
        # Calculate overall score
        validation.overall_score = (
            validation.structure_score * 0.3 +
            validation.ddd_score * 0.3 +
            validation.entry_points_score * 0.2 +
            validation.pyproject_score * 0.2
        )
        
        # Collect issues (failed validations)
        validation.issues = [
            result.details for result in validation.results 
            if not result.passed
        ]
        
        return validation
    
    def _calculate_category_score(self, results: List[ValidationResult]) -> float:
        """Calculate score for a category of validation results."""
        if not results:
            return 0.0
        
        total_score = sum(result.score for result in results)
        return (total_score / len(results)) * 100
    
    def validate_all_packages(self) -> List[PackageValidation]:
        """Validate all discovered packages."""
        packages = self.discover_packages()
        
        print(f"Discovered {len(packages)} packages for validation...")
        
        for package_path in packages:
            print(f"Validating {package_path.name}...")
            validation = self.validate_package(package_path)
            self.validation_results.append(validation)
        
        return self.validation_results
    
    def generate_report(self, output_format: str = 'console') -> str:
        """Generate validation report in specified format."""
        if output_format == 'console':
            return self._generate_console_report()
        elif output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'json':
            return self._generate_json_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_console_report(self) -> str:
        """Generate console-formatted report."""
        report = []
        report.append("=" * 80)
        report.append("PYNOMALY MONOREPO - COMPREHENSIVE PACKAGE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total packages validated: {len(self.validation_results)}")
        report.append("")
        
        # Overall statistics
        total_packages = len(self.validation_results)
        perfect_packages = len([v for v in self.validation_results if v.overall_score == 100])
        avg_score = sum(v.overall_score for v in self.validation_results) / total_packages if total_packages > 0 else 0
        
        report.append("OVERALL STATISTICS")
        report.append("-" * 50)
        report.append(f"Perfect compliance (100%): {perfect_packages}/{total_packages} ({perfect_packages/total_packages*100:.1f}%)")
        report.append(f"Average compliance score: {avg_score:.1f}%")
        report.append("")
        
        # Score distribution
        score_ranges = [(90, 100), (80, 90), (70, 80), (60, 70), (0, 60)]
        for min_score, max_score in score_ranges:
            count = len([v for v in self.validation_results 
                        if min_score <= v.overall_score < max_score or 
                        (max_score == 100 and v.overall_score == 100)])
            report.append(f"Score {min_score}-{max_score}%: {count} packages")
        report.append("")
        
        # Detailed results per package
        report.append("DETAILED PACKAGE VALIDATION RESULTS")
        report.append("-" * 50)
        
        # Sort packages by score (highest first)
        sorted_validations = sorted(self.validation_results, 
                                   key=lambda v: v.overall_score, reverse=True)
        
        for validation in sorted_validations:
            status = "✅ PASS" if validation.overall_score == 100 else "❌ FAIL"
            report.append(f"\n{validation.package_name} - {validation.overall_score:.1f}% {status}")
            report.append(f"  Location: {validation.package_path}")
            report.append(f"  Structure: {validation.structure_score:.1f}% | " +
                         f"DDD Layers: {validation.ddd_score:.1f}% | " +
                         f"Entry Points: {validation.entry_points_score:.1f}% | " +
                         f"pyproject.toml: {validation.pyproject_score:.1f}%")
            
            if validation.issues:
                report.append("  Issues:")
                for issue in validation.issues:
                    report.append(f"    • {issue}")
        
        return "\n".join(report)
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown-formatted report."""
        report = []
        report.append("# Monorepo Monorepo - Comprehensive Package Validation Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total packages validated:** {len(self.validation_results)}")
        report.append("")
        
        # Overall statistics
        total_packages = len(self.validation_results)
        perfect_packages = len([v for v in self.validation_results if v.overall_score == 100])
        avg_score = sum(v.overall_score for v in self.validation_results) / total_packages if total_packages > 0 else 0
        
        report.append("## Overall Statistics")
        report.append("")
        report.append(f"- **Perfect compliance (100%):** {perfect_packages}/{total_packages} ({perfect_packages/total_packages*100:.1f}%)")
        report.append(f"- **Average compliance score:** {avg_score:.1f}%")
        report.append("")
        
        # Results table
        report.append("## Package Validation Results")
        report.append("")
        report.append("| Package | Score | Structure | DDD | Entry Points | pyproject.toml | Status |")
        report.append("|---------|-------|-----------|-----|--------------|----------------|--------|")
        
        sorted_validations = sorted(self.validation_results, 
                                   key=lambda v: v.overall_score, reverse=True)
        
        for validation in sorted_validations:
            status = "✅ PASS" if validation.overall_score == 100 else "❌ FAIL"
            report.append(f"| {validation.package_name} | {validation.overall_score:.1f}% | "
                         f"{validation.structure_score:.1f}% | {validation.ddd_score:.1f}% | "
                         f"{validation.entry_points_score:.1f}% | {validation.pyproject_score:.1f}% | {status} |")
        
        # Issues section
        report.append("")
        report.append("## Issues by Package")
        report.append("")
        
        for validation in sorted_validations:
            if validation.issues:
                report.append(f"### {validation.package_name}")
                report.append("")
                for issue in validation.issues:
                    report.append(f"- {issue}")
                report.append("")
        
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """Generate JSON-formatted report."""
        report_data = {
            "generated": datetime.now().isoformat(),
            "total_packages": len(self.validation_results),
            "perfect_packages": len([v for v in self.validation_results if v.overall_score == 100]),
            "average_score": sum(v.overall_score for v in self.validation_results) / len(self.validation_results) if self.validation_results else 0,
            "packages": []
        }
        
        for validation in self.validation_results:
            package_data = {
                "name": validation.package_name,
                "path": str(validation.package_path),
                "scores": {
                    "overall": validation.overall_score,
                    "structure": validation.structure_score,
                    "ddd": validation.ddd_score,
                    "entry_points": validation.entry_points_score,
                    "pyproject": validation.pyproject_score
                },
                "passed": validation.overall_score == 100,
                "issues": validation.issues,
                "detailed_results": [
                    {
                        "name": result.name,
                        "passed": result.passed,
                        "details": result.details,
                        "score": result.score
                    }
                    for result in validation.results
                ]
            }
            report_data["packages"].append(package_data)
        
        return json.dumps(report_data, indent=2)


def main():
    """Main function to run package validation."""
    # Determine packages root directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent  # Go up from scripts/validation/
    packages_root = repo_root / 'src' / 'packages'
    
    if not packages_root.exists():
        print(f"ERROR: Packages directory not found at {packages_root}")
        sys.exit(1)
    
    print(f"Scanning packages in: {packages_root}")
    
    # Create validator and run validation
    validator = PackageValidator(packages_root)
    validator.validate_all_packages()
    
    # Generate and print console report
    console_report = validator.generate_report('console')
    print(console_report)
    
    # Save markdown report
    markdown_report = validator.generate_report('markdown')
    markdown_path = repo_root / 'PACKAGE_VALIDATION_REPORT.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    print(f"\nMarkdown report saved to: {markdown_path}")
    
    # Save JSON report
    json_report = validator.generate_report('json')
    json_path = repo_root / 'package_validation_results.json'
    with open(json_path, 'w') as f:
        f.write(json_report)
    print(f"JSON report saved to: {json_path}")
    
    # Exit with appropriate code
    perfect_packages = len([v for v in validator.validation_results if v.overall_score == 100])
    total_packages = len(validator.validation_results)
    
    if perfect_packages == total_packages:
        print(f"\n🎉 SUCCESS: All {total_packages} packages achieve 100% compliance!")
        sys.exit(0)
    else:
        print(f"\n⚠️  WARNING: {perfect_packages}/{total_packages} packages achieve 100% compliance")
        sys.exit(1)


if __name__ == '__main__':
    main()