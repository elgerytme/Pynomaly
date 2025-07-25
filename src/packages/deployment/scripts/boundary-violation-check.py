#!/usr/bin/env python3
"""
Automated Domain Boundary Violation Detection for CI/CD Pipeline.

This script analyzes the monorepo for domain boundary violations and fails
the CI/CD pipeline if violations are detected, ensuring clean architecture
is maintained.
"""

import ast
import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import subprocess

# Configuration for domain boundaries
DOMAIN_CONFIG = {
    "domains": {
        "data": ["anomaly_detection", "data_quality", "data_science", "quality", "observability"],
        "ai": ["machine_learning", "mlops", "nlp-advanced", "computer-vision", "neuro_symbolic"],
        "enterprise": ["security", "enterprise_auth", "enterprise_governance", "enterprise_scalability", "mlops_marketplace"],
        "integrations": ["cloud", "api_gateway"],
        "shared": ["shared", "interfaces", "configurations"],
        "infrastructure": ["infrastructure"],
        "ecosystem": ["ecosystem"],
        "clients": ["clients", "sdk_core"],
        "observability": ["observability"],
        "analytics": ["business_intelligence"]
    },
    "allowed_cross_domain_imports": {
        # Shared packages can be imported by any domain
        "shared": ["*"],
        "interfaces": ["*"],
        "configurations": ["*"],
        
        # Infrastructure can be imported by any domain
        "infrastructure": ["*"],
        
        # Specific cross-domain integrations
        "data_quality": ["shared", "interfaces", "infrastructure"],
        "machine_learning": ["shared", "interfaces", "infrastructure"],
        "security": ["shared", "interfaces", "infrastructure"],
        "cloud": ["shared", "interfaces", "infrastructure"],
        
        # Enterprise packages have broader access
        "enterprise_security": ["shared", "interfaces", "infrastructure", "data", "ai"],
        "mlops_marketplace": ["shared", "interfaces", "infrastructure", "ai", "data"],
    },
    "violation_patterns": [
        # Direct imports from other domains
        r"from\s+(?:src\.)?packages\.([^.]+)\.([^.\s]+)",
        r"import\s+(?:src\.)?packages\.([^.]+)\.([^.\s]+)",
        
        # Relative imports that cross domain boundaries
        r"from\s+\.\.\.\.([^.\s]+)",
        r"from\s+\.\.\.([^.\s]+)\.([^.\s]+)",
    ]
}


@dataclass
class ViolationLocation:
    """Location information for a boundary violation."""
    file_path: str
    line_number: int
    line_content: str
    column_offset: int = 0


@dataclass
class BoundaryViolation:
    """Represents a domain boundary violation."""
    violation_type: str
    source_domain: str
    target_domain: str
    target_package: str
    import_statement: str
    locations: List[ViolationLocation] = field(default_factory=list)
    severity: str = "medium"
    
    def __post_init__(self):
        """Set severity based on violation type."""
        if self.violation_type == "direct_cross_domain":
            self.severity = "high"
        elif self.violation_type == "relative_cross_domain":
            self.severity = "critical"
        elif self.violation_type == "circular_dependency":
            self.severity = "critical"


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    total_files_scanned: int = 0
    violations: List[BoundaryViolation] = field(default_factory=list)
    domain_statistics: Dict[str, Dict[str, int]] = field(default_factory=dict)
    allowed_imports: Dict[str, List[str]] = field(default_factory=dict)
    scan_timestamp: str = ""
    scan_duration_seconds: float = 0.0
    
    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return len(self.violations)
    
    @property
    def critical_violations(self) -> List[BoundaryViolation]:
        """Get critical violations."""
        return [v for v in self.violations if v.severity == "critical"]
    
    @property
    def high_violations(self) -> List[BoundaryViolation]:
        """Get high severity violations."""
        return [v for v in self.violations if v.severity == "high"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            **asdict(self),
            "violations": [asdict(v) for v in self.violations],
            "critical_count": len(self.critical_violations),
            "high_count": len(self.high_violations),
            "medium_count": len([v for v in self.violations if v.severity == "medium"]),
            "low_count": len([v for v in self.violations if v.severity == "low"])
        }


class DomainMapper:
    """Maps file paths to domain names."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_map = self._build_domain_map()
    
    def _build_domain_map(self) -> Dict[str, str]:
        """Build mapping from package names to domains."""
        domain_map = {}
        for domain, packages in self.config["domains"].items():
            for package in packages:
                domain_map[package] = domain
        return domain_map
    
    def get_domain_from_path(self, file_path: str) -> Optional[str]:
        """Extract domain from file path."""
        path_parts = Path(file_path).parts
        
        # Look for packages directory
        if "packages" in path_parts:
            pkg_index = path_parts.index("packages")
            if pkg_index + 1 < len(path_parts):
                package_name = path_parts[pkg_index + 1]
                return self.domain_map.get(package_name)
        
        return None
    
    def get_package_from_path(self, file_path: str) -> Optional[str]:
        """Extract package name from file path."""
        path_parts = Path(file_path).parts
        
        if "packages" in path_parts:
            pkg_index = path_parts.index("packages")
            if pkg_index + 1 < len(path_parts):
                return path_parts[pkg_index + 1]
        
        return None


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import statements."""
    
    def __init__(self, file_path: str, domain_mapper: DomainMapper):
        self.file_path = file_path
        self.domain_mapper = domain_mapper
        self.source_domain = domain_mapper.get_domain_from_path(file_path)
        self.source_package = domain_mapper.get_package_from_path(file_path)
        self.violations: List[BoundaryViolation] = []
        self.imports: List[Tuple[str, int, str]] = []
    
    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            self._analyze_import(alias.name, node.lineno, f"import {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statements."""
        if node.module:
            import_stmt = f"from {node.module} import {', '.join(alias.name for alias in node.names)}"
            self._analyze_import(node.module, node.lineno, import_stmt, is_from_import=True)
        self.generic_visit(node)
    
    def _analyze_import(self, module_name: str, line_no: int, import_stmt: str, is_from_import: bool = False):
        """Analyze a single import for boundary violations."""
        self.imports.append((module_name, line_no, import_stmt))
        
        # Skip if we can't determine source domain
        if not self.source_domain:
            return
        
        # Check for direct cross-domain imports
        violation = self._check_cross_domain_import(module_name, line_no, import_stmt)
        if violation:
            self.violations.append(violation)
    
    def _check_cross_domain_import(self, module_name: str, line_no: int, import_stmt: str) -> Optional[BoundaryViolation]:
        """Check if import violates domain boundaries."""
        # Parse module path
        module_parts = module_name.split('.')
        
        # Check for packages imports
        if "packages" in module_parts:
            pkg_index = module_parts.index("packages")
            if pkg_index + 1 < len(module_parts):
                target_package = module_parts[pkg_index + 1]
                target_domain = self.domain_mapper.domain_map.get(target_package)
                
                if target_domain and target_domain != self.source_domain:
                    # Check if this cross-domain import is allowed
                    if not self._is_import_allowed(self.source_package, target_package, target_domain):
                        return BoundaryViolation(
                            violation_type="direct_cross_domain",
                            source_domain=self.source_domain,
                            target_domain=target_domain,
                            target_package=target_package,
                            import_statement=import_stmt,
                            locations=[ViolationLocation(
                                file_path=self.file_path,
                                line_number=line_no,
                                line_content=import_stmt
                            )]
                        )
        
        # Check for relative imports that might cross boundaries
        if module_name.startswith('.'):
            # This would require more complex analysis of the file system structure
            # For now, we'll flag suspicious relative imports
            dot_count = len(module_name) - len(module_name.lstrip('.'))
            if dot_count >= 3:  # Going up 3+ levels is suspicious
                return BoundaryViolation(
                    violation_type="suspicious_relative_import",
                    source_domain=self.source_domain,
                    target_domain="unknown",
                    target_package="unknown",
                    import_statement=import_stmt,
                    locations=[ViolationLocation(
                        file_path=self.file_path,
                        line_number=line_no,
                        line_content=import_stmt
                    )],
                    severity="medium"
                )
        
        return None
    
    def _is_import_allowed(self, source_package: str, target_package: str, target_domain: str) -> bool:
        """Check if cross-domain import is allowed."""
        allowed_imports = DOMAIN_CONFIG.get("allowed_cross_domain_imports", {})
        
        # Check if source package has specific allowances
        if source_package in allowed_imports:
            allowed = allowed_imports[source_package]
            if "*" in allowed or target_package in allowed or target_domain in allowed:
                return True
        
        # Check if target is a universally allowed package
        universal_packages = ["shared", "interfaces", "configurations", "infrastructure"]
        if target_package in universal_packages:
            return True
        
        return False


class BoundaryViolationDetector:
    """Main class for detecting domain boundary violations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DOMAIN_CONFIG
        self.domain_mapper = DomainMapper(self.config)
        self.report = AnalysisReport()
    
    def scan_directory(self, directory: str, exclude_patterns: List[str] = None) -> AnalysisReport:
        """Scan directory for boundary violations."""
        import time
        start_time = time.time()
        
        exclude_patterns = exclude_patterns or [
            "*/tests/*", "*/test/*", "*/__pycache__/*", "*/node_modules/*",
            "*/venv/*", "*/virtualenv/*", "*/.git/*", "*/build/*", "*/dist/*"
        ]
        
        python_files = self._find_python_files(directory, exclude_patterns)
        
        for file_path in python_files:
            self._analyze_file(file_path)
            self.report.total_files_scanned += 1
        
        # Calculate statistics
        self._calculate_statistics()
        
        self.report.scan_duration_seconds = time.time() - start_time
        self.report.scan_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        return self.report
    
    def _find_python_files(self, directory: str, exclude_patterns: List[str]) -> List[str]:
        """Find all Python files in directory."""
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(os.path.join(root, d), pattern) 
                for pattern in exclude_patterns
            )]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if not any(self._matches_pattern(file_path, pattern) for pattern in exclude_patterns):
                        python_files.append(file_path)
        
        return python_files
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclusion pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def _analyze_file(self, file_path: str) -> None:
        """Analyze a single file for violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Analyze imports
            analyzer = ImportAnalyzer(file_path, self.domain_mapper)
            analyzer.visit(tree)
            
            # Add violations to report
            self.report.violations.extend(analyzer.violations)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not analyze {file_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
    
    def _calculate_statistics(self) -> None:
        """Calculate domain statistics."""
        stats = defaultdict(lambda: defaultdict(int))
        
        for violation in self.report.violations:
            stats[violation.source_domain]["violations"] += 1
            stats[violation.source_domain][f"{violation.severity}_violations"] += 1
            stats[violation.source_domain]["target_domains"] = len(set(
                v.target_domain for v in self.report.violations 
                if v.source_domain == violation.source_domain
            ))
        
        self.report.domain_statistics = dict(stats)


class ReportFormatter:
    """Formats analysis reports in different formats."""
    
    @staticmethod
    def format_console(report: AnalysisReport, show_details: bool = True) -> str:
        """Format report for console output."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("DOMAIN BOUNDARY VIOLATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Scan completed: {report.scan_timestamp}")
        lines.append(f"Files scanned: {report.total_files_scanned}")
        lines.append(f"Scan duration: {report.scan_duration_seconds:.2f} seconds")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total violations: {report.violation_count}")
        lines.append(f"Critical violations: {len(report.critical_violations)}")
        lines.append(f"High severity violations: {len(report.high_violations)}")
        lines.append("")
        
        if report.violation_count == 0:
            lines.append("✅ No domain boundary violations detected!")
            lines.append("")
        else:
            lines.append("❌ Domain boundary violations detected!")
            lines.append("")
            
            if show_details:
                # Group violations by domain
                violations_by_domain = defaultdict(list)
                for violation in report.violations:
                    violations_by_domain[violation.source_domain].append(violation)
                
                for domain, violations in violations_by_domain.items():
                    lines.append(f"DOMAIN: {domain.upper()}")
                    lines.append("-" * 40)
                    
                    for violation in violations:
                        severity_icon = {
                            "critical": "🔴",
                            "high": "🟠", 
                            "medium": "🟡",
                            "low": "🟢"
                        }.get(violation.severity, "⚪")
                        
                        lines.append(f"{severity_icon} {violation.severity.upper()}: {violation.violation_type}")
                        lines.append(f"   Import: {violation.import_statement}")
                        lines.append(f"   Target: {violation.target_domain}.{violation.target_package}")
                        
                        for location in violation.locations:
                            lines.append(f"   File: {location.file_path}:{location.line_number}")
                        
                        lines.append("")
        
        # Domain statistics
        if report.domain_statistics:
            lines.append("DOMAIN STATISTICS")
            lines.append("-" * 40)
            for domain, stats in report.domain_statistics.items():
                lines.append(f"{domain}: {stats.get('violations', 0)} violations")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_json(report: AnalysisReport) -> str:
        """Format report as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str)
    
    @staticmethod
    def format_github_annotation(report: AnalysisReport) -> str:
        """Format violations as GitHub Actions annotations."""
        annotations = []
        
        for violation in report.violations:
            for location in violation.locations:
                level = "error" if violation.severity in ["critical", "high"] else "warning"
                
                annotation = (
                    f"::{level} file={location.file_path},line={location.line_number}::"
                    f"Domain boundary violation: {violation.violation_type} - "
                    f"{violation.import_statement} (imports {violation.target_domain}.{violation.target_package})"
                )
                annotations.append(annotation)
        
        return "\n".join(annotations)


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Detect domain boundary violations in monorepo"
    )
    parser.add_argument(
        "directory",
        help="Directory to scan for violations"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (JSON)",
        default=None
    )
    parser.add_argument(
        "--format",
        choices=["console", "json", "github"],
        default="console",
        help="Output format"
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit with error code if violations found"
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true", 
        help="Exit with error code only on critical violations"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Exclude patterns (glob style)",
        default=[]
    )
    parser.add_argument(
        "--output",
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DOMAIN_CONFIG
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run analysis
    detector = BoundaryViolationDetector(config)
    
    if not args.quiet:
        print(f"Scanning {args.directory} for domain boundary violations...", file=sys.stderr)
    
    report = detector.scan_directory(args.directory, args.exclude)
    
    # Format output
    if args.format == "console":
        output = ReportFormatter.format_console(report, show_details=not args.quiet)
    elif args.format == "json":
        output = ReportFormatter.format_json(report)
    elif args.format == "github":
        output = ReportFormatter.format_github_annotation(report)
    else:
        output = ReportFormatter.format_console(report)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        if not args.quiet:
            print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)
    
    # Determine exit code
    if args.fail_on_critical and report.critical_violations:
        if not args.quiet:
            print(f"❌ Found {len(report.critical_violations)} critical violations", file=sys.stderr)
        sys.exit(1)
    elif args.fail_on_violations and report.violations:
        if not args.quiet:
            print(f"❌ Found {len(report.violations)} violations", file=sys.stderr)
        sys.exit(1)
    else:
        if not args.quiet:
            if report.violations:
                print(f"⚠️  Found {len(report.violations)} violations (not failing)", file=sys.stderr)
            else:
                print("✅ No violations found", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()