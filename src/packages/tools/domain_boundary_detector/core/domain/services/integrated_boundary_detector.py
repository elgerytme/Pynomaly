"""Integrated boundary detector that scans both code and documentation files."""

import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .scanner import ScanResult, DomainScanner
from .analyzer import Violation, ViolationAnalyzer
from .documentation_scanner import DocumentationScanner, DocumentationViolation


@dataclass
class IntegratedScanResult:
    """Combined results from code and documentation scanning."""
    code_violations: List[Violation]
    documentation_violations: List[DocumentationViolation]
    summary: Dict[str, int]
    
    @property
    def total_violations(self) -> int:
        """Get total number of violations across all types."""
        return len(self.code_violations) + len(self.documentation_violations)
    
    @property
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        critical_code = any(v.severity.value == 'critical' for v in self.code_violations)
        critical_docs = any(v.severity == 'critical' for v in self.documentation_violations)
        return critical_code or critical_docs


class IntegratedBoundaryDetector:
    """
    Integrated domain boundary detector that scans both Python code and documentation.
    
    This service combines the existing domain boundary detection for Python imports
    with the new documentation domain boundary checking to provide comprehensive
    boundary violation detection across the entire repository.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated boundary detector.
        
        Args:
            config_path: Path to the domain boundaries configuration file
        """
        self.config_path = config_path or '.domain-boundaries.yaml'
        self.documentation_scanner = DocumentationScanner(self.config_path)
        # Note: The existing code scanner would be initialized here
        # For now, we'll focus on the documentation scanning integration
    
    def scan_repository(
        self, 
        repository_path: str = ".",
        include_code: bool = True,
        include_docs: bool = True
    ) -> IntegratedScanResult:
        """
        Scan the entire repository for both code and documentation violations.
        
        Args:
            repository_path: Path to the repository root
            include_code: Whether to scan Python code files
            include_docs: Whether to scan documentation files
            
        Returns:
            Integrated scan results containing both types of violations
        """
        code_violations = []
        documentation_violations = []
        
        # Scan documentation if requested
        if include_docs:
            documentation_violations = self.documentation_scanner.scan_repository(repository_path)
        
        # Scan Python code if requested
        if include_code:
            # This would integrate with the existing code scanning functionality
            # For now, we'll leave this as a placeholder since we're focusing on docs
            pass
        
        # Generate summary
        summary = self._generate_summary(code_violations, documentation_violations)
        
        return IntegratedScanResult(
            code_violations=code_violations,
            documentation_violations=documentation_violations,
            summary=summary
        )
    
    def scan_package(
        self,
        package_path: str,
        include_code: bool = True,
        include_docs: bool = True
    ) -> IntegratedScanResult:
        """
        Scan a specific package for boundary violations.
        
        Args:
            package_path: Path to the package directory
            include_code: Whether to scan Python code files
            include_docs: Whether to scan documentation files
            
        Returns:
            Integrated scan results for the package
        """
        code_violations = []
        documentation_violations = []
        
        if include_docs:
            # Scan package documentation
            docs_path = os.path.join(package_path, "docs")
            if os.path.exists(docs_path):
                documentation_violations.extend(
                    self.documentation_scanner.scan_directory(docs_path)
                )
            
            # Scan package README
            readme_path = os.path.join(package_path, "README.md")
            if os.path.exists(readme_path):
                documentation_violations.extend(
                    self.documentation_scanner.scan_file(readme_path)
                )
        
        if include_code:
            # This would scan Python files in the package
            pass
        
        summary = self._generate_summary(code_violations, documentation_violations)
        
        return IntegratedScanResult(
            code_violations=code_violations,
            documentation_violations=documentation_violations,
            summary=summary
        )
    
    def scan_documentation_only(self, repository_path: str = ".") -> List[DocumentationViolation]:
        """
        Scan only documentation files for boundary violations.
        
        Args:
            repository_path: Path to the repository root
            
        Returns:
            List of documentation violations
        """
        return self.documentation_scanner.scan_repository(repository_path)
    
    def generate_report(
        self,
        scan_result: IntegratedScanResult,
        format_type: str = "console",
        include_suggestions: bool = True
    ) -> str:
        """
        Generate a comprehensive report from integrated scan results.
        
        Args:
            scan_result: Results from integrated scanning
            format_type: Format of the report ('console', 'json', 'markdown')
            include_suggestions: Whether to include fix suggestions
            
        Returns:
            Formatted report string
        """
        if format_type == "console":
            return self._generate_console_report(scan_result, include_suggestions)
        elif format_type == "json":
            return self._generate_json_report(scan_result)
        elif format_type == "markdown":
            return self._generate_markdown_report(scan_result, include_suggestions)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _generate_summary(
        self,
        code_violations: List[Violation],
        documentation_violations: List[DocumentationViolation]
    ) -> Dict[str, int]:
        """Generate a summary of violations by type and severity."""
        summary = {
            'total_violations': len(code_violations) + len(documentation_violations),
            'code_violations': len(code_violations),
            'documentation_violations': len(documentation_violations),
            'critical_violations': 0,
            'warning_violations': 0,
            'info_violations': 0
        }
        
        # Count code violations by severity
        for violation in code_violations:
            severity = violation.severity.value
            summary[f'{severity}_violations'] += 1
        
        # Count documentation violations by severity
        for violation in documentation_violations:
            severity = violation.severity
            summary[f'{severity}_violations'] += 1
        
        return summary
    
    def _generate_console_report(
        self,
        scan_result: IntegratedScanResult,
        include_suggestions: bool = True
    ) -> str:
        """Generate a console-formatted integrated report."""
        if scan_result.total_violations == 0:
            return "✅ No domain boundary violations found!"
        
        report = [
            "Integrated Domain Boundary Scan Report",
            "=" * 42,
            ""
        ]
        
        # Summary section
        summary = scan_result.summary
        report.extend([
            f"❌ TOTAL VIOLATIONS: {summary['total_violations']}",
            f"  ● Code violations: {summary['code_violations']}",
            f"  ● Documentation violations: {summary['documentation_violations']}",
            "",
            f"BY SEVERITY:",
            f"  ● Critical: {summary['critical_violations']}",
            f"  ● Warning: {summary['warning_violations']}",
            f"  ● Info: {summary['info_violations']}",
            ""
        ])
        
        # Documentation violations section
        if scan_result.documentation_violations:
            report.extend([
                "DOCUMENTATION VIOLATIONS:",
                "-" * 30,
                ""
            ])
            
            # Group by severity
            doc_violations_by_severity = {}
            for violation in scan_result.documentation_violations:
                severity = violation.severity
                if severity not in doc_violations_by_severity:
                    doc_violations_by_severity[severity] = []
                doc_violations_by_severity[severity].append(violation)
            
            for severity in ['critical', 'warning', 'info']:
                violations = doc_violations_by_severity.get(severity, [])
                if not violations:
                    continue
                
                report.extend([
                    f"{severity.upper()} DOCUMENTATION VIOLATIONS:",
                    "-" * 40,
                    ""
                ])
                
                for i, violation in enumerate(violations, 1):
                    report.extend([
                        f"{i}. {violation.message}",
                        f"   File: {violation.file_path}:{violation.line_number}",
                        f"   Rule: {violation.rule_name}",
                        f"   Content: {violation.line_content[:80]}..."
                    ])
                    
                    if include_suggestions and violation.suggestion:
                        report.append(f"   Suggestion: {violation.suggestion}")
                    
                    report.append("")
        
        # Code violations section (placeholder for future integration)
        if scan_result.code_violations:
            report.extend([
                "CODE VIOLATIONS:",
                "-" * 20,
                "(Code violation reporting would be integrated here)",
                ""
            ])
        
        return "\n".join(report)
    
    def _generate_json_report(self, scan_result: IntegratedScanResult) -> str:
        """Generate a JSON-formatted integrated report."""
        import json
        
        # Convert documentation violations to dict
        doc_violations_data = []
        for violation in scan_result.documentation_violations:
            doc_violations_data.append({
                'type': 'documentation',
                'file_path': violation.file_path,
                'line_number': violation.line_number,
                'line_content': violation.line_content,
                'violation_type': violation.violation_type,
                'message': violation.message,
                'severity': violation.severity,
                'rule_name': violation.rule_name,
                'pattern': violation.pattern,
                'suggestion': violation.suggestion
            })
        
        # Convert code violations to dict (placeholder)
        code_violations_data = []
        for violation in scan_result.code_violations:
            code_violations_data.append({
                'type': 'code',
                'file_path': violation.file_path,
                'line_number': violation.line_number,
                'from_package': violation.from_package,
                'to_package': violation.to_package,
                'severity': violation.severity.value,
                'description': violation.description,
                'suggestion': violation.suggestion
            })
        
        report_data = {
            'summary': scan_result.summary,
            'violations': {
                'documentation': doc_violations_data,
                'code': code_violations_data
            }
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_markdown_report(
        self,
        scan_result: IntegratedScanResult,
        include_suggestions: bool = True
    ) -> str:
        """Generate a Markdown-formatted integrated report."""
        if scan_result.total_violations == 0:
            return "# Integrated Domain Boundary Report\n\n✅ No violations found!"
        
        report = [
            "# Integrated Domain Boundary Report",
            "",
            "## Summary",
            ""
        ]
        
        summary = scan_result.summary
        report.extend([
            f"**Total Violations:** {summary['total_violations']}",
            f"- Code violations: {summary['code_violations']}",
            f"- Documentation violations: {summary['documentation_violations']}",
            "",
            "**By Severity:**",
            f"- Critical: {summary['critical_violations']}",
            f"- Warning: {summary['warning_violations']}",
            f"- Info: {summary['info_violations']}",
            ""
        ])
        
        # Documentation violations section
        if scan_result.documentation_violations:
            report.extend([
                "## Documentation Violations",
                ""
            ])
            
            # Group by severity
            doc_violations_by_severity = {}
            for violation in scan_result.documentation_violations:
                severity = violation.severity
                if severity not in doc_violations_by_severity:
                    doc_violations_by_severity[severity] = []
                doc_violations_by_severity[severity].append(violation)
            
            for severity in ['critical', 'warning', 'info']:
                violations = doc_violations_by_severity.get(severity, [])
                if not violations:
                    continue
                
                report.extend([
                    f"### {severity.title()} Documentation Violations",
                    ""
                ])
                
                for i, violation in enumerate(violations, 1):
                    report.extend([
                        f"#### {i}. {violation.message}",
                        "",
                        f"**File:** `{violation.file_path}:{violation.line_number}`",
                        f"**Rule:** {violation.rule_name}",
                        "",
                        "**Content:**",
                        "```",
                        violation.line_content,
                        "```",
                        ""
                    ])
                    
                    if include_suggestions and violation.suggestion:
                        report.extend([
                            f"**Suggestion:** {violation.suggestion}",
                            ""
                        ])
        
        # Code violations section (placeholder)
        if scan_result.code_violations:
            report.extend([
                "## Code Violations",
                "",
                "*(Code violation details would be shown here)*",
                ""
            ])
        
        return "\n".join(report)
    
    def check_exit_code(self, scan_result: IntegratedScanResult) -> int:
        """
        Determine appropriate exit code based on scan results.
        
        Args:
            scan_result: Results from integrated scanning
            
        Returns:
            Exit code (0 for success, >0 for violations)
        """
        if scan_result.has_critical_violations:
            return 1
        elif scan_result.total_violations > 0:
            return 2
        else:
            return 0