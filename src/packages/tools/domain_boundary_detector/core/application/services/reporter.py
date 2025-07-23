"""Console reporter for displaying boundary violations."""

from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from ...domain.services.analyzer import AnalysisResult, Violation, Severity, ViolationType


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


@dataclass
class ReportOptions:
    """Options for report generation."""
    show_exempted: bool = False
    show_suggestions: bool = True
    show_statistics: bool = True
    max_violations: Optional[int] = None
    group_by: str = 'severity'  # 'severity', 'type', 'domain'
    verbose: bool = False
    
    
class ConsoleReporter:
    """Reporter for displaying results in the console."""
    
    def __init__(self, options: Optional[ReportOptions] = None):
        self.options = options or ReportOptions()
        
    def report(self, result: AnalysisResult, title: str = "Domain Boundary Scan Report") -> None:
        """Generate and print console report."""
        self._print_header(title)
        
        if not result.violations:
            self._print_success("✅ No boundary violations found!")
            if self.options.show_statistics:
                self._print_statistics(result)
            return
            
        self._print_summary(result)
        self._print_violations(result)
        
        if self.options.show_statistics:
            self._print_statistics(result)
            
    def _print_header(self, title: str) -> None:
        """Print report header."""
        print(f"\n{Colors.BOLD}{title}{Colors.END}")
        print("=" * len(title))
        print()
        
    def _print_success(self, message: str) -> None:
        """Print success message."""
        print(f"{Colors.GREEN}{message}{Colors.END}\n")
        
    def _print_summary(self, result: AnalysisResult) -> None:
        """Print violation summary."""
        total = len(result.violations)
        critical = sum(1 for v in result.violations if v.severity == Severity.CRITICAL)
        warning = sum(1 for v in result.violations if v.severity == Severity.WARNING)
        info = sum(1 for v in result.violations if v.severity == Severity.INFO)
        exempted = sum(1 for v in result.violations if v.is_exempted())
        
        print(f"{Colors.RED}❌ VIOLATIONS FOUND: {total}{Colors.END}")
        print(f"  {Colors.RED}● Critical: {critical}{Colors.END}")
        print(f"  {Colors.YELLOW}● Warning: {warning}{Colors.END}")
        print(f"  {Colors.BLUE}● Info: {info}{Colors.END}")
        
        if exempted > 0:
            print(f"  {Colors.CYAN}● Exempted: {exempted}{Colors.END}")
        print()
        
    def _print_violations(self, result: AnalysisResult) -> None:
        """Print detailed violations."""
        violations = result.violations
        
        # Filter exempted if needed
        if not self.options.show_exempted:
            violations = [v for v in violations if not v.is_exempted()]
            
        # Limit violations if specified
        if self.options.max_violations:
            violations = violations[:self.options.max_violations]
            
        # Group violations
        if self.options.group_by == 'severity':
            self._print_by_severity(violations)
        elif self.options.group_by == 'type':
            self._print_by_type(violations)
        elif self.options.group_by == 'domain':
            self._print_by_domain(violations)
        else:
            self._print_flat(violations)
            
    def _print_by_severity(self, violations: List[Violation]) -> None:
        """Print violations grouped by severity."""
        for severity in [Severity.CRITICAL, Severity.WARNING, Severity.INFO]:
            severity_violations = [v for v in violations if v.severity == severity]
            if severity_violations:
                color = self._get_severity_color(severity)
                print(f"{color}{severity.value.upper()} VIOLATIONS:{Colors.END}")
                print("-" * 40)
                for i, v in enumerate(severity_violations, 1):
                    self._print_violation(v, i)
                print()
                
    def _print_by_type(self, violations: List[Violation]) -> None:
        """Print violations grouped by type."""
        types = {}
        for v in violations:
            if v.type not in types:
                types[v.type] = []
            types[v.type].append(v)
            
        for vtype, type_violations in types.items():
            print(f"{Colors.BOLD}{vtype.value.replace('_', ' ').title()}:{Colors.END}")
            print("-" * 40)
            for i, v in enumerate(type_violations, 1):
                self._print_violation(v, i)
            print()
            
    def _print_by_domain(self, violations: List[Violation]) -> None:
        """Print violations grouped by domain."""
        domains = {}
        for v in violations:
            key = f"{v.from_domain or 'unknown'} → {v.to_domain or 'unknown'}"
            if key not in domains:
                domains[key] = []
            domains[key].append(v)
            
        for domain_pair, domain_violations in domains.items():
            print(f"{Colors.BOLD}Domain: {domain_pair}{Colors.END}")
            print("-" * 40)
            for i, v in enumerate(domain_violations, 1):
                self._print_violation(v, i)
            print()
            
    def _print_flat(self, violations: List[Violation]) -> None:
        """Print violations in a flat list."""
        for i, v in enumerate(violations, 1):
            self._print_violation(v, i)
            
    def _print_violation(self, violation: Violation, index: int) -> None:
        """Print a single violation."""
        color = self._get_severity_color(violation.severity)
        
        print(f"{index}. {color}{violation.type.value.replace('_', ' ').title()}{Colors.END}")
        
        if violation.is_exempted():
            print(f"   {Colors.CYAN}[EXEMPTED: {violation.exception.reason}]{Colors.END}")
            
        print(f"   File: {violation.file_path}:{violation.line_number}")
        print(f"   Import: {violation.import_statement}")
        print(f"   Violation: {violation.description}")
        
        if self.options.show_suggestions and violation.suggestion:
            print(f"   {Colors.GREEN}Suggestion: {violation.suggestion}{Colors.END}")
            
        if self.options.verbose:
            print(f"   From: {violation.from_package} ({violation.from_domain or 'unknown'})")
            print(f"   To: {violation.to_package} ({violation.to_domain or 'unknown'})")
            
        print()
        
    def _print_statistics(self, result: AnalysisResult) -> None:
        """Print analysis statistics."""
        print(f"{Colors.BOLD}Statistics:{Colors.END}")
        print("-" * 40)
        
        stats = result.statistics
        print(f"Files scanned: {stats.get('total_files_scanned', 0)}")
        print(f"Total imports: {stats.get('total_imports', 0)}")
        print(f"Total packages: {stats.get('total_packages', 0)}")
        print(f"Total violations: {stats.get('total_violations', 0)}")
        
        if result.domain_packages:
            print(f"\n{Colors.BOLD}Packages by domain:{Colors.END}")
            for domain, packages in result.domain_packages.items():
                print(f"  {domain}: {len(packages)} packages")
                
        print()
        
    def _get_severity_color(self, severity: Severity) -> str:
        """Get color for severity level."""
        return {
            Severity.CRITICAL: Colors.RED,
            Severity.WARNING: Colors.YELLOW,
            Severity.INFO: Colors.BLUE
        }.get(severity, Colors.WHITE)


class JsonReporter:
    """Reporter for generating JSON output."""
    
    def report(self, result: AnalysisResult, output_path: Optional[Path] = None) -> str:
        """Generate JSON report."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_violations': len(result.violations),
                'critical': sum(1 for v in result.violations if v.severity == Severity.CRITICAL),
                'warning': sum(1 for v in result.violations if v.severity == Severity.WARNING),
                'info': sum(1 for v in result.violations if v.severity == Severity.INFO),
                'exempted': sum(1 for v in result.violations if v.is_exempted())
            },
            'violations': [
                {
                    'type': v.type.value,
                    'severity': v.severity.value,
                    'from_package': v.from_package,
                    'to_package': v.to_package,
                    'from_domain': v.from_domain,
                    'to_domain': v.to_domain,
                    'file': v.file_path,
                    'line': v.line_number,
                    'import': v.import_statement,
                    'description': v.description,
                    'suggestion': v.suggestion,
                    'exempted': v.is_exempted(),
                    'exemption_reason': v.exception.reason if v.exception else None
                }
                for v in result.violations
            ],
            'statistics': result.statistics,
            'domain_packages': result.domain_packages,
            'dependencies': {k: list(v) for k, v in result.dependencies.items()}
        }
        
        json_str = json.dumps(report_data, indent=2, sort_keys=True)
        
        if output_path:
            output_path.write_text(json_str)
            
        return json_str


class MarkdownReporter:
    """Reporter for generating Markdown output."""
    
    def report(self, result: AnalysisResult, output_path: Optional[Path] = None) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        lines.append("# Domain Boundary Scan Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        
        if not result.violations:
            lines.append("✅ **No boundary violations found!**")
        else:
            total = len(result.violations)
            critical = sum(1 for v in result.violations if v.severity == Severity.CRITICAL)
            warning = sum(1 for v in result.violations if v.severity == Severity.WARNING)
            info = sum(1 for v in result.violations if v.severity == Severity.INFO)
            
            lines.append(f"❌ **Total violations: {total}**")
            lines.append("")
            lines.append(f"- 🔴 Critical: {critical}")
            lines.append(f"- 🟡 Warning: {warning}")
            lines.append(f"- 🔵 Info: {info}")
            
        lines.append("")
        
        # Violations by severity
        for severity in [Severity.CRITICAL, Severity.WARNING, Severity.INFO]:
            severity_violations = [v for v in result.violations if v.severity == severity]
            if severity_violations:
                lines.append(f"## {severity.value.title()} Violations")
                lines.append("")
                
                for v in severity_violations:
                    lines.append(f"### {v.type.value.replace('_', ' ').title()}")
                    lines.append("")
                    lines.append(f"**File:** `{v.file_path}:{v.line_number}`")
                    lines.append("")
                    lines.append("```python")
                    lines.append(v.import_statement)
                    lines.append("```")
                    lines.append("")
                    lines.append(f"**Violation:** {v.description}")
                    lines.append("")
                    if v.suggestion:
                        lines.append(f"**Suggestion:** {v.suggestion}")
                        lines.append("")
                    lines.append("---")
                    lines.append("")
                    
        # Statistics
        lines.append("## Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in result.statistics.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
            
        markdown = "\n".join(lines)
        
        if output_path:
            output_path.write_text(markdown)
            
        return markdown