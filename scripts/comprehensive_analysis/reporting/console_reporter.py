"""Console reporter for comprehensive static analysis results."""

from typing import Dict, List, Any
from pathlib import Path
import sys

from ..config.manager import AnalysisConfig
from ..orchestrator import ComprehensiveAnalysisResult
from ..tools.adapter_base import Issue


class ConsoleReporter:
    """Generates console output for analysis results."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.colored_output = config.colored_output and sys.stdout.isatty()
    
    def generate_report(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate console report."""
        if not result.success:
            return self._generate_error_report(result)
        
        report_lines = []
        
        # Header
        report_lines.extend(self._generate_header())
        
        # Summary
        report_lines.extend(self._generate_summary(result))
        
        # Issues by severity
        report_lines.extend(self._generate_issues_by_severity(result))
        
        # Issues by file (if context is enabled)
        if self.config.show_context:
            report_lines.extend(self._generate_issues_by_file(result))
        
        # Footer
        report_lines.extend(self._generate_footer(result))
        
        return "\n".join(report_lines)
    
    def _generate_header(self) -> List[str]:
        """Generate report header."""
        return [
            "",
            self._colorize("🔍 Comprehensive Static Analysis Report", "bold"),
            "=" * 50,
            "",
        ]
    
    def _generate_summary(self, result: ComprehensiveAnalysisResult) -> List[str]:
        """Generate summary section."""
        lines = [
            self._colorize("📊 Analysis Summary", "bold"),
            "-" * 20,
        ]
        
        # Basic stats
        lines.append(f"Files analyzed: {result.files_analyzed}")
        lines.append(f"Execution time: {result.execution_time:.2f}s")
        lines.append(f"Tools used: {', '.join(r.tool for r in result.results)}")
        
        # Issue counts
        issues_by_severity = result.get_issues_by_severity()
        total_issues = sum(len(issues) for issues in issues_by_severity.values())
        
        lines.append(f"Total issues: {total_issues}")
        
        if issues_by_severity["error"]:
            lines.append(f"  {self._colorize('❌ Errors:', 'red')} {len(issues_by_severity['error'])}")
        
        if issues_by_severity["warning"]:
            lines.append(f"  {self._colorize('⚠️  Warnings:', 'yellow')} {len(issues_by_severity['warning'])}")
        
        if issues_by_severity["info"]:
            lines.append(f"  {self._colorize('ℹ️  Info:', 'blue')} {len(issues_by_severity['info'])}")
        
        # Files with issues
        files_with_issues = len(result.get_issues_by_file())
        if files_with_issues > 0:
            lines.append(f"Files with issues: {files_with_issues}")
        
        lines.append("")
        return lines
    
    def _generate_issues_by_severity(self, result: ComprehensiveAnalysisResult) -> List[str]:
        """Generate issues grouped by severity."""
        lines = []
        issues_by_severity = result.get_issues_by_severity()
        
        # Show errors first
        if issues_by_severity["error"]:
            lines.extend(self._generate_severity_section("Errors", issues_by_severity["error"], "red"))
        
        # Then warnings
        if issues_by_severity["warning"]:
            lines.extend(self._generate_severity_section("Warnings", issues_by_severity["warning"], "yellow"))
        
        # Finally info (if context is enabled)
        if self.config.show_context and issues_by_severity["info"]:
            lines.extend(self._generate_severity_section("Info", issues_by_severity["info"], "blue"))
        
        return lines
    
    def _generate_severity_section(self, title: str, issues: List[Issue], color: str) -> List[str]:
        """Generate a section for issues of a specific severity."""
        lines = [
            self._colorize(f"🚨 {title} ({len(issues)} issues)", "bold"),
            "-" * (len(title) + 20),
        ]
        
        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            if issue.file not in issues_by_file:
                issues_by_file[issue.file] = []
            issues_by_file[issue.file].append(issue)
        
        # Sort files by number of issues (descending)
        sorted_files = sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Show issues for each file
        for file_path, file_issues in sorted_files:
            lines.append(f"\n{self._colorize(str(file_path), 'bold')}")
            
            # Sort issues by line number
            sorted_issues = sorted(file_issues, key=lambda x: x.line)
            
            for issue in sorted_issues[:self.config.max_issues_per_file]:
                lines.append(self._format_issue(issue, color))
            
            # Show truncation message if needed
            if len(file_issues) > self.config.max_issues_per_file:
                remaining = len(file_issues) - self.config.max_issues_per_file
                lines.append(f"  ... and {remaining} more issues in this file")
        
        lines.append("")
        return lines
    
    def _generate_issues_by_file(self, result: ComprehensiveAnalysisResult) -> List[str]:
        """Generate issues grouped by file."""
        lines = [
            self._colorize("📁 Issues by File", "bold"),
            "-" * 20,
        ]
        
        issues_by_file = result.get_issues_by_file()
        
        # Sort files by number of issues (descending)
        sorted_files = sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)
        
        for file_path, file_issues in sorted_files:
            severity_counts = {"error": 0, "warning": 0, "info": 0}
            for issue in file_issues:
                severity_counts[issue.severity] += 1
            
            severity_summary = []
            if severity_counts["error"] > 0:
                severity_summary.append(f"{severity_counts['error']} errors")
            if severity_counts["warning"] > 0:
                severity_summary.append(f"{severity_counts['warning']} warnings")
            if severity_counts["info"] > 0:
                severity_summary.append(f"{severity_counts['info']} info")
            
            lines.append(f"{file_path} ({', '.join(severity_summary)})")
        
        lines.append("")
        return lines
    
    def _generate_footer(self, result: ComprehensiveAnalysisResult) -> List[str]:
        """Generate report footer."""
        lines = []
        
        # Performance summary
        if result.execution_time > 0:
            files_per_second = result.files_analyzed / result.execution_time
            lines.append(f"⚡ Performance: {files_per_second:.1f} files/second")
        
        # Success/failure status
        total_errors = len(result.get_issues_by_severity()["error"])
        
        if total_errors == 0:
            lines.append(self._colorize("✅ Analysis completed successfully!", "green"))
        else:
            lines.append(self._colorize(f"❌ Analysis completed with {total_errors} errors", "red"))
        
        lines.append("")
        return lines
    
    def _generate_error_report(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate error report for failed analysis."""
        lines = [
            "",
            self._colorize("❌ Analysis Failed", "red"),
            "=" * 20,
            "",
            f"Error: {result.error_message}",
            f"Execution time: {result.execution_time:.2f}s",
            "",
        ]
        
        return "\n".join(lines)
    
    def _format_issue(self, issue: Issue, color: str) -> str:
        """Format a single issue for display."""
        # Base format
        location = f"{issue.line}:{issue.column}"
        rule_info = f"[{issue.rule}]" if issue.rule else ""
        
        line = f"  {location:<8} {self._colorize(issue.message, color)} {rule_info}"
        
        # Add tool info if multiple tools are used
        if issue.tool:
            line += f" ({issue.tool})"
        
        # Add suggestion if available
        if issue.suggestion:
            line += f"\n           💡 {issue.suggestion}"
        
        return line
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colored output is enabled."""
        if not self.colored_output:
            return text
        
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }
        
        color_code = colors.get(color, "")
        reset_code = colors["reset"]
        
        return f"{color_code}{text}{reset_code}" if color_code else text