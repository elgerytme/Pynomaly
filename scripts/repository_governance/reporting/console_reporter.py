"""
Console reporter for repository governance.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

from .base_reporter import BaseReporter


class ConsoleReporter(BaseReporter):
    """Console reporter for governance results."""
    
    def __init__(self, output_path: Optional[Path] = None, use_colors: bool = True):
        """Initialize the console reporter."""
        super().__init__(output_path)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    @property
    def format_name(self) -> str:
        """Name of the report format."""
        return "Console"
    
    @property
    def file_extension(self) -> str:
        """File extension for this report format."""
        return "txt"
    
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate a console report from check and fix results."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(self._colorize("REPOSITORY GOVERNANCE REPORT", "bold"))
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.extend(self._generate_summary_section(check_results, fix_results))
        lines.append("")
        
        # Overall Score
        lines.extend(self._generate_score_section(check_results))
        lines.append("")
        
        # Check Results
        lines.extend(self._generate_check_results_section(check_results))
        lines.append("")
        
        # Fix Results
        if fix_results:
            lines.extend(self._generate_fix_results_section(fix_results))
            lines.append("")
        
        # Recommendations
        lines.extend(self._generate_recommendations_section(check_results))
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        
        return "\\n".join(lines)
    
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the report to a file."""
        if self.output_path is None:
            print(report_content)
            return True
        
        try:
            if filename is None:
                filename = self.create_default_filename()
            
            output_file = self.output_path / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove color codes when saving to file
            clean_content = self._remove_color_codes(report_content)
            output_file.write_text(clean_content, encoding='utf-8')
            
            print(f"Report saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return False
    
    def _generate_summary_section(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> List[str]:
        """Generate the summary section."""
        lines = []
        stats = self.extract_summary_stats(check_results)
        
        lines.append(self._colorize("SUMMARY", "bold"))
        lines.append("-" * 40)
        lines.append(f"Checkers Run: {stats['checkers_run']}")
        lines.append(f"Checkers Passed: {self._colorize(str(stats['checkers_passed']), 'green')}")
        lines.append(f"Checkers Failed: {self._colorize(str(stats['checkers_failed']), 'red')}")
        lines.append(f"Total Violations: {self._colorize(str(stats['total_violations']), 'red')}")
        lines.append("")
        
        # Severity breakdown
        lines.append("Violations by Severity:")
        lines.append(f"  High:   {self._colorize(str(stats['high_severity']), 'red')}")
        lines.append(f"  Medium: {self._colorize(str(stats['medium_severity']), 'yellow')}")
        lines.append(f"  Low:    {self._colorize(str(stats['low_severity']), 'cyan')}")
        lines.append(f"  Info:   {self._colorize(str(stats['info_severity']), 'green')}")
        
        # Fix summary if available
        if fix_results:
            fix_stats = self.extract_fix_summary(fix_results)
            lines.append("")
            lines.append("Fix Summary:")
            lines.append(f"  Fixes Attempted: {fix_stats['total_fixes_attempted']}")
            lines.append(f"  Successful: {self._colorize(str(fix_stats['successful_fixes']), 'green')}")
            lines.append(f"  Failed: {self._colorize(str(fix_stats['failed_fixes']), 'red')}")
            lines.append(f"  Files Changed: {fix_stats['files_changed']}")
        
        return lines
    
    def _generate_score_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the score section."""
        lines = []
        overall_score = self.calculate_overall_score(check_results)
        grade = self.get_score_grade(overall_score)
        
        lines.append(self._colorize("OVERALL SCORE", "bold"))
        lines.append("-" * 40)
        
        # Score with color based on grade
        score_color = "green" if overall_score >= 80 else "yellow" if overall_score >= 60 else "red"
        lines.append(f"Score: {self._colorize(f'{overall_score:.1f}/100', score_color)}")
        lines.append(f"Grade: {self._colorize(grade, score_color)}")
        
        return lines
    
    def _generate_check_results_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the check results section."""
        lines = []
        lines.append(self._colorize("CHECK RESULTS", "bold"))
        lines.append("-" * 40)
        
        for checker_name, result in check_results.items():
            if not isinstance(result, dict):
                continue
            
            violations = result.get("violations", [])
            score = result.get("score", 0)
            
            # Checker header
            status = "✓" if not violations else "✗"
            status_color = "green" if not violations else "red"
            lines.append(f"{self._colorize(status, status_color)} {checker_name} (Score: {score:.1f})")
            
            if violations:
                for violation in violations:
                    severity = violation.get("severity", "info")
                    message = violation.get("message", "No message")
                    
                    # Indent violation details
                    severity_color = self._get_severity_color(severity)
                    lines.append(f"    {self._colorize(severity.upper(), severity_color)}: {message}")
                    
                    # Show violation count if available
                    total_count = violation.get("total_count", 0)
                    if total_count > 0:
                        lines.append(f"        Count: {total_count}")
            
            lines.append("")
        
        return lines
    
    def _generate_fix_results_section(self, fix_results: Dict[str, Any]) -> List[str]:
        """Generate the fix results section."""
        lines = []
        lines.append(self._colorize("FIX RESULTS", "bold"))
        lines.append("-" * 40)
        
        for fixer_name, results in fix_results.items():
            if not isinstance(results, list):
                continue
            
            lines.append(f"Fixer: {fixer_name}")
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                success = result.get("success", False)
                message = result.get("message", "No message")
                files_changed = result.get("files_changed", [])
                
                # Fix result
                status = "✓" if success else "✗"
                status_color = "green" if success else "red"
                lines.append(f"  {self._colorize(status, status_color)} {message}")
                
                # Show changed files if any
                if files_changed:
                    lines.append(f"      Files changed: {len(files_changed)}")
                    for file_path in files_changed[:5]:  # Show first 5 files
                        lines.append(f"        - {file_path}")
                    if len(files_changed) > 5:
                        lines.append(f"        ... and {len(files_changed) - 5} more")
            
            lines.append("")
        
        return lines
    
    def _generate_recommendations_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the recommendations section."""
        lines = []
        recommendations = self.get_recommendations(check_results)
        
        if recommendations:
            lines.append(self._colorize("RECOMMENDATIONS", "bold"))
            lines.append("-" * 40)
            
            for i, recommendation in enumerate(recommendations, 1):
                lines.append(f"{i}. {recommendation}")
        
        return lines
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        
        colors = {
            "red": "\\033[31m",
            "green": "\\033[32m",
            "yellow": "\\033[33m",
            "blue": "\\033[34m",
            "magenta": "\\033[35m",
            "cyan": "\\033[36m",
            "white": "\\033[37m",
            "bold": "\\033[1m",
            "reset": "\\033[0m"
        }
        
        color_code = colors.get(color, "")
        reset_code = colors["reset"]
        
        return f"{color_code}{text}{reset_code}"
    
    def _get_severity_color(self, severity: str) -> str:
        """Get console color for severity."""
        color_map = {
            "high": "red",
            "medium": "yellow",
            "low": "cyan",
            "info": "green"
        }
        return color_map.get(severity, "white")
    
    def _remove_color_codes(self, text: str) -> str:
        """Remove ANSI color codes from text."""
        import re
        ansi_escape = re.compile(r'\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)