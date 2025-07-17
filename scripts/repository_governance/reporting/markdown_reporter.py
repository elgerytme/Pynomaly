"""
Markdown reporter for repository governance.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_reporter import BaseReporter


class MarkdownReporter(BaseReporter):
    """Markdown reporter for governance results."""
    
    def __init__(self, output_path: Optional[Path] = None, include_toc: bool = True):
        """Initialize the Markdown reporter."""
        super().__init__(output_path)
        self.include_toc = include_toc
    
    @property
    def format_name(self) -> str:
        """Name of the report format."""
        return "Markdown"
    
    @property
    def file_extension(self) -> str:
        """File extension for this report format."""
        return "md"
    
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate a Markdown report from check and fix results."""
        lines = []
        
        # Title
        lines.append("# Repository Governance Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Table of Contents
        if self.include_toc:
            lines.extend(self._generate_toc(fix_results is not None))
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
        
        return "\\n".join(lines)
    
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the report to a file."""
        try:
            if filename is None:
                filename = self.create_default_filename()
            
            if self.output_path is None:
                print(report_content)
                return True
            
            output_file = self.output_path / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            output_file.write_text(report_content, encoding='utf-8')
            print(f"Markdown report saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save Markdown report: {e}")
            return False
    
    def _generate_toc(self, has_fixes: bool) -> List[str]:
        """Generate table of contents."""
        lines = []
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("- [Summary](#summary)")
        lines.append("- [Overall Score](#overall-score)")
        lines.append("- [Check Results](#check-results)")
        if has_fixes:
            lines.append("- [Fix Results](#fix-results)")
        lines.append("- [Recommendations](#recommendations)")
        
        return lines
    
    def _generate_summary_section(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> List[str]:
        """Generate the summary section."""
        lines = []
        stats = self.extract_summary_stats(check_results)
        
        lines.append("## Summary")
        lines.append("")
        
        # Check summary table
        lines.append("### Check Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Checkers Run | {stats['checkers_run']} |")
        lines.append(f"| Checkers Passed | {stats['checkers_passed']} |")
        lines.append(f"| Checkers Failed | {stats['checkers_failed']} |")
        lines.append(f"| Total Violations | {stats['total_violations']} |")
        lines.append("")
        
        # Severity breakdown
        lines.append("### Violations by Severity")
        lines.append("")
        lines.append("| Severity | Count | Icon |")
        lines.append("|----------|-------|------|")
        lines.append(f"| High | {stats['high_severity']} | 🔴 |")
        lines.append(f"| Medium | {stats['medium_severity']} | 🟠 |")
        lines.append(f"| Low | {stats['low_severity']} | 🟡 |")
        lines.append(f"| Info | {stats['info_severity']} | 🟢 |")
        lines.append("")
        
        # Fix summary if available
        if fix_results:
            fix_stats = self.extract_fix_summary(fix_results)
            lines.append("### Fix Summary")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Fixes Attempted | {fix_stats['total_fixes_attempted']} |")
            lines.append(f"| Successful | {fix_stats['successful_fixes']} |")
            lines.append(f"| Failed | {fix_stats['failed_fixes']} |")
            lines.append(f"| Files Changed | {fix_stats['files_changed']} |")
            lines.append("")
        
        return lines
    
    def _generate_score_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the score section."""
        lines = []
        overall_score = self.calculate_overall_score(check_results)
        grade = self.get_score_grade(overall_score)
        
        lines.append("## Overall Score")
        lines.append("")
        
        # Score badge
        score_color = "green" if overall_score >= 80 else "yellow" if overall_score >= 60 else "red"
        lines.append(f"![Score](https://img.shields.io/badge/Score-{overall_score:.1f}%25-{score_color})")
        lines.append(f"![Grade](https://img.shields.io/badge/Grade-{grade}-{score_color})")
        lines.append("")
        
        # Score breakdown by checker
        lines.append("### Score Breakdown")
        lines.append("")
        lines.append("| Checker | Score | Status |")
        lines.append("|---------|-------|--------|")
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict):
                score = result.get("score", 0)
                violations = result.get("violations", [])
                status = "✅ Pass" if not violations else "❌ Fail"
                lines.append(f"| {checker_name} | {score:.1f} | {status} |")
        
        lines.append("")
        
        return lines
    
    def _generate_check_results_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the check results section."""
        lines = []
        lines.append("## Check Results")
        lines.append("")
        
        for checker_name, result in check_results.items():
            if not isinstance(result, dict):
                continue
            
            violations = result.get("violations", [])
            score = result.get("score", 0)
            
            # Checker subsection
            lines.append(f"### {checker_name}")
            lines.append("")
            lines.append(f"**Score:** {score:.1f}/100")
            lines.append("")
            
            if violations:
                lines.append(f"**Violations:** {len(violations)}")
                lines.append("")
                
                for violation in violations:
                    severity = violation.get("severity", "info")
                    message = violation.get("message", "No message")
                    total_count = violation.get("total_count", 0)
                    
                    severity_icon = self.get_severity_icon(severity)
                    lines.append(f"- {severity_icon} **{severity.upper()}**: {message}")
                    
                    if total_count > 0:
                        lines.append(f"  - Count: {total_count}")
                    
                    # Show details if available
                    if "violations" in violation:
                        violation_details = violation["violations"]
                        if isinstance(violation_details, list) and len(violation_details) > 0:
                            lines.append("  - Details:")
                            for detail in violation_details[:3]:  # Show first 3 details
                                if isinstance(detail, dict):
                                    file_path = detail.get("file", "")
                                    line = detail.get("line", "")
                                    if file_path:
                                        lines.append(f"    - `{file_path}`{f':{line}' if line else ''}")
                            if len(violation_details) > 3:
                                lines.append(f"    - ... and {len(violation_details) - 3} more")
                
                lines.append("")
            else:
                lines.append("✅ **No violations found**")
                lines.append("")
        
        return lines
    
    def _generate_fix_results_section(self, fix_results: Dict[str, Any]) -> List[str]:
        """Generate the fix results section."""
        lines = []
        lines.append("## Fix Results")
        lines.append("")
        
        for fixer_name, results in fix_results.items():
            if not isinstance(results, list):
                continue
            
            lines.append(f"### {fixer_name}")
            lines.append("")
            
            successful_fixes = [r for r in results if r.get("success", False)]
            failed_fixes = [r for r in results if not r.get("success", False)]
            
            lines.append(f"**Total Attempts:** {len(results)}")
            lines.append(f"**Successful:** {len(successful_fixes)}")
            lines.append(f"**Failed:** {len(failed_fixes)}")
            lines.append("")
            
            if successful_fixes:
                lines.append("#### Successful Fixes")
                for result in successful_fixes:
                    message = result.get("message", "No message")
                    files_changed = result.get("files_changed", [])
                    lines.append(f"- ✅ {message}")
                    if files_changed:
                        lines.append(f"  - Files changed: {len(files_changed)}")
                lines.append("")
            
            if failed_fixes:
                lines.append("#### Failed Fixes")
                for result in failed_fixes:
                    message = result.get("message", "No message")
                    lines.append(f"- ❌ {message}")
                lines.append("")
        
        return lines
    
    def _generate_recommendations_section(self, check_results: Dict[str, Any]) -> List[str]:
        """Generate the recommendations section."""
        lines = []
        recommendations = self.get_recommendations(check_results)
        
        lines.append("## Recommendations")
        lines.append("")
        
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                lines.append(f"{i}. {recommendation}")
        else:
            lines.append("✅ No recommendations - all checks passed!")
        
        lines.append("")
        
        return lines