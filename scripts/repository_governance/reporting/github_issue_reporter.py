"""
GitHub Issue reporter for repository governance.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_reporter import BaseReporter


class GitHubIssueReporter(BaseReporter):
    """GitHub Issue reporter for governance results."""
    
    def __init__(self, output_path: Optional[Path] = None, create_individual_issues: bool = True):
        """Initialize the GitHub Issue reporter."""
        super().__init__(output_path)
        self.create_individual_issues = create_individual_issues
    
    @property
    def format_name(self) -> str:
        """Name of the report format."""
        return "GitHub Issues"
    
    @property
    def file_extension(self) -> str:
        """File extension for this report format."""
        return "json"
    
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate GitHub issues from check and fix results."""
        issues = []
        
        if self.create_individual_issues:
            issues.extend(self._create_individual_issues(check_results, fix_results))
        else:
            issues.append(self._create_summary_issue(check_results, fix_results))
        
        return json.dumps(issues, indent=2, ensure_ascii=False)
    
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the GitHub issues to a file."""
        try:
            if filename is None:
                filename = self.create_default_filename("github_issues")
            
            if self.output_path is None:
                print(report_content)
                return True
            
            output_file = self.output_path / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            output_file.write_text(report_content, encoding='utf-8')
            print(f"GitHub issues saved to: {output_file}")
            
            # Also save as shell script for easy GitHub CLI usage
            self._save_github_cli_script(json.loads(report_content))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save GitHub issues: {e}")
            return False
    
    def _create_individual_issues(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Create individual GitHub issues for each violation type."""
        issues = []
        
        for checker_name, result in check_results.items():
            if not isinstance(result, dict):
                continue
            
            violations = result.get("violations", [])
            if not violations:
                continue
            
            for violation in violations:
                issue = self._create_issue_for_violation(checker_name, violation, fix_results)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def _create_summary_issue(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a single summary issue for all violations."""
        stats = self.extract_summary_stats(check_results)
        overall_score = self.calculate_overall_score(check_results)
        grade = self.get_score_grade(overall_score)
        
        title = f"Repository Governance Report - Score: {overall_score:.1f} ({grade})"
        
        body_lines = []
        body_lines.append("# Repository Governance Report")
        body_lines.append("")
        body_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        body_lines.append(f"**Overall Score:** {overall_score:.1f}/100 ({grade})")
        body_lines.append("")
        
        # Summary
        body_lines.append("## Summary")
        body_lines.append("")
        body_lines.append(f"- **Checkers Run:** {stats['checkers_run']}")
        body_lines.append(f"- **Checkers Passed:** {stats['checkers_passed']}")
        body_lines.append(f"- **Checkers Failed:** {stats['checkers_failed']}")
        body_lines.append(f"- **Total Violations:** {stats['total_violations']}")
        body_lines.append("")
        
        # Violations by severity
        body_lines.append("## Violations by Severity")
        body_lines.append("")
        body_lines.append(f"- 🔴 **High:** {stats['high_severity']}")
        body_lines.append(f"- 🟠 **Medium:** {stats['medium_severity']}")
        body_lines.append(f"- 🟡 **Low:** {stats['low_severity']}")
        body_lines.append(f"- 🟢 **Info:** {stats['info_severity']}")
        body_lines.append("")
        
        # Check results
        body_lines.append("## Check Results")
        body_lines.append("")
        
        for checker_name, result in check_results.items():
            if not isinstance(result, dict):
                continue
            
            violations = result.get("violations", [])
            score = result.get("score", 0)
            
            status = "✅" if not violations else "❌"
            body_lines.append(f"### {status} {checker_name} (Score: {score:.1f})")
            
            if violations:
                for violation in violations:
                    severity = violation.get("severity", "info")
                    message = violation.get("message", "No message")
                    total_count = violation.get("total_count", 0)
                    
                    severity_icon = self.get_severity_icon(severity)
                    body_lines.append(f"- {severity_icon} **{severity.upper()}:** {message}")
                    if total_count > 0:
                        body_lines.append(f"  - Count: {total_count}")
            
            body_lines.append("")
        
        # Fix results
        if fix_results:
            fix_stats = self.extract_fix_summary(fix_results)
            body_lines.append("## Fix Results")
            body_lines.append("")
            body_lines.append(f"- **Fixes Attempted:** {fix_stats['total_fixes_attempted']}")
            body_lines.append(f"- **Successful:** {fix_stats['successful_fixes']}")
            body_lines.append(f"- **Failed:** {fix_stats['failed_fixes']}")
            body_lines.append(f"- **Files Changed:** {fix_stats['files_changed']}")
            body_lines.append("")
        
        # Recommendations
        recommendations = self.get_recommendations(check_results)
        if recommendations:
            body_lines.append("## Recommendations")
            body_lines.append("")
            for i, recommendation in enumerate(recommendations, 1):
                body_lines.append(f"{i}. {recommendation}")
            body_lines.append("")
        
        # Labels
        labels = ["governance", "repository-quality"]
        if stats['high_severity'] > 0:
            labels.append("high-priority")
        elif stats['medium_severity'] > 0:
            labels.append("medium-priority")
        else:
            labels.append("low-priority")
        
        return {
            "title": title,
            "body": "\n".join(body_lines),
            "labels": labels,
            "assignees": []
        }
    
    def _create_issue_for_violation(self, checker_name: str, violation: Dict[str, Any], fix_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a GitHub issue for a specific violation."""
        violation_type = violation.get("type", "unknown")
        severity = violation.get("severity", "info")
        message = violation.get("message", "No message")
        total_count = violation.get("total_count", 0)
        
        title = f"[{severity.upper()}] {checker_name}: {violation_type}"
        
        body_lines = []
        body_lines.append(f"# {violation_type}")
        body_lines.append("")
        body_lines.append(f"**Checker:** {checker_name}")
        body_lines.append(f"**Severity:** {severity}")
        body_lines.append(f"**Count:** {total_count}")
        body_lines.append("")
        
        # Description
        body_lines.append("## Description")
        body_lines.append("")
        body_lines.append(message)
        body_lines.append("")
        
        # Violation details
        violation_details = violation.get("violations", [])
        if violation_details:
            body_lines.append("## Affected Files")
            body_lines.append("")
            
            for detail in violation_details[:10]:  # Show first 10 details
                if isinstance(detail, dict):
                    file_path = detail.get("file", "")
                    line = detail.get("line", "")
                    if file_path:
                        body_lines.append(f"- `{file_path}`{f':{line}' if line else ''}")
                        
                        # Add context if available
                        if "context" in detail:
                            body_lines.append(f"  - {detail['context']}")
            
            if len(violation_details) > 10:
                body_lines.append(f"- ... and {len(violation_details) - 10} more files")
            
            body_lines.append("")
        
        # Fix information
        if fix_results:
            body_lines.append("## Fix Information")
            body_lines.append("")
            
            # Check if this violation type has been addressed
            for fixer_name, results in fix_results.items():
                if isinstance(results, list):
                    relevant_fixes = [r for r in results if violation_type in r.get("message", "")]
                    if relevant_fixes:
                        body_lines.append(f"### {fixer_name}")
                        for fix in relevant_fixes:
                            success = fix.get("success", False)
                            fix_message = fix.get("message", "No message")
                            status = "✅" if success else "❌"
                            body_lines.append(f"- {status} {fix_message}")
                        body_lines.append("")
        
        # Labels
        labels = ["governance", f"checker-{checker_name.lower()}", f"severity-{severity}"]
        if violation_type:
            labels.append(f"type-{violation_type.replace('_', '-')}")
        
        return {
            "title": title,
            "body": "\n".join(body_lines),
            "labels": labels,
            "assignees": []
        }
    
    def _save_github_cli_script(self, issues: List[Dict[str, Any]]) -> None:
        """Save a shell script to create GitHub issues using GitHub CLI."""
        if self.output_path is None:
            return
        
        script_path = self.output_path / "create_github_issues.sh"
        
        script_lines = []
        script_lines.append("#!/bin/bash")
        script_lines.append("")
        script_lines.append("# Script to create GitHub issues for repository governance")
        script_lines.append("# Requires GitHub CLI (gh) to be installed and authenticated")
        script_lines.append("")
        script_lines.append("set -e")
        script_lines.append("")
        
        for i, issue in enumerate(issues, 1):
            title = issue.get("title", "")
            body = issue.get("body", "")
            labels = issue.get("labels", [])
            assignees = issue.get("assignees", [])
            
            script_lines.append(f"echo \"Creating issue {i}/{len(issues)}: {title}\"")
            script_lines.append("gh issue create \\\\")
            script_lines.append(f'  --title "{title}" \\\\')
            script_lines.append(f'  --body "{body}" \\\\')
            
            if labels:
                labels_str = ",".join(labels)
                script_lines.append(f'  --label "{labels_str}" \\\\')
            
            if assignees:
                assignees_str = ",".join(assignees)
                script_lines.append(f'  --assignee "{assignees_str}" \\\\')
            
            script_lines.append("  || echo \"Failed to create issue\"")
            script_lines.append("")
        
        script_lines.append("echo \"All issues created!\"")
        
        try:
            script_path.write_text("\n".join(script_lines), encoding='utf-8')
            # Make script executable
            script_path.chmod(0o755)
            print(f"GitHub CLI script saved to: {script_path}")
        except Exception as e:
            self.logger.error(f"Failed to save GitHub CLI script: {e}")
    
    def create_issue_template(self, issue_type: str) -> Dict[str, Any]:
        """Create a GitHub issue template for a specific issue type."""
        templates = {
            "governance_summary": {
                "title": "Repository Governance Summary",
                "body": "## Repository Governance Summary\n\n**Generated:** {timestamp}\n\n### Summary\n\n- **Overall Score:** {score}\n- **Grade:** {grade}\n\n### Actions Required\n\n- [ ] Review violations\n- [ ] Apply fixes\n- [ ] Update documentation\n",
                "labels": ["governance", "repository-quality"],
                "assignees": []
            },
            "high_priority_violation": {
                "title": "[HIGH] Repository Governance Violation",
                "body": "## High Priority Violation\\n\\n**Checker:** {checker}\\n**Type:** {type}\\n**Count:** {count}\\n\\n### Description\\n\\n{description}\\n\\n### Actions Required\\n\\n- [ ] Fix violations\\n- [ ] Update processes\\n- [ ] Document changes\\n",
                "labels": ["governance", "high-priority", "bug"],
                "assignees": []
            }
        }
        
        return templates.get(issue_type, {
            "title": f"Repository Governance Issue: {issue_type}",
            "body": "## Issue Description\\n\\nPlease provide details about this governance issue.\\n",
            "labels": ["governance"],
            "assignees": []
        })