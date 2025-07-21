#!/usr/bin/env python3
"""
Notification script for maintenance workflow violations and alerts.

This script handles notifications for quality violations, security issues,
and other maintenance-related alerts.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class NotificationManager:
    """Handles notifications for maintenance workflow violations."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = os.getenv("GITHUB_REPOSITORY")
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")

    def parse_quality_reports(self, reports_dir: Path) -> dict[str, Any]:
        """Parse quality reports and extract violations."""
        violations = {
            "ruff": [],
            "mypy": [],
            "bandit": [],
            "safety": [],
            "pip_audit": [],
        }

        total_violations = 0

        try:
            # Parse Ruff report
            ruff_report = reports_dir / "ruff-report.json"
            if ruff_report.exists():
                with open(ruff_report) as f:
                    ruff_data = json.load(f)
                    for item in ruff_data:
                        violations["ruff"].append(
                            {
                                "file": item.get("filename", "unknown"),
                                "line": item.get("location", {}).get("row", 0),
                                "message": item.get("message", ""),
                                "code": item.get("code", ""),
                                "severity": item.get("severity", "info"),
                            }
                        )
                    total_violations += len(violations["ruff"])

        except Exception as e:
            logger.warning(f"Failed to parse Ruff report: {e}")

        try:
            # Parse MyPy report
            mypy_report = reports_dir / "mypy-report.txt"
            if mypy_report.exists():
                with open(mypy_report) as f:
                    for line in f:
                        line = line.strip()
                        if line and ":" in line:
                            parts = line.split(":", 3)
                            if len(parts) >= 3:
                                violations["mypy"].append(
                                    {
                                        "file": parts[0],
                                        "line": parts[1],
                                        "message": parts[2] if len(parts) > 2 else "",
                                        "severity": "error"
                                        if "error:" in line
                                        else "warning",
                                    }
                                )
                    total_violations += len(violations["mypy"])

        except Exception as e:
            logger.warning(f"Failed to parse MyPy report: {e}")

        try:
            # Parse Bandit report
            bandit_report = reports_dir / "bandit-report.json"
            if bandit_report.exists():
                with open(bandit_report) as f:
                    bandit_data = json.load(f)
                    for result in bandit_data.get("results", []):
                        violations["bandit"].append(
                            {
                                "file": result.get("filename", "unknown"),
                                "line": result.get("line_number", 0),
                                "message": result.get("issue_text", ""),
                                "test_id": result.get("test_id", ""),
                                "severity": result.get(
                                    "issue_severity", "info"
                                ).lower(),
                                "confidence": result.get(
                                    "issue_confidence", "unknown"
                                ).lower(),
                            }
                        )
                    total_violations += len(violations["bandit"])

        except Exception as e:
            logger.warning(f"Failed to parse Bandit report: {e}")

        try:
            # Parse Safety report
            safety_report = reports_dir / "safety-report.json"
            if safety_report.exists():
                with open(safety_report) as f:
                    safety_data = json.load(f)
                    for vuln in safety_data.get("vulnerabilities", []):
                        violations["safety"].append(
                            {
                                "package": vuln.get("package_name", "unknown"),
                                "version": vuln.get("analyzed_version", "unknown"),
                                "vulnerability": vuln.get("vulnerability_id", ""),
                                "severity": vuln.get("severity", "unknown"),
                                "description": vuln.get("advisory", ""),
                            }
                        )
                    total_violations += len(violations["safety"])

        except Exception as e:
            logger.warning(f"Failed to parse Safety report: {e}")

        try:
            # Parse pip-audit report
            pip_audit_report = reports_dir / "pip-audit-report.json"
            if pip_audit_report.exists():
                with open(pip_audit_report) as f:
                    pip_audit_data = json.load(f)
                    for vuln in pip_audit_data:
                        violations["pip_audit"].append(
                            {
                                "package": vuln.get("name", "unknown"),
                                "version": vuln.get("version", "unknown"),
                                "vulnerability": vuln.get("id", ""),
                                "description": vuln.get("description", ""),
                                "aliases": vuln.get("aliases", []),
                            }
                        )
                    total_violations += len(violations["pip_audit"])

        except Exception as e:
            logger.warning(f"Failed to parse pip-audit report: {e}")

        return {
            "total_violations": total_violations,
            "violations": violations,
            "timestamp": datetime.now().isoformat(),
        }

    def create_github_issue(self, violations: dict[str, Any], threshold: int) -> bool:
        """Create a GitHub issue for quality violations."""
        if not self.github_token or not self.github_repo:
            logger.warning("GitHub token or repository not configured")
            return False

        total_violations = violations["total_violations"]

        if total_violations < threshold:
            logger.info(
                f"Violations ({total_violations}) below threshold ({threshold})"
            )
            return True

        # Create issue title and body
        title = f"Quality Violations Detected - {total_violations} issues found"

        body = f"""# 🚨 Quality Violations Detected

**Total violations**: {total_violations}
**Threshold**: {threshold}
**Timestamp**: {violations["timestamp"]}

## Summary

"""

        # Add violations by tool
        for tool, tool_violations in violations["violations"].items():
            if tool_violations:
                body += f"### {tool.upper()} - {len(tool_violations)} issues\n\n"

                for i, violation in enumerate(tool_violations[:5]):  # Limit to first 5
                    if tool == "ruff":
                        body += f"- **{violation['file']}:{violation['line']}** - {violation['message']} (`{violation['code']}`)\n"
                    elif tool == "mypy":
                        body += f"- **{violation['file']}:{violation['line']}** - {violation['message']}\n"
                    elif tool == "bandit":
                        body += f"- **{violation['file']}:{violation['line']}** - {violation['message']} ({violation['severity']})\n"
                    elif tool == "safety":
                        body += f"- **{violation['package']}** {violation['version']} - {violation['vulnerability']} ({violation['severity']})\n"
                    elif tool == "pip_audit":
                        body += f"- **{violation['package']}** {violation['version']} - {violation['vulnerability']}\n"

                if len(tool_violations) > 5:
                    body += f"... and {len(tool_violations) - 5} more issues\n\n"
                else:
                    body += "\n"

        body += """
## Action Required

Please review and fix the quality violations found during the maintenance scan.

### Next Steps:
1. Review the full reports in the GitHub Actions artifacts
2. Fix high-priority issues first (security vulnerabilities, type errors)
3. Address linting and formatting issues
4. Re-run the maintenance checks locally before pushing

### Local Testing:
```bash
# Run the same checks locally
python scripts/validation/validate_structure.py
ruff check src/ tests/
mypy src/anomaly_detection/
bandit -r src/
safety check --full-report
pip-audit
```

---
*This issue was automatically created by the maintenance workflow*
"""

        # Create the issue
        url = f"https://api.github.com/repos/{self.github_repo}/issues"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        data = {
            "title": title,
            "body": body,
            "labels": ["maintenance", "quality", "automated"],
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()

            issue_data = response.json()
            issue_number = issue_data["number"]
            issue_url = issue_data["html_url"]

            logger.info(f"Created GitHub issue #{issue_number}: {issue_url}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            return False

    def send_slack_notification(
        self, violations: dict[str, Any], threshold: int
    ) -> bool:
        """Send Slack notification for quality violations."""
        if not self.slack_webhook:
            logger.info("Slack webhook not configured")
            return True

        total_violations = violations["total_violations"]

        if total_violations < threshold:
            logger.info(
                f"Violations ({total_violations}) below threshold ({threshold})"
            )
            return True

        # Create Slack message
        message = {
            "text": f"🚨 Quality Violations Detected: {total_violations} issues",
            "attachments": [
                {
                    "color": "danger"
                    if total_violations > threshold * 2
                    else "warning",
                    "fields": [
                        {
                            "title": "Total Violations",
                            "value": str(total_violations),
                            "short": True,
                        },
                        {"title": "Threshold", "value": str(threshold), "short": True},
                        {
                            "title": "Repository",
                            "value": self.github_repo or "unknown",
                            "short": True,
                        },
                        {
                            "title": "Timestamp",
                            "value": violations["timestamp"],
                            "short": True,
                        },
                    ],
                }
            ],
        }

        # Add breakdown by tool
        breakdown = []
        for tool, tool_violations in violations["violations"].items():
            if tool_violations:
                breakdown.append(f"• {tool.upper()}: {len(tool_violations)} issues")

        if breakdown:
            message["attachments"].append(
                {
                    "color": "warning",
                    "title": "Breakdown by Tool",
                    "text": "\n".join(breakdown),
                }
            )

        try:
            response = requests.post(self.slack_webhook, json=message)
            response.raise_for_status()

            logger.info("Slack notification sent successfully")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def check_violations(self, reports_dir: Path, threshold: int) -> bool:
        """Check for violations and send notifications if needed."""
        logger.info(f"Checking violations in {reports_dir} with threshold {threshold}")

        # Parse reports
        violations = self.parse_quality_reports(reports_dir)

        total_violations = violations["total_violations"]
        logger.info(f"Found {total_violations} total violations")

        if total_violations < threshold:
            logger.info("No notifications needed - below threshold")
            return True

        # Send notifications
        success = True

        # Create GitHub issue
        if not self.create_github_issue(violations, threshold):
            success = False

        # Send Slack notification
        if not self.send_slack_notification(violations, threshold):
            success = False

        return success


def main():
    """Main entry point for notification script."""
    parser = argparse.ArgumentParser(
        description="Send notifications for maintenance violations"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check violations command
    check_parser = subparsers.add_parser(
        "check-violations", help="Check for violations and send notifications"
    )
    check_parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory containing quality reports",
    )
    check_parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Violation threshold for notifications",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create notification manager
    notifier = NotificationManager()

    if args.command == "check-violations":
        success = notifier.check_violations(args.reports_dir, args.threshold)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
