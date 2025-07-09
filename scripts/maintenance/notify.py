#!/usr/bin/env python3
"""
Notification and alerting helper for code quality monitoring.

This module provides reusable functions to:
- Open GitHub issues for quality violations
- Send Slack notifications via webhook
- Format quality reports for different channels
- Monitor violation thresholds
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
import structlog
import typer
from pydantic import BaseModel, Field, HttpUrl, validator
from rich.console import Console
from rich.table import Table

logger = structlog.get_logger()
console = Console()

# Configuration Models
class SlackConfig(BaseModel):
    """Configuration for Slack notifications."""
    webhook_url: HttpUrl
    channel: str = Field(default="#code-quality", description="Slack channel to post to")
    username: str = Field(default="Quality Bot", description="Bot username")
    icon_emoji: str = Field(default=":warning:", description="Bot icon emoji")
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        if not str(v).startswith('https://hooks.slack.com/'):
            raise ValueError('Invalid Slack webhook URL')
        return v

class GitHubConfig(BaseModel):
    """Configuration for GitHub issues."""
    token: str = Field(description="GitHub personal access token")
    repository: str = Field(description="Repository in format 'owner/repo'")
    api_url: str = Field(default="https://api.github.com", description="GitHub API URL")
    labels: List[str] = Field(default=["automated", "quality", "maintenance"], description="Issue labels")
    
    @validator('repository')
    def validate_repository(cls, v):
        if '/' not in v or len(v.split('/')) != 2:
            raise ValueError('Repository must be in format "owner/repo"')
        return v

class QualityViolation(BaseModel):
    """Represents a quality violation."""
    tool: str = Field(description="Tool that detected the violation")
    type: str = Field(description="Type of violation")
    file: str = Field(description="File path where violation occurred")
    line: Optional[int] = Field(description="Line number")
    column: Optional[int] = Field(description="Column number") 
    message: str = Field(description="Violation message")
    severity: str = Field(default="warning", description="Severity level")
    rule: Optional[str] = Field(description="Rule identifier")

class QualityReport(BaseModel):
    """Quality report containing violations and metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    total_violations: int = Field(description="Total number of violations")
    violations_by_type: Dict[str, int] = Field(description="Violations grouped by type")
    violations_by_tool: Dict[str, int] = Field(description="Violations grouped by tool")
    violations: List[QualityViolation] = Field(description="List of violations")
    threshold_exceeded: bool = Field(description="Whether threshold was exceeded")
    threshold_limit: int = Field(description="Configured threshold limit")
    
class NotificationConfig(BaseModel):
    """Complete notification configuration."""
    slack: Optional[SlackConfig] = Field(description="Slack configuration")
    github: Optional[GitHubConfig] = Field(description="GitHub configuration")
    violation_threshold: int = Field(default=50, description="Threshold for notifications")
    notification_cooldown: int = Field(default=24, description="Hours between notifications")
    
    @validator('violation_threshold')
    def validate_threshold(cls, v):
        if v < 0:
            raise ValueError('Violation threshold must be non-negative')
        return v

# Notification Classes
class SlackNotifier:
    """Send notifications to Slack via webhook."""
    
    def __init__(self, config: SlackConfig):
        self.config = config
        
    def send_quality_alert(self, report: QualityReport) -> bool:
        """Send quality alert to Slack."""
        try:
            payload = self._build_slack_payload(report)
            response = requests.post(
                str(self.config.webhook_url),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _build_slack_payload(self, report: QualityReport) -> Dict[str, Any]:
        """Build Slack message payload."""
        color = self._get_color_by_severity(report.total_violations)
        
        # Build violation summary
        violations_summary = []
        for tool, count in report.violations_by_tool.items():
            violations_summary.append(f"• {tool}: {count} violations")
        
        # Build top violation types
        top_types = sorted(
            report.violations_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        types_summary = []
        for violation_type, count in top_types:
            types_summary.append(f"• {violation_type}: {count}")
        
        return {
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "channel": self.config.channel,
            "attachments": [
                {
                    "color": color,
                    "title": f"Code Quality Alert - {report.total_violations} violations detected",
                    "text": f"Quality threshold exceeded (limit: {report.threshold_limit})",
                    "fields": [
                        {
                            "title": "Total Violations",
                            "value": str(report.total_violations),
                            "short": True
                        },
                        {
                            "title": "Threshold Status",
                            "value": "⚠️ EXCEEDED" if report.threshold_exceeded else "✅ OK",
                            "short": True
                        },
                        {
                            "title": "Violations by Tool",
                            "value": "\n".join(violations_summary),
                            "short": True
                        },
                        {
                            "title": "Top Violation Types",
                            "value": "\n".join(types_summary),
                            "short": True
                        }
                    ],
                    "footer": "Pynomaly Quality Monitor",
                    "ts": int(report.timestamp.timestamp())
                }
            ]
        }
    
    def _get_color_by_severity(self, violation_count: int) -> str:
        """Get color based on violation count."""
        if violation_count >= 100:
            return "danger"  # Red
        elif violation_count >= 50:
            return "warning"  # Yellow
        else:
            return "good"  # Green

class GitHubNotifier:
    """Create GitHub issues for quality violations."""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.config.token}",
            "Accept": "application/vnd.github.v3+json"
        })
        
    def create_quality_issue(self, report: QualityReport) -> bool:
        """Create GitHub issue for quality violations."""
        try:
            # Check if similar issue already exists
            if self._has_recent_quality_issue():
                logger.info("Recent quality issue already exists, skipping creation")
                return True
            
            issue_data = self._build_issue_data(report)
            owner, repo = self.config.repository.split('/')
            
            response = self.session.post(
                f"{self.config.api_url}/repos/{owner}/{repo}/issues",
                json=issue_data,
                timeout=30
            )
            response.raise_for_status()
            
            issue_data = response.json()
            issue_url = issue_data["html_url"]
            logger.info(f"GitHub issue created: {issue_url}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            return False
    
    def _has_recent_quality_issue(self) -> bool:
        """Check if a recent quality issue already exists."""
        try:
            owner, repo = self.config.repository.split('/')
            
            # Search for recent issues with quality labels
            params = {
                "state": "open",
                "labels": ",".join(self.config.labels),
                "sort": "created",
                "direction": "desc",
                "per_page": 10
            }
            
            response = self.session.get(
                f"{self.config.api_url}/repos/{owner}/{repo}/issues",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            issues = response.json()
            
            # Check if any issue was created in the last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for issue in issues:
                if "Quality Alert" in issue["title"]:
                    created_at = datetime.fromisoformat(
                        issue["created_at"].replace('Z', '+00:00')
                    )
                    if created_at > cutoff_time:
                        return True
            
            return False
            
        except requests.RequestException as e:
            logger.error(f"Error checking for recent issues: {e}")
            return False
    
    def _build_issue_data(self, report: QualityReport) -> Dict[str, Any]:
        """Build GitHub issue data."""
        title = f"Quality Alert - {report.total_violations} violations detected ({report.timestamp.strftime('%Y-%m-%d')})"
        
        # Build violation breakdown
        tool_breakdown = []
        for tool, count in sorted(report.violations_by_tool.items()):
            tool_breakdown.append(f"- **{tool}**: {count} violations")
        
        type_breakdown = []
        for violation_type, count in sorted(report.violations_by_type.items()):
            type_breakdown.append(f"- **{violation_type}**: {count}")
        
        # Build top violations sample
        violations_sample = []
        for violation in report.violations[:10]:  # Top 10 violations
            location = f"{violation.file}"
            if violation.line:
                location += f":{violation.line}"
            if violation.column:
                location += f":{violation.column}"
            
            violations_sample.append(
                f"- `{violation.tool}` ({violation.severity}): {violation.message} - {location}"
            )
        
        body = f"""## 🚨 Code Quality Alert

**Total Violations**: {report.total_violations}  
**Threshold Limit**: {report.threshold_limit}  
**Status**: {'⚠️ EXCEEDED' if report.threshold_exceeded else '✅ OK'}  
**Timestamp**: {report.timestamp.isoformat()}

### Violations by Tool
{chr(10).join(tool_breakdown)}

### Violations by Type
{chr(10).join(type_breakdown)}

### Sample Violations
{chr(10).join(violations_sample)}

### Action Required
- Review and fix the violations above
- Consider updating quality gates if needed
- Check if new patterns need to be added to exclusions

### Automated Report
This issue was automatically generated by the code quality monitoring system.
"""
        
        return {
            "title": title,
            "body": body,
            "labels": self.config.labels
        }

class QualityNotifier:
    """Main notification orchestrator."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.slack_notifier = SlackNotifier(config.slack) if config.slack else None
        self.github_notifier = GitHubNotifier(config.github) if config.github else None
        
    def process_quality_report(self, report: QualityReport) -> bool:
        """Process quality report and send notifications if needed."""
        if not self._should_notify(report):
            logger.info("Quality report below threshold, no notification needed")
            return True
        
        success = True
        
        # Send Slack notification
        if self.slack_notifier:
            if not self.slack_notifier.send_quality_alert(report):
                success = False
        
        # Create GitHub issue
        if self.github_notifier:
            if not self.github_notifier.create_quality_issue(report):
                success = False
        
        return success
    
    def _should_notify(self, report: QualityReport) -> bool:
        """Check if notification should be sent."""
        if not report.threshold_exceeded:
            return False
        
        # Check cooldown period
        cooldown_file = Path("reports/quality/.last_notification")
        if cooldown_file.exists():
            try:
                last_notification = datetime.fromtimestamp(cooldown_file.stat().st_mtime)
                cooldown_period = timedelta(hours=self.config.notification_cooldown)
                
                if datetime.now() - last_notification < cooldown_period:
                    logger.info("Notification cooldown period active, skipping")
                    return False
            except (OSError, ValueError):
                pass
        
        # Update cooldown file
        cooldown_file.parent.mkdir(parents=True, exist_ok=True)
        cooldown_file.touch()
        
        return True

# Utility Functions
def parse_sarif_report(sarif_file: Path) -> List[QualityViolation]:
    """Parse SARIF report and extract violations."""
    violations = []
    
    try:
        with open(sarif_file, 'r') as f:
            sarif_data = json.load(f)
        
        for run in sarif_data.get('runs', []):
            tool_name = run.get('tool', {}).get('driver', {}).get('name', 'unknown')
            
            for result in run.get('results', []):
                for location in result.get('locations', []):
                    physical_location = location.get('physicalLocation', {})
                    artifact_location = physical_location.get('artifactLocation', {})
                    region = physical_location.get('region', {})
                    
                    violation = QualityViolation(
                        tool=tool_name,
                        type=result.get('ruleId', 'unknown'),
                        file=artifact_location.get('uri', 'unknown'),
                        line=region.get('startLine'),
                        column=region.get('startColumn'),
                        message=result.get('message', {}).get('text', 'No message'),
                        severity=result.get('level', 'warning'),
                        rule=result.get('ruleId')
                    )
                    violations.append(violation)
    
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.error(f"Error parsing SARIF report: {e}")
    
    return violations

def parse_json_report(json_file: Path, tool_name: str) -> List[QualityViolation]:
    """Parse JSON report and extract violations."""
    violations = []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON report formats
        if tool_name.lower() == 'bandit':
            for result in data.get('results', []):
                violation = QualityViolation(
                    tool=tool_name,
                    type=result.get('test_id', 'unknown'),
                    file=result.get('filename', 'unknown'),
                    line=result.get('line_number'),
                    column=result.get('col_offset'),
                    message=result.get('issue_text', 'No message'),
                    severity=result.get('issue_severity', 'warning'),
                    rule=result.get('test_id')
                )
                violations.append(violation)
        
        elif tool_name.lower() == 'ruff':
            for result in data:
                violation = QualityViolation(
                    tool=tool_name,
                    type=result.get('code', 'unknown'),
                    file=result.get('filename', 'unknown'),
                    line=result.get('location', {}).get('row'),
                    column=result.get('location', {}).get('column'),
                    message=result.get('message', 'No message'),
                    severity='error' if result.get('code', '').startswith('E') else 'warning',
                    rule=result.get('code')
                )
                violations.append(violation)
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.error(f"Error parsing JSON report: {e}")
    
    return violations

def load_config_from_env() -> NotificationConfig:
    """Load notification configuration from environment variables."""
    config_data = {
        "violation_threshold": int(os.getenv("QUALITY_VIOLATION_THRESHOLD", "50")),
        "notification_cooldown": int(os.getenv("QUALITY_NOTIFICATION_COOLDOWN", "24"))
    }
    
    # Slack configuration
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        config_data["slack"] = {
            "webhook_url": slack_webhook,
            "channel": os.getenv("SLACK_CHANNEL", "#code-quality"),
            "username": os.getenv("SLACK_USERNAME", "Quality Bot"),
            "icon_emoji": os.getenv("SLACK_ICON_EMOJI", ":warning:")
        }
    
    # GitHub configuration
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPOSITORY")
    if github_token and github_repo:
        config_data["github"] = {
            "token": github_token,
            "repository": github_repo,
            "api_url": os.getenv("GITHUB_API_URL", "https://api.github.com"),
            "labels": os.getenv("GITHUB_ISSUE_LABELS", "automated,quality,maintenance").split(',')
        }
    
    return NotificationConfig(**config_data)

# CLI Interface
app = typer.Typer(help="Code quality notification and alerting system")

@app.command()
def check_violations(
    reports_dir: Path = typer.Option("reports/quality", help="Directory containing quality reports"),
    threshold: int = typer.Option(50, help="Violation threshold for notifications"),
    config_file: Optional[Path] = typer.Option(None, help="Configuration file path"),
    dry_run: bool = typer.Option(False, help="Dry run mode - don't send notifications")
):
    """
    Check quality violations and send notifications if threshold is exceeded.
    
    This command processes quality reports and sends notifications to configured
    channels (Slack, GitHub) if the violation count exceeds the threshold.
    """
    try:
        # Load configuration
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = NotificationConfig(**config_data)
        else:
            config = load_config_from_env()
        
        # Override threshold if provided
        if threshold != 50:
            config.violation_threshold = threshold
        
        # Find and process reports
        violations = []
        violations_by_tool = {}
        violations_by_type = {}
        
        # Process SARIF reports
        sarif_files = list(reports_dir.glob("*.sarif"))
        for sarif_file in sarif_files:
            console.print(f"Processing SARIF report: {sarif_file}")
            sarif_violations = parse_sarif_report(sarif_file)
            violations.extend(sarif_violations)
        
        # Process JSON reports
        json_files = list(reports_dir.glob("*-report.json"))
        for json_file in json_files:
            tool_name = json_file.stem.replace('-report', '')
            console.print(f"Processing JSON report: {json_file} (tool: {tool_name})")
            json_violations = parse_json_report(json_file, tool_name)
            violations.extend(json_violations)
        
        # Build aggregated data
        for violation in violations:
            violations_by_tool[violation.tool] = violations_by_tool.get(violation.tool, 0) + 1
            violations_by_type[violation.type] = violations_by_type.get(violation.type, 0) + 1
        
        # Create quality report
        report = QualityReport(
            total_violations=len(violations),
            violations_by_type=violations_by_type,
            violations_by_tool=violations_by_tool,
            violations=violations,
            threshold_exceeded=len(violations) > config.violation_threshold,
            threshold_limit=config.violation_threshold
        )
        
        # Display summary
        table = Table(title="Quality Violations Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Violations", str(report.total_violations))
        table.add_row("Threshold", str(config.violation_threshold))
        table.add_row("Threshold Exceeded", "✅ Yes" if report.threshold_exceeded else "❌ No")
        table.add_row("Notifications", "Disabled (dry-run)" if dry_run else "Enabled")
        
        console.print(table)
        
        # Show violations by tool
        if violations_by_tool:
            tool_table = Table(title="Violations by Tool")
            tool_table.add_column("Tool", style="cyan")
            tool_table.add_column("Count", style="magenta")
            
            for tool, count in sorted(violations_by_tool.items()):
                tool_table.add_row(tool, str(count))
            
            console.print(tool_table)
        
        # Send notifications if needed
        if not dry_run:
            notifier = QualityNotifier(config)
            if notifier.process_quality_report(report):
                console.print("[green]✅ Notifications processed successfully[/green]")
            else:
                console.print("[red]❌ Failed to send some notifications[/red]")
                sys.exit(1)
        else:
            console.print("[yellow]Dry run mode - notifications not sent[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Error in check_violations: {e}")
        sys.exit(1)

@app.command()
def test_notifications(
    config_file: Optional[Path] = typer.Option(None, help="Configuration file path"),
    slack_only: bool = typer.Option(False, help="Test only Slack notifications"),
    github_only: bool = typer.Option(False, help="Test only GitHub notifications")
):
    """Test notification configuration with dummy data."""
    try:
        # Load configuration
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = NotificationConfig(**config_data)
        else:
            config = load_config_from_env()
        
        # Create test report
        test_violations = [
            QualityViolation(
                tool="bandit",
                type="B101",
                file="src/example.py",
                line=42,
                column=10,
                message="Use of assert detected",
                severity="low",
                rule="B101"
            ),
            QualityViolation(
                tool="ruff",
                type="E501",
                file="src/example.py",
                line=85,
                column=80,
                message="Line too long",
                severity="error",
                rule="E501"
            )
        ]
        
        report = QualityReport(
            total_violations=len(test_violations),
            violations_by_type={"B101": 1, "E501": 1},
            violations_by_tool={"bandit": 1, "ruff": 1},
            violations=test_violations,
            threshold_exceeded=True,
            threshold_limit=config.violation_threshold
        )
        
        # Test notifications
        success = True
        
        if not github_only and config.slack:
            console.print("Testing Slack notification...")
            slack_notifier = SlackNotifier(config.slack)
            if slack_notifier.send_quality_alert(report):
                console.print("[green]✅ Slack notification sent successfully[/green]")
            else:
                console.print("[red]❌ Failed to send Slack notification[/red]")
                success = False
        
        if not slack_only and config.github:
            console.print("Testing GitHub issue creation...")
            github_notifier = GitHubNotifier(config.github)
            if github_notifier.create_quality_issue(report):
                console.print("[green]✅ GitHub issue created successfully[/green]")
            else:
                console.print("[red]❌ Failed to create GitHub issue[/red]")
                success = False
        
        if success:
            console.print("[green]✅ All tests passed[/green]")
        else:
            console.print("[red]❌ Some tests failed[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Error in test_notifications: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()
