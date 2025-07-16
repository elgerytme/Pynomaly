#!/usr/bin/env python3
"""
Repository Organization Monitor

Continuous monitoring system for repository organization health.
Runs periodic checks and generates reports on organization compliance.
"""

import os
import json
import time
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import argparse


@dataclass
class OrganizationMetrics:
    """Repository organization health metrics."""
    timestamp: str
    root_cleanliness_score: float  # 0-100
    file_placement_score: float    # 0-100
    naming_compliance_score: float # 0-100
    temp_file_score: float         # 0-100
    overall_health_score: float    # 0-100
    
    total_files: int
    misplaced_files: int
    temp_files: int
    config_violations: int
    naming_violations: int
    
    issues: List[str]
    recommendations: List[str]


class OrganizationMonitor:
    """Monitors repository organization health."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.metrics_file = repo_root / 'scripts' / 'monitoring' / 'organization_metrics.json'
        self.reports_dir = repo_root / 'scripts' / 'monitoring' / 'reports'
        self.validation_script = repo_root / 'scripts' / 'validation' / 'validate_organization.py'
        self.cleanup_script = repo_root / 'scripts' / 'cleanup' / 'auto_organize.py'
        
        # Ensure directories exist
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def collect_metrics(self) -> OrganizationMetrics:
        """Collect current organization metrics."""
        timestamp = datetime.now().isoformat()
        
        # Run validation script to get detailed results
        validation_results = self._run_validation()
        
        # Calculate metrics
        total_files = self._count_files()
        misplaced_files = len([r for r in validation_results if 'placement' in r.get('category', '')])
        temp_files = len([r for r in validation_results if 'temporary' in r.get('category', '')])
        config_violations = len([r for r in validation_results if 'config' in r.get('category', '')])
        naming_violations = len([r for r in validation_results if 'naming' in r.get('category', '')])
        
        # Calculate scores (0-100)
        root_cleanliness_score = self._calculate_root_cleanliness_score()
        file_placement_score = max(0, 100 - (misplaced_files * 10))
        naming_compliance_score = max(0, 100 - (naming_violations * 5))
        temp_file_score = max(0, 100 - (temp_files * 20))
        
        # Overall health score (weighted average)
        overall_health_score = (
            root_cleanliness_score * 0.3 +
            file_placement_score * 0.3 +
            naming_compliance_score * 0.2 +
            temp_file_score * 0.2
        )
        
        # Extract issues and recommendations
        issues = [r.get('message', '') for r in validation_results if r.get('level') == 'error']
        recommendations = self._generate_recommendations(validation_results)
        
        return OrganizationMetrics(
            timestamp=timestamp,
            root_cleanliness_score=root_cleanliness_score,
            file_placement_score=file_placement_score,
            naming_compliance_score=naming_compliance_score,
            temp_file_score=temp_file_score,
            overall_health_score=overall_health_score,
            total_files=total_files,
            misplaced_files=misplaced_files,
            temp_files=temp_files,
            config_violations=config_violations,
            naming_violations=naming_violations,
            issues=issues[:10],  # Limit to top 10 issues
            recommendations=recommendations[:5]  # Limit to top 5 recommendations
        )

    def _run_validation(self) -> List[Dict]:
        """Run validation script and parse results."""
        try:
            if not self.validation_script.exists():
                return []
            
            result = subprocess.run([
                'python3', str(self.validation_script), '--json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
            
        except Exception as e:
            print(f"Failed to run validation: {e}")
        
        return []

    def _count_files(self) -> int:
        """Count total files in repository."""
        count = 0
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            count += len(files)
        return count

    def _calculate_root_cleanliness_score(self) -> float:
        """Calculate root directory cleanliness score."""
        allowed_items = {
            'README.md', 'CHANGELOG.md', 'LICENSE', 'pyproject.toml', '.gitignore',
            '.python-version', '.claude', '.github', '.hypothesis', '.project-rules',
            '.ruff_cache', '.storybook', '.vscode', 'docs', 'pkg', 'scripts', 'src'
        }
        
        root_items = set(os.listdir(self.repo_root))
        unauthorized_items = root_items - allowed_items
        
        # Score based on cleanliness
        if not unauthorized_items:
            return 100.0
        else:
            # Penalize based on number of unauthorized items
            penalty = min(len(unauthorized_items) * 10, 100)
            return max(0, 100 - penalty)

    def _generate_recommendations(self, validation_results: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        error_count = len([r for r in validation_results if r.get('level') == 'error'])
        warning_count = len([r for r in validation_results if r.get('level') == 'warning'])
        
        if error_count > 0:
            recommendations.append(f"Fix {error_count} critical organization errors")
        
        if warning_count > 5:
            recommendations.append(f"Address {warning_count} organization warnings")
        
        # Check for common patterns
        config_issues = [r for r in validation_results if 'config' in r.get('category', '')]
        if config_issues:
            recommendations.append("Move configuration files to scripts/config/")
        
        temp_issues = [r for r in validation_results if 'temporary' in r.get('category', '')]
        if temp_issues:
            recommendations.append("Clean up temporary files")
        
        naming_issues = [r for r in validation_results if 'naming' in r.get('category', '')]
        if naming_issues:
            recommendations.append("Fix naming convention violations")
        
        return recommendations

    def save_metrics(self, metrics: OrganizationMetrics):
        """Save metrics to file."""
        # Load existing metrics
        metrics_data = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            except Exception:
                metrics_data = []
        
        # Add new metrics
        metrics_data.append(asdict(metrics))
        
        # Keep only last 30 days of metrics
        cutoff_date = datetime.now() - timedelta(days=30)
        metrics_data = [
            m for m in metrics_data 
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        # Save updated metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    def generate_health_report(self) -> str:
        """Generate current health status report."""
        metrics = self.collect_metrics()
        
        # Health status emoji
        if metrics.overall_health_score >= 90:
            status_emoji = "🟢"
            status_text = "EXCELLENT"
        elif metrics.overall_health_score >= 75:
            status_emoji = "🟡"
            status_text = "GOOD"
        elif metrics.overall_health_score >= 60:
            status_emoji = "🟠"
            status_text = "NEEDS ATTENTION"
        else:
            status_emoji = "🔴"
            status_text = "CRITICAL"
        
        report = f"""# Repository Organization Health Report

{status_emoji} **Overall Health: {metrics.overall_health_score:.1f}/100 ({status_text})**

**Generated**: {metrics.timestamp}

## Health Metrics

| Metric | Score | Status |
|--------|-------|--------|
| Root Cleanliness | {metrics.root_cleanliness_score:.1f}/100 | {'✅' if metrics.root_cleanliness_score >= 90 else '⚠️' if metrics.root_cleanliness_score >= 75 else '❌'} |
| File Placement | {metrics.file_placement_score:.1f}/100 | {'✅' if metrics.file_placement_score >= 90 else '⚠️' if metrics.file_placement_score >= 75 else '❌'} |
| Naming Compliance | {metrics.naming_compliance_score:.1f}/100 | {'✅' if metrics.naming_compliance_score >= 90 else '⚠️' if metrics.naming_compliance_score >= 75 else '❌'} |
| Temporary Files | {metrics.temp_file_score:.1f}/100 | {'✅' if metrics.temp_file_score >= 90 else '⚠️' if metrics.temp_file_score >= 75 else '❌'} |

## Repository Statistics

- **Total Files**: {metrics.total_files:,}
- **Misplaced Files**: {metrics.misplaced_files}
- **Temporary Files**: {metrics.temp_files}
- **Configuration Violations**: {metrics.config_violations}
- **Naming Violations**: {metrics.naming_violations}

"""
        
        if metrics.issues:
            report += "## Critical Issues\n\n"
            for issue in metrics.issues:
                report += f"- ❌ {issue}\n"
            report += "\n"
        
        if metrics.recommendations:
            report += "## Recommendations\n\n"
            for rec in metrics.recommendations:
                report += f"- 💡 {rec}\n"
            report += "\n"
        
        report += """## Quick Actions

```bash
# Run validation
python3 scripts/validation/validate_organization.py

# Auto-organize repository
python3 scripts/cleanup/auto_organize.py --execute

# Check detailed metrics
python3 scripts/monitoring/organization_monitor.py --status
```

---
*This report is automatically generated by the repository organization monitor.*
"""
        
        return report

    def save_health_report(self):
        """Save current health report to file."""
        report = self.generate_health_report()
        report_file = self.reports_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Also save as latest report
        latest_file = self.reports_dir / "latest_health_report.md"
        with open(latest_file, 'w') as f:
            f.write(report)
        
        return report_file

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle."""
        print(f"🔍 Running organization monitoring cycle at {datetime.now()}")
        
        try:
            # Collect metrics
            metrics = self.collect_metrics()
            
            # Save metrics
            self.save_metrics(metrics)
            
            # Generate and save report
            report_file = self.save_health_report()
            
            print(f"📊 Health Score: {metrics.overall_health_score:.1f}/100")
            print(f"📄 Report saved: {report_file}")
            
            # Alert if health is critical
            if metrics.overall_health_score < 60:
                print(f"🚨 ALERT: Repository organization health is critical!")
                print(f"📧 Consider running: scripts/cleanup/auto_organize.py --execute")
            
        except Exception as e:
            print(f"❌ Monitoring cycle failed: {e}")

    def start_monitoring(self, interval_hours: int = 24):
        """Start continuous monitoring."""
        print(f"🚀 Starting repository organization monitoring (every {interval_hours} hours)")
        
        # Schedule monitoring
        schedule.every(interval_hours).hours.do(self.run_monitoring_cycle)
        
        # Run initial cycle
        self.run_monitoring_cycle()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def get_metrics_history(self, days: int = 7) -> List[OrganizationMetrics]:
        """Get metrics history for specified days."""
        if not self.metrics_file.exists():
            return []
        
        try:
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_metrics = []
            for data in metrics_data:
                if datetime.fromisoformat(data['timestamp']) > cutoff_date:
                    recent_metrics.append(OrganizationMetrics(**data))
            
            return sorted(recent_metrics, key=lambda x: x.timestamp)
            
        except Exception as e:
            print(f"Failed to load metrics history: {e}")
            return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Repository organization monitor")
    parser.add_argument('--start', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--report', action='store_true', help='Generate health report')
    parser.add_argument('--interval', type=int, default=24, help='Monitoring interval in hours')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.parent
    monitor = OrganizationMonitor(repo_root)
    
    if args.start:
        monitor.start_monitoring(args.interval)
    elif args.status:
        metrics = monitor.collect_metrics()
        print(f"📊 Repository Organization Health: {metrics.overall_health_score:.1f}/100")
        print(f"📁 Total Files: {metrics.total_files:,}")
        print(f"⚠️  Issues: {len(metrics.issues)}")
        print(f"💡 Recommendations: {len(metrics.recommendations)}")
    elif args.report:
        report_file = monitor.save_health_report()
        print(f"📄 Health report generated: {report_file}")
    else:
        # Run single monitoring cycle
        monitor.run_monitoring_cycle()


if __name__ == "__main__":
    main()