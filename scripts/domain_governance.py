#!/usr/bin/env python3
"""
Domain Boundary Governance System

Provides governance, monitoring, and reporting for domain boundary compliance.
"""

import os
import sys
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import our domain boundary validator
from domain_boundary_validator import DomainBoundaryValidator, DomainViolation

@dataclass
class ComplianceMetrics:
    """Compliance metrics for tracking"""
    timestamp: datetime
    total_violations: int
    packages_with_violations: int
    compliance_percentage: float
    most_common_violations: Dict[str, int]
    packages_status: Dict[str, int]

@dataclass
class ComplianceAlert:
    """Alert for compliance issues"""
    timestamp: datetime
    severity: str  # critical, high, medium, low
    message: str
    violation_count: int
    affected_packages: List[str]
    recommended_actions: List[str]

class DomainGovernanceSystem:
    """Main governance system for domain boundaries"""
    
    def __init__(self, config_path: str = "governance_config.json"):
        self.config = self._load_config(config_path)
        self.validator = DomainBoundaryValidator()
        self.metrics_history: List[ComplianceMetrics] = []
        self.alerts: List[ComplianceAlert] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('domain_governance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load governance configuration"""
        default_config = {
            "compliance_thresholds": {
                "critical": 1000,   # > 1000 violations = critical
                "high": 500,        # > 500 violations = high
                "medium": 100,      # > 100 violations = medium
                "low": 0            # > 0 violations = low
            },
            "monitoring": {
                "check_interval_hours": 24,
                "trend_analysis_days": 7,
                "alert_on_regression": True,
                "alert_on_new_violations": True
            },
            "notifications": {
                "email_enabled": False,
                "slack_enabled": False,
                "webhook_enabled": False
            },
            "governance": {
                "require_approval_for_new_domains": True,
                "enforce_pre_commit_hooks": True,
                "block_prs_with_violations": True,
                "auto_create_issues": True
            }
        }
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        return default_config
    
    def run_compliance_check(self) -> ComplianceMetrics:
        """Run comprehensive compliance check"""
        self.logger.info("Running domain boundary compliance check...")
        
        # Run validation
        results = self.validator.validate_all_packages()
        
        # Calculate metrics
        total_violations = sum(len(violations) for violations in results.values())
        packages_with_violations = len(results)
        
        # Calculate compliance percentage (assuming baseline of 26062 violations)
        baseline_violations = 26062
        compliance_percentage = max(0, (1 - total_violations / baseline_violations) * 100)
        
        # Get most common violations
        violation_counts = {}
        for violations in results.values():
            for violation in violations:
                term = violation.prohibited_term
                violation_counts[term] = violation_counts.get(term, 0) + 1
        
        most_common = dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Package status
        packages_status = {}
        for package_name, violations in results.items():
            packages_status[package_name] = len(violations)
        
        # Create metrics
        metrics = ComplianceMetrics(
            timestamp=datetime.now(),
            total_violations=total_violations,
            packages_with_violations=packages_with_violations,
            compliance_percentage=compliance_percentage,
            most_common_violations=most_common,
            packages_status=packages_status
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        self.logger.info(f"Compliance check completed: {total_violations} violations, {compliance_percentage:.1f}% compliance")
        
        return metrics
    
    def _check_alerts(self, metrics: ComplianceMetrics):
        """Check for compliance alerts"""
        alerts = []
        
        # Check violation thresholds
        violation_count = metrics.total_violations
        thresholds = self.config["compliance_thresholds"]
        
        if violation_count > thresholds["critical"]:
            alerts.append(ComplianceAlert(
                timestamp=datetime.now(),
                severity="critical",
                message=f"Critical compliance issue: {violation_count} violations detected",
                violation_count=violation_count,
                affected_packages=list(metrics.packages_status.keys()),
                recommended_actions=[
                    "Immediate intervention required",
                    "Review DOMAIN_COMPLIANCE_PLAN.md",
                    "Assign dedicated team to fix violations",
                    "Block all non-compliance related development"
                ]
            ))
        elif violation_count > thresholds["high"]:
            alerts.append(ComplianceAlert(
                timestamp=datetime.now(),
                severity="high",
                message=f"High compliance issue: {violation_count} violations detected",
                violation_count=violation_count,
                affected_packages=list(metrics.packages_status.keys()),
                recommended_actions=[
                    "Schedule compliance sprint",
                    "Review most common violations",
                    "Implement additional automated checks",
                    "Update development guidelines"
                ]
            ))
        elif violation_count > thresholds["medium"]:
            alerts.append(ComplianceAlert(
                timestamp=datetime.now(),
                severity="medium",
                message=f"Medium compliance issue: {violation_count} violations detected",
                violation_count=violation_count,
                affected_packages=list(metrics.packages_status.keys()),
                recommended_actions=[
                    "Schedule compliance review",
                    "Focus on high-impact violations",
                    "Update documentation",
                    "Provide developer training"
                ]
            ))
        
        # Check for regression
        if len(self.metrics_history) >= 2:
            previous_metrics = self.metrics_history[-2]
            if metrics.total_violations > previous_metrics.total_violations:
                alerts.append(ComplianceAlert(
                    timestamp=datetime.now(),
                    severity="high",
                    message=f"Compliance regression detected: {metrics.total_violations - previous_metrics.total_violations} new violations",
                    violation_count=metrics.total_violations - previous_metrics.total_violations,
                    affected_packages=list(metrics.packages_status.keys()),
                    recommended_actions=[
                        "Review recent code changes",
                        "Strengthen pre-commit hooks",
                        "Audit development practices",
                        "Provide immediate training"
                    ]
                ))
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Send notifications
        for alert in alerts:
            self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: ComplianceAlert):
        """Send alert notification"""
        self.logger.warning(f"COMPLIANCE ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Email notification
        if self.config["notifications"]["email_enabled"]:
            self._send_email_alert(alert)
        
        # Slack notification
        if self.config["notifications"]["slack_enabled"]:
            self._send_slack_alert(alert)
        
        # Webhook notification
        if self.config["notifications"]["webhook_enabled"]:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: ComplianceAlert):
        """Send email alert"""
        try:
            # Email configuration should be in config
            email_config = self.config.get("email", {})
            
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from', 'governance@domain.com')
            msg['To'] = email_config.get('to', 'team@domain.com')
            msg['Subject'] = f"Domain Boundary Compliance Alert - {alert.severity.upper()}"
            
            body = f"""
Domain Boundary Compliance Alert

Severity: {alert.severity.upper()}
Message: {alert.message}
Timestamp: {alert.timestamp}
Violations: {alert.violation_count}
Affected Packages: {', '.join(alert.affected_packages)}

Recommended Actions:
{chr(10).join(f'- {action}' for action in alert.recommended_actions)}

For more details, run: python scripts/domain_boundary_validator.py
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config.get('smtp_host', 'localhost'), 
                                 email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username', ''), 
                        email_config.get('password', ''))
            text = msg.as_string()
            server.sendmail(msg['From'], msg['To'], text)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert.severity} compliance issue")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def generate_compliance_report(self, output_file: str = "compliance_report.html"):
        """Generate comprehensive compliance report"""
        if not self.metrics_history:
            self.logger.warning("No metrics history available for report")
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Generate trend data
        trend_data = []
        for metrics in self.metrics_history[-30:]:  # Last 30 checks
            trend_data.append({
                'timestamp': metrics.timestamp.isoformat(),
                'violations': metrics.total_violations,
                'compliance': metrics.compliance_percentage
            })
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Domain Boundary Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 20px; border-radius: 5px; }}
                .metric h3 {{ font-size: 36px; margin: 0; }}
                .critical {{ background: #dc3545; color: white; }}
                .high {{ background: #fd7e14; color: white; }}
                .medium {{ background: #ffc107; color: black; }}
                .good {{ background: #28a745; color: white; }}
                .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert.critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
                .alert.high {{ background: #fff3cd; border-left: 4px solid #fd7e14; }}
                .alert.medium {{ background: #d4edda; border-left: 4px solid #ffc107; }}
                .violation-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .violation-table th, .violation-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .violation-table th {{ background-color: #f2f2f2; }}
                .recommendation {{ background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Domain Boundary Compliance Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report Period: {self.metrics_history[0].timestamp.strftime('%Y-%m-%d')} to {latest_metrics.timestamp.strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric {'critical' if latest_metrics.total_violations > 1000 else 'high' if latest_metrics.total_violations > 500 else 'medium' if latest_metrics.total_violations > 100 else 'good'}">
                    <h3>{latest_metrics.total_violations}</h3>
                    <p>Total Violations</p>
                </div>
                <div class="metric {'good' if latest_metrics.compliance_percentage >= 90 else 'medium' if latest_metrics.compliance_percentage >= 70 else 'high' if latest_metrics.compliance_percentage >= 50 else 'critical'}">
                    <h3>{latest_metrics.compliance_percentage:.1f}%</h3>
                    <p>Compliance Rate</p>
                </div>
                <div class="metric">
                    <h3>{latest_metrics.packages_with_violations}</h3>
                    <p>Packages with Violations</p>
                </div>
            </div>
            
            <h2>Compliance Trend</h2>
            <div class="chart">
                <canvas id="trendChart" width="800" height="400"></canvas>
            </div>
            
            <h2>Recent Alerts</h2>
        """
        
        # Add recent alerts
        recent_alerts = self.alerts[-10:]  # Last 10 alerts
        if recent_alerts:
            for alert in recent_alerts:
                html_content += f"""
                <div class="alert {alert.severity}">
                    <h4>{alert.severity.upper()}: {alert.message}</h4>
                    <p><strong>Timestamp:</strong> {alert.timestamp}</p>
                    <p><strong>Violations:</strong> {alert.violation_count}</p>
                    <p><strong>Affected Packages:</strong> {', '.join(alert.affected_packages)}</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        {''.join(f'<li>{action}</li>' for action in alert.recommended_actions)}
                    </ul>
                </div>
                """
        else:
            html_content += "<p>No recent alerts</p>"
        
        html_content += """
            <h2>Most Common Violations</h2>
            <table class="violation-table">
                <thead>
                    <tr>
                        <th>Violation Term</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        total_violations = sum(latest_metrics.most_common_violations.values())
        for term, count in latest_metrics.most_common_violations.items():
            percentage = (count / total_violations * 100) if total_violations > 0 else 0
            html_content += f"""
                <tr>
                    <td>{term}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
            
            <h2>Package Status</h2>
            <table class="violation-table">
                <thead>
                    <tr>
                        <th>Package</th>
                        <th>Violations</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for package, violations in latest_metrics.packages_status.items():
            status = "✅ Compliant" if violations == 0 else f"❌ {violations} violations"
            html_content += f"""
                <tr>
                    <td>{package}</td>
                    <td>{violations}</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
            
            <h2>Recommendations</h2>
            <div class="recommendation">
                <h4>Immediate Actions</h4>
                <ul>
                    <li>Focus on the most common violations: {}</li>
                    <li>Review packages with highest violation counts</li>
                    <li>Implement automated fixes where possible</li>
                    <li>Provide targeted training to development teams</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h4>Long-term Improvements</h4>
                <ul>
                    <li>Strengthen pre-commit hooks</li>
                    <li>Implement real-time monitoring</li>
                    <li>Create domain-specific tooling</li>
                    <li>Regular compliance audits</li>
                </ul>
            </div>
            
            <h2>Governance Actions</h2>
            <div class="recommendation">
                <h4>Policy Enforcement</h4>
                <ul>
                    <li>Block PRs with violations: {}</li>
                    <li>Require pre-commit hooks: {}</li>
                    <li>Auto-create issues: {}</li>
                    <li>New domain approval required: {}</li>
                </ul>
            </div>
        </body>
        </html>
        """.format(
            ', '.join(list(latest_metrics.most_common_violations.keys())[:3]),
            '✅ Enabled' if self.config['governance']['block_prs_with_violations'] else '❌ Disabled',
            '✅ Enabled' if self.config['governance']['enforce_pre_commit_hooks'] else '❌ Disabled',
            '✅ Enabled' if self.config['governance']['auto_create_issues'] else '❌ Disabled',
            '✅ Enabled' if self.config['governance']['require_approval_for_new_domains'] else '❌ Disabled'
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Compliance report generated: {output_file}")
    
    def start_monitoring(self):
        """Start continuous compliance monitoring"""
        self.logger.info("Starting domain boundary compliance monitoring...")
        
        # Schedule regular checks
        interval_hours = self.config["monitoring"]["check_interval_hours"]
        schedule.every(interval_hours).hours.do(self.run_compliance_check)
        
        # Run initial check
        self.run_compliance_check()
        
        # Start monitoring loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def generate_governance_summary(self) -> Dict[str, Any]:
        """Generate governance summary"""
        if not self.metrics_history:
            return {"error": "No metrics history available"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends
        trend_data = []
        if len(self.metrics_history) >= 2:
            for i in range(1, len(self.metrics_history)):
                prev_metrics = self.metrics_history[i-1]
                curr_metrics = self.metrics_history[i]
                
                trend_data.append({
                    'timestamp': curr_metrics.timestamp.isoformat(),
                    'violation_change': curr_metrics.total_violations - prev_metrics.total_violations,
                    'compliance_change': curr_metrics.compliance_percentage - prev_metrics.compliance_percentage
                })
        
        return {
            "current_status": {
                "timestamp": latest_metrics.timestamp.isoformat(),
                "total_violations": latest_metrics.total_violations,
                "compliance_percentage": latest_metrics.compliance_percentage,
                "packages_with_violations": latest_metrics.packages_with_violations
            },
            "trends": trend_data[-7:],  # Last 7 data points
            "alerts": [asdict(alert) for alert in self.alerts[-5:]],  # Last 5 alerts
            "governance_config": self.config["governance"],
            "recommendations": self._generate_recommendations(latest_metrics)
        }
    
    def _generate_recommendations(self, metrics: ComplianceMetrics) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics.total_violations > 1000:
            recommendations.extend([
                "Critical: Immediate intervention required",
                "Assign dedicated compliance team",
                "Halt non-essential development",
                "Implement emergency compliance measures"
            ])
        elif metrics.total_violations > 500:
            recommendations.extend([
                "High priority: Schedule compliance sprint",
                "Focus on top 10 violation types",
                "Strengthen automated validation",
                "Provide team training"
            ])
        elif metrics.total_violations > 100:
            recommendations.extend([
                "Medium priority: Regular compliance review",
                "Target specific violation patterns",
                "Update development documentation",
                "Enhance tooling"
            ])
        else:
            recommendations.extend([
                "Good compliance: Maintain current practices",
                "Continue monitoring",
                "Optimize validation performance",
                "Share best practices"
            ])
        
        return recommendations

def main():
    """Main function for governance system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain Boundary Governance System")
    parser.add_argument("--check", action="store_true", help="Run single compliance check")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--report", action="store_true", help="Generate compliance report")
    parser.add_argument("--summary", action="store_true", help="Generate governance summary")
    parser.add_argument("--config", default="governance_config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create governance system
    governance = DomainGovernanceSystem(args.config)
    
    if args.check:
        # Run single compliance check
        metrics = governance.run_compliance_check()
        print(f"Compliance check completed: {metrics.total_violations} violations")
        
    elif args.monitor:
        # Start continuous monitoring
        governance.start_monitoring()
        
    elif args.report:
        # Generate compliance report
        governance.generate_compliance_report()
        print("Compliance report generated: compliance_report.html")
        
    elif args.summary:
        # Generate governance summary
        summary = governance.generate_governance_summary()
        print(json.dumps(summary, indent=2))
        
    else:
        # Default: run check and generate report
        governance.run_compliance_check()
        governance.generate_compliance_report()
        print("Compliance check and report completed")

if __name__ == "__main__":
    main()