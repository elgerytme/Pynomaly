#!/usr/bin/env python3
"""
Monitor structure enforcement system in CI.

This script monitors the effectiveness of the structure enforcement system
over time and provides reports on violations and fixes.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def run_command(cmd: str, cwd: str = None) -> tuple[bool, str, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or Path.cwd(),
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def analyze_structure_violations() -> Dict:
    """Analyze current structure violations."""
    print("🔍 Analyzing structure violations...")
    
    # Run structure validation
    success, stdout, stderr = run_command("python scripts/validation/validate_structure.py")
    
    violations = {
        "timestamp": datetime.now().isoformat(),
        "validation_passed": success,
        "violations_found": not success,
        "stdout": stdout,
        "stderr": stderr,
        "violation_count": 0,
        "violation_categories": {},
    }
    
    if not success:
        # Parse violations from stdout
        lines = stdout.split('\n')
        for line in lines:
            if "Stray file in root:" in line or "Stray directory in root:" in line:
                violations["violation_count"] += 1
                
                # Categorize violations
                if "Stray file" in line:
                    violations["violation_categories"]["stray_files"] = violations["violation_categories"].get("stray_files", 0) + 1
                elif "Stray directory" in line:
                    violations["violation_categories"]["stray_directories"] = violations["violation_categories"].get("stray_directories", 0) + 1
    
    return violations


def check_ci_status() -> Dict:
    """Check the status of CI runs related to structure enforcement."""
    print("📊 Checking CI status...")
    
    # Check if GitHub CLI is available
    success, stdout, stderr = run_command("gh --version")
    if not success:
        return {"available": False, "reason": "GitHub CLI not available"}
    
    # Get recent workflow runs
    success, stdout, stderr = run_command("gh run list --workflow=tests.yml --limit=10 --json status,conclusion,createdAt,headBranch")
    
    if not success:
        return {"available": False, "reason": f"Failed to get workflow runs: {stderr}"}
    
    try:
        runs = json.loads(stdout)
        
        # Analyze runs
        total_runs = len(runs)
        successful_runs = sum(1 for run in runs if run.get("conclusion") == "success")
        failed_runs = sum(1 for run in runs if run.get("conclusion") == "failure")
        
        return {
            "available": True,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
            "recent_runs": runs[:5],  # Most recent 5 runs
        }
    except json.JSONDecodeError:
        return {"available": False, "reason": "Failed to parse workflow data"}


def analyze_file_organization_trends() -> Dict:
    """Analyze trends in file organization over time."""
    print("📈 Analyzing file organization trends...")
    
    # Check if there are historical reports
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return {"available": False, "reason": "No reports directory found"}
    
    # Look for structure validation reports
    validation_reports = list(reports_dir.glob("structure_validation*.json"))
    
    if not validation_reports:
        return {"available": False, "reason": "No historical validation reports found"}
    
    trends = {
        "available": True,
        "total_reports": len(validation_reports),
        "latest_report": None,
        "violation_trend": "stable",  # stable, improving, degrading
    }
    
    # Load the latest report
    latest_report_path = max(validation_reports, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest_report_path, 'r') as f:
            latest_report = json.load(f)
            trends["latest_report"] = latest_report
    except Exception as e:
        trends["latest_report_error"] = str(e)
    
    return trends


def generate_monitoring_report() -> Dict:
    """Generate a comprehensive monitoring report."""
    print("📋 Generating monitoring report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "report_type": "structure_enforcement_monitoring",
        "structure_violations": analyze_structure_violations(),
        "ci_status": check_ci_status(),
        "file_organization_trends": analyze_file_organization_trends(),
        "recommendations": [],
    }
    
    # Generate recommendations based on findings
    if report["structure_violations"]["violations_found"]:
        report["recommendations"].append({
            "priority": "high",
            "action": "Run file organization tool",
            "command": "python scripts/analysis/organize_files.py --execute",
            "reason": f"Found {report['structure_violations']['violation_count']} violations"
        })
    
    if report["ci_status"]["available"]:
        success_rate = report["ci_status"]["success_rate"]
        if success_rate < 80:
            report["recommendations"].append({
                "priority": "medium",
                "action": "Investigate CI failures",
                "reason": f"CI success rate is {success_rate:.1f}%, below 80% threshold"
            })
    
    # Check for frequent violations
    if report["structure_violations"]["violation_count"] > 10:
        report["recommendations"].append({
            "priority": "medium",
            "action": "Review FILE_ORGANIZATION_STANDARDS",
            "reason": "High number of violations may indicate standards need adjustment"
        })
    
    return report


def save_report(report: Dict) -> None:
    """Save the monitoring report to file."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"structure_enforcement_monitoring_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to: {report_path}")


def print_summary(report: Dict) -> None:
    """Print a summary of the monitoring report."""
    print("\n" + "=" * 60)
    print("📊 STRUCTURE ENFORCEMENT MONITORING SUMMARY")
    print("=" * 60)
    
    # Structure violations
    violations = report["structure_violations"]
    if violations["violations_found"]:
        print(f"❌ Structure Violations: {violations['violation_count']} found")
        for category, count in violations["violation_categories"].items():
            print(f"   • {category}: {count}")
    else:
        print("✅ Structure Violations: None found")
    
    # CI status
    ci = report["ci_status"]
    if ci["available"]:
        print(f"📊 CI Status: {ci['success_rate']:.1f}% success rate ({ci['successful_runs']}/{ci['total_runs']})")
    else:
        print(f"📊 CI Status: Not available ({ci['reason']})")
    
    # Recommendations
    if report["recommendations"]:
        print("\n📋 Recommendations:")
        for rec in report["recommendations"]:
            priority_emoji = "🔴" if rec["priority"] == "high" else "🟡"
            print(f"  {priority_emoji} {rec['action']}")
            if "command" in rec:
                print(f"     Command: {rec['command']}")
            print(f"     Reason: {rec['reason']}")
    else:
        print("\n✅ No recommendations - system is healthy")
    
    print("\n" + "=" * 60)


def main():
    """Main monitoring function."""
    print("🔍 Structure Enforcement Monitoring System")
    print("=" * 60)
    
    try:
        # Generate comprehensive report
        report = generate_monitoring_report()
        
        # Save report
        save_report(report)
        
        # Print summary
        print_summary(report)
        
        # Exit with appropriate code
        if report["structure_violations"]["violations_found"]:
            print("\n⚠️  Structure violations detected - see recommendations above")
            return 1
        else:
            print("\n✅ Structure enforcement system is healthy")
            return 0
            
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
