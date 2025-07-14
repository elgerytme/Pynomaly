#!/usr/bin/env python3
"""Coverage monitoring and automation setup script for Pynomaly."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class CoverageManager:
    """Manages test coverage monitoring and reporting."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """Initialize coverage manager."""
        self.root_dir = root_dir or Path.cwd()
        self.reports_dir = self.root_dir / "reports"
        self.coverage_dir = self.reports_dir / "coverage"
        
        # Ensure directories exist
        self.coverage_dir.mkdir(parents=True, exist_ok=True)
    
    def run_coverage_tests(self, test_paths: Optional[List[str]] = None) -> bool:
        """Run tests with coverage measurement."""
        print("🧪 Running tests with coverage measurement...")
        
        # Default test paths
        if not test_paths:
            test_paths = ["tests/", "src/pynomaly/tests/"]
        
        # Filter existing paths
        existing_paths = [path for path in test_paths if Path(path).exists()]
        
        if not existing_paths:
            print("❌ No test directories found")
            return False
        
        cmd = [
            "coverage", "run", 
            "--source", "src/pynomaly",
            "--branch",
            "--parallel-mode",
            "-m", "pytest"
        ] + existing_paths + [
            "--verbose",
            "--tb=short",
            f"--junitxml={self.reports_dir}/test-results.xml"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Tests completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False
    
    def generate_reports(self) -> bool:
        """Generate coverage reports in multiple formats."""
        print("📊 Generating coverage reports...")
        
        try:
            # Combine parallel coverage data
            subprocess.run(["coverage", "combine"], check=True)
            
            # Generate terminal report
            result = subprocess.run(
                ["coverage", "report", "--show-missing", "--precision=2"],
                capture_output=True, text=True, check=True
            )
            print("Terminal Coverage Report:")
            print(result.stdout)
            
            # Generate HTML report
            subprocess.run([
                "coverage", "html", 
                f"--directory={self.coverage_dir}/html"
            ], check=True)
            print(f"✅ HTML report generated: {self.coverage_dir}/html/index.html")
            
            # Generate XML report for CI/CD
            subprocess.run([
                "coverage", "xml", 
                f"--output={self.coverage_dir}/coverage.xml"
            ], check=True)
            print(f"✅ XML report generated: {self.coverage_dir}/coverage.xml")
            
            # Generate JSON report for analysis
            subprocess.run([
                "coverage", "json", 
                f"--output={self.coverage_dir}/coverage.json"
            ], check=True)
            print(f"✅ JSON report generated: {self.coverage_dir}/coverage.json")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Report generation failed: {e}")
            return False
    
    def analyze_coverage(self) -> Dict:
        """Analyze coverage data and return metrics."""
        print("🔍 Analyzing coverage data...")
        
        json_file = self.coverage_dir / "coverage.json"
        if not json_file.exists():
            print("❌ Coverage JSON file not found")
            return {}
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Extract key metrics
        totals = data.get("totals", {})
        metrics = {
            "coverage_percent": totals.get("percent_covered", 0),
            "lines_total": totals.get("num_statements", 0),
            "lines_covered": totals.get("covered_lines", 0),
            "lines_missing": totals.get("missing_lines", 0),
            "branches_total": totals.get("num_branches", 0),
            "branches_covered": totals.get("covered_branches", 0),
            "branches_missing": totals.get("missing_branches", 0),
        }
        
        # Calculate additional metrics
        if metrics["lines_total"] > 0:
            metrics["line_coverage_percent"] = (
                metrics["lines_covered"] / metrics["lines_total"] * 100
            )
        
        if metrics["branches_total"] > 0:
            metrics["branch_coverage_percent"] = (
                metrics["branches_covered"] / metrics["branches_total"] * 100
            )
        
        print(f"📈 Overall Coverage: {metrics['coverage_percent']:.2f}%")
        print(f"📈 Line Coverage: {metrics.get('line_coverage_percent', 0):.2f}%")
        print(f"📈 Branch Coverage: {metrics.get('branch_coverage_percent', 0):.2f}%")
        
        return metrics
    
    def check_thresholds(self, metrics: Dict) -> bool:
        """Check if coverage meets required thresholds."""
        print("🎯 Checking coverage thresholds...")
        
        # Define thresholds
        thresholds = {
            "overall": 90.0,
            "line": 95.0,
            "branch": 85.0
        }
        
        overall = metrics.get("coverage_percent", 0)
        line = metrics.get("line_coverage_percent", 0)
        branch = metrics.get("branch_coverage_percent", 0)
        
        passed = True
        
        if overall < thresholds["overall"]:
            print(f"❌ Overall coverage {overall:.2f}% below threshold {thresholds['overall']}%")
            passed = False
        else:
            print(f"✅ Overall coverage {overall:.2f}% meets threshold {thresholds['overall']}%")
        
        if line < thresholds["line"]:
            print(f"❌ Line coverage {line:.2f}% below threshold {thresholds['line']}%")
            passed = False
        else:
            print(f"✅ Line coverage {line:.2f}% meets threshold {thresholds['line']}%")
        
        if branch < thresholds["branch"]:
            print(f"❌ Branch coverage {branch:.2f}% below threshold {thresholds['branch']}%")
            passed = False
        else:
            print(f"✅ Branch coverage {branch:.2f}% meets threshold {thresholds['branch']}%")
        
        return passed
    
    def generate_badge_data(self, metrics: Dict) -> None:
        """Generate badge data for README."""
        coverage = metrics.get("coverage_percent", 0)
        
        # Determine badge color
        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellow"
        elif coverage >= 60:
            color = "orange"
        else:
            color = "red"
        
        badge_data = {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{coverage:.1f}%",
            "color": color
        }
        
        badge_file = self.coverage_dir / "badge.json"
        with open(badge_file, "w") as f:
            json.dump(badge_data, f, indent=2)
        
        print(f"🏆 Badge data generated: {badge_file}")
    
    def setup_pre_commit_hook(self) -> None:
        """Setup pre-commit hook for coverage checks."""
        hook_content = '''#!/bin/bash
# Pre-commit coverage check

echo "🧪 Running coverage check..."
python scripts/coverage_setup.py --check-only

if [ $? -ne 0 ]; then
    echo "❌ Coverage check failed. Commit aborted."
    exit 1
fi

echo "✅ Coverage check passed."
'''
        
        hooks_dir = self.root_dir / ".git" / "hooks"
        if hooks_dir.exists():
            hook_file = hooks_dir / "pre-commit"
            with open(hook_file, "w") as f:
                f.write(hook_content)
            hook_file.chmod(0o755)
            print(f"✅ Pre-commit hook installed: {hook_file}")
        else:
            print("⚠️  Git hooks directory not found. Pre-commit hook not installed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Coverage monitoring and automation")
    parser.add_argument("--test-paths", nargs="*", help="Test paths to run")
    parser.add_argument("--check-only", action="store_true", help="Only check existing coverage")
    parser.add_argument("--setup-hooks", action="store_true", help="Setup pre-commit hooks")
    parser.add_argument("--threshold", type=float, default=90.0, help="Coverage threshold")
    
    args = parser.parse_args()
    
    manager = CoverageManager()
    
    if args.setup_hooks:
        manager.setup_pre_commit_hook()
        return
    
    if not args.check_only:
        # Run tests with coverage
        if not manager.run_coverage_tests(args.test_paths):
            sys.exit(1)
        
        # Generate reports
        if not manager.generate_reports():
            sys.exit(1)
    
    # Analyze coverage
    metrics = manager.analyze_coverage()
    if not metrics:
        sys.exit(1)
    
    # Generate badge data
    manager.generate_badge_data(metrics)
    
    # Check thresholds
    if not manager.check_thresholds(metrics):
        sys.exit(1)
    
    print("🎉 Coverage monitoring completed successfully!")


if __name__ == "__main__":
    main()