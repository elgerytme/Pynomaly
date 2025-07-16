#!/usr/bin/env python3
"""Performance gate checker for CI/CD pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class PerformanceGate:
    """Performance gate checker to validate performance regression limits."""
    
    def __init__(self, max_regression_percent: float = 20.0, max_failures: int = 5):
        self.max_regression_percent = max_regression_percent
        self.max_failures = max_failures
        self.violations: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        
    def check_regression_report(self, report_path: Path) -> bool:
        """Check performance regression report against gate criteria.
        
        Args:
            report_path: Path to the regression report JSON file
            
        Returns:
            True if performance gate passes, False otherwise
        """
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error reading regression report: {e}")
            return False
        
        # Check for regressions
        regressions = report.get('regressions', [])
        critical_regressions = []
        warning_regressions = []
        
        for regression in regressions:
            change_percent = abs(regression.get('change_percent', 0))
            
            if change_percent > self.max_regression_percent:
                critical_regressions.append(regression)
                self.violations.append({
                    'type': 'regression',
                    'test': regression.get('test', 'Unknown'),
                    'change_percent': change_percent,
                    'threshold': self.max_regression_percent,
                    'severity': 'critical'
                })
            elif change_percent > self.max_regression_percent * 0.5:  # Warning at 50% of limit
                warning_regressions.append(regression)
                self.warnings.append({
                    'type': 'regression',
                    'test': regression.get('test', 'Unknown'),
                    'change_percent': change_percent,
                    'threshold': self.max_regression_percent * 0.5,
                    'severity': 'warning'
                })
        
        # Check for test failures
        failures = report.get('failures', [])
        if len(failures) > self.max_failures:
            self.violations.append({
                'type': 'failures',
                'count': len(failures),
                'threshold': self.max_failures,
                'severity': 'critical',
                'tests': [f.get('test', 'Unknown') for f in failures[:10]]  # First 10
            })
        
        # Check for memory issues
        memory_issues = report.get('memory_issues', [])
        for issue in memory_issues:
            if issue.get('severity') == 'critical':
                self.violations.append({
                    'type': 'memory',
                    'issue': issue.get('description', 'Unknown memory issue'),
                    'severity': 'critical'
                })
        
        # Overall performance score check
        overall_score = report.get('overall_performance_score', 100)
        if overall_score < 70:  # Less than 70% of baseline performance
            self.violations.append({
                'type': 'overall_performance',
                'score': overall_score,
                'threshold': 70,
                'severity': 'critical'
            })
        elif overall_score < 85:  # Warning threshold
            self.warnings.append({
                'type': 'overall_performance',
                'score': overall_score,
                'threshold': 85,
                'severity': 'warning'
            })
        
        return len(self.violations) == 0
    
    def check_benchmark_results(self, benchmark_path: Path) -> bool:
        """Check benchmark results for performance issues.
        
        Args:
            benchmark_path: Path to benchmark JSON file
            
        Returns:
            True if benchmarks pass gate criteria
        """
        try:
            with open(benchmark_path, 'r') as f:
                benchmark = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️  Warning: Could not read benchmark file {benchmark_path}: {e}")
            return True  # Don't fail gate if benchmark file is missing
        
        # Check for extremely slow tests
        benchmarks = benchmark.get('benchmarks', [])
        for bench in benchmarks:
            stats = bench.get('stats', {})
            mean_time = stats.get('mean', 0)
            
            # Flag tests taking more than 5 seconds
            if mean_time > 5.0:
                self.warnings.append({
                    'type': 'slow_test',
                    'test': bench.get('name', 'Unknown'),
                    'mean_time': mean_time,
                    'threshold': 5.0,
                    'severity': 'warning'
                })
            
            # Critical flag for tests taking more than 30 seconds
            if mean_time > 30.0:
                self.violations.append({
                    'type': 'extremely_slow_test',
                    'test': bench.get('name', 'Unknown'),
                    'mean_time': mean_time,
                    'threshold': 30.0,
                    'severity': 'critical'
                })
        
        return True
    
    def check_load_test_results(self, load_test_path: Path) -> bool:
        """Check load test results for performance issues.
        
        Args:
            load_test_path: Path to load test results
            
        Returns:
            True if load tests pass gate criteria
        """
        if not load_test_path.exists():
            print(f"⚠️  Warning: Load test results not found at {load_test_path}")
            return True
        
        try:
            with open(load_test_path, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Warning: Could not parse load test results")
            return True
        
        # Check error rates
        error_rate = results.get('error_rate', 0)
        if error_rate > 5.0:  # More than 5% errors
            self.violations.append({
                'type': 'high_error_rate',
                'error_rate': error_rate,
                'threshold': 5.0,
                'severity': 'critical'
            })
        elif error_rate > 1.0:  # Warning for >1% errors
            self.warnings.append({
                'type': 'elevated_error_rate',
                'error_rate': error_rate,
                'threshold': 1.0,
                'severity': 'warning'
            })
        
        # Check response times
        avg_response_time = results.get('avg_response_time', 0)
        if avg_response_time > 2000:  # More than 2 seconds
            self.violations.append({
                'type': 'slow_response_time',
                'avg_response_time': avg_response_time,
                'threshold': 2000,
                'severity': 'critical'
            })
        elif avg_response_time > 1000:  # Warning for >1 second
            self.warnings.append({
                'type': 'elevated_response_time',
                'avg_response_time': avg_response_time,
                'threshold': 1000,
                'severity': 'warning'
            })
        
        return True
    
    def generate_report(self) -> str:
        """Generate a human-readable performance gate report."""
        report = ["# Performance Gate Report\n"]
        
        if not self.violations and not self.warnings:
            report.append("✅ **PASS**: All performance criteria met!\n")
            return "\n".join(report)
        
        if self.violations:
            report.append("❌ **FAIL**: Performance gate violations detected!\n")
            report.append("## Critical Issues\n")
            
            for violation in self.violations:
                if violation['type'] == 'regression':
                    report.append(
                        f"- **Regression**: {violation['test']} is "
                        f"{violation['change_percent']:.1f}% slower "
                        f"(threshold: {violation['threshold']:.1f}%)"
                    )
                elif violation['type'] == 'failures':
                    report.append(
                        f"- **Test Failures**: {violation['count']} failures "
                        f"(threshold: {violation['threshold']})"
                    )
                    if 'tests' in violation:
                        report.append(f"  - Failed tests: {', '.join(violation['tests'])}")
                elif violation['type'] == 'memory':
                    report.append(f"- **Memory Issue**: {violation['issue']}")
                elif violation['type'] == 'overall_performance':
                    report.append(
                        f"- **Overall Performance**: {violation['score']}% of baseline "
                        f"(threshold: {violation['threshold']}%)"
                    )
                elif violation['type'] == 'extremely_slow_test':
                    report.append(
                        f"- **Extremely Slow Test**: {violation['test']} takes "
                        f"{violation['mean_time']:.1f}s (threshold: {violation['threshold']}s)"
                    )
                elif violation['type'] == 'high_error_rate':
                    report.append(
                        f"- **High Error Rate**: {violation['error_rate']:.1f}% "
                        f"(threshold: {violation['threshold']}%)"
                    )
                elif violation['type'] == 'slow_response_time':
                    report.append(
                        f"- **Slow Response Time**: {violation['avg_response_time']}ms "
                        f"(threshold: {violation['threshold']}ms)"
                    )
            
            report.append("")
        
        if self.warnings:
            report.append("⚠️  **Warnings**:\n")
            
            for warning in self.warnings:
                if warning['type'] == 'regression':
                    report.append(
                        f"- **Minor Regression**: {warning['test']} is "
                        f"{warning['change_percent']:.1f}% slower"
                    )
                elif warning['type'] == 'overall_performance':
                    report.append(
                        f"- **Performance Decrease**: {warning['score']}% of baseline"
                    )
                elif warning['type'] == 'slow_test':
                    report.append(
                        f"- **Slow Test**: {warning['test']} takes "
                        f"{warning['mean_time']:.1f}s"
                    )
                elif warning['type'] == 'elevated_error_rate':
                    report.append(
                        f"- **Elevated Error Rate**: {warning['error_rate']:.1f}%"
                    )
                elif warning['type'] == 'elevated_response_time':
                    report.append(
                        f"- **Elevated Response Time**: {warning['avg_response_time']}ms"
                    )
            
            report.append("")
        
        # Add recommendations
        report.append("## Recommendations\n")
        
        if self.violations:
            report.append("1. **Immediate Action Required**: Address critical performance issues before merging")
            report.append("2. **Profile Code**: Use memory profilers and performance analyzers to identify bottlenecks")
            report.append("3. **Review Recent Changes**: Check commits that might have introduced performance regressions")
        
        if self.warnings:
            report.append("4. **Monitor Trends**: Track performance metrics to prevent issues from becoming critical")
            report.append("5. **Optimize**: Consider optimizing flagged components in upcoming sprints")
        
        return "\n".join(report)
    
    def run_gate_check(self, 
                      regression_report: Path = None,
                      benchmark_results: List[Path] = None,
                      load_test_results: Path = None) -> bool:
        """Run comprehensive performance gate check.
        
        Args:
            regression_report: Path to regression analysis report
            benchmark_results: List of paths to benchmark result files
            load_test_results: Path to load test results
            
        Returns:
            True if all checks pass
        """
        all_passed = True
        
        # Check regression report
        if regression_report and regression_report.exists():
            if not self.check_regression_report(regression_report):
                all_passed = False
        
        # Check benchmark results
        if benchmark_results:
            for benchmark_path in benchmark_results:
                if benchmark_path.exists():
                    self.check_benchmark_results(benchmark_path)
        
        # Check load test results  
        if load_test_results and load_test_results.exists():
            self.check_load_test_results(load_test_results)
        
        # Violations always fail the gate
        if self.violations:
            all_passed = False
        
        return all_passed


def main():
    """Main entry point for performance gate checker."""
    parser = argparse.ArgumentParser(description="Performance gate checker")
    parser.add_argument("--report", type=Path, help="Path to regression report JSON")
    parser.add_argument("--benchmarks", type=Path, nargs="*", help="Paths to benchmark result files")
    parser.add_argument("--load-test", type=Path, help="Path to load test results")
    parser.add_argument("--max-regression", type=float, default=20.0, 
                       help="Maximum allowed regression percentage (default: 20.0)")
    parser.add_argument("--max-failures", type=int, default=5,
                       help="Maximum allowed test failures (default: 5)")
    parser.add_argument("--output", type=Path, help="Output file for gate report")
    
    args = parser.parse_args()
    
    # Create performance gate
    gate = PerformanceGate(
        max_regression_percent=args.max_regression,
        max_failures=args.max_failures
    )
    
    # Run gate check
    print("🚦 Running performance gate check...")
    
    passed = gate.run_gate_check(
        regression_report=args.report,
        benchmark_results=args.benchmarks or [],
        load_test_results=args.load_test
    )
    
    # Generate and display report
    report = gate.generate_report()
    print(report)
    
    # Save report if requested
    if args.output:
        args.output.write_text(report)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    if passed:
        print("\n✅ Performance gate: PASSED")
        sys.exit(0)
    else:
        print("\n❌ Performance gate: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()