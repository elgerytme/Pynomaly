#!/usr/bin/env python3
"""Generate performance test summary reports."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class PerformanceSummaryGenerator:
    """Generate comprehensive performance test summaries."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.summary_data = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "overall_metrics": {},
            "recommendations": [],
            "trends": {}
        }
    
    def process_benchmark_results(self) -> Dict[str, Any]:
        """Process all benchmark result files."""
        benchmark_files = list(self.results_dir.glob("**/benchmark-*.json"))
        
        all_benchmarks = {}
        suite_summaries = {}
        
        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                
                # Extract suite name from filename
                suite_name = benchmark_file.stem.replace("benchmark-", "")
                
                # Process benchmarks
                benchmarks = data.get('benchmarks', [])
                suite_metrics = self._process_suite_benchmarks(benchmarks)
                
                suite_summaries[suite_name] = {
                    "total_tests": len(benchmarks),
                    "metrics": suite_metrics,
                    "file": str(benchmark_file)
                }
                
                all_benchmarks[suite_name] = benchmarks
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not process {benchmark_file}: {e}")
        
        return {
            "suites": suite_summaries,
            "all_benchmarks": all_benchmarks
        }
    
    def _process_suite_benchmarks(self, benchmarks: List[Dict]) -> Dict[str, Any]:
        """Process benchmarks for a specific suite."""
        if not benchmarks:
            return {}
        
        # Extract performance metrics
        mean_times = []
        min_times = []
        max_times = []
        std_devs = []
        
        for benchmark in benchmarks:
            stats = benchmark.get('stats', {})
            if stats:
                mean_times.append(stats.get('mean', 0))
                min_times.append(stats.get('min', 0))
                max_times.append(stats.get('max', 0))
                std_devs.append(stats.get('stddev', 0))
        
        if not mean_times:
            return {}
        
        return {
            "avg_execution_time": statistics.mean(mean_times),
            "fastest_test": min(min_times),
            "slowest_test": max(max_times),
            "total_execution_time": sum(mean_times),
            "performance_stability": statistics.mean(std_devs),
            "test_count": len(benchmarks)
        }
    
    def process_regression_reports(self) -> Dict[str, Any]:
        """Process regression analysis reports."""
        regression_files = list(self.results_dir.glob("**/regression-report-*.json"))
        
        all_regressions = []
        all_improvements = []
        suite_regressions = {}
        
        for regression_file in regression_files:
            try:
                with open(regression_file, 'r') as f:
                    data = json.load(f)
                
                suite_name = regression_file.stem.replace("regression-report-", "")
                
                regressions = data.get('regressions', [])
                improvements = data.get('improvements', [])
                
                all_regressions.extend(regressions)
                all_improvements.extend(improvements)
                
                suite_regressions[suite_name] = {
                    "regressions": len(regressions),
                    "improvements": len(improvements),
                    "details": data
                }
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not process {regression_file}: {e}")
        
        return {
            "total_regressions": len(all_regressions),
            "total_improvements": len(all_improvements),
            "by_suite": suite_regressions,
            "critical_regressions": [r for r in all_regressions if r.get('change_percent', 0) > 20],
            "significant_improvements": [i for i in all_improvements if i.get('change_percent', 0) > 10]
        }
    
    def process_load_test_results(self) -> Optional[Dict[str, Any]]:
        """Process load test results."""
        # Check for k6 results
        k6_file = self.results_dir / "k6-results.json"
        locust_files = list(self.results_dir.glob("**/locust-results_*.csv"))
        
        load_test_summary = {}
        
        # Process k6 results
        if k6_file.exists():
            try:
                with open(k6_file, 'r') as f:
                    k6_data = json.load(f)
                
                # Extract key metrics from k6
                metrics = k6_data.get('metrics', {})
                load_test_summary['k6'] = {
                    "total_requests": metrics.get('http_reqs', {}).get('count', 0),
                    "avg_response_time": metrics.get('http_req_duration', {}).get('avg', 0),
                    "error_rate": metrics.get('http_req_failed', {}).get('rate', 0) * 100,
                    "throughput": metrics.get('http_reqs', {}).get('rate', 0)
                }
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not process k6 results: {e}")
        
        # Process Locust results (basic parsing)
        if locust_files:
            load_test_summary['locust'] = {
                "files_found": len(locust_files),
                "note": "Locust CSV results require additional parsing"
            }
        
        return load_test_summary if load_test_summary else None
    
    def generate_recommendations(self, 
                               benchmark_data: Dict, 
                               regression_data: Dict,
                               load_test_data: Optional[Dict]) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        # Benchmark-based recommendations
        for suite_name, suite_data in benchmark_data.get('suites', {}).items():
            metrics = suite_data.get('metrics', {})
            
            # Check for slow tests
            avg_time = metrics.get('avg_execution_time', 0)
            if avg_time > 1.0:  # More than 1 second average
                recommendations.append(
                    f"🐌 **{suite_name} Suite**: Average execution time is {avg_time:.2f}s. "
                    "Consider optimizing slow algorithms or test setup."
                )
            
            # Check for performance instability
            stability = metrics.get('performance_stability', 0)
            if stability > 0.1:  # High standard deviation
                recommendations.append(
                    f"📊 **{suite_name} Suite**: Performance is unstable (stddev: {stability:.3f}). "
                    "Review test environment consistency and algorithm determinism."
                )
        
        # Regression-based recommendations
        if regression_data.get('total_regressions', 0) > 0:
            critical_count = len(regression_data.get('critical_regressions', []))
            if critical_count > 0:
                recommendations.append(
                    f"🚨 **Critical**: {critical_count} critical performance regressions detected. "
                    "Immediate investigation required before deployment."
                )
            else:
                recommendations.append(
                    f"⚠️  **Minor Regressions**: {regression_data['total_regressions']} performance "
                    "regressions detected. Monitor trends and consider optimization."
                )
        
        # Improvement recognition
        if regression_data.get('total_improvements', 0) > 0:
            recommendations.append(
                f"✅ **Good**: {regression_data['total_improvements']} performance improvements "
                "detected. Great work on optimization!"
            )
        
        # Load test recommendations
        if load_test_data:
            k6_data = load_test_data.get('k6', {})
            if k6_data:
                error_rate = k6_data.get('error_rate', 0)
                avg_response = k6_data.get('avg_response_time', 0)
                
                if error_rate > 1.0:
                    recommendations.append(
                        f"🚨 **Load Test**: Error rate is {error_rate:.2f}%. "
                        "System stability under load needs attention."
                    )
                
                if avg_response > 1000:  # More than 1 second
                    recommendations.append(
                        f"🐌 **Load Test**: Average response time is {avg_response:.0f}ms. "
                        "Consider optimizing API endpoints and database queries."
                    )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ **Excellent**: All performance metrics are within acceptable ranges. "
                "Continue monitoring and maintain current practices."
            )
        
        return recommendations
    
    def generate_markdown_summary(self) -> str:
        """Generate a comprehensive markdown summary."""
        # Process all data
        benchmark_data = self.process_benchmark_results()
        regression_data = self.process_regression_reports()
        load_test_data = self.process_load_test_results()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            benchmark_data, regression_data, load_test_data
        )
        
        # Build markdown report
        markdown = [
            "# Performance Test Summary",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📊 Overview",
            ""
        ]
        
        # Overall metrics
        total_tests = sum(
            suite['total_tests'] for suite in benchmark_data.get('suites', {}).values()
        )
        total_regressions = regression_data.get('total_regressions', 0)
        total_improvements = regression_data.get('total_improvements', 0)
        
        markdown.extend([
            f"- **Total Performance Tests**: {total_tests}",
            f"- **Performance Regressions**: {total_regressions}",
            f"- **Performance Improvements**: {total_improvements}",
            ""
        ])
        
        # Test suite breakdown
        if benchmark_data.get('suites'):
            markdown.extend([
                "## 🧪 Test Suite Performance",
                "",
                "| Suite | Tests | Avg Time (s) | Total Time (s) | Status |",
                "|-------|-------|--------------|----------------|--------|"
            ])
            
            for suite_name, suite_data in benchmark_data['suites'].items():
                metrics = suite_data.get('metrics', {})
                total_tests = suite_data.get('total_tests', 0)
                avg_time = metrics.get('avg_execution_time', 0)
                total_time = metrics.get('total_execution_time', 0)
                
                # Determine status
                status = "✅ Good"
                if avg_time > 2.0:
                    status = "🐌 Slow"
                elif avg_time > 1.0:
                    status = "⚠️ Moderate"
                
                markdown.append(
                    f"| {suite_name} | {total_tests} | {avg_time:.3f} | {total_time:.3f} | {status} |"
                )
            
            markdown.append("")
        
        # Regression details
        if regression_data.get('total_regressions', 0) > 0:
            markdown.extend([
                "## 📉 Performance Regressions",
                ""
            ])
            
            critical_regressions = regression_data.get('critical_regressions', [])
            if critical_regressions:
                markdown.extend([
                    "### 🚨 Critical Regressions (>20% slower)",
                    ""
                ])
                
                for regression in critical_regressions:
                    test_name = regression.get('test', 'Unknown')
                    change = regression.get('change_percent', 0)
                    markdown.append(f"- **{test_name}**: {change:.1f}% slower")
                
                markdown.append("")
            
            # By suite breakdown
            markdown.extend([
                "### By Test Suite",
                "",
                "| Suite | Regressions | Improvements |",
                "|-------|-------------|-------------|"
            ])
            
            for suite_name, suite_reg in regression_data.get('by_suite', {}).items():
                regressions = suite_reg.get('regressions', 0)
                improvements = suite_reg.get('improvements', 0)
                markdown.append(f"| {suite_name} | {regressions} | {improvements} |")
            
            markdown.append("")
        
        # Load test results
        if load_test_data:
            markdown.extend([
                "## 🚀 Load Test Results",
                ""
            ])
            
            if 'k6' in load_test_data:
                k6 = load_test_data['k6']
                markdown.extend([
                    "### K6 Load Test",
                    "",
                    f"- **Total Requests**: {k6.get('total_requests', 0):,}",
                    f"- **Average Response Time**: {k6.get('avg_response_time', 0):.0f}ms",
                    f"- **Error Rate**: {k6.get('error_rate', 0):.2f}%",
                    f"- **Throughput**: {k6.get('throughput', 0):.1f} req/s",
                    ""
                ])
            
            if 'locust' in load_test_data:
                markdown.extend([
                    "### Locust Load Test",
                    "",
                    f"- **Result Files**: {load_test_data['locust']['files_found']}",
                    ""
                ])
        
        # Recommendations
        markdown.extend([
            "## 💡 Recommendations",
            ""
        ])
        
        for i, recommendation in enumerate(recommendations, 1):
            markdown.append(f"{i}. {recommendation}")
        
        markdown.extend([
            "",
            "---",
            "",
            "Generated by Pynomaly Performance Testing Suite"
        ])
        
        return "\n".join(markdown)
    
    def generate_json_summary(self) -> Dict[str, Any]:
        """Generate a JSON summary for programmatic processing."""
        benchmark_data = self.process_benchmark_results()
        regression_data = self.process_regression_reports()
        load_test_data = self.process_load_test_results()
        
        return {
            "timestamp": self.summary_data["timestamp"],
            "benchmarks": benchmark_data,
            "regressions": regression_data,
            "load_tests": load_test_data,
            "recommendations": self.generate_recommendations(
                benchmark_data, regression_data, load_test_data
            ),
            "overall_status": "PASS" if regression_data.get('total_regressions', 0) == 0 else "FAIL"
        }


def main():
    """Main entry point for performance summary generator."""
    parser = argparse.ArgumentParser(description="Generate performance test summary")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing performance test results")
    parser.add_argument("--output", type=Path, default="performance-summary.md",
                       help="Output file for summary (default: performance-summary.md)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format (default: markdown)")
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        return 1
    
    print(f"📊 Generating performance summary from {args.results_dir}")
    
    # Create summary generator
    generator = PerformanceSummaryGenerator(args.results_dir)
    
    # Generate summary
    if args.format == "markdown":
        summary = generator.generate_markdown_summary()
    else:
        summary = json.dumps(generator.generate_json_summary(), indent=2)
    
    # Write output
    args.output.write_text(summary)
    
    print(f"✅ Summary generated: {args.output}")
    
    # Print brief overview
    if args.format == "markdown":
        lines = summary.split('\n')
        overview_start = next((i for i, line in enumerate(lines) if "## 📊 Overview" in line), None)
        if overview_start:
            for line in lines[overview_start:overview_start+10]:
                if line.startswith("- **"):
                    print(f"  {line}")
    
    return 0


if __name__ == "__main__":
    exit(main())