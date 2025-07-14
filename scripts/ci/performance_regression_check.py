#!/usr/bin/env python3
"""
Performance Regression Check for CI/CD Pipeline.

This script integrates with the CI/CD pipeline to automatically detect performance
regressions, manage baselines, and provide comprehensive reporting with alerts.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynomaly.infrastructure.performance.regression_framework import (
    PerformanceRegressionFramework,
    APIPerformanceTest,
    DatabasePerformanceTest,
    SystemResourceTest
)
from pynomaly.infrastructure.performance.baseline_tracker import (
    AdaptiveBaselineTracker,
    BaselineConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CIPerformanceChecker:
    """CI/CD integrated performance regression checker."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.framework = PerformanceRegressionFramework(
            baseline_storage_path=self.config.get('baseline_storage', 'performance_baselines')
        )
        
        baseline_config = BaselineConfig(
            min_samples=self.config.get('min_baseline_samples', 20),
            outlier_threshold=self.config.get('outlier_threshold', 3.0),
            trend_window_days=self.config.get('trend_window_days', 30),
            baseline_update_threshold=self.config.get('baseline_update_threshold', 0.15)
        )
        
        self.baseline_tracker = AdaptiveBaselineTracker(
            db_path=self.config.get('tracking_db', 'performance_baselines/tracking.db'),
            config=baseline_config
        )
        
        self.test_run_id = f"ci_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        default_config = {
            'base_url': os.getenv('PYNOMALY_TEST_URL', 'http://localhost:8000'),
            'test_duration': int(os.getenv('PERF_TEST_DURATION', '30')),
            'concurrent_users': int(os.getenv('PERF_CONCURRENT_USERS', '5')),
            'regression_threshold': float(os.getenv('PERF_REGRESSION_THRESHOLD', '20.0')),
            'fail_on_regression': os.getenv('PERF_FAIL_ON_REGRESSION', 'true').lower() == 'true',
            'baseline_storage': 'performance_baselines',
            'tracking_db': 'performance_baselines/tracking.db',
            'min_baseline_samples': 20,
            'outlier_threshold': 3.0,
            'trend_window_days': 30,
            'baseline_update_threshold': 0.15,
            'enable_alerts': True,
            'alert_webhooks': [],
            'test_scenarios': [
                {
                    'name': 'health_check',
                    'type': 'api',
                    'endpoint': '/health',
                    'method': 'GET',
                    'concurrent_users': 2,
                    'duration': 10
                },
                {
                    'name': 'dashboard_api',
                    'type': 'api', 
                    'endpoint': '/api/v1/dashboard',
                    'method': 'GET',
                    'concurrent_users': 3,
                    'duration': 15
                },
                {
                    'name': 'system_resources',
                    'type': 'system',
                    'duration': 10
                }
            ]
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def create_tests_from_config(self) -> List:
        """Create performance tests from configuration."""
        tests = []
        base_url = self.config['base_url']
        
        for scenario in self.config['test_scenarios']:
            if scenario['type'] == 'api':
                test = APIPerformanceTest(
                    name=scenario['name'],
                    endpoint=f"{base_url}{scenario['endpoint']}",
                    method=scenario.get('method', 'GET'),
                    concurrent_users=scenario.get('concurrent_users', 1),
                    duration_seconds=scenario.get('duration', 10)
                )
                tests.append(test)
            
            elif scenario['type'] == 'system':
                test = SystemResourceTest(
                    name=scenario['name'],
                    duration_seconds=scenario.get('duration', 10)
                )
                tests.append(test)
            
            elif scenario['type'] == 'database':
                test = DatabasePerformanceTest(
                    name=scenario['name'],
                    query=scenario.get('query', 'SELECT 1'),
                    iterations=scenario.get('iterations', 100)
                )
                tests.append(test)
        
        return tests
    
    async def run_performance_check(self) -> Dict[str, Any]:
        """Run complete performance regression check."""
        logger.info(f"Starting performance regression check (run_id: {self.test_run_id})")
        
        # Create and add tests
        tests = self.create_tests_from_config()
        for test in tests:
            self.framework.add_test(test)
        
        # Run tests
        start_time = time.time()
        results = await self.framework.run_tests()
        
        # Record metrics in baseline tracker
        metrics_for_tracking = []
        for test_name, test_data in results.get('test_results', {}).items():
            if test_data.get('status') == 'success':
                for metric in test_data.get('metrics', []):
                    metric['test_run_id'] = self.test_run_id
                    metrics_for_tracking.append(metric)
        
        if metrics_for_tracking:
            self.baseline_tracker.record_metrics(metrics_for_tracking, self.test_run_id)
        
        # Enhanced analysis
        regression_summary = self.framework.get_regression_summary(results)
        baseline_status = self.baseline_tracker.get_all_baselines_status()
        
        # Compile final report
        report = {
            'run_id': self.test_run_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'config': {
                'base_url': self.config['base_url'],
                'test_duration': self.config['test_duration'],
                'regression_threshold': self.config['regression_threshold']
            },
            'test_results': results,
            'regression_summary': regression_summary,
            'baseline_status': baseline_status,
            'recommendations': self._generate_recommendations(regression_summary, baseline_status),
            'ci_status': self._determine_ci_status(regression_summary),
            'environment': results.get('environment', {})
        }
        
        return report
    
    def _generate_recommendations(self, regression_summary: Dict[str, Any], 
                                 baseline_status: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if regression_summary['has_critical_regressions']:
            recommendations.append(
                "🚨 CRITICAL: Performance regressions detected. "
                "Review recent changes and optimize before deployment."
            )
        
        if regression_summary['total_regressions'] > 0:
            recommendations.append(
                f"⚠️ {regression_summary['total_regressions']} performance regressions found. "
                "Consider investigating affected components."
            )
        
        if baseline_status['average_health_score'] < 0.7:
            recommendations.append(
                "📊 Baseline health is below optimal. Consider refreshing baselines "
                "with recent stable performance data."
            )
        
        if baseline_status['degraded_baselines'] > 0:
            recommendations.append(
                f"🔧 {baseline_status['degraded_baselines']} baselines need attention. "
                "Review metrics with poor baseline health scores."
            )
        
        if regression_summary['total_improvements'] > 0:
            recommendations.append(
                f"✅ {regression_summary['total_improvements']} performance improvements detected! "
                "Consider updating baselines to reflect improvements."
            )
        
        if not recommendations:
            recommendations.append(
                "🎉 All performance metrics are within expected ranges. "
                "No immediate action required."
            )
        
        return recommendations
    
    def _determine_ci_status(self, regression_summary: Dict[str, Any]) -> str:
        """Determine CI pipeline status based on regression analysis."""
        if regression_summary['has_critical_regressions']:
            return 'FAILED'
        elif regression_summary['total_regressions'] > 0:
            return 'WARNING'
        else:
            return 'PASSED'
    
    def save_report(self, report: Dict[str, Any], output_path: str = None) -> Path:
        """Save performance report to file."""
        if output_path is None:
            output_dir = Path('performance_reports')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"performance_report_{self.test_run_id}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {output_path}")
        return output_path
    
    def generate_markdown_summary(self, report: Dict[str, Any]) -> str:
        """Generate a markdown summary for CI/CD comments."""
        regression_summary = report['regression_summary']
        ci_status = report['ci_status']
        
        # Status emoji
        status_emoji = {
            'PASSED': '✅',
            'WARNING': '⚠️', 
            'FAILED': '❌'
        }
        
        markdown = f"""
## {status_emoji.get(ci_status, '❓')} Performance Regression Report

**Status:** {ci_status}  
**Run ID:** `{report['run_id']}`  
**Duration:** {report['duration_seconds']:.1f}s  
**Timestamp:** {report['timestamp']}

### 📊 Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report['test_results']['total_tests']} |
| Successful Tests | {report['test_results']['successful_tests']} |
| Total Regressions | {regression_summary['total_regressions']} |
| Critical Regressions | {regression_summary['regressions_by_severity']['critical']} |
| Performance Improvements | {regression_summary['total_improvements']} |
| Baseline Health Score | {report['baseline_status']['average_health_score']:.2f} |

### 🎯 Regression Breakdown

"""
        
        # Add severity breakdown
        for severity, count in regression_summary['regressions_by_severity'].items():
            if count > 0:
                icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                markdown += f"- {icon} **{severity.title()}:** {count}\n"
        
        if regression_summary['total_regressions'] == 0:
            markdown += "- 🎉 No regressions detected!\n"
        
        markdown += "\n### 💡 Recommendations\n\n"
        for rec in report['recommendations']:
            markdown += f"- {rec}\n"
        
        # Add environment info
        env = report['environment']
        markdown += f"""
### 🖥️ Environment

- **Python:** {env.get('python_version', 'unknown')}
- **CPU Cores:** {env.get('cpu_count', 'unknown')}
- **Memory:** {env.get('memory_gb', 'unknown')}GB
- **Platform:** {env.get('platform', 'unknown')}
"""
        
        return markdown


async def main():
    """Main CLI function for performance regression checking."""
    parser = argparse.ArgumentParser(description='Performance Regression Checker for CI/CD')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output report path')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with error code on regression')
    parser.add_argument('--establish-baselines', action='store_true',
                       help='Establish baselines from test results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize checker
        checker = CIPerformanceChecker(args.config)
        
        # Run performance check
        report = await checker.run_performance_check()
        
        # Establish baselines if requested
        if args.establish_baselines:
            logger.info("Establishing baselines from test results...")
            # Extract metric names from results
            metric_names = set()
            for test_data in report['test_results']['test_results'].values():
                if test_data.get('status') == 'success':
                    for metric in test_data.get('metrics', []):
                        metric_names.add(metric['name'])
            
            for metric_name in metric_names:
                checker.baseline_tracker.establish_baseline(metric_name)
        
        # Save report
        if args.format == 'json':
            report_path = checker.save_report(report, args.output)
            print(f"Report saved to: {report_path}")
        elif args.format == 'markdown':
            markdown = checker.generate_markdown_summary(report)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(markdown)
                print(f"Markdown report saved to: {args.output}")
            else:
                print(markdown)
        
        # Print summary to console
        ci_status = report['ci_status']
        regression_summary = report['regression_summary']
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REGRESSION CHECK: {ci_status}")
        print(f"{'='*60}")
        print(f"Total Regressions: {regression_summary['total_regressions']}")
        print(f"Critical Regressions: {regression_summary['regressions_by_severity']['critical']}")
        print(f"Performance Improvements: {regression_summary['total_improvements']}")
        print(f"Overall Health Score: {report['baseline_status']['average_health_score']:.2f}")
        
        # Exit with appropriate code
        if args.fail_on_regression and ci_status in ['FAILED', 'WARNING']:
            print(f"\nExiting with error due to performance regressions")
            sys.exit(1)
        elif ci_status == 'FAILED':
            print(f"\nCritical performance regressions detected")
            sys.exit(1)
        else:
            print(f"\nPerformance check completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Performance check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())