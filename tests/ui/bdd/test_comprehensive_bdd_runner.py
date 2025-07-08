"""Comprehensive BDD Test Runner for All User Workflows."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page
from pytest_bdd import scenarios

# Import all feature files
FEATURES_DIR = Path(__file__).parent / "features"

# Load all scenarios from feature files
scenarios(FEATURES_DIR / "data_scientist_workflows.feature")
scenarios(FEATURES_DIR / "security_analyst_workflows.feature")
scenarios(FEATURES_DIR / "ml_engineer_workflows.feature")
scenarios(FEATURES_DIR / "user_workflows.feature")
scenarios(FEATURES_DIR / "accessibility_compliance.feature")
scenarios(FEATURES_DIR / "performance_optimization.feature")

# Test configuration
BDD_CONFIG = {
    "parallel_execution": True,
    "screenshot_on_failure": True,
    "video_recording": False,
    "trace_collection": True,
    "retry_failed_scenarios": 2,
    "timeout_per_scenario": 300,  # 5 minutes
    "browsers": ["chromium", "firefox", "webkit"],
    "devices": ["desktop", "tablet", "mobile"],
    "environments": ["local", "staging", "production"],
}

# Global test results tracking
test_results = {
    "scenarios_run": 0,
    "scenarios_passed": 0,
    "scenarios_failed": 0,
    "scenarios_skipped": 0,
    "execution_time": 0,
    "browser_results": {},
    "feature_results": {},
    "error_summary": [],
}


class BDDTestRunner:
    """Comprehensive BDD test runner with advanced reporting."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or BDD_CONFIG
        self.results = test_results.copy()
        self.start_time = None
        self.reports_dir = Path("test_reports/bdd")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def setup_test_environment(self, page: Page, scenario_name: str):
        """Setup test environment for each scenario."""
        # Clear any existing state
        page.evaluate("localStorage.clear()")
        page.evaluate("sessionStorage.clear()")

        # Set test metadata
        page.add_init_script(
            f"""
            window.testMetadata = {{
                scenario: '{scenario_name}',
                timestamp: '{time.time()}',
                browser: '{page.context.browser.browser_type.name}',
                viewport: {{
                    width: {page.viewport_size['width']},
                    height: {page.viewport_size['height']}
                }}
            }};
        """
        )

        # Setup error tracking
        page.on("pageerror", lambda error: self._log_page_error(error, scenario_name))
        page.on("console", lambda msg: self._log_console_message(msg, scenario_name))

    def _log_page_error(self, error, scenario_name: str):
        """Log page errors during scenario execution."""
        error_info = {
            "scenario": scenario_name,
            "error": str(error),
            "timestamp": time.time(),
            "type": "page_error",
        }
        self.results["error_summary"].append(error_info)

    def _log_console_message(self, msg, scenario_name: str):
        """Log console messages during scenario execution."""
        if msg.type in ["error", "warning"]:
            log_info = {
                "scenario": scenario_name,
                "message": msg.text,
                "type": msg.type,
                "timestamp": time.time(),
            }
            self.results["error_summary"].append(log_info)

    async def run_scenario_with_retry(self, scenario_func, *args, **kwargs):
        """Run scenario with retry logic."""
        max_retries = self.config.get("retry_failed_scenarios", 2)

        for attempt in range(max_retries + 1):
            try:
                await scenario_func(*args, **kwargs)
                return True
            except Exception as e:
                if attempt == max_retries:
                    self.results["scenarios_failed"] += 1
                    self._log_scenario_failure(
                        scenario_func.__name__, str(e), attempt + 1
                    )
                    return False
                else:
                    # Wait before retry
                    await asyncio.sleep(2**attempt)

        return False

    def _log_scenario_failure(self, scenario_name: str, error: str, attempts: int):
        """Log scenario failure details."""
        failure_info = {
            "scenario": scenario_name,
            "error": error,
            "attempts": attempts,
            "timestamp": time.time(),
        }
        self.results["error_summary"].append(failure_info)

    def generate_comprehensive_report(self):
        """Generate comprehensive BDD test report."""
        self.results["execution_time"] = time.time() - (self.start_time or time.time())

        # Calculate success rate
        total_scenarios = self.results["scenarios_run"]
        if total_scenarios > 0:
            success_rate = (self.results["scenarios_passed"] / total_scenarios) * 100
        else:
            success_rate = 0

        report = {
            "summary": {
                "total_scenarios": total_scenarios,
                "passed": self.results["scenarios_passed"],
                "failed": self.results["scenarios_failed"],
                "skipped": self.results["scenarios_skipped"],
                "success_rate": round(success_rate, 2),
                "execution_time": round(self.results["execution_time"], 2),
            },
            "browser_breakdown": self.results.get("browser_results", {}),
            "feature_breakdown": self.results.get("feature_results", {}),
            "error_analysis": self._analyze_errors(),
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_recommendations(),
            "timestamp": time.time(),
        }

        # Save JSON report
        json_report_path = self.reports_dir / "bdd_comprehensive_report.json"
        with open(json_report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate HTML report
        html_report_path = self._generate_html_report(report)

        print("\n🎯 BDD Test Execution Summary")
        print(f"{'='*50}")
        print(f"Total Scenarios: {total_scenarios}")
        print(f"Passed: {self.results['scenarios_passed']} ✅")
        print(f"Failed: {self.results['scenarios_failed']} ❌")
        print(f"Skipped: {self.results['scenarios_skipped']} ⏭️")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Execution Time: {self.results['execution_time']:.1f}s")
        print("\n📊 Reports Generated:")
        print(f"  JSON: {json_report_path}")
        print(f"  HTML: {html_report_path}")

        return report

    def _analyze_errors(self):
        """Analyze error patterns and frequency."""
        error_analysis = {
            "total_errors": len(self.results["error_summary"]),
            "error_types": {},
            "most_common_errors": [],
            "error_patterns": [],
        }

        # Group errors by type
        for error in self.results["error_summary"]:
            error_type = error.get("type", "unknown")
            if error_type not in error_analysis["error_types"]:
                error_analysis["error_types"][error_type] = 0
            error_analysis["error_types"][error_type] += 1

        # Find most common errors
        error_messages = [
            error.get("error", error.get("message", ""))
            for error in self.results["error_summary"]
        ]
        from collections import Counter

        error_counter = Counter(error_messages)
        error_analysis["most_common_errors"] = error_counter.most_common(5)

        return error_analysis

    def _calculate_performance_metrics(self):
        """Calculate performance metrics from test execution."""
        return {
            "average_scenario_time": self.results["execution_time"]
            / max(self.results["scenarios_run"], 1),
            "scenarios_per_minute": (
                (self.results["scenarios_run"] / (self.results["execution_time"] / 60))
                if self.results["execution_time"] > 0
                else 0
            ),
            "failure_rate": (
                self.results["scenarios_failed"] / max(self.results["scenarios_run"], 1)
            )
            * 100,
            "retry_rate": len(
                [
                    e
                    for e in self.results["error_summary"]
                    if "attempts" in e and e["attempts"] > 1
                ]
            ),
        }

    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        # Performance recommendations
        avg_time = self.results["execution_time"] / max(
            self.results["scenarios_run"], 1
        )
        if avg_time > 60:
            recommendations.append(
                {
                    "category": "Performance",
                    "priority": "high",
                    "recommendation": "Scenario execution time is high. Consider parallelization or test optimization.",
                    "details": f"Average scenario time: {avg_time:.1f}s",
                }
            )

        # Reliability recommendations
        failure_rate = (
            self.results["scenarios_failed"] / max(self.results["scenarios_run"], 1)
        ) * 100
        if failure_rate > 10:
            recommendations.append(
                {
                    "category": "Reliability",
                    "priority": "high",
                    "recommendation": "High failure rate detected. Review failing scenarios and improve test stability.",
                    "details": f"Failure rate: {failure_rate:.1f}%",
                }
            )

        # Error pattern recommendations
        error_types = self._analyze_errors()["error_types"]
        if "page_error" in error_types and error_types["page_error"] > 5:
            recommendations.append(
                {
                    "category": "Application Quality",
                    "priority": "medium",
                    "recommendation": "Multiple page errors detected. Review application error handling.",
                    "details": f"Page errors: {error_types['page_error']}",
                }
            )

        return recommendations

    def _generate_html_report(self, report_data: dict[str, Any]) -> Path:
        """Generate HTML report from test data."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pynomaly BDD Test Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .header h1 {{ color: #2c3e50; margin-bottom: 10px; }}
                .header .subtitle {{ color: #7f8c8d; font-size: 16px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .metric {{ background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }}
                .metric .number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric .label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
                .success {{ color: #27ae60; }}
                .error {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .chart {{ background: #ecf0f1; padding: 20px; border-radius: 6px; margin: 20px 0; }}
                .recommendations {{ background: #fff8dc; border-left: 4px solid #f39c12; padding: 20px; margin: 20px 0; }}
                .recommendation {{ margin-bottom: 15px; }}
                .recommendation .priority {{ font-weight: bold; text-transform: uppercase; }}
                .priority.high {{ color: #e74c3c; }}
                .priority.medium {{ color: #f39c12; }}
                .priority.low {{ color: #27ae60; }}
                .error-list {{ background: #fdf2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 15px; }}
                .error-item {{ background: white; margin: 10px 0; padding: 15px; border-radius: 4px; border-left: 4px solid #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: 600; }}
                .timestamp {{ color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Pynomaly BDD Test Report</h1>
                    <div class="subtitle">Comprehensive Behavior-Driven Development Test Results</div>
                    <div class="timestamp">Generated: {timestamp}</div>
                </div>

                <div class="summary">
                    <div class="metric">
                        <div class="number">{total_scenarios}</div>
                        <div class="label">Total Scenarios</div>
                    </div>
                    <div class="metric">
                        <div class="number success">{passed}</div>
                        <div class="label">Passed</div>
                    </div>
                    <div class="metric">
                        <div class="number error">{failed}</div>
                        <div class="label">Failed</div>
                    </div>
                    <div class="metric">
                        <div class="number">{success_rate}%</div>
                        <div class="label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="number">{execution_time}s</div>
                        <div class="label">Execution Time</div>
                    </div>
                </div>

                <div class="section">
                    <h2>📊 Performance Metrics</h2>
                    <div class="chart">
                        <p><strong>Average Scenario Time:</strong> {avg_scenario_time:.1f}s</p>
                        <p><strong>Scenarios per Minute:</strong> {scenarios_per_minute:.1f}</p>
                        <p><strong>Failure Rate:</strong> {failure_rate:.1f}%</p>
                    </div>
                </div>

                {recommendations_section}

                {errors_section}

                <div class="section">
                    <h2>📋 Feature Coverage</h2>
                    <p>BDD scenarios cover the following user workflows:</p>
                    <ul>
                        <li><strong>Data Scientist Workflows:</strong> Research, analysis, model training, explainability</li>
                        <li><strong>Security Analyst Workflows:</strong> Threat detection, incident response, monitoring</li>
                        <li><strong>ML Engineer Workflows:</strong> Deployment, monitoring, MLOps, scaling</li>
                        <li><strong>Cross-cutting Concerns:</strong> Accessibility, performance, mobile responsiveness</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        # Format recommendations section
        recommendations_html = ""
        if report_data.get("recommendations"):
            recommendations_html = """
                <div class="section">
                    <h2>💡 Recommendations</h2>
                    <div class="recommendations">
            """
            for rec in report_data["recommendations"]:
                recommendations_html += f"""
                    <div class="recommendation">
                        <span class="priority {rec['priority']}">{rec['priority']}</span> -
                        <strong>{rec['category']}:</strong> {rec['recommendation']}
                        <br><small>{rec['details']}</small>
                    </div>
                """
            recommendations_html += "</div></div>"

        # Format errors section
        errors_html = ""
        error_analysis = report_data.get("error_analysis", {})
        if error_analysis.get("total_errors", 0) > 0:
            errors_html = f"""
                <div class="section">
                    <h2>🐛 Error Analysis</h2>
                    <div class="error-list">
                        <p><strong>Total Errors:</strong> {error_analysis['total_errors']}</p>
                        <h4>Most Common Errors:</h4>
            """
            for error, count in error_analysis.get("most_common_errors", [])[:3]:
                errors_html += (
                    f"<div class='error-item'>{error} (occurred {count} times)</div>"
                )
            errors_html += "</div></div>"

        # Fill template
        html_content = html_template.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_scenarios=report_data["summary"]["total_scenarios"],
            passed=report_data["summary"]["passed"],
            failed=report_data["summary"]["failed"],
            success_rate=report_data["summary"]["success_rate"],
            execution_time=report_data["summary"]["execution_time"],
            avg_scenario_time=report_data["performance_metrics"][
                "average_scenario_time"
            ],
            scenarios_per_minute=report_data["performance_metrics"][
                "scenarios_per_minute"
            ],
            failure_rate=report_data["performance_metrics"]["failure_rate"],
            recommendations_section=recommendations_html,
            errors_section=errors_html,
        )

        html_report_path = self.reports_dir / "bdd_comprehensive_report.html"
        with open(html_report_path, "w") as f:
            f.write(html_content)

        return html_report_path


# Test fixtures and configuration
@pytest.fixture(scope="session")
def bdd_runner():
    """Create BDD test runner instance."""
    return BDDTestRunner()


@pytest.fixture(autouse=True)
def setup_scenario(request, page: Page, bdd_runner: BDDTestRunner):
    """Setup each scenario with proper context."""
    scenario_name = request.node.name
    bdd_runner.setup_test_environment(page, scenario_name)
    bdd_runner.results["scenarios_run"] += 1

    if not bdd_runner.start_time:
        bdd_runner.start_time = time.time()

    yield

    # Check if scenario passed or failed
    if hasattr(request.node, "rep_call") and request.node.rep_call.passed:
        bdd_runner.results["scenarios_passed"] += 1
    elif hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        bdd_runner.results["scenarios_failed"] += 1


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


def pytest_sessionfinish(session, exitstatus):
    """Generate final report after all tests complete."""
    if hasattr(session, "_bdd_runner"):
        session._bdd_runner.generate_comprehensive_report()


@pytest.fixture(scope="session", autouse=True)
def session_bdd_runner(request):
    """Session-scoped BDD runner for final reporting."""
    runner = BDDTestRunner()
    request.session._bdd_runner = runner
    yield runner
    runner.generate_comprehensive_report()


# Example test class for organizing BDD tests
class TestDataScientistWorkflows:
    """Data scientist workflow BDD tests."""

    @pytest.mark.critical
    @pytest.mark.smoke
    def test_complete_research_workflow_financial_fraud_detection(self):
        """Test complete research workflow for financial fraud detection."""
        pass  # Implementation handled by step definitions

    @pytest.mark.analysis
    @pytest.mark.advanced
    def test_multi_algorithm_comparison_study(self):
        """Test multi-algorithm comparison study."""
        pass

    @pytest.mark.preprocessing
    @pytest.mark.data_quality
    def test_advanced_data_preprocessing_workflow(self):
        """Test advanced data preprocessing workflow."""
        pass


class TestSecurityAnalystWorkflows:
    """Security analyst workflow BDD tests."""

    @pytest.mark.critical
    @pytest.mark.security
    @pytest.mark.real_time
    def test_real_time_network_traffic_monitoring(self):
        """Test real-time network traffic monitoring."""
        pass

    @pytest.mark.incident_response
    @pytest.mark.investigation
    def test_security_incident_investigation_workflow(self):
        """Test security incident investigation workflow."""
        pass


class TestMLEngineerWorkflows:
    """ML engineer workflow BDD tests."""

    @pytest.mark.deployment
    @pytest.mark.production
    @pytest.mark.critical
    def test_production_model_deployment_pipeline(self):
        """Test production model deployment pipeline."""
        pass

    @pytest.mark.monitoring
    @pytest.mark.observability
    def test_production_model_monitoring_and_observability(self):
        """Test production model monitoring and observability."""
        pass


# CLI integration for running BDD tests
if __name__ == "__main__":
    # Run BDD tests with comprehensive reporting
    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--html=test_reports/bdd/bdd_test_report.html",
            "--self-contained-html",
            "--tb=short",
            "--maxfail=10",
        ]
    )

    print("\n🎯 BDD Test Execution Complete")
    print(f"Exit Code: {exit_code}")
    print("Reports available in: test_reports/bdd/")

    exit(exit_code)
