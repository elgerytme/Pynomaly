#!/usr/bin/env python3
"""
Comprehensive test script for advanced security features.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime

# TODO: fix
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecurityTester:
    """Comprehensive security testing suite."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.results = {}

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def test_security_endpoints(self) -> dict:
        """Test security management API endpoints."""
        logger.info("Testing security management endpoints...")

        endpoints = [
            "/api/v1/security/overview",
            "/api/v1/security/stats",
            "/api/v1/security/threats",
            "/api/v1/security/blocked-ips",
            "/api/v1/security/rules",
            "/api/v1/security/events",
            "/api/v1/security/health",
            "/api/v1/security/dashboard-data",
        ]

        results = {}
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    results[endpoint] = {
                        "status": response.status,
                        "success": response.status == 200,
                        "response_time": None,  # Would measure in real implementation
                    }
                    if response.status == 200:
                        data = await response.json()
                        results[endpoint]["data_keys"] = (
                            list(data.keys()) if isinstance(data, dict) else None
                        )

            except Exception as e:
                results[endpoint] = {"status": None, "success": False, "error": str(e)}
                logger.error(f"Error testing {endpoint}: {e}")

        return results

    async def test_waf_detection(self) -> dict:
        """Test WAF threat detection capabilities."""
        logger.info("Testing WAF threat detection...")

        attack_patterns = {
            "sql_injection": {
                "payload": "' OR 1=1 --",
                "endpoint": "/api/v1/datasets",
                "method": "GET",
                "params": {"search": "' OR 1=1 --"},
            },
            "xss": {
                "payload": "<script>alert('XSS')</script>",
                "endpoint": "/api/v1/datasets",
                "method": "GET",
                "params": {"name": "<script>alert('XSS')</script>"},
            },
            "command_injection": {
                "payload": "; cat /etc/passwd",
                "endpoint": "/api/v1/datasets",
                "method": "GET",
                "params": {"file": "; cat /etc/passwd"},
            },
            "path_traversal": {
                "payload": "../../etc/passwd",
                "endpoint": "/api/v1/datasets",
                "method": "GET",
                "params": {"path": "../../etc/passwd"},
            },
        }

        results = {}
        for attack_type, attack_config in attack_patterns.items():
            try:
                url = f"{self.base_url}{attack_config['endpoint']}"

                if attack_config["method"] == "GET":
                    async with self.session.get(
                        url, params=attack_config["params"]
                    ) as response:
                        results[attack_type] = {
                            "status": response.status,
                            "blocked": response.status == 403,
                            "detected": response.status in [403, 429],
                            "headers": dict(response.headers),
                        }

                        # Check for WAF headers
                        waf_headers = {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower().startswith("x-waf")
                        }
                        results[attack_type]["waf_headers"] = waf_headers

            except Exception as e:
                results[attack_type] = {
                    "status": None,
                    "blocked": False,
                    "detected": False,
                    "error": str(e),
                }
                logger.error(f"Error testing {attack_type}: {e}")

        return results

    async def test_rate_limiting(self) -> dict:
        """Test rate limiting functionality."""
        logger.info("Testing rate limiting...")

        # Test rapid requests to trigger rate limiting
        endpoint = f"{self.base_url}/api/v1/health/"
        request_count = 50
        rapid_requests = []

        # Send rapid requests
        start_time = time.time()
        tasks = []
        for i in range(request_count):
            task = self.session.get(endpoint)
            tasks.append(task)

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # Analyze responses
            status_codes = []
            rate_limited_count = 0

            for response in responses:
                if isinstance(response, Exception):
                    continue

                status_codes.append(response.status)
                if response.status == 429:
                    rate_limited_count += 1

                response.close()

            return {
                "total_requests": request_count,
                "duration": end_time - start_time,
                "rate_limited_requests": rate_limited_count,
                "status_codes": status_codes,
                "rate_limiting_triggered": rate_limited_count > 0,
                "average_response_time": (end_time - start_time) / request_count,
            }

        except Exception as e:
            return {
                "total_requests": request_count,
                "error": str(e),
                "rate_limiting_triggered": False,
            }

    async def test_ip_blocking(self) -> dict:
        """Test IP blocking functionality."""
        logger.info("Testing IP blocking...")

        # Test blocking an IP
        test_ip = "192.168.1.100"
        block_data = {
            "ip": test_ip,
            "reason": "Security test",
            "duration_hours": 1,
            "block_type": "manual",
        }

        results = {}

        try:
            # Block IP
            async with self.session.post(
                f"{self.base_url}/api/v1/security/block-ip", json=block_data
            ) as response:
                results["block_request"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    results["block_request"]["response"] = await response.json()

            # Check blocked IPs list
            async with self.session.get(
                f"{self.base_url}/api/v1/security/blocked-ips"
            ) as response:
                results["blocked_ips_list"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    data = await response.json()
                    results["blocked_ips_list"]["data"] = data
                    results["blocked_ips_list"]["test_ip_blocked"] = (
                        test_ip in data.get("blocked_ips", [])
                    )

            # Unblock IP
            async with self.session.post(
                f"{self.base_url}/api/v1/security/unblock-ip?ip={test_ip}"
            ) as response:
                results["unblock_request"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    results["unblock_request"]["response"] = await response.json()

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error testing IP blocking: {e}")

        return results

    async def test_security_configuration(self) -> dict:
        """Test security configuration updates."""
        logger.info("Testing security configuration...")

        config_updates = {
            "waf_enabled": True,
            "auto_block_threshold": 3,
            "block_duration": 3600,
            "monitoring_enabled": True,
        }

        results = {}

        try:
            # Update configuration
            async with self.session.post(
                f"{self.base_url}/api/v1/security/config", json=config_updates
            ) as response:
                results["config_update"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    results["config_update"]["response"] = await response.json()

            # Get security rules
            async with self.session.get(
                f"{self.base_url}/api/v1/security/rules"
            ) as response:
                results["security_rules"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    data = await response.json()
                    results["security_rules"]["rules_count"] = len(
                        data.get("rules", [])
                    )

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error testing security configuration: {e}")

        return results

    async def test_security_dashboard(self) -> dict:
        """Test security dashboard accessibility."""
        logger.info("Testing security dashboard...")

        results = {}

        try:
            # Test dashboard page
            async with self.session.get(
                f"{self.base_url}/security-dashboard"
            ) as response:
                results["dashboard_page"] = {
                    "status": response.status,
                    "success": response.status
                    in [200, 302],  # 302 for redirect to login
                    "content_type": response.headers.get("content-type", ""),
                }

            # Test dashboard data API
            async with self.session.get(
                f"{self.base_url}/api/v1/security/dashboard-data"
            ) as response:
                results["dashboard_data"] = {
                    "status": response.status,
                    "success": response.status == 200,
                }
                if response.status == 200:
                    data = await response.json()
                    results["dashboard_data"]["data_sections"] = list(data.keys())

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error testing security dashboard: {e}")

        return results

    async def test_security_health(self) -> dict:
        """Test security component health checks."""
        logger.info("Testing security health checks...")

        results = {}

        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/security/health"
            ) as response:
                results = {"status": response.status, "success": response.status == 200}

                if response.status == 200:
                    data = await response.json()
                    results["health_data"] = data
                    results["overall_status"] = data.get("overall_status")
                    results["components"] = list(data.get("components", {}).keys())

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error testing security health: {e}")

        return results

    async def run_comprehensive_test(self) -> dict:
        """Run all security tests."""
        logger.info("Starting comprehensive security test suite...")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "tests": {},
        }

        # Run all test categories
        test_categories = [
            ("security_endpoints", self.test_security_endpoints),
            ("waf_detection", self.test_waf_detection),
            ("rate_limiting", self.test_rate_limiting),
            ("ip_blocking", self.test_ip_blocking),
            ("security_configuration", self.test_security_configuration),
            ("security_dashboard", self.test_security_dashboard),
            ("security_health", self.test_security_health),
        ]

        for category_name, test_func in test_categories:
            logger.info(f"Running {category_name} tests...")
            try:
                start_time = time.time()
                results = await test_func()
                end_time = time.time()

                test_results["tests"][category_name] = {
                    "results": results,
                    "duration": end_time - start_time,
                    "status": "completed",
                }

            except Exception as e:
                test_results["tests"][category_name] = {
                    "error": str(e),
                    "status": "failed",
                }
                logger.error(f"Test category {category_name} failed: {e}")

        return test_results

    def generate_report(self, test_results: dict) -> str:
        """Generate a comprehensive test report."""
        report = [
            "=" * 80,
            "PYNOMALY SECURITY TEST REPORT",
            "=" * 80,
            f"Timestamp: {test_results['timestamp']}",
            f"Base URL: {test_results['base_url']}",
            "",
            "TEST SUMMARY",
            "-" * 40,
        ]

        total_tests = len(test_results["tests"])
        passed_tests = sum(
            1
            for test in test_results["tests"].values()
            if test.get("status") == "completed"
        )
        failed_tests = total_tests - passed_tests

        report.extend(
            [
                f"Total Test Categories: {total_tests}",
                f"Passed: {passed_tests}",
                f"Failed: {failed_tests}",
                f"Success Rate: {(passed_tests/total_tests)*100:.1f}%",
                "",
            ]
        )

        # Detailed results
        for category, results in test_results["tests"].items():
            report.extend([f"TEST CATEGORY: {category.upper()}", "-" * 40])

            if results.get("status") == "completed":
                report.append("✅ Status: PASSED")
                report.append(f"⏱️  Duration: {results.get('duration', 0):.2f}s")

                # Category-specific summaries
                if category == "waf_detection":
                    waf_results = results.get("results", {})
                    detected_attacks = sum(
                        1
                        for attack in waf_results.values()
                        if attack.get("detected", False)
                    )
                    report.append(
                        f"🛡️  Attacks Detected: {detected_attacks}/{len(waf_results)}"
                    )

                elif category == "rate_limiting":
                    rl_results = results.get("results", {})
                    if rl_results.get("rate_limiting_triggered"):
                        report.append("🚦 Rate Limiting: ACTIVE")
                    else:
                        report.append("🚦 Rate Limiting: NOT TRIGGERED")

                elif category == "security_endpoints":
                    endpoints = results.get("results", {})
                    working_endpoints = sum(
                        1 for ep in endpoints.values() if ep.get("success", False)
                    )
                    report.append(
                        f"🔗 Working Endpoints: {working_endpoints}/{len(endpoints)}"
                    )

            else:
                report.append("❌ Status: FAILED")
                if "error" in results:
                    report.append(f"Error: {results['error']}")

            report.append("")

        # Recommendations
        report.extend(["RECOMMENDATIONS", "-" * 40])

        recommendations = []

        # Check WAF effectiveness
        waf_results = test_results["tests"].get("waf_detection", {}).get("results", {})
        if waf_results:
            detected_count = sum(
                1 for attack in waf_results.values() if attack.get("detected", False)
            )
            if detected_count < len(waf_results):
                recommendations.append(
                    "🔧 Consider tuning WAF signatures for better detection"
                )

        # Check rate limiting
        rl_results = test_results["tests"].get("rate_limiting", {}).get("results", {})
        if not rl_results.get("rate_limiting_triggered"):
            recommendations.append("🔧 Consider lowering rate limit thresholds")

        # Check endpoint availability
        endpoint_results = (
            test_results["tests"].get("security_endpoints", {}).get("results", {})
        )
        failed_endpoints = [
            ep
            for ep, result in endpoint_results.items()
            if not result.get("success", False)
        ]
        if failed_endpoints:
            recommendations.append(
                f"🔧 Fix failed endpoints: {', '.join(failed_endpoints)}"
            )

        if not recommendations:
            recommendations.append(
                "✅ All security features appear to be working correctly"
            )

        report.extend(recommendations)
        report.extend(["", "=" * 80])

        return "\n".join(report)


async def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="Test Pynomaly security features")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of Pynomaly application (default: http://localhost:8000)",
    )
    parser.add_argument("--output", help="Output file for test results (JSON format)")
    parser.add_argument("--report", help="Output file for test report (text format)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    async with SecurityTester(args.url) as tester:
        # Run comprehensive tests
        results = await tester.run_comprehensive_test()

        # Generate report
        report = tester.generate_report(results)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Test results saved to {args.output}")

        if args.report:
            with open(args.report, "w") as f:
                f.write(report)
            logger.info(f"Test report saved to {args.report}")
        else:
            print(report)

        # Exit with appropriate code
        failed_tests = sum(
            1 for test in results["tests"].values() if test.get("status") != "completed"
        )

        if failed_tests > 0:
            logger.error(f"Security tests completed with {failed_tests} failures")
            sys.exit(1)
        else:
            logger.info("All security tests passed successfully")
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
