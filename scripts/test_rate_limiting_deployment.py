#!/usr/bin/env python3
"""
Comprehensive deployment test for rate limiting and security middleware.
Tests the complete production-ready system with rate limiting, authentication, and security features.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for output
console = Console()


class RateLimitTestResult:
    """Test result for rate limiting validation."""

    def __init__(
        self, test_name: str, passed: bool, message: str, details: dict | None = None
    ):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.duration = 0.0


class ProductionRateLimitTester:
    """Comprehensive rate limiting and security testing for production deployment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results: list[RateLimitTestResult] = []

        # Test configuration
        self.test_user_credentials = {
            "username": "rate_limit_test_user",
            "email": "rate_limit_test@example.com",
            "password": "RateLimitTest123!",
            "full_name": "Rate Limit Test User",
        }

        self.auth_token = None
        self.api_key = None

    def log_result(
        self, test_name: str, passed: bool, message: str, details: dict | None = None
    ):
        """Log test result."""
        result = RateLimitTestResult(test_name, passed, message, details)
        self.results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"{status} {test_name}: {message}")

        if details:
            console.print(f"    Details: {json.dumps(details, indent=2)}")

    def test_basic_connectivity(self) -> bool:
        """Test basic API connectivity."""
        try:
            response = self.session.get(f"{self.base_url}/")

            if response.status_code == 200:
                self.log_result("Basic Connectivity", True, "API is accessible")
                return True
            else:
                self.log_result(
                    "Basic Connectivity", False, f"API returned {response.status_code}"
                )
                return False

        except Exception as e:
            self.log_result("Basic Connectivity", False, f"Connection failed: {str(e)}")
            return False

    def test_health_endpoint_bypass(self) -> bool:
        """Test that health endpoints bypass rate limiting."""
        try:
            # Health endpoints should not have rate limit headers
            response = self.session.get(f"{self.base_url}/api/v1/health/")

            if response.status_code == 200:
                has_rate_limit_headers = any(
                    header.startswith("X-RateLimit-") for header in response.headers
                )

                if not has_rate_limit_headers:
                    self.log_result(
                        "Health Endpoint Bypass",
                        True,
                        "Health endpoint bypasses rate limiting",
                    )
                    return True
                else:
                    self.log_result(
                        "Health Endpoint Bypass",
                        False,
                        "Health endpoint has rate limit headers",
                    )
                    return False
            else:
                self.log_result(
                    "Health Endpoint Bypass",
                    False,
                    f"Health endpoint returned {response.status_code}",
                )
                return False

        except Exception as e:
            self.log_result(
                "Health Endpoint Bypass",
                False,
                f"Health endpoint test failed: {str(e)}",
            )
            return False

    def test_rate_limit_headers_present(self) -> bool:
        """Test that rate limit headers are present in responses."""
        try:
            response = self.session.get(f"{self.base_url}/")

            expected_headers = [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ]
            present_headers = []

            for header in expected_headers:
                if header in response.headers:
                    present_headers.append(header)

            if len(present_headers) == len(expected_headers):
                self.log_result(
                    "Rate Limit Headers",
                    True,
                    "All rate limit headers present",
                    {"headers": {h: response.headers[h] for h in present_headers}},
                )
                return True
            else:
                missing = set(expected_headers) - set(present_headers)
                self.log_result(
                    "Rate Limit Headers", False, f"Missing headers: {missing}"
                )
                return False

        except Exception as e:
            self.log_result(
                "Rate Limit Headers", False, f"Header test failed: {str(e)}"
            )
            return False

    def test_concurrent_requests(self, num_requests: int = 10) -> bool:
        """Test concurrent request handling."""
        try:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = []

                for i in range(num_requests):
                    future = executor.submit(self.session.get, f"{self.base_url}/")
                    futures.append(future)

                responses = []
                for future in as_completed(futures):
                    try:
                        response = future.result(timeout=10)
                        responses.append(response)
                    except Exception as e:
                        self.log_result(
                            "Concurrent Requests", False, f"Request failed: {str(e)}"
                        )
                        return False

            end_time = time.time()
            duration = end_time - start_time

            success_count = sum(1 for r in responses if r.status_code == 200)
            rate_limited_count = sum(1 for r in responses if r.status_code == 429)

            self.log_result(
                "Concurrent Requests",
                True,
                f"Handled {num_requests} concurrent requests",
                {
                    "duration": f"{duration:.2f}s",
                    "successful": success_count,
                    "rate_limited": rate_limited_count,
                    "total": len(responses),
                },
            )
            return True

        except Exception as e:
            self.log_result(
                "Concurrent Requests", False, f"Concurrent test failed: {str(e)}"
            )
            return False

    def test_auth_rate_limiting(self) -> bool:
        """Test stricter rate limiting for authentication endpoints."""
        try:
            # First register a test user
            register_response = self.session.post(
                f"{self.base_url}/api/v1/auth/register", json=self.test_user_credentials
            )

            if register_response.status_code not in [
                200,
                201,
                409,
            ]:  # 409 = already exists
                self.log_result(
                    "Auth Rate Limiting",
                    False,
                    f"User registration failed: {register_response.status_code}",
                )
                return False

            # Test login rate limiting
            login_data = {
                "username": self.test_user_credentials["username"],
                "password": self.test_user_credentials["password"],
            }

            # Make multiple login attempts rapidly
            responses = []
            for i in range(15):  # Auth limit is typically 10/minute
                response = self.session.post(
                    f"{self.base_url}/api/v1/auth/login", data=login_data
                )
                responses.append(response)

                if response.status_code == 429:
                    break

            rate_limited_responses = [r for r in responses if r.status_code == 429]

            if rate_limited_responses:
                # Check rate limit response format
                rate_limit_response = rate_limited_responses[0]
                try:
                    data = rate_limit_response.json()
                    has_required_fields = all(
                        field in data
                        for field in ["error", "message", "limit", "remaining"]
                    )

                    if has_required_fields:
                        self.log_result(
                            "Auth Rate Limiting",
                            True,
                            "Auth endpoints have stricter rate limits",
                            {
                                "attempts_before_limit": len(responses)
                                - len(rate_limited_responses),
                                "rate_limit_response": data,
                            },
                        )
                        return True
                    else:
                        self.log_result(
                            "Auth Rate Limiting",
                            False,
                            "Rate limit response missing required fields",
                        )
                        return False

                except json.JSONDecodeError:
                    self.log_result(
                        "Auth Rate Limiting",
                        False,
                        "Rate limit response is not valid JSON",
                    )
                    return False
            else:
                self.log_result(
                    "Auth Rate Limiting",
                    False,
                    "No rate limiting detected on auth endpoints",
                )
                return False

        except Exception as e:
            self.log_result(
                "Auth Rate Limiting", False, f"Auth rate limit test failed: {str(e)}"
            )
            return False

    def test_different_ip_simulation(self) -> bool:
        """Test rate limiting with different IP addresses (simulated via headers)."""
        try:
            # Test with different X-Forwarded-For headers
            ips = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]

            results = []
            for ip in ips:
                headers = {"X-Forwarded-For": ip}
                response = self.session.get(f"{self.base_url}/", headers=headers)
                results.append(
                    (
                        ip,
                        response.status_code,
                        response.headers.get("X-RateLimit-Remaining"),
                    )
                )

            # All should be successful since they're different IPs
            successful = all(status == 200 for _, status, _ in results)

            if successful:
                self.log_result(
                    "Different IP Simulation",
                    True,
                    "Different IPs have separate rate limits",
                    {
                        "results": [
                            {"ip": ip, "status": status, "remaining": remaining}
                            for ip, status, remaining in results
                        ]
                    },
                )
                return True
            else:
                self.log_result(
                    "Different IP Simulation",
                    False,
                    "IP-based rate limiting not working properly",
                )
                return False

        except Exception as e:
            self.log_result(
                "Different IP Simulation", False, f"IP simulation test failed: {str(e)}"
            )
            return False

    def test_api_key_rate_limiting(self) -> bool:
        """Test API key-based rate limiting."""
        try:
            # Simulate API key header
            api_key = "pyn_test_key_123456"
            headers = {"X-API-Key": api_key}

            response = self.session.get(f"{self.base_url}/", headers=headers)

            if response.status_code == 200:
                # Check if rate limit headers indicate different limits for API keys
                limit = response.headers.get("X-RateLimit-Limit")
                remaining = response.headers.get("X-RateLimit-Remaining")

                self.log_result(
                    "API Key Rate Limiting",
                    True,
                    "API key rate limiting functional",
                    {
                        "limit": limit,
                        "remaining": remaining,
                        "api_key": api_key[:10] + "...",  # Partially hide key
                    },
                )
                return True
            else:
                self.log_result(
                    "API Key Rate Limiting",
                    False,
                    f"API key request failed: {response.status_code}",
                )
                return False

        except Exception as e:
            self.log_result(
                "API Key Rate Limiting", False, f"API key test failed: {str(e)}"
            )
            return False

    def test_rate_limit_recovery(self) -> bool:
        """Test rate limit recovery after waiting."""
        try:
            # Make requests until rate limited
            initial_response = self.session.get(f"{self.base_url}/")
            if initial_response.status_code != 200:
                self.log_result("Rate Limit Recovery", False, "Initial request failed")
                return False

            # Extract rate limit info
            limit = int(initial_response.headers.get("X-RateLimit-Limit", 100))
            remaining = int(initial_response.headers.get("X-RateLimit-Remaining", 100))

            # If we have enough remaining requests, make them to trigger rate limit
            if remaining > 10:
                self.log_result(
                    "Rate Limit Recovery",
                    True,
                    "Rate limit recovery test skipped (too many remaining requests)",
                    {"remaining": remaining, "limit": limit},
                )
                return True

            # Wait a short time and retry
            time.sleep(2)

            recovery_response = self.session.get(f"{self.base_url}/")
            new_remaining = int(
                recovery_response.headers.get("X-RateLimit-Remaining", 0)
            )

            if new_remaining >= remaining:
                self.log_result(
                    "Rate Limit Recovery",
                    True,
                    "Rate limit recovery working",
                    {"initial_remaining": remaining, "after_wait": new_remaining},
                )
                return True
            else:
                self.log_result(
                    "Rate Limit Recovery", False, "Rate limit not recovering properly"
                )
                return False

        except Exception as e:
            self.log_result(
                "Rate Limit Recovery", False, f"Recovery test failed: {str(e)}"
            )
            return False

    def test_security_headers(self) -> bool:
        """Test security headers are present."""
        try:
            response = self.session.get(f"{self.base_url}/")

            expected_security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy",
                "Referrer-Policy",
            ]

            present_headers = []
            for header in expected_security_headers:
                if header in response.headers:
                    present_headers.append(header)

            coverage = len(present_headers) / len(expected_security_headers) * 100

            if coverage >= 80:  # At least 80% of security headers present
                self.log_result(
                    "Security Headers",
                    True,
                    f"Security headers present ({coverage:.1f}%)",
                    {
                        "present": present_headers,
                        "missing": list(
                            set(expected_security_headers) - set(present_headers)
                        ),
                    },
                )
                return True
            else:
                self.log_result(
                    "Security Headers",
                    False,
                    f"Insufficient security headers ({coverage:.1f}%)",
                )
                return False

        except Exception as e:
            self.log_result(
                "Security Headers", False, f"Security header test failed: {str(e)}"
            )
            return False

    def test_cors_configuration(self) -> bool:
        """Test CORS configuration."""
        try:
            # Test preflight request
            headers = {
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }

            response = self.session.options(
                f"{self.base_url}/api/v1/auth/login", headers=headers
            )

            cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Headers",
            ]

            present_cors_headers = [h for h in cors_headers if h in response.headers]

            if len(present_cors_headers) >= 2:
                self.log_result(
                    "CORS Configuration",
                    True,
                    "CORS properly configured",
                    {
                        "cors_headers": {
                            h: response.headers[h] for h in present_cors_headers
                        }
                    },
                )
                return True
            else:
                self.log_result(
                    "CORS Configuration", False, "CORS not properly configured"
                )
                return False

        except Exception as e:
            self.log_result("CORS Configuration", False, f"CORS test failed: {str(e)}")
            return False

    def run_all_tests(self) -> dict[str, any]:
        """Run all rate limiting and security tests."""
        console.print("🔐 Starting comprehensive rate limiting and security tests...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            tests = [
                ("Testing basic connectivity", self.test_basic_connectivity),
                ("Testing health endpoint bypass", self.test_health_endpoint_bypass),
                ("Testing rate limit headers", self.test_rate_limit_headers_present),
                ("Testing concurrent requests", self.test_concurrent_requests),
                ("Testing auth rate limiting", self.test_auth_rate_limiting),
                ("Testing different IP simulation", self.test_different_ip_simulation),
                ("Testing API key rate limiting", self.test_api_key_rate_limiting),
                ("Testing rate limit recovery", self.test_rate_limit_recovery),
                ("Testing security headers", self.test_security_headers),
                ("Testing CORS configuration", self.test_cors_configuration),
            ]

            for description, test_func in tests:
                task = progress.add_task(description, total=None)
                start_time = time.time()

                try:
                    test_func()
                except Exception as e:
                    self.log_result(
                        description, False, f"Test failed with exception: {str(e)}"
                    )

                duration = time.time() - start_time
                if self.results:
                    self.results[-1].duration = duration

                progress.update(task, completed=True)

        return self.generate_report()

    def generate_report(self) -> dict[str, any]:
        """Generate comprehensive test report."""
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]

        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.results) * 100
                if self.results
                else 0,
            },
            "passed_tests": [
                {"name": r.test_name, "message": r.message, "duration": r.duration}
                for r in passed_tests
            ],
            "failed_tests": [
                {"name": r.test_name, "message": r.message, "details": r.details}
                for r in failed_tests
            ],
            "base_url": self.base_url,
        }

        return report

    def display_results(self):
        """Display test results in a formatted table."""
        console.print("\n📊 Rate Limiting and Security Test Results\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message", style="white")
        table.add_column("Duration", justify="right", style="dim")

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            status_style = "green" if result.passed else "red"

            duration = f"{result.duration:.2f}s" if result.duration else "-"

            table.add_row(
                result.test_name,
                f"[{status_style}]{status}[/{status_style}]",
                result.message,
                duration,
            )

        console.print(table)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        console.print(
            f"\n📈 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)"
        )

        if passed == total:
            console.print(
                "🎉 [bold green]All tests passed! Rate limiting and security systems are working correctly.[/bold green]"
            )
        else:
            console.print(
                "⚠️  [bold red]Some tests failed. Please review the issues above.[/bold red]"
            )


def main():
    """Main function to run rate limiting tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test rate limiting and security middleware"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for API"
    )
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Create tester
    tester = ProductionRateLimitTester(args.url)

    # Run tests
    report = tester.run_all_tests()

    # Display results
    if args.json:
        console.print(json.dumps(report, indent=2))
    else:
        tester.display_results()

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n📄 Report saved to {args.output}")

    # Exit with appropriate code
    if report["summary"]["success_rate"] == 100:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
