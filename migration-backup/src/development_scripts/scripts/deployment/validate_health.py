#!/usr/bin/env python3
"""
Health Validation Script for Production Deployment
Validates application health after deployment
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Health check configuration"""

    name: str
    url: str
    method: str = "GET"
    timeout: int = 30
    expected_status: int = 200
    expected_response: dict | None = None
    headers: dict | None = None
    required: bool = True


@dataclass
class HealthCheckResult:
    """Health check result"""

    check: HealthCheck
    success: bool
    status_code: int | None = None
    response_time: float | None = None
    response_body: str | None = None
    error: str | None = None


class HealthValidator:
    """Validates application health through comprehensive checks"""

    def __init__(self):
        self.base_url = os.getenv("HEALTH_CHECK_URL", "http://localhost:8000")
        self.timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "300"))
        self.project_root = Path(__file__).parent.parent.parent

    def get_health_checks(self) -> list[HealthCheck]:
        """Define all health checks to perform"""
        return [
            # Basic health checks
            HealthCheck(
                name="Application Health",
                url=f"{self.base_url}/health",
                expected_response={"status": "healthy"},
            ),
            HealthCheck(
                name="API Health",
                url=f"{self.base_url}/api/v1/health",
                expected_response={"status": "healthy"},
            ),
            HealthCheck(
                name="Metrics Endpoint",
                url=f"{self.base_url}/api/v1/health/metrics",
                required=False,
            ),
            # Authentication endpoints
            HealthCheck(
                name="Auth Endpoint Accessible",
                url=f"{self.base_url}/api/v1/auth/login",
                method="POST",
                expected_status=422,  # Validation error expected without credentials
                required=False,
            ),
            # API endpoints
            HealthCheck(
                name="Detection Endpoint Accessible",
                url=f"{self.base_url}/api/v1/detection/detect",
                method="POST",
                expected_status=422,  # Validation error expected without data
                required=False,
            ),
            HealthCheck(
                name="Models Endpoint",
                url=f"{self.base_url}/api/v1/models",
                required=False,
            ),
            # Documentation endpoints
            HealthCheck(
                name="API Documentation", url=f"{self.base_url}/docs", required=False
            ),
            HealthCheck(
                name="OpenAPI Spec", url=f"{self.base_url}/openapi.json", required=False
            ),
            # Web UI endpoints (if applicable)
            HealthCheck(name="Web UI Root", url=f"{self.base_url}/", required=False),
        ]

    async def perform_health_check(self, check: HealthCheck) -> HealthCheckResult:
        """Perform a single health check"""
        logger.info(f"Checking: {check.name}")

        try:
            start_time = time.time()

            async with httpx.AsyncClient(timeout=check.timeout) as client:
                response = await client.request(
                    method=check.method, url=check.url, headers=check.headers or {}
                )

                response_time = time.time() - start_time

                # Check status code
                if response.status_code != check.expected_status:
                    return HealthCheckResult(
                        check=check,
                        success=False,
                        status_code=response.status_code,
                        response_time=response_time,
                        response_body=response.text[:1000],
                        error=f"Expected status {check.expected_status}, got {response.status_code}",
                    )

                # Check response content if specified
                if check.expected_response:
                    try:
                        response_json = response.json()
                        for key, expected_value in check.expected_response.items():
                            if (
                                key not in response_json
                                or response_json[key] != expected_value
                            ):
                                return HealthCheckResult(
                                    check=check,
                                    success=False,
                                    status_code=response.status_code,
                                    response_time=response_time,
                                    response_body=response.text[:1000],
                                    error=f"Expected {key}={expected_value}, got {response_json.get(key)}",
                                )
                    except json.JSONDecodeError:
                        return HealthCheckResult(
                            check=check,
                            success=False,
                            status_code=response.status_code,
                            response_time=response_time,
                            response_body=response.text[:1000],
                            error="Response is not valid JSON",
                        )

                return HealthCheckResult(
                    check=check,
                    success=True,
                    status_code=response.status_code,
                    response_time=response_time,
                    response_body=response.text[:1000],
                )

        except TimeoutError:
            return HealthCheckResult(
                check=check,
                success=False,
                error=f"Request timed out after {check.timeout} seconds",
            )
        except Exception as e:
            return HealthCheckResult(check=check, success=False, error=str(e))

    async def wait_for_application_start(self) -> bool:
        """Wait for application to start up"""
        logger.info(f"Waiting for application to start at {self.base_url}")

        basic_check = HealthCheck(
            name="Basic Health", url=f"{self.base_url}/health", timeout=10
        )

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            result = await self.perform_health_check(basic_check)

            if result.success:
                logger.info("Application is responding")
                return True

            logger.info(f"Application not ready yet, waiting... ({result.error})")
            await asyncio.sleep(5)

        logger.error(f"Application failed to start within {self.timeout} seconds")
        return False

    async def validate_all_health_checks(self) -> tuple[bool, list[HealthCheckResult]]:
        """Perform all health checks"""
        logger.info("Starting comprehensive health validation")

        # Wait for application to start
        if not await self.wait_for_application_start():
            return False, []

        # Get all health checks
        health_checks = self.get_health_checks()

        # Perform checks concurrently
        tasks = [self.perform_health_check(check) for check in health_checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_results.append(
                    HealthCheckResult(
                        check=health_checks[i], success=False, error=str(result)
                    )
                )
            else:
                health_results.append(result)

        # Determine overall success
        required_checks = [r for r in health_results if r.check.required]
        required_passed = [r for r in required_checks if r.success]

        overall_success = len(required_passed) == len(required_checks)

        return overall_success, health_results

    def generate_health_report(self, results: list[HealthCheckResult]) -> dict:
        """Generate health validation report"""
        total_checks = len(results)
        passed_checks = len([r for r in results if r.success])
        failed_checks = total_checks - passed_checks

        required_checks = [r for r in results if r.check.required]
        required_passed = len([r for r in required_checks if r.success])
        required_total = len(required_checks)

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_url": self.base_url,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "required_checks": required_total,
                "required_passed": required_passed,
                "success_rate": round(passed_checks / total_checks * 100, 2)
                if total_checks > 0
                else 0,
                "overall_success": required_passed == required_total,
            },
            "checks": [
                {
                    "name": result.check.name,
                    "url": result.check.url,
                    "method": result.check.method,
                    "required": result.check.required,
                    "success": result.success,
                    "status_code": result.status_code,
                    "response_time": round(result.response_time, 3)
                    if result.response_time
                    else None,
                    "error": result.error,
                }
                for result in results
            ],
        }

        return report

    def save_health_report(self, report: dict, output_path: str | None = None):
        """Save health report to file"""
        try:
            report_path = (
                output_path
                or self.project_root
                / "artifacts"
                / "deployment"
                / "health_validation_report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Health report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save health report: {e}")

    def print_health_summary(self, results: list[HealthCheckResult]):
        """Print health check summary"""
        print("\n" + "=" * 60)
        print("HEALTH VALIDATION SUMMARY")
        print("=" * 60)

        for result in results:
            status_symbol = "✓" if result.success else "✗"
            required_text = "[REQUIRED]" if result.check.required else "[OPTIONAL]"

            print(f"{status_symbol} {result.check.name} {required_text}")

            if result.response_time:
                print(f"    Response time: {result.response_time:.3f}s")

            if result.status_code:
                print(f"    Status code: {result.status_code}")

            if result.error:
                print(f"    Error: {result.error}")

            print()

        # Summary
        total = len(results)
        passed = len([r for r in results if r.success])
        required_checks = [r for r in results if r.check.required]
        required_passed = len([r for r in required_checks if r.success])

        print(f"Total checks: {total}")
        print(f"Passed: {passed}")
        print(f"Required checks: {len(required_checks)}")
        print(f"Required passed: {required_passed}")
        print(f"Success rate: {passed/total*100:.1f}%")

        overall_success = required_passed == len(required_checks)
        print(f"Overall status: {'HEALTHY' if overall_success else 'UNHEALTHY'}")
        print("=" * 60)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate application health")
    parser.add_argument("--url", help="Base URL for health checks")
    parser.add_argument("--timeout", type=int, help="Timeout for health checks")
    parser.add_argument("--output", help="Output path for health report")

    args = parser.parse_args()

    # Override defaults with command line arguments
    if args.url:
        os.environ["HEALTH_CHECK_URL"] = args.url
    if args.timeout:
        os.environ["HEALTH_CHECK_TIMEOUT"] = str(args.timeout)

    validator = HealthValidator()

    try:
        success, results = await validator.validate_all_health_checks()

        # Generate and save report
        report = validator.generate_health_report(results)
        validator.save_health_report(report, args.output)

        # Print summary
        validator.print_health_summary(results)

        if success:
            logger.info("🎉 All required health checks passed!")
            sys.exit(0)
        else:
            logger.error("❌ Some required health checks failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Health validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
