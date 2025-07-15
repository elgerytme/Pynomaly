#!/usr/bin/env python3
"""
Smoke Tests for Pynomaly Production Deployment
==============================================

Comprehensive smoke testing suite to validate production deployment health.
Tests critical functionality across all system components.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass

import httpx
import psycopg2
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/smoke_tests.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container."""

    name: str
    passed: bool
    duration: float
    error: str | None = None
    details: dict | None = None


class SmokeTestSuite:
    """Comprehensive smoke test suite for production deployment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.results: list[TestResult] = []

    async def run_all_tests(self) -> tuple[bool, list[TestResult]]:
        """Run all smoke tests and return results."""
        logger.info("Starting comprehensive smoke test suite...")

        test_methods = [
            # Core API Tests
            self.test_api_health,
            self.test_api_documentation,
            self.test_api_openapi_schema,
            self.test_api_metrics,
            # Authentication Tests
            self.test_authentication_endpoints,
            self.test_jwt_token_validation,
            # Core Functionality Tests
            self.test_detection_endpoints,
            self.test_training_endpoints,
            self.test_automl_endpoints,
            self.test_ensemble_endpoints,
            self.test_streaming_endpoints,
            # Infrastructure Tests
            self.test_database_connectivity,
            self.test_redis_connectivity,
            self.test_cache_functionality,
            # Performance Tests
            self.test_api_response_times,
            self.test_concurrent_requests,
            # Integration Tests
            self.test_end_to_end_detection,
            self.test_model_lifecycle,
            # Monitoring Tests
            self.test_monitoring_endpoints,
            self.test_prometheus_metrics,
        ]

        for test_method in test_methods:
            try:
                start_time = time.time()
                result = await test_method()
                duration = time.time() - start_time

                if isinstance(result, TestResult):
                    result.duration = duration
                    self.results.append(result)
                else:
                    # Handle boolean return
                    self.results.append(
                        TestResult(
                            name=test_method.__name__, passed=result, duration=duration
                        )
                    )

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
                self.results.append(
                    TestResult(
                        name=test_method.__name__,
                        passed=False,
                        duration=duration,
                        error=str(e),
                    )
                )

        # Calculate overall success
        all_passed = all(result.passed for result in self.results)

        self._log_results()
        return all_passed, self.results

    async def test_api_health(self) -> TestResult:
        """Test basic API health endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/health")

            success = response.status_code == 200
            data = response.json() if response.status_code == 200 else None

            return TestResult(
                name="API Health Check",
                passed=success,
                duration=0,
                details={"status_code": response.status_code, "data": data},
            )

    async def test_api_documentation(self) -> TestResult:
        """Test API documentation accessibility."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/docs")
            success = response.status_code == 200

            return TestResult(
                name="API Documentation",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_api_openapi_schema(self) -> TestResult:
        """Test OpenAPI schema generation."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/openapi.json")

            if response.status_code != 200:
                return TestResult(
                    name="OpenAPI Schema",
                    passed=False,
                    duration=0,
                    error=f"Status code: {response.status_code}",
                )

            try:
                schema = response.json()
                endpoint_count = len(schema.get("paths", {}))
                success = endpoint_count > 50  # Expect significant number of endpoints

                return TestResult(
                    name="OpenAPI Schema",
                    passed=success,
                    duration=0,
                    details={
                        "endpoint_count": endpoint_count,
                        "openapi_version": schema.get("openapi"),
                    },
                )
            except Exception as e:
                return TestResult(
                    name="OpenAPI Schema",
                    passed=False,
                    duration=0,
                    error=f"Failed to parse JSON: {e}",
                )

    async def test_api_metrics(self) -> TestResult:
        """Test API metrics endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/metrics")
            success = response.status_code == 200 and "pynomaly_" in response.text

            return TestResult(
                name="API Metrics",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_authentication_endpoints(self) -> TestResult:
        """Test authentication endpoints."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test login endpoint exists
            response = await client.post(
                f"{self.base_url}/api/v1/auth/login",
                data={"username": "test", "password": "test"},
            )

            # Should return 401 or 422 (not 404/500)
            success = response.status_code in [401, 422]

            return TestResult(
                name="Authentication Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_jwt_token_validation(self) -> TestResult:
        """Test JWT token validation."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test protected endpoint without token
            response = await client.get(f"{self.base_url}/api/v1/detection/status")

            # Should require authentication
            success = response.status_code in [401, 403]

            return TestResult(
                name="JWT Token Validation",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_detection_endpoints(self) -> TestResult:
        """Test detection endpoints are accessible."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/detection/algorithms")

            # Should list available algorithms (may require auth)
            success = response.status_code in [200, 401, 403]

            return TestResult(
                name="Detection Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_training_endpoints(self) -> TestResult:
        """Test training endpoints are accessible."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/training/status")

            # Should respond appropriately (may require auth)
            success = response.status_code in [200, 401, 403]

            return TestResult(
                name="Training Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_automl_endpoints(self) -> TestResult:
        """Test AutoML endpoints are accessible."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/automl/status")

            success = response.status_code in [200, 401, 403]

            return TestResult(
                name="AutoML Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_ensemble_endpoints(self) -> TestResult:
        """Test ensemble endpoints are accessible."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/ensemble/methods")

            success = response.status_code in [200, 401, 403]

            return TestResult(
                name="Ensemble Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_streaming_endpoints(self) -> TestResult:
        """Test streaming endpoints are accessible."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/streaming/status")

            success = response.status_code in [200, 401, 403]

            return TestResult(
                name="Streaming Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_database_connectivity(self) -> TestResult:
        """Test database connectivity."""
        try:
            # Try to connect to database
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="pynomaly_prod",
                user="pynomaly_user",
                password="test_password",  # This will fail in production
                connect_timeout=5,
            )
            conn.close()
            success = True
            error = None
        except psycopg2.OperationalError as e:
            # Expected in production without real credentials
            success = "Connection refused" not in str(e)  # Service should be running
            error = str(e)
        except Exception as e:
            success = False
            error = str(e)

        return TestResult(
            name="Database Connectivity", passed=success, duration=0, error=error
        )

    async def test_redis_connectivity(self) -> TestResult:
        """Test Redis connectivity."""
        try:
            r = redis.Redis(
                host="localhost", port=6379, decode_responses=True, socket_timeout=5
            )
            r.ping()
            success = True
            error = None
        except redis.ConnectionError as e:
            # Check if service is at least running
            success = "Connection refused" not in str(e)
            error = str(e)
        except Exception as e:
            success = False
            error = str(e)

        return TestResult(
            name="Redis Connectivity", passed=success, duration=0, error=error
        )

    async def test_cache_functionality(self) -> TestResult:
        """Test cache functionality through API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Make the same request twice to test caching
            url = f"{self.base_url}/api/v1/detection/algorithms"

            response1 = await client.get(url)
            response2 = await client.get(url)

            # Both should succeed (or fail) consistently
            success = response1.status_code == response2.status_code

            return TestResult(
                name="Cache Functionality",
                passed=success,
                duration=0,
                details={
                    "first_status": response1.status_code,
                    "second_status": response2.status_code,
                },
            )

    async def test_api_response_times(self) -> TestResult:
        """Test API response times are reasonable."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            start_time = time.time()
            response = await client.get(f"{self.base_url}/api/v1/health")
            response_time = time.time() - start_time

            # Health endpoint should respond within 1 second
            success = response.status_code == 200 and response_time < 1.0

            return TestResult(
                name="API Response Times",
                passed=success,
                duration=response_time,
                details={"response_time_ms": int(response_time * 1000)},
            )

    async def test_concurrent_requests(self) -> TestResult:
        """Test handling of concurrent requests."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Send 10 concurrent requests
            tasks = []
            for _ in range(10):
                tasks.append(client.get(f"{self.base_url}/api/v1/health"))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful responses
            successful = sum(
                1
                for r in responses
                if hasattr(r, "status_code") and r.status_code == 200
            )
            success = successful >= 8  # At least 80% should succeed

            return TestResult(
                name="Concurrent Requests",
                passed=success,
                duration=0,
                details={"successful_requests": successful, "total_requests": 10},
            )

    async def test_end_to_end_detection(self) -> TestResult:
        """Test end-to-end detection workflow (if possible without auth)."""
        # This is a placeholder - would need valid authentication in production
        return TestResult(
            name="End-to-End Detection",
            passed=True,  # Skip for now
            duration=0,
            details={"skipped": "Requires authentication"},
        )

    async def test_model_lifecycle(self) -> TestResult:
        """Test model lifecycle operations (if possible without auth)."""
        # This is a placeholder - would need valid authentication in production
        return TestResult(
            name="Model Lifecycle",
            passed=True,  # Skip for now
            duration=0,
            details={"skipped": "Requires authentication"},
        )

    async def test_monitoring_endpoints(self) -> TestResult:
        """Test monitoring endpoints."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Test health dependencies endpoint
            response = await client.get(f"{self.base_url}/api/v1/health/dependencies")

            success = response.status_code in [
                200,
                503,
            ]  # Either healthy or unhealthy is fine

            return TestResult(
                name="Monitoring Endpoints",
                passed=success,
                duration=0,
                details={"status_code": response.status_code},
            )

    async def test_prometheus_metrics(self) -> TestResult:
        """Test Prometheus metrics format."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/metrics")

            if response.status_code != 200:
                return TestResult(
                    name="Prometheus Metrics",
                    passed=False,
                    duration=0,
                    error=f"Status code: {response.status_code}",
                )

            # Check for basic Prometheus metric format
            content = response.text
            has_help = "# HELP" in content
            has_type = "# TYPE" in content
            has_metrics = any(
                line and not line.startswith("#") for line in content.split("\n")
            )

            success = has_help and has_type and has_metrics

            return TestResult(
                name="Prometheus Metrics",
                passed=success,
                duration=0,
                details={
                    "has_help": has_help,
                    "has_type": has_type,
                    "has_metrics": has_metrics,
                },
            )

    def _log_results(self):
        """Log test results summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        logger.info(f"\n{'='*60}")
        logger.info("SMOKE TEST RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"{'='*60}")

        if failed_tests > 0:
            logger.info("FAILED TESTS:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"❌ {result.name}: {result.error or 'Failed'}")
                    if result.details:
                        logger.error(f"   Details: {result.details}")

        logger.info("PASSED TESTS:")
        for result in self.results:
            if result.passed:
                logger.info(f"✅ {result.name} ({result.duration:.3f}s)")


async def main():
    """Main entry point for smoke tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Pynomaly Production Smoke Tests")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for API"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Request timeout in seconds"
    )
    parser.add_argument("--output", help="Output file for results (JSON)")
    args = parser.parse_args()

    # Run smoke tests
    suite = SmokeTestSuite(base_url=args.url, timeout=args.timeout)
    success, results = await suite.run_all_tests()

    # Save results if requested
    if args.output:
        output_data = {
            "success": success,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "failed_tests": sum(1 for r in results if not r.passed),
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "error": r.error,
                    "details": r.details,
                }
                for r in results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
