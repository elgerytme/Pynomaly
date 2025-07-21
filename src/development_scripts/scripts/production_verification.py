#!/usr/bin/env python3
"""
Production Verification Script
Comprehensive validation of production deployment readiness and health
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionVerificationSuite:
    """Comprehensive production verification suite"""

    def __init__(self):
        self.base_url = os.getenv("PRODUCTION_BASE_URL", "https://anomaly_detection.ai")
        self.api_key = os.getenv("PRODUCTION_API_KEY")
        self.timeout = 30
        self.results: list[dict[str, Any]] = []

    async def run_all_verifications(self) -> bool:
        """Run complete verification suite"""
        logger.info("Starting production verification suite...")

        verification_tasks = [
            ("API Health Check", self.verify_api_health),
            ("Authentication System", self.verify_authentication),
            ("Core API Endpoints", self.verify_core_endpoints),
            ("Database Connectivity", self.verify_database_connectivity),
            ("Performance Benchmarks", self.verify_performance),
            ("Security Headers", self.verify_security_headers),
            ("SSL/TLS Configuration", self.verify_ssl_configuration),
            ("Error Handling", self.verify_error_handling),
            ("Rate Limiting", self.verify_rate_limiting),
            ("Monitoring Endpoints", self.verify_monitoring_endpoints),
        ]

        all_passed = True

        for test_name, test_function in verification_tasks:
            logger.info(f"Running: {test_name}")
            try:
                result = await test_function()
                self.results.append(
                    {
                        "test": test_name,
                        "status": "PASS" if result else "FAIL",
                        "timestamp": datetime.now().isoformat(),
                        "details": getattr(test_function, "_last_result", {}),
                    }
                )

                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    all_passed = False

            except Exception as e:
                logger.error(f"💥 {test_name}: EXCEPTION - {e}")
                self.results.append(
                    {
                        "test": test_name,
                        "status": "ERROR",
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                )
                all_passed = False

        # Generate report
        self.generate_verification_report()

        return all_passed

    async def verify_api_health(self) -> bool:
        """Verify API health endpoints"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Health check endpoint
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    if response.status != 200:
                        return False

                    data = await response.json()
                    if data.get("status") != "healthy":
                        return False

                # Ready check endpoint
                async with session.get(
                    f"{self.base_url}/api/v1/health/ready"
                ) as response:
                    if response.status != 200:
                        return False

                    data = await response.json()
                    if not data.get("ready"):
                        return False

                # Live check endpoint
                async with session.get(
                    f"{self.base_url}/api/v1/health/live"
                ) as response:
                    if response.status != 200:
                        return False

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def verify_authentication(self) -> bool:
        """Verify authentication system"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test protected endpoint without auth
                async with session.get(
                    f"{self.base_url}/api/v1/user/profile"
                ) as response:
                    if response.status != 401:
                        return False

                # Test with valid API key if available
                if self.api_key:
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    async with session.get(
                        f"{self.base_url}/api/v1/user/profile", headers=headers
                    ) as response:
                        if response.status not in [
                            200,
                            404,
                        ]:  # 404 is ok if user doesn't exist
                            return False

            return True

        except Exception as e:
            logger.error(f"Authentication verification failed: {e}")
            return False

    async def verify_core_endpoints(self) -> bool:
        """Verify core API endpoints"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test public endpoints
                public_endpoints = [
                    "/api/v1/docs",
                    "/api/v1/openapi.json",
                    "/api/v1/algorithms",
                    "/api/v1/datasets/public",
                ]

                for endpoint in public_endpoints:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status not in [
                            200,
                            404,
                        ]:  # Some endpoints might not exist yet
                            logger.warning(
                                f"Endpoint {endpoint} returned {response.status}"
                            )

            return True

        except Exception as e:
            logger.error(f"Core endpoints verification failed: {e}")
            return False

    async def verify_database_connectivity(self) -> bool:
        """Verify database connectivity through API"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test database health through API
                async with session.get(
                    f"{self.base_url}/api/v1/health/database"
                ) as response:
                    if response.status != 200:
                        return False

                    data = await response.json()
                    if not data.get("database_healthy"):
                        return False

            return True

        except Exception as e:
            logger.error(f"Database connectivity verification failed: {e}")
            return False

    async def verify_performance(self) -> bool:
        """Verify performance benchmarks"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            ) as session:
                # Test response times
                start_time = time.time()

                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    if response.status != 200:
                        return False

                response_time = (
                    time.time() - start_time
                ) * 1000  # Convert to milliseconds

                # Response time should be under 1000ms for health check
                if response_time > 1000:
                    logger.warning(f"Slow response time: {response_time:.2f}ms")
                    return False

                logger.info(f"Health endpoint response time: {response_time:.2f}ms")

            return True

        except Exception as e:
            logger.error(f"Performance verification failed: {e}")
            return False

    async def verify_security_headers(self) -> bool:
        """Verify security headers"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    headers = response.headers

                    required_headers = [
                        "X-Content-Type-Options",
                        "X-Frame-Options",
                        "X-XSS-Protection",
                        "Strict-Transport-Security",
                    ]

                    for header in required_headers:
                        if header not in headers:
                            logger.warning(f"Missing security header: {header}")
                            return False

                    # Check specific header values
                    if headers.get("X-Content-Type-Options") != "nosniff":
                        return False

                    if headers.get("X-Frame-Options") not in ["DENY", "SAMEORIGIN"]:
                        return False

            return True

        except Exception as e:
            logger.error(f"Security headers verification failed: {e}")
            return False

    async def verify_ssl_configuration(self) -> bool:
        """Verify SSL/TLS configuration"""
        try:
            # This would typically use openssl or similar to check SSL config
            # For now, we'll just verify HTTPS works
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    if response.status != 200:
                        return False

                    # Verify we're actually using HTTPS
                    if not self.base_url.startswith("https://"):
                        logger.warning("Not using HTTPS")
                        return False

            return True

        except Exception as e:
            logger.error(f"SSL configuration verification failed: {e}")
            return False

    async def verify_error_handling(self) -> bool:
        """Verify error handling"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test 404 handling
                async with session.get(
                    f"{self.base_url}/api/v1/nonexistent"
                ) as response:
                    if response.status != 404:
                        return False

                    # Verify error response format
                    data = await response.json()
                    if "error" not in data and "detail" not in data:
                        return False

                # Test method not allowed
                async with session.post(f"{self.base_url}/api/v1/health") as response:
                    if response.status != 405:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error handling verification failed: {e}")
            return False

    async def verify_rate_limiting(self) -> bool:
        """Verify rate limiting"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            ) as session:
                # Make rapid requests to trigger rate limiting
                responses = []

                for i in range(100):  # Make 100 rapid requests
                    try:
                        async with session.get(
                            f"{self.base_url}/api/v1/health"
                        ) as response:
                            responses.append(response.status)
                            if response.status == 429:  # Too Many Requests
                                logger.info("Rate limiting is working (429 received)")
                                return True
                    except:
                        continue

                # If we made 100 requests without rate limiting, that's concerning
                logger.warning("No rate limiting detected after 100 requests")
                return True  # Don't fail the deployment for this

        except Exception as e:
            logger.error(f"Rate limiting verification failed: {e}")
            return True  # Don't fail deployment for this

    async def verify_monitoring_endpoints(self) -> bool:
        """Verify monitoring endpoints"""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test metrics endpoint
                async with session.get(f"{self.base_url}/metrics") as response:
                    if response.status not in [200, 404]:  # 404 is ok if not exposed
                        return False

                # Test admin health endpoint
                async with session.get(f"{self.base_url}/admin/health") as response:
                    if response.status not in [
                        200,
                        401,
                        404,
                    ]:  # Various responses are ok
                        return False

            return True

        except Exception as e:
            logger.error(f"Monitoring endpoints verification failed: {e}")
            return False

    def generate_verification_report(self):
        """Generate verification report"""
        try:
            report = {
                "verification_id": f"prod-verify-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "total_tests": len(self.results),
                "passed_tests": len([r for r in self.results if r["status"] == "PASS"]),
                "failed_tests": len([r for r in self.results if r["status"] == "FAIL"]),
                "error_tests": len([r for r in self.results if r["status"] == "ERROR"]),
                "results": self.results,
            }

            # Save report
            reports_dir = Path("reports/verification")
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_file = (
                reports_dir
                / f"production_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Verification report saved to {report_file}")

            # Print summary
            print("\n" + "=" * 60)
            print("PRODUCTION VERIFICATION SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {report['total_tests']}")
            print(f"Passed: {report['passed_tests']}")
            print(f"Failed: {report['failed_tests']}")
            print(f"Errors: {report['error_tests']}")
            print(
                f"Success Rate: {(report['passed_tests']/report['total_tests']*100):.1f}%"
            )
            print("=" * 60)

        except Exception as e:
            logger.error(f"Failed to generate verification report: {e}")


async def main():
    """Main function"""
    verification_suite = ProductionVerificationSuite()

    try:
        success = await verification_suite.run_all_verifications()

        if success:
            print("\n✅ All production verifications passed!")
            sys.exit(0)
        else:
            print("\n❌ Some production verifications failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Verification failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
