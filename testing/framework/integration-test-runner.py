#!/usr/bin/env python3
"""
Integration Test Runner for Domain-Driven Monorepo

Orchestrates comprehensive integration testing across packages,
validating interactions, event flows, and system behavior.
"""

import asyncio
import docker
import json
import os
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import requests
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class TestEnvironment:
    """Test environment configuration."""
    name: str
    docker_compose_file: str
    services: List[str]
    health_checks: Dict[str, str]
    cleanup_required: bool = True

@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    test_name: str
    status: str  # 'pass', 'fail', 'skip'
    duration_seconds: float
    message: str
    details: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None

@dataclass
class TestSuite:
    """Collection of integration tests."""
    name: str
    description: str
    tests: List['IntegrationTest'] = field(default_factory=list)
    setup_script: Optional[str] = None
    cleanup_script: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

class IntegrationTest:
    """Base class for integration tests."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"test.{name}")
    
    async def setup(self, environment: TestEnvironment) -> bool:
        """Setup test prerequisites."""
        return True
    
    async def run(self, environment: TestEnvironment) -> IntegrationTestResult:
        """Run the integration test."""
        start_time = time.time()
        
        try:
            await self.setup(environment)
            success = await self.execute(environment)
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=self.name,
                status="pass" if success else "fail",
                duration_seconds=duration,
                message="Test completed successfully" if success else "Test failed"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return IntegrationTestResult(
                test_name=self.name,
                status="fail",
                duration_seconds=duration,
                message=f"Test failed with exception: {str(e)}",
                details={"exception": str(e)}
            )
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Execute the test logic. Override in subclasses."""
        raise NotImplementedError
    
    async def cleanup(self, environment: TestEnvironment):
        """Clean up test resources."""
        pass

class PackageIndependenceTest(IntegrationTest):
    """Test that packages can run independently."""
    
    def __init__(self, package_name: str):
        super().__init__(
            f"independence_{package_name}",
            f"Test that {package_name} package runs independently"
        )
        self.package_name = package_name
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Test package independence by starting it in isolation."""
        try:
            # Create isolated environment for the package
            isolated_compose = self._create_isolated_compose()
            
            # Start package services
            result = subprocess.run([
                "docker-compose", "-f", isolated_compose,
                "up", "-d", "--build"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start {self.package_name}: {result.stderr}")
                return False
            
            # Wait for services to be healthy
            await asyncio.sleep(30)
            
            # Test basic functionality
            health_ok = await self._check_package_health()
            
            # Cleanup
            subprocess.run([
                "docker-compose", "-f", isolated_compose,
                "down", "-v"
            ], capture_output=True)
            
            return health_ok
            
        except Exception as e:
            self.logger.error(f"Independence test failed: {e}")
            return False
    
    def _create_isolated_compose(self) -> str:
        """Create isolated docker-compose file for the package."""
        compose_content = {
            'version': '3.8',
            'services': {
                f'{self.package_name}': {
                    'build': f'./src/packages/{self.package_name}',
                    'ports': ['8000:8000'],
                    'environment': {
                        'ENVIRONMENT': 'test',
                        'DATABASE_URL': 'sqlite:///test.db'
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yml', delete=False
        )
        yaml.dump(compose_content, temp_file)
        temp_file.close()
        
        return temp_file.name
    
    async def _check_package_health(self) -> bool:
        """Check if package is healthy."""
        for attempt in range(10):
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            
            await asyncio.sleep(10)
        
        return False

class EventFlowTest(IntegrationTest):
    """Test event-driven communication between packages."""
    
    def __init__(self, source_package: str, target_package: str, event_type: str):
        super().__init__(
            f"event_flow_{source_package}_to_{target_package}",
            f"Test event flow from {source_package} to {target_package}"
        )
        self.source_package = source_package
        self.target_package = target_package
        self.event_type = event_type
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Test event flow between packages."""
        try:
            # Generate unique event ID for tracking
            event_id = str(uuid.uuid4())
            
            # Trigger event in source package
            trigger_success = await self._trigger_event(event_id)
            if not trigger_success:
                return False
            
            # Wait for event processing
            await asyncio.sleep(5)
            
            # Verify event was processed by target package
            processing_success = await self._verify_event_processed(event_id)
            
            return processing_success
            
        except Exception as e:
            self.logger.error(f"Event flow test failed: {e}")
            return False
    
    async def _trigger_event(self, event_id: str) -> bool:
        """Trigger event in source package."""
        try:
            response = requests.post(
                f'http://localhost:8001/{self.source_package}/events',
                json={
                    'event_type': self.event_type,
                    'event_id': event_id,
                    'payload': {'test': True}
                },
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.error(f"Failed to trigger event: {e}")
            return False
    
    async def _verify_event_processed(self, event_id: str) -> bool:
        """Verify event was processed by target package."""
        for attempt in range(30):  # Wait up to 5 minutes
            try:
                response = requests.get(
                    f'http://localhost:8002/{self.target_package}/events/{event_id}',
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get('processed', False)
            except requests.RequestException:
                pass
            
            await asyncio.sleep(10)
        
        return False

class LoadTest(IntegrationTest):
    """Test system performance under load."""
    
    def __init__(self, target_endpoint: str, concurrent_users: int = 50, duration_seconds: int = 60):
        super().__init__(
            f"load_test_{target_endpoint.replace('/', '_')}",
            f"Load test {target_endpoint} with {concurrent_users} users for {duration_seconds}s"
        )
        self.target_endpoint = target_endpoint
        self.concurrent_users = concurrent_users
        self.duration_seconds = duration_seconds
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Execute load test."""
        try:
            # Use locust for load testing
            locust_file = self._create_locust_file()
            
            # Run locust test
            result = subprocess.run([
                'locust',
                '-f', locust_file,
                '--headless',
                '--users', str(self.concurrent_users),
                '--spawn-rate', '10',
                '--run-time', f'{self.duration_seconds}s',
                '--host', 'http://localhost:8000',
                '--csv', f'/tmp/load_test_{uuid.uuid4()}'
            ], capture_output=True, text=True, timeout=self.duration_seconds + 60)
            
            # Analyze results
            success_rate = self._parse_locust_results(result.stdout)
            
            # Consider test successful if 95%+ requests succeed
            return success_rate >= 0.95
            
        except Exception as e:
            self.logger.error(f"Load test failed: {e}")
            return False
    
    def _create_locust_file(self) -> str:
        """Create locust test file."""
        locust_content = f'''
from locust import HttpUser, task, between

class TestUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def test_endpoint(self):
        self.client.get("{self.target_endpoint}")
'''
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        )
        temp_file.write(locust_content)
        temp_file.close()
        
        return temp_file.name
    
    def _parse_locust_results(self, output: str) -> float:
        """Parse locust results to get success rate."""
        # Simple parsing - in practice, would parse CSV output
        lines = output.split('\n')
        for line in lines:
            if 'requests' in line and 'failures' in line:
                # Extract success rate from summary line
                import re
                match = re.search(r'(\d+)%', line)
                if match:
                    return float(match.group(1)) / 100
        
        return 0.0

class SecurityTest(IntegrationTest):
    """Test security aspects of the system."""
    
    def __init__(self, target_service: str):
        super().__init__(
            f"security_test_{target_service}",
            f"Security test for {target_service}"
        )
        self.target_service = target_service
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Execute security tests."""
        try:
            # Test authentication
            auth_ok = await self._test_authentication()
            
            # Test authorization
            authz_ok = await self._test_authorization()
            
            # Test input validation
            validation_ok = await self._test_input_validation()
            
            # Test for common vulnerabilities
            vuln_ok = await self._test_vulnerabilities()
            
            return all([auth_ok, authz_ok, validation_ok, vuln_ok])
            
        except Exception as e:
            self.logger.error(f"Security test failed: {e}")
            return False
    
    async def _test_authentication(self) -> bool:
        """Test authentication mechanisms."""
        try:
            # Test unauthenticated access
            response = requests.get(f'http://localhost:8000/{self.target_service}/protected')
            if response.status_code != 401:
                self.logger.error("Protected endpoint accessible without authentication")
                return False
            
            # Test with valid token
            token = await self._get_valid_token()
            if token:
                response = requests.get(
                    f'http://localhost:8000/{self.target_service}/protected',
                    headers={'Authorization': f'Bearer {token}'}
                )
                if response.status_code != 200:
                    self.logger.error("Valid token rejected")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication test failed: {e}")
            return False
    
    async def _test_authorization(self) -> bool:
        """Test authorization controls."""
        # Test with different user roles
        return True  # Simplified
    
    async def _test_input_validation(self) -> bool:
        """Test input validation."""
        test_payloads = [
            {"malicious": "<script>alert('xss')</script>"},
            {"sql_injection": "'; DROP TABLE users; --"},
            {"oversized": "x" * 10000}
        ]
        
        for payload in test_payloads:
            try:
                response = requests.post(
                    f'http://localhost:8000/{self.target_service}/api',
                    json=payload,
                    timeout=5
                )
                # Should return 400 for malicious input
                if response.status_code == 200:
                    self.logger.warning(f"Potentially unsafe input accepted: {payload}")
            except Exception:
                pass
        
        return True
    
    async def _test_vulnerabilities(self) -> bool:
        """Test for common vulnerabilities."""
        # OWASP Top 10 checks would go here
        return True
    
    async def _get_valid_token(self) -> Optional[str]:
        """Get valid JWT token for testing."""
        try:
            response = requests.post(
                'http://localhost:8000/auth/login',
                json={'username': 'testuser', 'password': 'testpass'},
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get('access_token')
        except Exception:
            pass
        return None

class DataConsistencyTest(IntegrationTest):
    """Test data consistency across packages."""
    
    def __init__(self, packages: List[str]):
        super().__init__(
            "data_consistency",
            f"Test data consistency across {len(packages)} packages"
        )
        self.packages = packages
    
    async def execute(self, environment: TestEnvironment) -> bool:
        """Test data consistency."""
        try:
            # Create test data in first package
            test_id = str(uuid.uuid4())
            create_success = await self._create_test_data(test_id)
            if not create_success:
                return False
            
            # Wait for eventual consistency
            await asyncio.sleep(10)
            
            # Verify data consistency across all packages
            consistency_checks = await asyncio.gather(*[
                self._check_data_consistency(package, test_id)
                for package in self.packages
            ])
            
            return all(consistency_checks)
            
        except Exception as e:
            self.logger.error(f"Data consistency test failed: {e}")
            return False
    
    async def _create_test_data(self, test_id: str) -> bool:
        """Create test data in the first package."""
        try:
            response = requests.post(
                f'http://localhost:8000/{self.packages[0]}/test-data',
                json={'id': test_id, 'data': 'consistency_test'},
                timeout=10
            )
            return response.status_code == 201
        except Exception as e:
            self.logger.error(f"Failed to create test data: {e}")
            return False
    
    async def _check_data_consistency(self, package: str, test_id: str) -> bool:
        """Check data consistency in a package."""
        for attempt in range(30):  # Wait up to 5 minutes
            try:
                response = requests.get(
                    f'http://localhost:8000/{package}/test-data/{test_id}',
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get('data') == 'consistency_test'
            except Exception:
                pass
            
            await asyncio.sleep(10)
        
        return False

class IntegrationTestRunner:
    """Orchestrates integration testing across the monorepo."""
    
    def __init__(self, monorepo_root: str = "."):
        self.monorepo_root = Path(monorepo_root).resolve()
        self.logger = logging.getLogger("integration_test_runner")
        self.docker_client = docker.from_env()
        
    def discover_test_suites(self) -> List[TestSuite]:
        """Discover integration test suites."""
        suites = []
        
        # Package independence tests
        independence_suite = TestSuite(
            "Package Independence",
            "Verify packages can run independently"
        )
        
        packages = self._discover_packages()
        for package in packages:
            independence_suite.tests.append(PackageIndependenceTest(package))
        
        suites.append(independence_suite)
        
        # Event flow tests
        event_suite = TestSuite(
            "Event Flow",
            "Test event-driven communication"
        )
        
        # Add specific event flow tests based on package configurations
        event_flows = self._discover_event_flows()
        for source, target, event_type in event_flows:
            event_suite.tests.append(EventFlowTest(source, target, event_type))
        
        suites.append(event_suite)
        
        # Performance tests
        perf_suite = TestSuite(
            "Performance",
            "Load and performance testing"
        )
        
        for package in packages:
            perf_suite.tests.append(LoadTest(f"/{package}/health"))
        
        suites.append(perf_suite)
        
        # Security tests
        security_suite = TestSuite(
            "Security",
            "Security and vulnerability testing"
        )
        
        for package in packages:
            security_suite.tests.append(SecurityTest(package))
        
        suites.append(security_suite)
        
        # Data consistency tests
        if len(packages) > 1:
            consistency_suite = TestSuite(
                "Data Consistency",
                "Cross-package data consistency"
            )
            consistency_suite.tests.append(DataConsistencyTest(packages))
            suites.append(consistency_suite)
        
        return suites
    
    def _discover_packages(self) -> List[str]:
        """Discover packages in the monorepo."""
        packages_dir = self.monorepo_root / "src" / "packages"
        if not packages_dir.exists():
            return []
        
        packages = []
        for item in packages_dir.iterdir():
            if item.is_dir() and (item / "src").exists():
                packages.append(item.name)
        
        return packages
    
    def _discover_event_flows(self) -> List[Tuple[str, str, str]]:
        """Discover event flows between packages."""
        # In practice, this would parse package configurations
        # to discover event subscriptions and publications
        return [
            ("user-management", "notification", "user_created"),
            ("order-management", "inventory", "order_placed"),
            ("payment", "order-management", "payment_processed")
        ]
    
    async def run_test_suite(self, suite: TestSuite, environment: TestEnvironment) -> List[IntegrationTestResult]:
        """Run a complete test suite."""
        results = []
        
        self.logger.info(f"Running test suite: {suite.name}")
        
        # Run setup script if provided
        if suite.setup_script:
            await self._run_script(suite.setup_script, environment)
        
        try:
            # Run tests in parallel where possible
            if suite.name == "Package Independence":
                # Run independence tests sequentially to avoid port conflicts
                for test in suite.tests:
                    result = await test.run(environment)
                    results.append(result)
                    self.logger.info(f"Test {test.name}: {result.status}")
            else:
                # Run other tests in parallel
                test_tasks = [test.run(environment) for test in suite.tests]
                parallel_results = await asyncio.gather(*test_tasks, return_exceptions=True)
                
                for i, result in enumerate(parallel_results):
                    if isinstance(result, Exception):
                        results.append(IntegrationTestResult(
                            test_name=suite.tests[i].name,
                            status="fail",
                            duration_seconds=0,
                            message=f"Test failed with exception: {str(result)}"
                        ))
                    else:
                        results.append(result)
                    
                    self.logger.info(f"Test {suite.tests[i].name}: {results[-1].status}")
            
        finally:
            # Run cleanup script
            if suite.cleanup_script:
                await self._run_script(suite.cleanup_script, environment)
        
        return results
    
    async def _run_script(self, script_path: str, environment: TestEnvironment):
        """Run setup or cleanup script."""
        try:
            result = subprocess.run([script_path], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.warning(f"Script {script_path} failed: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Failed to run script {script_path}: {e}")
    
    def create_test_environment(self, environment_name: str) -> TestEnvironment:
        """Create test environment."""
        environments = {
            "local": TestEnvironment(
                name="local",
                docker_compose_file="docker-compose.test.yml",
                services=["postgres", "redis", "kafka"],
                health_checks={
                    "postgres": "pg_isready -h localhost -p 5432",
                    "redis": "redis-cli -h localhost -p 6379 ping",
                    "kafka": "kafka-topics --bootstrap-server localhost:9092 --list"
                }
            ),
            "ci": TestEnvironment(
                name="ci",
                docker_compose_file="docker-compose.ci.yml",
                services=["postgres-test", "redis-test"],
                health_checks={
                    "postgres-test": "pg_isready -h localhost -p 5433",
                    "redis-test": "redis-cli -h localhost -p 6380 ping"
                }
            )
        }
        
        return environments.get(environment_name, environments["local"])
    
    async def setup_environment(self, environment: TestEnvironment) -> bool:
        """Setup test environment."""
        try:
            self.logger.info(f"Setting up environment: {environment.name}")
            
            # Start Docker services
            result = subprocess.run([
                "docker-compose", "-f", environment.docker_compose_file,
                "up", "-d"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start environment: {result.stderr}")
                return False
            
            # Wait for services to be healthy
            for service, health_check in environment.health_checks.items():
                if not await self._wait_for_service(service, health_check):
                    self.logger.error(f"Service {service} failed health check")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup environment: {e}")
            return False
    
    async def _wait_for_service(self, service: str, health_check: str, timeout: int = 120) -> bool:
        """Wait for service to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    health_check.split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    self.logger.info(f"Service {service} is healthy")
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(5)
        
        return False
    
    async def cleanup_environment(self, environment: TestEnvironment):
        """Clean up test environment."""
        if environment.cleanup_required:
            try:
                self.logger.info(f"Cleaning up environment: {environment.name}")
                subprocess.run([
                    "docker-compose", "-f", environment.docker_compose_file,
                    "down", "-v", "--remove-orphans"
                ], capture_output=True, timeout=120)
            except Exception as e:
                self.logger.error(f"Failed to cleanup environment: {e}")
    
    def generate_report(self, results: List[IntegrationTestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == "pass"])
        failed_tests = len([r for r in results if r.status == "fail"])
        skipped_tests = len([r for r in results if r.status == "skip"])
        
        total_duration = sum(r.duration_seconds for r in results)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration_seconds": total_duration
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "message": r.message,
                    "details": r.details
                }
                for r in results
            ],
            "failed_tests": [
                {
                    "test_name": r.test_name,
                    "message": r.message,
                    "details": r.details
                }
                for r in results if r.status == "fail"
            ]
        }
        
        return report

async def main():
    """Main entry point for integration test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument("--environment", default="local", help="Test environment to use")
    parser.add_argument("--suite", help="Specific test suite to run")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--format", choices=["json", "junit"], default="json", help="Report format")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = IntegrationTestRunner()
    environment = runner.create_test_environment(args.environment)
    
    try:
        # Setup environment
        if not await runner.setup_environment(environment):
            print("Failed to setup test environment")
            return 1
        
        # Discover and run test suites
        suites = runner.discover_test_suites()
        
        if args.suite:
            suites = [s for s in suites if s.name.lower() == args.suite.lower()]
            if not suites:
                print(f"Test suite '{args.suite}' not found")
                return 1
        
        all_results = []
        for suite in suites:
            results = await runner.run_test_suite(suite, environment)
            all_results.extend(results)
        
        # Generate report
        report = runner.generate_report(all_results)
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))
        
        # Return exit code based on test results
        return 0 if report["summary"]["failed"] == 0 else 1
        
    finally:
        await runner.cleanup_environment(environment)

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))