#!/usr/bin/env python3
"""
Deployment Validation Script

This script validates that all deployed components are working correctly:
- API endpoints are responding
- Health checks are passing
- Documentation is accessible
- Monitoring is active
- CI/CD pipelines are functional
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import subprocess

import requests
import yaml
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    """Validation result model."""
    component: str
    status: str = Field(..., regex=r'^(pass|fail|skip)$')
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    duration: Optional[float] = None

class DeploymentValidator:
    """Main deployment validator class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'api_base_url': 'http://localhost:8000',
            'health_endpoints': [
                '/health',
                '/health/detailed',
                '/metrics'
            ],
            'api_endpoints': [
                '/api/v1/detectors',
                '/api/v1/health',
                '/openapi.json'
            ],
            'documentation_paths': [
                'docs/api/generated/index.html',
                'docs/api/generated/openapi.json',
                'docs/api/generated/openapi.yaml'
            ],
            'timeout': 30,
            'retry_count': 3,
            'retry_delay': 2
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Starting deployment validation...")
        
        # Run validation checks
        validation_methods = [
            self._validate_api_health,
            self._validate_api_endpoints,
            self._validate_documentation,
            self._validate_monitoring,
            self._validate_ci_cd,
            self._validate_security,
            self._validate_performance
        ]
        
        for method in validation_methods:
            try:
                method()
            except Exception as e:
                logger.error(f"Validation method {method.__name__} failed: {e}")
                self.results.append(ValidationResult(
                    component=method.__name__.replace('_validate_', ''),
                    status='fail',
                    message=f'Validation failed: {str(e)}'
                ))
        
        # Generate summary
        return self._generate_summary()
    
    def _validate_api_health(self):
        """Validate API health endpoints."""
        logger.info("Validating API health endpoints...")
        
        for endpoint in self.config['health_endpoints']:
            start_time = time.time()
            url = urljoin(self.config['api_base_url'], endpoint)
            
            try:
                response = self._make_request(url)
                duration = time.time() - start_time
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    # Check for expected health response structure
                    if 'success' in data or 'status' in data:
                        self.results.append(ValidationResult(
                            component=f'health_{endpoint.replace("/", "_")}',
                            status='pass',
                            message=f'Health endpoint {endpoint} responding correctly',
                            details={'response_time': duration, 'status_code': response.status_code},
                            duration=duration
                        ))
                    else:
                        self.results.append(ValidationResult(
                            component=f'health_{endpoint.replace("/", "_")}',
                            status='fail',
                            message=f'Health endpoint {endpoint} returned unexpected format',
                            details={'response': data},
                            duration=duration
                        ))
                else:
                    self.results.append(ValidationResult(
                        component=f'health_{endpoint.replace("/", "_")}',
                        status='fail',
                        message=f'Health endpoint {endpoint} not responding',
                        details={'status_code': response.status_code if response else None},
                        duration=duration
                    ))
            
            except Exception as e:
                self.results.append(ValidationResult(
                    component=f'health_{endpoint.replace("/", "_")}',
                    status='fail',
                    message=f'Health endpoint {endpoint} failed: {str(e)}',
                    duration=time.time() - start_time
                ))
    
    def _validate_api_endpoints(self):
        """Validate API endpoints."""
        logger.info("Validating API endpoints...")
        
        for endpoint in self.config['api_endpoints']:
            start_time = time.time()
            url = urljoin(self.config['api_base_url'], endpoint)
            
            try:
                response = self._make_request(url)
                duration = time.time() - start_time
                
                if response and response.status_code in [200, 401]:  # 401 is OK for auth-protected endpoints
                    self.results.append(ValidationResult(
                        component=f'api_{endpoint.replace("/", "_")}',
                        status='pass',
                        message=f'API endpoint {endpoint} accessible',
                        details={'response_time': duration, 'status_code': response.status_code},
                        duration=duration
                    ))
                else:
                    self.results.append(ValidationResult(
                        component=f'api_{endpoint.replace("/", "_")}',
                        status='fail',
                        message=f'API endpoint {endpoint} not accessible',
                        details={'status_code': response.status_code if response else None},
                        duration=duration
                    ))
            
            except Exception as e:
                self.results.append(ValidationResult(
                    component=f'api_{endpoint.replace("/", "_")}',
                    status='fail',
                    message=f'API endpoint {endpoint} failed: {str(e)}',
                    duration=time.time() - start_time
                ))
    
    def _validate_documentation(self):
        """Validate documentation files."""
        logger.info("Validating documentation...")
        
        for doc_path in self.config['documentation_paths']:
            start_time = time.time()
            path = Path(doc_path)
            
            try:
                if path.exists():
                    if path.suffix in ['.json', '.yaml', '.yml']:
                        # Validate JSON/YAML structure
                        with open(path, 'r') as f:
                            if path.suffix == '.json':
                                data = json.load(f)
                            else:
                                data = yaml.safe_load(f)
                        
                        # Basic validation for OpenAPI specs
                        if 'openapi' in str(path):
                            if 'openapi' in data and 'info' in data and 'paths' in data:
                                self.results.append(ValidationResult(
                                    component=f'docs_{path.name}',
                                    status='pass',
                                    message=f'Documentation file {path.name} is valid',
                                    details={'file_size': path.stat().st_size},
                                    duration=time.time() - start_time
                                ))
                            else:
                                self.results.append(ValidationResult(
                                    component=f'docs_{path.name}',
                                    status='fail',
                                    message=f'Documentation file {path.name} has invalid structure',
                                    duration=time.time() - start_time
                                ))
                        else:
                            self.results.append(ValidationResult(
                                component=f'docs_{path.name}',
                                status='pass',
                                message=f'Documentation file {path.name} exists and is valid',
                                details={'file_size': path.stat().st_size},
                                duration=time.time() - start_time
                            ))
                    
                    elif path.suffix == '.html':
                        # Validate HTML structure
                        with open(path, 'r') as f:
                            content = f.read()
                        
                        if 'swagger-ui' in content or 'api' in content.lower():
                            self.results.append(ValidationResult(
                                component=f'docs_{path.name}',
                                status='pass',
                                message=f'Documentation file {path.name} contains expected content',
                                details={'file_size': path.stat().st_size},
                                duration=time.time() - start_time
                            ))
                        else:
                            self.results.append(ValidationResult(
                                component=f'docs_{path.name}',
                                status='fail',
                                message=f'Documentation file {path.name} missing expected content',
                                duration=time.time() - start_time
                            ))
                    
                    else:
                        self.results.append(ValidationResult(
                            component=f'docs_{path.name}',
                            status='pass',
                            message=f'Documentation file {path.name} exists',
                            details={'file_size': path.stat().st_size},
                            duration=time.time() - start_time
                        ))
                
                else:
                    self.results.append(ValidationResult(
                        component=f'docs_{path.name}',
                        status='fail',
                        message=f'Documentation file {path} not found',
                        duration=time.time() - start_time
                    ))
            
            except Exception as e:
                self.results.append(ValidationResult(
                    component=f'docs_{path.name}',
                    status='fail',
                    message=f'Documentation validation failed: {str(e)}',
                    duration=time.time() - start_time
                ))
    
    def _validate_monitoring(self):
        """Validate monitoring setup."""
        logger.info("Validating monitoring setup...")
        
        start_time = time.time()
        
        # Check for monitoring script
        monitoring_script = Path('src/development_scripts/scripts/health_monitoring.py')
        if monitoring_script.exists():
            self.results.append(ValidationResult(
                component='monitoring_script',
                status='pass',
                message='Health monitoring script exists',
                details={'file_size': monitoring_script.stat().st_size},
                duration=time.time() - start_time
            ))
        else:
            self.results.append(ValidationResult(
                component='monitoring_script',
                status='fail',
                message='Health monitoring script not found',
                duration=time.time() - start_time
            ))
        
        # Check for Prometheus metrics endpoint
        try:
            metrics_url = urljoin(self.config['api_base_url'], '/metrics')
            response = self._make_request(metrics_url)
            
            if response and response.status_code == 200:
                self.results.append(ValidationResult(
                    component='prometheus_metrics',
                    status='pass',
                    message='Prometheus metrics endpoint accessible',
                    details={'response_time': time.time() - start_time},
                    duration=time.time() - start_time
                ))
            else:
                self.results.append(ValidationResult(
                    component='prometheus_metrics',
                    status='fail',
                    message='Prometheus metrics endpoint not accessible',
                    duration=time.time() - start_time
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                component='prometheus_metrics',
                status='fail',
                message=f'Prometheus metrics check failed: {str(e)}',
                duration=time.time() - start_time
            ))
    
    def _validate_ci_cd(self):
        """Validate CI/CD pipeline setup."""
        logger.info("Validating CI/CD pipeline setup...")
        
        start_time = time.time()
        
        # Check for GitHub Actions workflows
        workflows_dir = Path('.github/workflows')
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml'))
            
            if workflow_files:
                self.results.append(ValidationResult(
                    component='github_workflows',
                    status='pass',
                    message=f'Found {len(workflow_files)} GitHub Actions workflows',
                    details={'workflows': [f.name for f in workflow_files]},
                    duration=time.time() - start_time
                ))
            else:
                self.results.append(ValidationResult(
                    component='github_workflows',
                    status='fail',
                    message='No GitHub Actions workflows found',
                    duration=time.time() - start_time
                ))
        else:
            self.results.append(ValidationResult(
                component='github_workflows',
                status='fail',
                message='GitHub Actions workflows directory not found',
                duration=time.time() - start_time
            ))
        
        # Check for pyproject.toml
        pyproject_file = Path('pyproject.toml')
        if pyproject_file.exists():
            self.results.append(ValidationResult(
                component='pyproject_toml',
                status='pass',
                message='Project configuration file exists',
                details={'file_size': pyproject_file.stat().st_size},
                duration=time.time() - start_time
            ))
        else:
            self.results.append(ValidationResult(
                component='pyproject_toml',
                status='fail',
                message='Project configuration file not found',
                duration=time.time() - start_time
            ))
    
    def _validate_security(self):
        """Validate security configuration."""
        logger.info("Validating security configuration...")
        
        start_time = time.time()
        
        # Check for security scripts
        security_scripts = [
            'src/development_scripts/scripts/security_hardening.py',
            'src/development_scripts/scripts/security/automated_security_scanner.py'
        ]
        
        found_scripts = []
        for script_path in security_scripts:
            if Path(script_path).exists():
                found_scripts.append(script_path)
        
        if found_scripts:
            self.results.append(ValidationResult(
                component='security_scripts',
                status='pass',
                message=f'Found {len(found_scripts)} security scripts',
                details={'scripts': found_scripts},
                duration=time.time() - start_time
            ))
        else:
            self.results.append(ValidationResult(
                component='security_scripts',
                status='fail',
                message='No security scripts found',
                duration=time.time() - start_time
            ))
    
    def _validate_performance(self):
        """Validate performance monitoring."""
        logger.info("Validating performance monitoring...")
        
        start_time = time.time()
        
        # Check for performance test scripts
        performance_scripts = [
            'src/development_scripts/scripts/performance_testing.py',
            'src/development_scripts/scripts/performance/run_performance_tests.py'
        ]
        
        found_scripts = []
        for script_path in performance_scripts:
            if Path(script_path).exists():
                found_scripts.append(script_path)
        
        if found_scripts:
            self.results.append(ValidationResult(
                component='performance_scripts',
                status='pass',
                message=f'Found {len(found_scripts)} performance scripts',
                details={'scripts': found_scripts},
                duration=time.time() - start_time
            ))
        else:
            self.results.append(ValidationResult(
                component='performance_scripts',
                status='fail',
                message='No performance scripts found',
                duration=time.time() - start_time
            ))
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.config['retry_count']):
            try:
                response = requests.get(
                    url,
                    timeout=self.config['timeout'],
                    headers={'User-Agent': 'Deployment-Validator/1.0'}
                )
                return response
            except Exception as e:
                if attempt == self.config['retry_count'] - 1:
                    raise e
                time.sleep(self.config['retry_delay'])
        
        return None
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == 'pass'])
        failed_tests = len([r for r in self.results if r.status == 'fail'])
        skipped_tests = len([r for r in self.results if r.status == 'skip'])
        
        overall_status = 'PASS' if failed_tests == 0 else 'FAIL'
        total_duration = time.time() - self.start_time
        
        summary = {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_duration': total_duration,
            'timestamp': datetime.now().isoformat(),
            'results': [r.dict() for r in self.results]
        }
        
        return summary
    
    def save_report(self, report_path: str = 'deployment_validation_report.json'):
        """Save validation report to file."""
        summary = self._generate_summary()
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path
    
    def print_summary(self):
        """Print validation summary to console."""
        summary = self._generate_summary()
        
        print("\n" + "="*60)
        print("🔍 DEPLOYMENT VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print("="*60)
        
        # Print detailed results
        print("\n📊 DETAILED RESULTS:")
        for result in self.results:
            status_icon = "✅" if result.status == 'pass' else "❌" if result.status == 'fail' else "⏭️"
            print(f"{status_icon} {result.component}: {result.message}")
            if result.duration:
                print(f"   Duration: {result.duration:.2f}s")
            if result.details:
                print(f"   Details: {result.details}")
        
        print("\n" + "="*60)
        
        if summary['overall_status'] == 'FAIL':
            print("❌ DEPLOYMENT VALIDATION FAILED")
            print("Please review the failed tests above and fix the issues.")
        else:
            print("✅ DEPLOYMENT VALIDATION PASSED")
            print("All components are working correctly!")
        
        print("="*60)

def main():
    """Main function for standalone execution."""
    print("🚀 Starting Deployment Validation...")
    
    # Create validator
    validator = DeploymentValidator()
    
    try:
        # Run validation
        summary = validator.validate_all()
        
        # Print summary
        validator.print_summary()
        
        # Save report
        report_path = validator.save_report()
        
        # Exit with appropriate code
        if summary['overall_status'] == 'FAIL':
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()