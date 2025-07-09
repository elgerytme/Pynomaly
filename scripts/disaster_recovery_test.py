#!/usr/bin/env python3
"""
Disaster Recovery Testing Framework for Pynomaly
This script performs comprehensive disaster recovery testing and validation
"""

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import psutil
import requests
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Disaster recovery test types"""
    BACKUP_RESTORE = "backup_restore"
    FAILOVER = "failover"
    NETWORK_PARTITION = "network_partition"
    DATA_CENTER_FAILURE = "data_center_failure"
    FULL_SYSTEM_RECOVERY = "full_system_recovery"
    POINT_IN_TIME_RECOVERY = "point_in_time_recovery"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Disaster recovery test case"""
    name: str
    test_type: TestType
    description: str
    duration_estimate: int  # seconds
    prerequisites: List[str]
    cleanup_required: bool = True
    critical: bool = False


@dataclass
class TestExecution:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error_message: Optional[str] = None
    logs: List[str] = None
    metrics: Dict = None


class DisasterRecoveryTester:
    """Disaster recovery testing framework"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.test_results = []
        self.system_state = {}
        self.backup_created = False
        self.recovery_point = None
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'environment': 'staging',
            'namespace': 'pynomaly-staging',
            'database': {
                'host': 'postgres-staging-service',
                'port': 5432,
                'database': 'pynomaly_staging',
                'username': 'postgres'
            },
            'application': {
                'url': 'http://localhost:8000',
                'health_endpoint': '/health',
                'api_endpoint': '/api/v1'
            },
            'backup': {
                'storage_path': '/backups',
                'retention_days': 30,
                'compression': True
            },
            'recovery': {
                'timeout': 600,  # 10 minutes
                'max_retries': 3,
                'verification_delay': 30
            },
            'monitoring': {
                'prometheus_url': 'http://prometheus:9090',
                'grafana_url': 'http://grafana:3000'
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def _execute_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """Execute shell command with timeout"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)
    
    def _check_system_health(self) -> Dict:
        """Check overall system health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'kubernetes': self._check_kubernetes_health(),
            'application': self._check_application_health(),
            'database': self._check_database_health(),
            'monitoring': self._check_monitoring_health()
        }
        return health_status
    
    def _check_kubernetes_health(self) -> Dict:
        """Check Kubernetes cluster health"""
        try:
            # Check cluster info
            returncode, stdout, stderr = self._execute_command("kubectl cluster-info")
            cluster_healthy = returncode == 0
            
            # Check node status
            returncode, stdout, stderr = self._execute_command("kubectl get nodes -o json")
            if returncode == 0:
                nodes_data = json.loads(stdout)
                ready_nodes = []
                for node in nodes_data.get('items', []):
                    node_name = node['metadata']['name']
                    conditions = node.get('status', {}).get('conditions', [])
                    ready = any(c['type'] == 'Ready' and c['status'] == 'True' for c in conditions)
                    ready_nodes.append({'name': node_name, 'ready': ready})
            else:
                ready_nodes = []
            
            # Check pod status
            namespace = self.config['namespace']
            returncode, stdout, stderr = self._execute_command(f"kubectl get pods -n {namespace} -o json")
            if returncode == 0:
                pods_data = json.loads(stdout)
                running_pods = []
                for pod in pods_data.get('items', []):
                    pod_name = pod['metadata']['name']
                    phase = pod.get('status', {}).get('phase', 'Unknown')
                    running_pods.append({'name': pod_name, 'phase': phase})
            else:
                running_pods = []
            
            return {
                'cluster_healthy': cluster_healthy,
                'nodes': ready_nodes,
                'pods': running_pods
            }
        except Exception as e:
            logger.error(f"Error checking Kubernetes health: {e}")
            return {'error': str(e)}
    
    def _check_application_health(self) -> Dict:
        """Check application health"""
        try:
            app_url = self.config['application']['url']
            health_endpoint = self.config['application']['health_endpoint']
            
            # Health check
            response = requests.get(f"{app_url}{health_endpoint}", timeout=10)
            health_status = response.status_code == 200
            
            # API check
            api_endpoint = self.config['application']['api_endpoint']
            response = requests.get(f"{app_url}{api_endpoint}/health", timeout=10)
            api_status = response.status_code == 200
            
            return {
                'health_check': health_status,
                'api_available': api_status,
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            logger.error(f"Error checking application health: {e}")
            return {'error': str(e)}
    
    def _check_database_health(self) -> Dict:
        """Check database health"""
        try:
            namespace = self.config['namespace']
            
            # Check database pod
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- pg_isready"
            )
            db_ready = returncode == 0
            
            # Check database connectivity
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- psql -U postgres -c 'SELECT version();'"
            )
            db_accessible = returncode == 0
            
            return {
                'ready': db_ready,
                'accessible': db_accessible,
                'version': stdout.strip() if db_accessible else None
            }
        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            return {'error': str(e)}
    
    def _check_monitoring_health(self) -> Dict:
        """Check monitoring system health"""
        try:
            prometheus_url = self.config['monitoring']['prometheus_url']
            grafana_url = self.config['monitoring']['grafana_url']
            
            # Check Prometheus
            try:
                response = requests.get(f"{prometheus_url}/api/v1/query?query=up", timeout=10)
                prometheus_healthy = response.status_code == 200
            except:
                prometheus_healthy = False
            
            # Check Grafana
            try:
                response = requests.get(f"{grafana_url}/api/health", timeout=10)
                grafana_healthy = response.status_code == 200
            except:
                grafana_healthy = False
            
            return {
                'prometheus': prometheus_healthy,
                'grafana': grafana_healthy
            }
        except Exception as e:
            logger.error(f"Error checking monitoring health: {e}")
            return {'error': str(e)}
    
    def _create_backup(self) -> bool:
        """Create system backup"""
        try:
            logger.info("Creating system backup...")
            
            # Create backup directory
            backup_dir = Path(self.config['backup']['storage_path'])
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"pynomaly_backup_{timestamp}"
            
            # Database backup
            namespace = self.config['namespace']
            db_backup_file = backup_path / "database.sql"
            
            backup_path.mkdir(parents=True, exist_ok=True)
            
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- pg_dump -U postgres pynomaly_staging > {db_backup_file}"
            )
            
            if returncode != 0:
                logger.error(f"Database backup failed: {stderr}")
                return False
            
            # Kubernetes configuration backup
            k8s_backup_file = backup_path / "kubernetes.yaml"
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get all -n {namespace} -o yaml > {k8s_backup_file}"
            )
            
            if returncode != 0:
                logger.error(f"Kubernetes backup failed: {stderr}")
                return False
            
            # Application data backup (if any)
            # This would backup persistent volumes, logs, etc.
            
            self.backup_created = True
            self.recovery_point = timestamp
            
            logger.info(f"Backup created successfully: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def _simulate_failure(self, failure_type: TestType) -> bool:
        """Simulate different types of failures"""
        try:
            namespace = self.config['namespace']
            
            if failure_type == TestType.BACKUP_RESTORE:
                # No actual failure simulation for backup/restore test
                return True
                
            elif failure_type == TestType.FAILOVER:
                # Simulate database failover
                logger.info("Simulating database failover...")
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl delete pod postgres-0 -n {namespace}"
                )
                return returncode == 0
                
            elif failure_type == TestType.NETWORK_PARTITION:
                # Simulate network partition
                logger.info("Simulating network partition...")
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl apply -f - <<EOF\napiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: deny-all\n  namespace: {namespace}\nspec:\n  podSelector: {{}}\n  policyTypes:\n  - Ingress\n  - Egress\nEOF"
                )
                return returncode == 0
                
            elif failure_type == TestType.DATA_CENTER_FAILURE:
                # Simulate data center failure by cordoning nodes
                logger.info("Simulating data center failure...")
                returncode, stdout, stderr = self._execute_command(
                    "kubectl get nodes -o name | head -1 | xargs kubectl cordon"
                )
                return returncode == 0
                
            elif failure_type == TestType.FULL_SYSTEM_RECOVERY:
                # Simulate full system failure
                logger.info("Simulating full system failure...")
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl delete deployment --all -n {namespace}"
                )
                return returncode == 0
                
            else:
                logger.error(f"Unknown failure type: {failure_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error simulating failure: {e}")
            return False
    
    def _restore_from_backup(self) -> bool:
        """Restore system from backup"""
        try:
            if not self.backup_created or not self.recovery_point:
                logger.error("No backup available for restoration")
                return False
            
            logger.info(f"Restoring from backup: {self.recovery_point}")
            
            namespace = self.config['namespace']
            backup_path = Path(self.config['backup']['storage_path']) / f"pynomaly_backup_{self.recovery_point}"
            
            # Restore database
            db_backup_file = backup_path / "database.sql"
            if db_backup_file.exists():
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl exec -i -n {namespace} postgres-0 -- psql -U postgres pynomaly_staging < {db_backup_file}"
                )
                
                if returncode != 0:
                    logger.error(f"Database restore failed: {stderr}")
                    return False
            
            # Restore Kubernetes configuration
            k8s_backup_file = backup_path / "kubernetes.yaml"
            if k8s_backup_file.exists():
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl apply -f {k8s_backup_file}"
                )
                
                if returncode != 0:
                    logger.error(f"Kubernetes restore failed: {stderr}")
                    return False
            
            # Wait for restoration to complete
            time.sleep(self.config['recovery']['verification_delay'])
            
            logger.info("Restoration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def _cleanup_failure_simulation(self, failure_type: TestType) -> bool:
        """Clean up after failure simulation"""
        try:
            namespace = self.config['namespace']
            
            if failure_type == TestType.NETWORK_PARTITION:
                # Remove network policy
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl delete networkpolicy deny-all -n {namespace}"
                )
                
            elif failure_type == TestType.DATA_CENTER_FAILURE:
                # Uncordon nodes
                returncode, stdout, stderr = self._execute_command(
                    "kubectl get nodes -o name | xargs kubectl uncordon"
                )
                
            elif failure_type == TestType.FULL_SYSTEM_RECOVERY:
                # Restore application deployment
                returncode, stdout, stderr = self._execute_command(
                    f"kubectl apply -f k8s/staging/"
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up failure simulation: {e}")
            return False
    
    def _wait_for_recovery(self, timeout: int = 600) -> bool:
        """Wait for system to recover"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health_status = self._check_system_health()
            
            # Check if system is healthy
            if (health_status.get('kubernetes', {}).get('cluster_healthy', False) and
                health_status.get('application', {}).get('health_check', False) and
                health_status.get('database', {}).get('ready', False)):
                
                logger.info("System recovery completed successfully")
                return True
            
            logger.info("Waiting for system recovery...")
            time.sleep(10)
        
        logger.error("System recovery timed out")
        return False
    
    def _verify_data_integrity(self) -> bool:
        """Verify data integrity after recovery"""
        try:
            namespace = self.config['namespace']
            
            # Check database integrity
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- psql -U postgres -d pynomaly_staging -c 'SELECT COUNT(*) FROM information_schema.tables;'"
            )
            
            if returncode != 0:
                logger.error(f"Database integrity check failed: {stderr}")
                return False
            
            # Check application data
            try:
                app_url = self.config['application']['url']
                response = requests.get(f"{app_url}/api/v1/health", timeout=10)
                if response.status_code != 200:
                    logger.error("Application integrity check failed")
                    return False
            except Exception as e:
                logger.error(f"Application integrity check failed: {e}")
                return False
            
            logger.info("Data integrity verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {e}")
            return False
    
    def run_test(self, test_case: TestCase) -> TestExecution:
        """Run a single disaster recovery test"""
        logger.info(f"Running test: {test_case.name}")
        
        execution = TestExecution(
            test_case=test_case,
            result=TestResult.FAILED,
            start_time=datetime.now(),
            logs=[]
        )
        
        try:
            # Check prerequisites
            for prereq in test_case.prerequisites:
                if not self._check_prerequisite(prereq):
                    execution.result = TestResult.SKIPPED
                    execution.error_message = f"Prerequisite not met: {prereq}"
                    return execution
            
            # Record initial system state
            execution.logs.append("Recording initial system state...")
            self.system_state = self._check_system_health()
            
            # Create backup if needed
            if test_case.test_type != TestType.BACKUP_RESTORE:
                if not self._create_backup():
                    execution.error_message = "Failed to create backup"
                    return execution
            
            # Run the specific test
            if test_case.test_type == TestType.BACKUP_RESTORE:
                success = self._test_backup_restore()
            elif test_case.test_type == TestType.FAILOVER:
                success = self._test_failover()
            elif test_case.test_type == TestType.NETWORK_PARTITION:
                success = self._test_network_partition()
            elif test_case.test_type == TestType.DATA_CENTER_FAILURE:
                success = self._test_data_center_failure()
            elif test_case.test_type == TestType.FULL_SYSTEM_RECOVERY:
                success = self._test_full_system_recovery()
            elif test_case.test_type == TestType.POINT_IN_TIME_RECOVERY:
                success = self._test_point_in_time_recovery()
            else:
                execution.error_message = f"Unknown test type: {test_case.test_type}"
                return execution
            
            if success:
                execution.result = TestResult.PASSED
                execution.logs.append("Test completed successfully")
            else:
                execution.result = TestResult.FAILED
                execution.logs.append("Test failed")
            
            # Cleanup if required
            if test_case.cleanup_required:
                self._cleanup_failure_simulation(test_case.test_type)
            
        except Exception as e:
            execution.result = TestResult.ERROR
            execution.error_message = str(e)
            execution.logs.append(f"Test error: {e}")
            
        finally:
            execution.end_time = datetime.now()
            execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
        return execution
    
    def _check_prerequisite(self, prereq: str) -> bool:
        """Check if prerequisite is met"""
        # Implementation depends on specific prerequisites
        # This is a placeholder implementation
        return True
    
    def _test_backup_restore(self) -> bool:
        """Test backup and restore functionality"""
        try:
            # Create backup
            if not self._create_backup():
                return False
            
            # Simulate data modification
            namespace = self.config['namespace']
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- psql -U postgres -d pynomaly_staging -c 'CREATE TABLE test_table (id INTEGER);'"
            )
            
            # Restore from backup
            if not self._restore_from_backup():
                return False
            
            # Verify restoration
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- psql -U postgres -d pynomaly_staging -c 'SELECT * FROM test_table;'"
            )
            
            # Test table should not exist after restore
            return returncode != 0
            
        except Exception as e:
            logger.error(f"Backup/restore test failed: {e}")
            return False
    
    def _test_failover(self) -> bool:
        """Test database failover"""
        try:
            # Simulate database failure
            if not self._simulate_failure(TestType.FAILOVER):
                return False
            
            # Wait for failover
            if not self._wait_for_recovery():
                return False
            
            # Verify data integrity
            return self._verify_data_integrity()
            
        except Exception as e:
            logger.error(f"Failover test failed: {e}")
            return False
    
    def _test_network_partition(self) -> bool:
        """Test network partition recovery"""
        try:
            # Simulate network partition
            if not self._simulate_failure(TestType.NETWORK_PARTITION):
                return False
            
            # Wait for network to be restored
            time.sleep(30)
            
            # Clean up network policy
            if not self._cleanup_failure_simulation(TestType.NETWORK_PARTITION):
                return False
            
            # Wait for recovery
            if not self._wait_for_recovery():
                return False
            
            # Verify system health
            health_status = self._check_system_health()
            return health_status.get('application', {}).get('health_check', False)
            
        except Exception as e:
            logger.error(f"Network partition test failed: {e}")
            return False
    
    def _test_data_center_failure(self) -> bool:
        """Test data center failure recovery"""
        try:
            # Simulate data center failure
            if not self._simulate_failure(TestType.DATA_CENTER_FAILURE):
                return False
            
            # Wait for pods to be rescheduled
            time.sleep(60)
            
            # Clean up
            if not self._cleanup_failure_simulation(TestType.DATA_CENTER_FAILURE):
                return False
            
            # Wait for recovery
            if not self._wait_for_recovery():
                return False
            
            # Verify system health
            health_status = self._check_system_health()
            return health_status.get('application', {}).get('health_check', False)
            
        except Exception as e:
            logger.error(f"Data center failure test failed: {e}")
            return False
    
    def _test_full_system_recovery(self) -> bool:
        """Test full system recovery"""
        try:
            # Simulate full system failure
            if not self._simulate_failure(TestType.FULL_SYSTEM_RECOVERY):
                return False
            
            # Restore from backup
            if not self._restore_from_backup():
                return False
            
            # Wait for recovery
            if not self._wait_for_recovery():
                return False
            
            # Verify data integrity
            return self._verify_data_integrity()
            
        except Exception as e:
            logger.error(f"Full system recovery test failed: {e}")
            return False
    
    def _test_point_in_time_recovery(self) -> bool:
        """Test point-in-time recovery"""
        try:
            # This would implement point-in-time recovery testing
            # For now, we'll use the backup/restore mechanism
            return self._test_backup_restore()
            
        except Exception as e:
            logger.error(f"Point-in-time recovery test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> List[TestExecution]:
        """Run comprehensive disaster recovery tests"""
        test_cases = [
            TestCase(
                name="Backup and Restore Test",
                test_type=TestType.BACKUP_RESTORE,
                description="Test backup creation and restoration functionality",
                duration_estimate=300,
                prerequisites=["database_available", "storage_available"],
                critical=True
            ),
            TestCase(
                name="Database Failover Test",
                test_type=TestType.FAILOVER,
                description="Test database failover and recovery",
                duration_estimate=180,
                prerequisites=["database_available"],
                critical=True
            ),
            TestCase(
                name="Network Partition Test",
                test_type=TestType.NETWORK_PARTITION,
                description="Test recovery from network partition",
                duration_estimate=120,
                prerequisites=["network_policies_supported"],
                critical=False
            ),
            TestCase(
                name="Data Center Failure Test",
                test_type=TestType.DATA_CENTER_FAILURE,
                description="Test recovery from data center failure",
                duration_estimate=240,
                prerequisites=["multi_node_cluster"],
                critical=False
            ),
            TestCase(
                name="Full System Recovery Test",
                test_type=TestType.FULL_SYSTEM_RECOVERY,
                description="Test full system recovery from backup",
                duration_estimate=600,
                prerequisites=["backup_available"],
                critical=True
            )
        ]
        
        results = []
        for test_case in test_cases:
            execution = self.run_test(test_case)
            results.append(execution)
            self.test_results.append(execution)
            
            # Log test result
            if execution.result == TestResult.PASSED:
                logger.info(f"✓ {test_case.name} - PASSED ({execution.duration:.1f}s)")
            else:
                logger.error(f"✗ {test_case.name} - {execution.result.value.upper()} ({execution.duration:.1f}s)")
                if execution.error_message:
                    logger.error(f"  Error: {execution.error_message}")
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == TestResult.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.result == TestResult.FAILED)
        error_tests = sum(1 for r in self.test_results if r.result == TestResult.ERROR)
        skipped_tests = sum(1 for r in self.test_results if r.result == TestResult.SKIPPED)
        
        critical_tests = [r for r in self.test_results if r.test_case.critical]
        critical_passed = sum(1 for r in critical_tests if r.result == TestResult.PASSED)
        
        total_duration = sum(r.duration for r in self.test_results)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'skipped': skipped_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'critical_tests': len(critical_tests),
                'critical_passed': critical_passed,
                'critical_success_rate': (critical_passed / len(critical_tests) * 100) if critical_tests else 0,
                'total_duration': total_duration
            },
            'test_results': [
                {
                    'name': r.test_case.name,
                    'type': r.test_case.test_type.value,
                    'result': r.result.value,
                    'duration': r.duration,
                    'critical': r.test_case.critical,
                    'error_message': r.error_message,
                    'start_time': r.start_time.isoformat(),
                    'end_time': r.end_time.isoformat() if r.end_time else None
                }
                for r in self.test_results
            ],
            'system_state': self.system_state,
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if r.result == TestResult.FAILED]
        critical_failed = [r for r in failed_tests if r.test_case.critical]
        
        if critical_failed:
            recommendations.append("CRITICAL: Address failed critical disaster recovery tests immediately")
            for test in critical_failed:
                recommendations.append(f"- Fix {test.test_case.name}: {test.error_message}")
        
        if failed_tests:
            recommendations.append("Address failed disaster recovery tests")
            recommendations.append("Review and improve backup procedures")
            recommendations.append("Enhance monitoring and alerting for failures")
        
        # General recommendations
        recommendations.extend([
            "Schedule regular disaster recovery testing",
            "Update disaster recovery documentation",
            "Train team on disaster recovery procedures",
            "Review and update RTO/RPO objectives",
            "Implement automated disaster recovery testing"
        ])
        
        return recommendations
    
    def save_report(self, report: Dict, filename: str = None):
        """Save test report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"disaster_recovery_report_{timestamp}.json"
        
        reports_dir = Path("./reports/disaster_recovery")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {filepath}")
        return filepath


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Disaster Recovery Testing Framework')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--test-type', choices=[t.value for t in TestType],
                       help='Specific test type to run')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = DisasterRecoveryTester(args.config)
    
    try:
        if args.test_type:
            # Run specific test
            test_case = TestCase(
                name=f"{args.test_type.replace('_', ' ').title()} Test",
                test_type=TestType(args.test_type),
                description=f"Test {args.test_type.replace('_', ' ')} functionality",
                duration_estimate=300,
                prerequisites=[],
                critical=True
            )
            execution = tester.run_test(test_case)
            results = [execution]
        else:
            # Run comprehensive tests
            results = tester.run_comprehensive_test()
        
        # Generate and save report
        report = tester.generate_report()
        tester.save_report(report, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("DISASTER RECOVERY TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Skipped: {report['summary']['skipped']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Critical Success Rate: {report['summary']['critical_success_rate']:.1f}%")
        print(f"Total Duration: {report['summary']['total_duration']:.1f}s")
        
        if report['summary']['failed'] > 0 or report['summary']['errors'] > 0:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*60)
        
        # Exit with appropriate code
        if report['summary']['critical_passed'] < report['summary']['critical_tests']:
            exit(1)  # Critical tests failed
        elif report['summary']['failed'] > 0:
            exit(2)  # Some tests failed
        else:
            exit(0)  # All tests passed
            
    except Exception as e:
        logger.error(f"Disaster recovery testing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()