#!/usr/bin/env python3
"""
Production Deployment Script for Pynomaly
This script handles automated deployment to production environment
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import requests
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentPhase(Enum):
    """Deployment phases"""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str = "production"
    namespace: str = "pynomaly-prod"
    image_tag: str = "latest"
    replicas: int = 5
    min_replicas: int = 3
    max_replicas: int = 50
    cpu_request: str = "500m"
    cpu_limit: str = "1000m"
    memory_request: str = "1Gi"
    memory_limit: str = "2Gi"
    timeout: int = 1800  # 30 minutes
    rollback_on_failure: bool = True
    run_tests: bool = True
    skip_confirmation: bool = False
    dry_run: bool = False


@dataclass
class DeploymentResult:
    """Deployment result"""
    status: DeploymentStatus
    phase: DeploymentPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error_message: Optional[str] = None
    logs: List[str] = None
    metrics: Dict = None
    rollback_performed: bool = False


class ProductionDeployer:
    """Production deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.k8s_manifests_path = Path(__file__).parent.parent / "k8s" / "production"
        self.previous_deployment = None
        self.deployment_logs = []
        
    def _execute_command(self, command: str, timeout: int = 300) -> Tuple[int, str, str]:
        """Execute shell command with timeout"""
        try:
            logger.info(f"Executing: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
                
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)
    
    def _log_deployment_step(self, message: str, level: str = "INFO"):
        """Log deployment step"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_logs.append(log_entry)
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def _check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        self._log_deployment_step("Checking deployment prerequisites...")
        
        # Check kubectl access
        returncode, stdout, stderr = self._execute_command("kubectl cluster-info")
        if returncode != 0:
            self._log_deployment_step(f"kubectl cluster access failed: {stderr}", "ERROR")
            return False
        
        # Check namespace exists
        returncode, stdout, stderr = self._execute_command(f"kubectl get namespace {self.config.namespace}")
        if returncode != 0:
            self._log_deployment_step(f"Namespace {self.config.namespace} not found", "ERROR")
            return False
        
        # Check Docker image exists
        if not self.config.dry_run:
            returncode, stdout, stderr = self._execute_command(f"docker pull pynomaly:{self.config.image_tag}")
            if returncode != 0:
                self._log_deployment_step(f"Docker image pynomaly:{self.config.image_tag} not found", "ERROR")
                return False
        
        # Check database connectivity
        if not self._check_database_connectivity():
            return False
        
        # Check monitoring systems
        if not self._check_monitoring_systems():
            return False
        
        self._log_deployment_step("All prerequisites satisfied")
        return True
    
    def _check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        try:
            # Check PostgreSQL
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {self.config.namespace} postgres-prod-0 -- pg_isready"
            )
            if returncode != 0:
                self._log_deployment_step("PostgreSQL not ready", "ERROR")
                return False
            
            # Check Redis
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {self.config.namespace} redis-prod-0 -- redis-cli ping"
            )
            if returncode != 0:
                self._log_deployment_step("Redis not ready", "ERROR")
                return False
            
            # Check MongoDB
            returncode, stdout, stderr = self._execute_command(
                f"kubectl exec -n {self.config.namespace} mongodb-prod-0 -- mongo --eval 'db.adminCommand(\"ping\")'"
            )
            if returncode != 0:
                self._log_deployment_step("MongoDB not ready", "ERROR")
                return False
            
            self._log_deployment_step("All databases are ready")
            return True
        except Exception as e:
            self._log_deployment_step(f"Database connectivity check failed: {e}", "ERROR")
            return False
    
    def _check_monitoring_systems(self) -> bool:
        """Check monitoring systems"""
        try:
            # Check Prometheus
            returncode, stdout, stderr = self._execute_command(
                "kubectl get pods -n monitoring -l app=prometheus --field-selector=status.phase=Running"
            )
            if returncode != 0:
                self._log_deployment_step("Prometheus not running", "WARNING")
            
            # Check Grafana
            returncode, stdout, stderr = self._execute_command(
                "kubectl get pods -n monitoring -l app=grafana --field-selector=status.phase=Running"
            )
            if returncode != 0:
                self._log_deployment_step("Grafana not running", "WARNING")
            
            return True
        except Exception as e:
            self._log_deployment_step(f"Monitoring systems check failed: {e}", "WARNING")
            return True  # Non-critical for deployment
    
    def _backup_current_deployment(self) -> bool:
        """Backup current deployment configuration"""
        try:
            self._log_deployment_step("Backing up current deployment...")
            
            backup_dir = Path(f"/tmp/pynomaly-backup-{self.deployment_id}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup current deployment
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get deployment pynomaly-prod-app -n {self.config.namespace} -o yaml > {backup_dir}/deployment.yaml"
            )
            if returncode != 0:
                self._log_deployment_step(f"Failed to backup deployment: {stderr}", "ERROR")
                return False
            
            # Backup current configmap
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get configmap pynomaly-prod-config -n {self.config.namespace} -o yaml > {backup_dir}/configmap.yaml"
            )
            if returncode != 0:
                self._log_deployment_step(f"Failed to backup configmap: {stderr}", "ERROR")
                return False
            
            # Store backup reference
            self.previous_deployment = str(backup_dir)
            self._log_deployment_step(f"Backup created at {backup_dir}")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Backup failed: {e}", "ERROR")
            return False
    
    def _run_pre_deployment_tests(self) -> bool:
        """Run pre-deployment tests"""
        if not self.config.run_tests:
            self._log_deployment_step("Skipping pre-deployment tests")
            return True
        
        self._log_deployment_step("Running pre-deployment tests...")
        
        try:
            # Run security tests
            returncode, stdout, stderr = self._execute_command(
                "python scripts/run_security_tests.sh",
                timeout=600
            )
            if returncode != 0:
                self._log_deployment_step(f"Security tests failed: {stderr}", "ERROR")
                return False
            
            # Run performance tests
            returncode, stdout, stderr = self._execute_command(
                "python scripts/run_performance_tests.sh --duration 60",
                timeout=300
            )
            if returncode != 0:
                self._log_deployment_step(f"Performance tests failed: {stderr}", "ERROR")
                return False
            
            # Run smoke tests
            returncode, stdout, stderr = self._execute_command(
                "python scripts/smoke_tests.py",
                timeout=300
            )
            if returncode != 0:
                self._log_deployment_step(f"Smoke tests failed: {stderr}", "ERROR")
                return False
            
            self._log_deployment_step("All pre-deployment tests passed")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Pre-deployment tests failed: {e}", "ERROR")
            return False
    
    def _deploy_manifests(self) -> bool:
        """Deploy Kubernetes manifests"""
        self._log_deployment_step("Deploying Kubernetes manifests...")
        
        try:
            # Apply manifests in order
            manifest_files = [
                "namespace.yaml",
                "configmap.yaml",
                "secrets.yaml",
                "databases.yaml",
                "pynomaly-production.yaml",
                "ingress.yaml"
            ]
            
            for manifest_file in manifest_files:
                manifest_path = self.k8s_manifests_path / manifest_file
                
                if not manifest_path.exists():
                    self._log_deployment_step(f"Manifest file not found: {manifest_path}", "ERROR")
                    return False
                
                # Replace placeholders with actual values
                self._update_manifest_placeholders(manifest_path)
                
                # Apply manifest
                if self.config.dry_run:
                    command = f"kubectl apply -f {manifest_path} --dry-run=client"
                else:
                    command = f"kubectl apply -f {manifest_path}"
                
                returncode, stdout, stderr = self._execute_command(command)
                if returncode != 0:
                    self._log_deployment_step(f"Failed to apply {manifest_file}: {stderr}", "ERROR")
                    return False
                
                self._log_deployment_step(f"Applied {manifest_file}")
                time.sleep(2)  # Brief pause between manifests
            
            self._log_deployment_step("All manifests deployed successfully")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Manifest deployment failed: {e}", "ERROR")
            return False
    
    def _update_manifest_placeholders(self, manifest_path: Path):
        """Update manifest placeholders with actual values"""
        try:
            with open(manifest_path, 'r') as f:
                content = f.read()
            
            # Replace placeholders
            content = content.replace("{{IMAGE_TAG}}", self.config.image_tag)
            content = content.replace("{{REPLICAS}}", str(self.config.replicas))
            content = content.replace("{{MIN_REPLICAS}}", str(self.config.min_replicas))
            content = content.replace("{{MAX_REPLICAS}}", str(self.config.max_replicas))
            content = content.replace("{{CPU_REQUEST}}", self.config.cpu_request)
            content = content.replace("{{CPU_LIMIT}}", self.config.cpu_limit)
            content = content.replace("{{MEMORY_REQUEST}}", self.config.memory_request)
            content = content.replace("{{MEMORY_LIMIT}}", self.config.memory_limit)
            content = content.replace("{{DEPLOYMENT_ID}}", self.deployment_id)
            
            with open(manifest_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            self._log_deployment_step(f"Failed to update placeholders in {manifest_path}: {e}", "ERROR")
    
    def _wait_for_deployment(self) -> bool:
        """Wait for deployment to complete"""
        self._log_deployment_step("Waiting for deployment to complete...")
        
        try:
            # Wait for deployment rollout
            command = f"kubectl rollout status deployment/pynomaly-prod-app -n {self.config.namespace} --timeout={self.config.timeout}s"
            returncode, stdout, stderr = self._execute_command(command, timeout=self.config.timeout)
            
            if returncode != 0:
                self._log_deployment_step(f"Deployment rollout failed: {stderr}", "ERROR")
                return False
            
            # Wait for pods to be ready
            time.sleep(30)  # Give pods time to start
            
            # Check pod status
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get pods -n {self.config.namespace} -l app=pynomaly,environment=production"
            )
            
            if returncode != 0:
                self._log_deployment_step(f"Failed to check pod status: {stderr}", "ERROR")
                return False
            
            # Count running pods
            running_pods = stdout.count("Running")
            if running_pods < self.config.min_replicas:
                self._log_deployment_step(f"Insufficient running pods: {running_pods}/{self.config.min_replicas}", "ERROR")
                return False
            
            self._log_deployment_step(f"Deployment completed successfully with {running_pods} running pods")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Deployment waiting failed: {e}", "ERROR")
            return False
    
    def _run_post_deployment_tests(self) -> bool:
        """Run post-deployment tests"""
        if not self.config.run_tests:
            self._log_deployment_step("Skipping post-deployment tests")
            return True
        
        self._log_deployment_step("Running post-deployment tests...")
        
        try:
            # Health check
            if not self._verify_application_health():
                return False
            
            # API tests
            if not self._run_api_tests():
                return False
            
            # Performance verification
            if not self._verify_performance():
                return False
            
            self._log_deployment_step("All post-deployment tests passed")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Post-deployment tests failed: {e}", "ERROR")
            return False
    
    def _verify_application_health(self) -> bool:
        """Verify application health"""
        try:
            # Get service endpoint
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get service pynomaly-prod-service -n {self.config.namespace} -o jsonpath='{{.spec.clusterIP}}'"
            )
            
            if returncode != 0:
                self._log_deployment_step(f"Failed to get service IP: {stderr}", "ERROR")
                return False
            
            service_ip = stdout.strip()
            
            # Test health endpoint
            for attempt in range(5):
                try:
                    # Use port-forward for testing
                    returncode, stdout, stderr = self._execute_command(
                        f"kubectl port-forward -n {self.config.namespace} service/pynomaly-prod-service 8080:8000 &"
                    )
                    
                    time.sleep(5)  # Wait for port-forward
                    
                    response = requests.get("http://localhost:8080/health", timeout=10)
                    if response.status_code == 200:
                        self._log_deployment_step("Health check passed")
                        return True
                    else:
                        self._log_deployment_step(f"Health check failed with status {response.status_code}")
                        
                except Exception as e:
                    self._log_deployment_step(f"Health check attempt {attempt + 1} failed: {e}")
                    time.sleep(10)
            
            self._log_deployment_step("Health check failed after 5 attempts", "ERROR")
            return False
            
        except Exception as e:
            self._log_deployment_step(f"Health verification failed: {e}", "ERROR")
            return False
    
    def _run_api_tests(self) -> bool:
        """Run API tests"""
        try:
            # Run API test suite
            returncode, stdout, stderr = self._execute_command(
                "python -m pytest tests/api/ -v --tb=short",
                timeout=300
            )
            
            if returncode != 0:
                self._log_deployment_step(f"API tests failed: {stderr}", "ERROR")
                return False
            
            self._log_deployment_step("API tests passed")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"API tests failed: {e}", "ERROR")
            return False
    
    def _verify_performance(self) -> bool:
        """Verify performance metrics"""
        try:
            # Run performance validation
            returncode, stdout, stderr = self._execute_command(
                "python tests/performance/performance_validation.py --duration 60",
                timeout=120
            )
            
            if returncode != 0:
                self._log_deployment_step(f"Performance verification failed: {stderr}", "ERROR")
                return False
            
            self._log_deployment_step("Performance verification passed")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Performance verification failed: {e}", "ERROR")
            return False
    
    def _rollback_deployment(self) -> bool:
        """Rollback deployment"""
        if not self.config.rollback_on_failure:
            self._log_deployment_step("Rollback disabled, skipping")
            return True
        
        self._log_deployment_step("Rolling back deployment...")
        
        try:
            # Rollback to previous deployment
            returncode, stdout, stderr = self._execute_command(
                f"kubectl rollout undo deployment/pynomaly-prod-app -n {self.config.namespace}"
            )
            
            if returncode != 0:
                self._log_deployment_step(f"Rollback failed: {stderr}", "ERROR")
                return False
            
            # Wait for rollback to complete
            returncode, stdout, stderr = self._execute_command(
                f"kubectl rollout status deployment/pynomaly-prod-app -n {self.config.namespace} --timeout=300s"
            )
            
            if returncode != 0:
                self._log_deployment_step(f"Rollback status check failed: {stderr}", "ERROR")
                return False
            
            self._log_deployment_step("Rollback completed successfully")
            return True
            
        except Exception as e:
            self._log_deployment_step(f"Rollback failed: {e}", "ERROR")
            return False
    
    def _cleanup_resources(self):
        """Cleanup temporary resources"""
        try:
            # Kill port-forward processes
            self._execute_command("pkill -f 'kubectl port-forward'")
            
            # Clean up temporary files
            if self.previous_deployment:
                backup_dir = Path(self.previous_deployment)
                if backup_dir.exists():
                    import shutil
                    shutil.rmtree(backup_dir)
                    
        except Exception as e:
            self._log_deployment_step(f"Cleanup failed: {e}", "WARNING")
    
    def _generate_deployment_report(self, result: DeploymentResult) -> Dict:
        """Generate deployment report"""
        report = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "result": asdict(result),
            "logs": self.deployment_logs,
            "k8s_resources": self._get_k8s_resources_status(),
            "metrics": self._collect_deployment_metrics()
        }
        
        return report
    
    def _get_k8s_resources_status(self) -> Dict:
        """Get Kubernetes resources status"""
        try:
            resources = {}
            
            # Get deployment status
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get deployment pynomaly-prod-app -n {self.config.namespace} -o json"
            )
            if returncode == 0:
                resources["deployment"] = json.loads(stdout)
            
            # Get pod status
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get pods -n {self.config.namespace} -l app=pynomaly -o json"
            )
            if returncode == 0:
                resources["pods"] = json.loads(stdout)
            
            # Get service status
            returncode, stdout, stderr = self._execute_command(
                f"kubectl get service pynomaly-prod-service -n {self.config.namespace} -o json"
            )
            if returncode == 0:
                resources["service"] = json.loads(stdout)
            
            return resources
            
        except Exception as e:
            self._log_deployment_step(f"Failed to get K8s resources status: {e}", "ERROR")
            return {}
    
    def _collect_deployment_metrics(self) -> Dict:
        """Collect deployment metrics"""
        try:
            metrics = {
                "deployment_time": datetime.now().isoformat(),
                "deployment_duration": 0,
                "pod_count": 0,
                "ready_pods": 0,
                "resource_usage": {}
            }
            
            # Get pod metrics
            returncode, stdout, stderr = self._execute_command(
                f"kubectl top pods -n {self.config.namespace} -l app=pynomaly --no-headers"
            )
            if returncode == 0:
                lines = stdout.strip().split('\n')
                metrics["pod_count"] = len([line for line in lines if line])
                
                # Parse resource usage
                total_cpu = 0
                total_memory = 0
                for line in lines:
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            cpu_str = parts[1].replace('m', '')
                            memory_str = parts[2].replace('Mi', '')
                            try:
                                total_cpu += int(cpu_str)
                                total_memory += int(memory_str)
                            except ValueError:
                                pass
                
                metrics["resource_usage"] = {
                    "total_cpu_millicores": total_cpu,
                    "total_memory_mb": total_memory
                }
            
            return metrics
            
        except Exception as e:
            self._log_deployment_step(f"Failed to collect deployment metrics: {e}", "ERROR")
            return {}
    
    def deploy(self) -> DeploymentResult:
        """Execute production deployment"""
        start_time = datetime.now()
        result = DeploymentResult(
            status=DeploymentStatus.PENDING,
            phase=DeploymentPhase.PREPARATION,
            start_time=start_time,
            logs=[]
        )
        
        try:
            self._log_deployment_step(f"Starting production deployment {self.deployment_id}")
            
            # Phase 1: Preparation
            result.phase = DeploymentPhase.PREPARATION
            result.status = DeploymentStatus.IN_PROGRESS
            
            if not self._check_prerequisites():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Prerequisites check failed"
                return result
            
            if not self._backup_current_deployment():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Backup failed"
                return result
            
            # Phase 2: Validation
            result.phase = DeploymentPhase.VALIDATION
            
            if not self._run_pre_deployment_tests():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Pre-deployment tests failed"
                return result
            
            # Phase 3: Deployment
            result.phase = DeploymentPhase.DEPLOYMENT
            
            if not self._deploy_manifests():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Manifest deployment failed"
                return result
            
            if not self._wait_for_deployment():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Deployment timeout or failure"
                return result
            
            # Phase 4: Verification
            result.phase = DeploymentPhase.VERIFICATION
            
            if not self._run_post_deployment_tests():
                result.status = DeploymentStatus.FAILED
                result.error_message = "Post-deployment tests failed"
                return result
            
            # Phase 5: Completion
            result.phase = DeploymentPhase.COMPLETION
            result.status = DeploymentStatus.SUCCESS
            
            self._log_deployment_step("Production deployment completed successfully")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            self._log_deployment_step(f"Deployment failed: {e}", "ERROR")
            
        finally:
            # Handle rollback if needed
            if result.status == DeploymentStatus.FAILED and self.config.rollback_on_failure:
                if self._rollback_deployment():
                    result.status = DeploymentStatus.ROLLED_BACK
                    result.rollback_performed = True
            
            # Cleanup
            self._cleanup_resources()
            
            # Calculate duration
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.logs = self.deployment_logs
            
            # Generate report
            report = self._generate_deployment_report(result)
            self._save_deployment_report(report)
        
        return result
    
    def _save_deployment_report(self, report: Dict):
        """Save deployment report"""
        try:
            reports_dir = Path("./reports/deployment")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"production_deployment_{self.deployment_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self._log_deployment_step(f"Deployment report saved to {report_file}")
            
        except Exception as e:
            self._log_deployment_step(f"Failed to save deployment report: {e}", "ERROR")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pynomaly Production Deployment')
    parser.add_argument('--image-tag', default='latest', help='Docker image tag to deploy')
    parser.add_argument('--replicas', type=int, default=5, help='Number of replicas')
    parser.add_argument('--timeout', type=int, default=1800, help='Deployment timeout in seconds')
    parser.add_argument('--skip-tests', action='store_true', help='Skip pre/post deployment tests')
    parser.add_argument('--no-rollback', action='store_true', help='Disable rollback on failure')
    parser.add_argument('--skip-confirmation', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create deployment configuration
    config = DeploymentConfig(
        image_tag=args.image_tag,
        replicas=args.replicas,
        timeout=args.timeout,
        run_tests=not args.skip_tests,
        rollback_on_failure=not args.no_rollback,
        skip_confirmation=args.skip_confirmation,
        dry_run=args.dry_run
    )
    
    # Confirm deployment
    if not config.skip_confirmation and not config.dry_run:
        print(f"\nProduction Deployment Configuration:")
        print(f"  Image Tag: {config.image_tag}")
        print(f"  Replicas: {config.replicas}")
        print(f"  Timeout: {config.timeout}s")
        print(f"  Run Tests: {config.run_tests}")
        print(f"  Rollback on Failure: {config.rollback_on_failure}")
        
        confirmation = input("\nProceed with production deployment? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Deployment cancelled")
            sys.exit(0)
    
    # Execute deployment
    deployer = ProductionDeployer(config)
    result = deployer.deploy()
    
    # Print results
    print(f"\nDeployment Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Phase: {result.phase.value}")
    print(f"  Duration: {result.duration:.1f}s")
    
    if result.error_message:
        print(f"  Error: {result.error_message}")
    
    if result.rollback_performed:
        print(f"  Rollback: Performed")
    
    # Exit with appropriate code
    if result.status == DeploymentStatus.SUCCESS:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()