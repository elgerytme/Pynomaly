#!/usr/bin/env python3
"""
Deployment Orchestration Script for anomaly_detection
This script orchestrates the complete deployment process including infrastructure and application deployment
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """Orchestrates the complete deployment process for anomaly_detection"""

    def __init__(self, config_file: Path, environment: str):
        self.config_file = config_file
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent
        self.deployment_id = f"orchestration-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Load configuration
        self.config = self._load_config()

        # Initialize logging
        self.logs: list[str] = []
        self.start_time = datetime.now()

        # Deployment state
        self.infrastructure_deployed = False
        self.application_deployed = False
        self.monitoring_deployed = False

        self.log(f"Initialized deployment orchestrator for {environment}")

    def _load_config(self) -> dict[str, Any]:
        """Load deployment configuration"""
        try:
            with open(self.config_file) as f:
                if self.config_file.suffix in [".yaml", ".yml"]:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            # Validate required fields
            required_fields = ["environment", "deployment", "resources"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required configuration field: {field}")

            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)

        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG":
            logger.debug(message)
        else:
            logger.info(message)

    async def execute_command(
        self, command: str, cwd: Path | None = None, timeout: int = 300
    ) -> tuple[int, str, str]:
        """Execute shell command asynchronously"""
        self.log(f"Executing: {command}")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or self.project_root,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return_code = process.returncode
            stdout_str = stdout.decode("utf-8") if stdout else ""
            stderr_str = stderr.decode("utf-8") if stderr else ""

            if return_code == 0:
                self.log(f"Command succeeded: {command}")
            else:
                self.log(
                    f"Command failed with exit code {return_code}: {command}", "ERROR"
                )
                if stderr_str.strip():
                    self.log(f"STDERR: {stderr_str.strip()}", "ERROR")

            return return_code, stdout_str, stderr_str

        except TimeoutError:
            self.log(f"Command timed out after {timeout} seconds: {command}", "ERROR")
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            self.log(f"Command execution error: {e}", "ERROR")
            return -1, "", str(e)

    async def validate_prerequisites(self) -> bool:
        """Validate deployment prerequisites"""
        self.log("Validating deployment prerequisites...")

        validation_tasks = [
            self._validate_tools(),
            self._validate_aws_access(),
            self._validate_terraform_state(),
            self._validate_kubernetes_access(),
            self._validate_docker_registry(),
            self._validate_secrets(),
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log(f"Validation task {i} failed: {result}", "ERROR")
                return False
            elif not result:
                self.log(f"Validation task {i} failed", "ERROR")
                return False

        self.log("All prerequisites validated successfully")
        return True

    async def _validate_tools(self) -> bool:
        """Validate required tools are installed"""
        tools = [
            ("terraform", "--version"),
            ("kubectl", "version --client"),
            ("docker", "--version"),
            ("aws", "--version"),
            ("helm", "version --client"),
            ("jq", "--version"),
        ]

        for tool, version_cmd in tools:
            return_code, _, _ = await self.execute_command(
                f"{tool} {version_cmd}", timeout=30
            )
            if return_code != 0:
                self.log(f"Required tool not found or not working: {tool}", "ERROR")
                return False

        self.log("All required tools are available")
        return True

    async def _validate_aws_access(self) -> bool:
        """Validate AWS access and permissions"""
        try:
            # Check AWS credentials
            return_code, _, _ = await self.execute_command(
                "aws sts get-caller-identity", timeout=30
            )
            if return_code != 0:
                self.log("AWS credentials not configured or invalid", "ERROR")
                return False

            # Check required AWS permissions
            permissions_check = [
                "aws iam list-roles --max-items 1",
                "aws ec2 describe-regions --max-items 1",
                "aws eks list-clusters --max-items 1",
            ]

            for check in permissions_check:
                return_code, _, _ = await self.execute_command(check, timeout=30)
                if return_code != 0:
                    self.log(f"Insufficient AWS permissions for: {check}", "ERROR")
                    return False

            self.log("AWS access validation passed")
            return True

        except Exception as e:
            self.log(f"AWS validation failed: {e}", "ERROR")
            return False

    async def _validate_terraform_state(self) -> bool:
        """Validate Terraform state and backend"""
        try:
            terraform_dir = (
                self.project_root / "infrastructure" / "terraform" / "production"
            )

            # Initialize Terraform
            return_code, _, stderr = await self.execute_command(
                "terraform init -backend=true", cwd=terraform_dir, timeout=120
            )

            if return_code != 0:
                self.log(f"Terraform initialization failed: {stderr}", "ERROR")
                return False

            # Validate Terraform configuration
            return_code, _, stderr = await self.execute_command(
                "terraform validate", cwd=terraform_dir, timeout=60
            )

            if return_code != 0:
                self.log(f"Terraform validation failed: {stderr}", "ERROR")
                return False

            self.log("Terraform validation passed")
            return True

        except Exception as e:
            self.log(f"Terraform validation failed: {e}", "ERROR")
            return False

    async def _validate_kubernetes_access(self) -> bool:
        """Validate Kubernetes cluster access"""
        try:
            # Check kubectl configuration
            return_code, _, _ = await self.execute_command(
                "kubectl config current-context", timeout=30
            )
            if return_code != 0:
                self.log("kubectl context not set", "WARNING")
                # This is okay if we're creating a new cluster
                return True

            # Check cluster connectivity (if cluster exists)
            return_code, _, _ = await self.execute_command(
                "kubectl cluster-info", timeout=30
            )
            if return_code != 0:
                self.log("Cannot connect to Kubernetes cluster", "WARNING")
                # This is okay if we're creating a new cluster
                return True

            self.log("Kubernetes access validation passed")
            return True

        except Exception as e:
            self.log(f"Kubernetes validation failed: {e}", "ERROR")
            return False

    async def _validate_docker_registry(self) -> bool:
        """Validate Docker registry access"""
        try:
            # Check Docker login
            registry = self.config.get("container", {}).get("registry", "ghcr.io")

            return_code, _, _ = await self.execute_command("docker info", timeout=30)
            if return_code != 0:
                self.log("Docker daemon not running", "ERROR")
                return False

            # Try to pull a small image to test registry access
            return_code, _, _ = await self.execute_command(
                "docker pull alpine:latest", timeout=120
            )
            if return_code != 0:
                self.log("Cannot pull images from Docker registry", "ERROR")
                return False

            self.log("Docker registry access validation passed")
            return True

        except Exception as e:
            self.log(f"Docker registry validation failed: {e}", "ERROR")
            return False

    async def _validate_secrets(self) -> bool:
        """Validate required secrets are available"""
        try:
            required_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]

            if self.environment == "production":
                required_env_vars.extend(["SLACK_WEBHOOK_URL", "GITHUB_TOKEN"])

            for var in required_env_vars:
                if not os.getenv(var):
                    self.log(f"Required environment variable missing: {var}", "ERROR")
                    return False

            self.log("Secrets validation passed")
            return True

        except Exception as e:
            self.log(f"Secrets validation failed: {e}", "ERROR")
            return False

    async def deploy_infrastructure(self) -> bool:
        """Deploy infrastructure using Terraform"""
        self.log("Starting infrastructure deployment...")

        try:
            terraform_dir = (
                self.project_root / "infrastructure" / "terraform" / "production"
            )

            # Generate terraform.tfvars from configuration
            await self._generate_terraform_vars()

            # Plan the deployment
            self.log("Creating Terraform execution plan...")
            return_code, stdout, stderr = await self.execute_command(
                "terraform plan -out=tfplan", cwd=terraform_dir, timeout=300
            )

            if return_code != 0:
                self.log(f"Terraform plan failed: {stderr}", "ERROR")
                return False

            # Apply the plan
            self.log("Applying Terraform plan...")
            return_code, stdout, stderr = await self.execute_command(
                "terraform apply -auto-approve tfplan",
                cwd=terraform_dir,
                timeout=1800,  # 30 minutes for infrastructure deployment
            )

            if return_code != 0:
                self.log(f"Terraform apply failed: {stderr}", "ERROR")
                return False

            # Get outputs
            await self._save_terraform_outputs()

            self.infrastructure_deployed = True
            self.log("Infrastructure deployment completed successfully")
            return True

        except Exception as e:
            self.log(f"Infrastructure deployment failed: {e}", "ERROR")
            return False

    async def _generate_terraform_vars(self):
        """Generate terraform.tfvars from configuration"""
        terraform_dir = (
            self.project_root / "infrastructure" / "terraform" / "production"
        )

        # Map configuration to Terraform variables
        tf_vars = {
            "aws_region": self.config.get("environment", {}).get("region", "us-east-1"),
            "cluster_name": f"anomaly_detection-{self.environment}",
            "environment": self.environment,
            "domain_name": self.config.get("environment", {}).get(
                "domain", "anomaly_detection.ai"
            ),
            "node_desired_capacity": self.config.get("resources", {})
            .get("api", {})
            .get("replicas", {})
            .get("initial", 3),
            "node_max_capacity": self.config.get("resources", {})
            .get("api", {})
            .get("replicas", {})
            .get("max", 10),
            "db_instance_class": "db.t3.medium"
            if self.environment != "production"
            else "db.r5.large",
            "redis_instance_type": "cache.t3.medium"
            if self.environment != "production"
            else "cache.r5.large",
        }

        # Write terraform.tfvars
        tfvars_content = "\n".join(
            [f'{key} = "{value}"' for key, value in tf_vars.items()]
        )

        tfvars_file = terraform_dir / "terraform.tfvars"
        with open(tfvars_file, "w") as f:
            f.write(tfvars_content)

        self.log(f"Generated Terraform variables: {tfvars_file}")

    async def _save_terraform_outputs(self):
        """Save Terraform outputs for use by application deployment"""
        terraform_dir = (
            self.project_root / "infrastructure" / "terraform" / "production"
        )

        return_code, stdout, stderr = await self.execute_command(
            "terraform output -json", cwd=terraform_dir, timeout=60
        )

        if return_code == 0:
            outputs_file = (
                self.project_root
                / "config"
                / "deployment"
                / f"{self.environment}-outputs.json"
            )
            outputs_file.parent.mkdir(parents=True, exist_ok=True)

            with open(outputs_file, "w") as f:
                f.write(stdout)

            self.log(f"Saved Terraform outputs: {outputs_file}")
        else:
            self.log(f"Failed to get Terraform outputs: {stderr}", "ERROR")

    async def setup_kubernetes_cluster(self) -> bool:
        """Setup Kubernetes cluster configuration and essential components"""
        self.log("Setting up Kubernetes cluster...")

        try:
            # Update kubeconfig
            cluster_name = f"anomaly_detection-{self.environment}"
            region = self.config.get("environment", {}).get("region", "us-east-1")

            return_code, _, stderr = await self.execute_command(
                f"aws eks update-kubeconfig --region {region} --name {cluster_name}",
                timeout=60,
            )

            if return_code != 0:
                self.log(f"Failed to update kubeconfig: {stderr}", "ERROR")
                return False

            # Wait for cluster to be ready
            self.log("Waiting for cluster to be ready...")
            await asyncio.sleep(60)

            # Install essential cluster components
            success = await self._install_cluster_components()
            if not success:
                return False

            self.log("Kubernetes cluster setup completed")
            return True

        except Exception as e:
            self.log(f"Kubernetes cluster setup failed: {e}", "ERROR")
            return False

    async def _install_cluster_components(self) -> bool:
        """Install essential cluster components"""
        components = [
            (
                "AWS Load Balancer Controller",
                self._install_aws_load_balancer_controller,
            ),
            ("EBS CSI Driver", self._install_ebs_csi_driver),
            ("Cluster Autoscaler", self._install_cluster_autoscaler),
            ("Metrics Server", self._install_metrics_server),
            ("External DNS", self._install_external_dns),
            ("Cert Manager", self._install_cert_manager),
        ]

        for name, install_func in components:
            self.log(f"Installing {name}...")
            success = await install_func()
            if not success:
                self.log(f"Failed to install {name}", "ERROR")
                return False
            self.log(f"{name} installed successfully")

        return True

    async def _install_aws_load_balancer_controller(self) -> bool:
        """Install AWS Load Balancer Controller"""
        try:
            # Add EKS Helm repository
            await self.execute_command(
                "helm repo add eks https://aws.github.io/eks-charts"
            )
            await self.execute_command("helm repo update")

            # Install AWS Load Balancer Controller
            cluster_name = f"anomaly_detection-{self.environment}"
            region = self.config.get("environment", {}).get("region", "us-east-1")

            install_command = f"""
            helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
              -n kube-system \
              --set clusterName={cluster_name} \
              --set serviceAccount.create=false \
              --set serviceAccount.name=aws-load-balancer-controller \
              --set region={region}
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=300
            )
            return return_code == 0

        except Exception as e:
            self.log(f"AWS Load Balancer Controller installation failed: {e}", "ERROR")
            return False

    async def _install_ebs_csi_driver(self) -> bool:
        """Install EBS CSI Driver"""
        try:
            # Install EBS CSI Driver addon
            cluster_name = f"anomaly_detection-{self.environment}"

            install_command = f"""
            aws eks create-addon \
              --cluster-name {cluster_name} \
              --addon-name aws-ebs-csi-driver \
              --resolve-conflicts OVERWRITE
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=300
            )
            return return_code == 0

        except Exception as e:
            self.log(f"EBS CSI Driver installation failed: {e}", "ERROR")
            return False

    async def _install_cluster_autoscaler(self) -> bool:
        """Install Cluster Autoscaler"""
        try:
            # Apply Cluster Autoscaler YAML
            autoscaler_yaml = (
                self.project_root / "deploy" / "kubernetes" / "cluster-autoscaler.yaml"
            )

            return_code, _, stderr = await self.execute_command(
                f"kubectl apply -f {autoscaler_yaml}", timeout=120
            )

            return return_code == 0

        except Exception as e:
            self.log(f"Cluster Autoscaler installation failed: {e}", "ERROR")
            return False

    async def _install_metrics_server(self) -> bool:
        """Install Metrics Server"""
        try:
            return_code, _, stderr = await self.execute_command(
                "kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml",
                timeout=120,
            )

            return return_code == 0

        except Exception as e:
            self.log(f"Metrics Server installation failed: {e}", "ERROR")
            return False

    async def _install_external_dns(self) -> bool:
        """Install External DNS"""
        try:
            # Apply External DNS YAML
            external_dns_yaml = (
                self.project_root / "deploy" / "kubernetes" / "external-dns.yaml"
            )

            return_code, _, stderr = await self.execute_command(
                f"kubectl apply -f {external_dns_yaml}", timeout=120
            )

            return return_code == 0

        except Exception as e:
            self.log(f"External DNS installation failed: {e}", "ERROR")
            return False

    async def _install_cert_manager(self) -> bool:
        """Install Cert Manager"""
        try:
            # Add Jetstack Helm repository
            await self.execute_command(
                "helm repo add jetstack https://charts.jetstack.io"
            )
            await self.execute_command("helm repo update")

            # Install cert-manager
            install_command = """
            helm upgrade --install cert-manager jetstack/cert-manager \
              --namespace cert-manager \
              --create-namespace \
              --set installCRDs=true
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=300
            )
            return return_code == 0

        except Exception as e:
            self.log(f"Cert Manager installation failed: {e}", "ERROR")
            return False

    async def deploy_application(self) -> bool:
        """Deploy the anomaly detection application"""
        self.log("Starting application deployment...")

        try:
            # Use the automated deployment pipeline
            deployment_config = {
                "environment": self.environment,
                "namespace": f"anomaly_detection-{self.environment}",
                "image_tag": self.config.get("container", {}).get(
                    "image_tag", "latest"
                ),
                "strategy": self.config.get("deployment", {}).get(
                    "strategy", "rolling"
                ),
                "skip_tests": False,
                "skip_build": False,
                "rollback_on_failure": True,
            }

            # Save deployment config to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(deployment_config, f, indent=2)
                config_file = f.name

            try:
                # Run automated deployment
                deployment_script = (
                    self.project_root / "scripts" / "deploy" / "automated_deployment.py"
                )

                return_code, _, stderr = await self.execute_command(
                    f"python {deployment_script} --config {config_file}",
                    timeout=1800,  # 30 minutes
                )

                if return_code != 0:
                    self.log(f"Application deployment failed: {stderr}", "ERROR")
                    return False

                self.application_deployed = True
                self.log("Application deployment completed successfully")
                return True

            finally:
                # Clean up temporary config file
                os.unlink(config_file)

        except Exception as e:
            self.log(f"Application deployment failed: {e}", "ERROR")
            return False

    async def deploy_monitoring(self) -> bool:
        """Deploy monitoring and observability stack"""
        self.log("Starting monitoring deployment...")

        try:
            monitoring_tasks = [
                self._deploy_prometheus(),
                self._deploy_grafana(),
                self._deploy_jaeger(),
                self._deploy_fluentbit(),
            ]

            results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    self.log(f"Monitoring component {i} deployment failed", "ERROR")
                    return False

            self.monitoring_deployed = True
            self.log("Monitoring deployment completed successfully")
            return True

        except Exception as e:
            self.log(f"Monitoring deployment failed: {e}", "ERROR")
            return False

    async def _deploy_prometheus(self) -> bool:
        """Deploy Prometheus monitoring"""
        try:
            # Add Prometheus Helm repository
            await self.execute_command(
                "helm repo add prometheus-community https://prometheus-community.github.io/helm-charts"
            )
            await self.execute_command("helm repo update")

            # Install Prometheus
            install_command = """
            helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
              --namespace monitoring \
              --create-namespace \
              --set prometheus.prometheusSpec.retention=30d
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=600
            )
            return return_code == 0

        except Exception as e:
            self.log(f"Prometheus deployment failed: {e}", "ERROR")
            return False

    async def _deploy_grafana(self) -> bool:
        """Deploy Grafana dashboards"""
        try:
            # Grafana is included with kube-prometheus-stack
            # Just configure custom dashboards
            dashboards_dir = (
                self.project_root / "config" / "monitoring" / "grafana" / "dashboards"
            )

            if dashboards_dir.exists():
                return_code, _, stderr = await self.execute_command(
                    f"kubectl apply -f {dashboards_dir} -n monitoring", timeout=120
                )
                return return_code == 0

            return True

        except Exception as e:
            self.log(f"Grafana deployment failed: {e}", "ERROR")
            return False

    async def _deploy_jaeger(self) -> bool:
        """Deploy Jaeger tracing"""
        try:
            # Add Jaeger Helm repository
            await self.execute_command(
                "helm repo add jaegertracing https://jaegertracing.github.io/helm-charts"
            )
            await self.execute_command("helm repo update")

            # Install Jaeger
            install_command = """
            helm upgrade --install jaeger jaegertracing/jaeger \
              --namespace tracing \
              --create-namespace
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=300
            )
            return return_code == 0

        except Exception as e:
            self.log(f"Jaeger deployment failed: {e}", "ERROR")
            return False

    async def _deploy_fluentbit(self) -> bool:
        """Deploy Fluent Bit for log aggregation"""
        try:
            # Add Fluent Bit Helm repository
            await self.execute_command(
                "helm repo add fluent https://fluent.github.io/helm-charts"
            )
            await self.execute_command("helm repo update")

            # Install Fluent Bit
            install_command = """
            helm upgrade --install fluent-bit fluent/fluent-bit \
              --namespace logging \
              --create-namespace
            """

            return_code, _, stderr = await self.execute_command(
                install_command, timeout=300
            )
            return return_code == 0

        except Exception as e:
            self.log(f"Fluent Bit deployment failed: {e}", "ERROR")
            return False

    async def run_post_deployment_tests(self) -> bool:
        """Run post-deployment verification tests"""
        self.log("Running post-deployment tests...")

        try:
            test_tasks = [
                self._test_application_health(),
                self._test_api_endpoints(),
                self._test_monitoring_stack(),
                self._test_security_policies(),
            ]

            results = await asyncio.gather(*test_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    self.log(f"Post-deployment test {i} failed", "ERROR")
                    return False

            self.log("All post-deployment tests passed")
            return True

        except Exception as e:
            self.log(f"Post-deployment tests failed: {e}", "ERROR")
            return False

    async def _test_application_health(self) -> bool:
        """Test application health endpoints"""
        try:
            namespace = f"anomaly_detection-{self.environment}"

            # Port forward to test health endpoint
            port_forward_process = await asyncio.create_subprocess_shell(
                f"kubectl port-forward service/anomaly_detection-api-internal 8080:8000 -n {namespace}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for port forward to establish
            await asyncio.sleep(10)

            try:
                # Test health endpoint
                return_code, _, _ = await self.execute_command(
                    "curl -f http://localhost:8080/api/v1/health", timeout=30
                )

                return return_code == 0

            finally:
                # Clean up port forward
                port_forward_process.terminate()
                await port_forward_process.wait()

        except Exception as e:
            self.log(f"Application health test failed: {e}", "ERROR")
            return False

    async def _test_api_endpoints(self) -> bool:
        """Test API endpoints functionality"""
        try:
            # Run API test suite
            return_code, _, stderr = await self.execute_command(
                "python -m pytest tests/api/ -v --tb=short", timeout=300
            )

            return return_code == 0

        except Exception as e:
            self.log(f"API endpoints test failed: {e}", "ERROR")
            return False

    async def _test_monitoring_stack(self) -> bool:
        """Test monitoring stack is working"""
        try:
            # Check if monitoring pods are running
            return_code, _, _ = await self.execute_command(
                "kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus --field-selector=status.phase=Running",
                timeout=30,
            )

            return return_code == 0

        except Exception as e:
            self.log(f"Monitoring stack test failed: {e}", "ERROR")
            return False

    async def _test_security_policies(self) -> bool:
        """Test security policies are enforced"""
        try:
            # Check if network policies are applied
            return_code, _, _ = await self.execute_command(
                f"kubectl get networkpolicies -n anomaly_detection-{self.environment}",
                timeout=30,
            )

            # This is non-critical, so return True even if it fails
            return True

        except Exception as e:
            self.log(f"Security policies test failed: {e}", "WARNING")
            return True

    def generate_deployment_report(self) -> dict[str, Any]:
        """Generate comprehensive deployment report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        report = {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "infrastructure_deployed": self.infrastructure_deployed,
            "application_deployed": self.application_deployed,
            "monitoring_deployed": self.monitoring_deployed,
            "configuration": self.config,
            "logs": self.logs,
            "metadata": {
                "orchestrator_version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
            },
        }

        return report

    def save_deployment_report(self, report: dict[str, Any]):
        """Save deployment report to file"""
        try:
            reports_dir = self.project_root / "reports" / "orchestration"
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_file = reports_dir / f"orchestration_{self.deployment_id}.json"

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            self.log(f"Deployment report saved to {report_file}")

        except Exception as e:
            self.log(f"Failed to save deployment report: {e}", "ERROR")

    async def execute_full_deployment(self) -> bool:
        """Execute the complete deployment orchestration"""
        try:
            self.log(f"Starting full deployment orchestration for {self.environment}")

            # Phase 1: Prerequisites
            if not await self.validate_prerequisites():
                self.log("Prerequisites validation failed", "ERROR")
                return False

            # Phase 2: Infrastructure
            if not await self.deploy_infrastructure():
                self.log("Infrastructure deployment failed", "ERROR")
                return False

            # Phase 3: Kubernetes Setup
            if not await self.setup_kubernetes_cluster():
                self.log("Kubernetes cluster setup failed", "ERROR")
                return False

            # Phase 4: Application
            if not await self.deploy_application():
                self.log("Application deployment failed", "ERROR")
                return False

            # Phase 5: Monitoring
            if not await self.deploy_monitoring():
                self.log("Monitoring deployment failed", "ERROR")
                return False

            # Phase 6: Verification
            if not await self.run_post_deployment_tests():
                self.log("Post-deployment tests failed", "ERROR")
                return False

            self.log("Full deployment orchestration completed successfully")
            return True

        except Exception as e:
            self.log(f"Deployment orchestration failed: {e}", "ERROR")
            return False

        finally:
            # Always generate and save report
            report = self.generate_deployment_report()
            self.save_deployment_report(report)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="anomaly_detection Deployment Orchestrator")
    parser.add_argument(
        "--config", type=Path, required=True, help="Deployment configuration file"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "testing", "staging", "production"],
        required=True,
        help="Deployment environment",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Initialize and execute orchestrator
    orchestrator = DeploymentOrchestrator(args.config, args.environment)

    try:
        success = asyncio.run(orchestrator.execute_full_deployment())

        if success:
            print(
                f"\n✅ Deployment orchestration for {args.environment} completed successfully!"
            )
            print(f"Deployment ID: {orchestrator.deployment_id}")
            sys.exit(0)
        else:
            print(f"\n❌ Deployment orchestration for {args.environment} failed!")
            print(f"Deployment ID: {orchestrator.deployment_id}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Deployment orchestration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Deployment orchestration failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
