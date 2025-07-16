"""Comprehensive Docker pipeline tests.

This module contains comprehensive tests for Docker containerization,
image building, multi-stage builds, security scanning, and container
orchestration.
"""

import time
import uuid
from typing import Any

import pytest


class TestDockerPipeline:
    """Test Docker pipeline functionality."""

    @pytest.fixture
    def mock_docker_environment(self):
        """Create mock Docker environment."""

        class MockDockerEnvironment:
            def __init__(self):
                self.images = {}
                self.containers = {}
                self.build_logs = []
                self.registry_images = {}
                self.docker_config = {
                    "base_images": {
                        "python": "python:3.11-slim",
                        "alpine": "python:3.11-alpine",
                        "ubuntu": "ubuntu:22.04",
                    },
                    "build_stages": ["dependencies", "build", "test", "production"],
                    "security_scanners": ["trivy", "grype", "snyk"],
                }
                self.build_contexts = {}

            def create_dockerfile(self, dockerfile_type: str = "production") -> str:
                """Create Dockerfile content for different purposes."""
                if dockerfile_type == "production":
                    return self._create_production_dockerfile()
                elif dockerfile_type == "development":
                    return self._create_development_dockerfile()
                elif dockerfile_type == "testing":
                    return self._create_testing_dockerfile()
                elif dockerfile_type == "multi-stage":
                    return self._create_multistage_dockerfile()
                else:
                    raise ValueError(f"Unknown dockerfile type: {dockerfile_type}")

            def _create_production_dockerfile(self) -> str:
                """Create production Dockerfile."""
                return """
# Production Dockerfile for Pynomaly
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 pynomaly && \\
    useradd --uid 1000 --gid pynomaly --shell /bin/bash --create-home pynomaly

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        gcc \\
        g++ \\
        libc6-dev \\
        libffi-dev \\
        libssl-dev && \\
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Install the application
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER pynomaly

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import monorepo; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "monorepo", "--help"]
"""

            def _create_development_dockerfile(self) -> str:
                """Create development Dockerfile."""
                return """
# Development Dockerfile for Pynomaly
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including dev tools
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        gcc \\
        g++ \\
        git \\
        vim \\
        curl \\
        wget \\
        build-essential \\
        libffi-dev \\
        libssl-dev && \\
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install development Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install the application in development mode
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Default command for development
CMD ["bash"]
"""

            def _create_testing_dockerfile(self) -> str:
                """Create testing Dockerfile."""
                return """
# Testing Dockerfile for Pynomaly
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        gcc \\
        g++ \\
        git \\
        curl \\
        libffi-dev \\
        libssl-dev && \\
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy source code and tests
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .
COPY README.md .

# Install the application
RUN pip install --no-cache-dir -e ".[test]"

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "--cov=pynomaly", "--cov-report=xml"]
"""

            def _create_multistage_dockerfile(self) -> str:
                """Create multi-stage Dockerfile."""
                return """
# Multi-stage Dockerfile for Pynomaly

# Stage 1: Dependencies
FROM python:3.11-slim as dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Build
FROM dependencies as build
COPY src/ src/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --user -e .

# Stage 3: Test
FROM build as test
COPY tests/ tests/
COPY requirements-test.txt .
RUN pip install --no-cache-dir --user -r requirements-test.txt
RUN python -m pytest tests/ --cov=pynomaly

# Stage 4: Production
FROM python:3.11-slim as production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd --gid 1000 pynomaly && \\
    useradd --uid 1000 --gid pynomaly --shell /bin/bash --create-home pynomaly

# Copy installed packages from build stage
COPY --from=build /root/.local /home/pynomaly/.local

# Set PATH
ENV PATH=/home/pynomaly/.local/bin:$PATH

# Switch to non-root user
USER pynomaly
WORKDIR /home/pynomaly

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import monorepo; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "monorepo", "--help"]
"""

            def build_image(
                self,
                dockerfile_content: str,
                image_name: str,
                build_args: dict[str, str] = None,
                target_stage: str = None,
            ) -> dict[str, Any]:
                """Build Docker image."""
                image_id = str(uuid.uuid4())
                build_args = build_args or {}

                # Simulate build process
                build_result = {
                    "image_id": image_id,
                    "image_name": image_name,
                    "build_args": build_args,
                    "target_stage": target_stage,
                    "build_success": True,
                    "build_time": 45.0,  # Mock build time
                    "image_size": 256 * 1024 * 1024,  # 256MB mock size
                    "layers": [],
                    "build_logs": [],
                    "warnings": [],
                    "errors": [],
                }

                # Simulate dockerfile parsing and layer creation
                dockerfile_lines = dockerfile_content.strip().split("\n")
                layer_count = 0

                for line in dockerfile_lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if any(
                            line.startswith(cmd)
                            for cmd in ["FROM", "RUN", "COPY", "ADD"]
                        ):
                            layer_id = f"sha256:{str(uuid.uuid4()).replace('-', '')}"
                            layer_size = 10 * 1024 * 1024  # 10MB mock layer size

                            build_result["layers"].append(
                                {
                                    "layer_id": layer_id,
                                    "instruction": (
                                        line[:50] + "..." if len(line) > 50 else line
                                    ),
                                    "size": layer_size,
                                }
                            )
                            layer_count += 1

                # Simulate potential build issues
                if (
                    "apt-get update" in dockerfile_content
                    and "--no-install-recommends" not in dockerfile_content
                ):
                    build_result["warnings"].append(
                        "Consider using --no-install-recommends to reduce image size"
                    )

                if (
                    "pip install" in dockerfile_content
                    and "--no-cache-dir" not in dockerfile_content
                ):
                    build_result["warnings"].append(
                        "Consider using --no-cache-dir to reduce image size"
                    )

                if (
                    "USER" not in dockerfile_content
                    and dockerfile_content.count("FROM") == 1
                ):
                    build_result["warnings"].append(
                        "Running as root user may pose security risks"
                    )

                # Calculate total image size
                build_result["image_size"] = sum(
                    layer["size"] for layer in build_result["layers"]
                )

                # Store image
                self.images[image_id] = build_result

                return build_result

            def scan_image_security(
                self, image_id: str, scanner: str = "trivy"
            ) -> dict[str, Any]:
                """Scan image for security vulnerabilities."""
                if image_id not in self.images:
                    return {"error": "Image not found"}

                # Mock security scan results
                vulnerabilities = self._generate_mock_vulnerabilities(scanner)

                scan_result = {
                    "image_id": image_id,
                    "scanner": scanner,
                    "scan_time": time.time(),
                    "vulnerabilities": vulnerabilities,
                    "summary": {
                        "total": len(vulnerabilities),
                        "critical": len(
                            [v for v in vulnerabilities if v["severity"] == "CRITICAL"]
                        ),
                        "high": len(
                            [v for v in vulnerabilities if v["severity"] == "HIGH"]
                        ),
                        "medium": len(
                            [v for v in vulnerabilities if v["severity"] == "MEDIUM"]
                        ),
                        "low": len(
                            [v for v in vulnerabilities if v["severity"] == "LOW"]
                        ),
                    },
                    "passed": len(
                        [
                            v
                            for v in vulnerabilities
                            if v["severity"] in ["CRITICAL", "HIGH"]
                        ]
                    )
                    == 0,
                }

                return scan_result

            def _generate_mock_vulnerabilities(
                self, scanner: str
            ) -> list[dict[str, Any]]:
                """Generate mock vulnerability data."""
                base_vulnerabilities = [
                    {
                        "id": "CVE-2023-1001",
                        "severity": "MEDIUM",
                        "package": "openssl",
                        "version": "1.1.1",
                        "fixed_version": "1.1.1k",
                        "description": "OpenSSL vulnerability in certificate validation",
                    },
                    {
                        "id": "CVE-2023-1002",
                        "severity": "LOW",
                        "package": "curl",
                        "version": "7.68.0",
                        "fixed_version": "7.74.0",
                        "description": "curl vulnerability in URL parsing",
                    },
                ]

                if scanner == "trivy":
                    return base_vulnerabilities
                elif scanner == "grype":
                    # Grype might find different vulnerabilities
                    return base_vulnerabilities + [
                        {
                            "id": "GHSA-xxxx-yyyy-zzzz",
                            "severity": "HIGH",
                            "package": "setuptools",
                            "version": "45.0.0",
                            "fixed_version": "65.5.1",
                            "description": "setuptools vulnerability in package processing",
                        }
                    ]
                elif scanner == "snyk":
                    # Snyk might have different format
                    return [
                        {
                            "id": "SNYK-PYTHON-SETUPTOOLS-123456",
                            "severity": "MEDIUM",
                            "package": "setuptools",
                            "version": "45.0.0",
                            "fixed_version": "65.5.1",
                            "description": "setuptools Remote Code Execution",
                        }
                    ]
                else:
                    return []

            def run_container(
                self,
                image_id: str,
                command: list[str] = None,
                environment: dict[str, str] = None,
                ports: dict[str, str] = None,
            ) -> dict[str, Any]:
                """Run container from image."""
                if image_id not in self.images:
                    return {"error": "Image not found"}

                container_id = str(uuid.uuid4())
                command = command or ["python", "-m", "monorepo", "--help"]
                environment = environment or {}
                ports = ports or {}

                # Simulate container execution
                container_result = {
                    "container_id": container_id,
                    "image_id": image_id,
                    "command": command,
                    "environment": environment,
                    "ports": ports,
                    "status": "running",
                    "start_time": time.time(),
                    "exit_code": None,
                    "logs": [],
                    "health_status": "starting",
                }

                # Simulate command execution
                if command[0] == "python":
                    if "--help" in command:
                        container_result["logs"].append(
                            "Pynomaly - Anomaly Detection Library"
                        )
                        container_result["logs"].append(
                            "Usage: python -m pynomaly [OPTIONS]"
                        )
                        container_result["exit_code"] = 0
                        container_result["status"] = "exited"
                    elif "-m pytest" in " ".join(command):
                        container_result["logs"].append("collecting tests...")
                        container_result["logs"].append("test session starts")
                        container_result["logs"].append("collected 250 items")
                        container_result["logs"].append(
                            "245 passed, 3 failed, 2 skipped"
                        )
                        container_result["exit_code"] = (
                            1 if "failed" in " ".join(command) else 0
                        )
                        container_result["status"] = "exited"
                    else:
                        container_result["logs"].append("Python application started")
                        container_result["status"] = "running"

                # Simulate health check
                if container_result["status"] == "running":
                    container_result["health_status"] = "healthy"

                # Store container
                self.containers[container_id] = container_result

                return container_result

            def push_image_to_registry(
                self, image_id: str, registry_url: str, image_tag: str
            ) -> dict[str, Any]:
                """Push image to container registry."""
                if image_id not in self.images:
                    return {"error": "Image not found", "success": False}

                image = self.images[image_id]

                # Simulate push process
                push_result = {
                    "image_id": image_id,
                    "registry_url": registry_url,
                    "image_tag": image_tag,
                    "full_image_name": f"{registry_url}/{image_tag}",
                    "push_success": True,
                    "push_time": 30.0,  # Mock push time
                    "compressed_size": image["image_size"]
                    * 0.7,  # Assume 30% compression
                    "layers_pushed": len(image["layers"]),
                    "manifest_digest": f"sha256:{str(uuid.uuid4()).replace('-', '')}",
                }

                # Store in registry
                self.registry_images[push_result["full_image_name"]] = {
                    "image_id": image_id,
                    "pushed_at": time.time(),
                    "size": push_result["compressed_size"],
                    "manifest_digest": push_result["manifest_digest"],
                }

                return push_result

            def create_docker_compose(self, services: dict[str, Any]) -> str:
                """Create docker-compose.yml content."""
                compose_data = {"version": "3.8", "services": services}

                # Convert to YAML-like string (simplified)
                compose_content = f"version: '{compose_data['version']}'\n\nservices:\n"

                for service_name, service_config in services.items():
                    compose_content += f"  {service_name}:\n"
                    for key, value in service_config.items():
                        if isinstance(value, dict):
                            compose_content += f"    {key}:\n"
                            for sub_key, sub_value in value.items():
                                compose_content += f"      {sub_key}: {sub_value}\n"
                        elif isinstance(value, list):
                            compose_content += f"    {key}:\n"
                            for item in value:
                                compose_content += f"      - {item}\n"
                        else:
                            compose_content += f"    {key}: {value}\n"
                    compose_content += "\n"

                return compose_content

            def validate_docker_best_practices(
                self, dockerfile_content: str
            ) -> dict[str, Any]:
                """Validate Dockerfile against best practices."""
                violations = []
                recommendations = []

                lines = dockerfile_content.strip().split("\n")

                # Check for best practices
                has_user_instruction = any("USER" in line for line in lines)
                has_healthcheck = any("HEALTHCHECK" in line for line in lines)
                has_no_cache_dir = any("--no-cache-dir" in line for line in lines)
                has_no_install_recommends = any(
                    "--no-install-recommends" in line for line in lines
                )

                # Security checks
                if not has_user_instruction:
                    violations.append(
                        {
                            "rule": "USER_REQUIRED",
                            "severity": "HIGH",
                            "message": "Dockerfile should specify a non-root user",
                            "line": None,
                        }
                    )

                # Performance checks
                if not has_no_cache_dir:
                    recommendations.append(
                        {
                            "rule": "PIP_NO_CACHE",
                            "severity": "MEDIUM",
                            "message": "Use --no-cache-dir with pip to reduce image size",
                            "line": None,
                        }
                    )

                if not has_no_install_recommends:
                    recommendations.append(
                        {
                            "rule": "APT_NO_RECOMMENDS",
                            "severity": "MEDIUM",
                            "message": "Use --no-install-recommends with apt-get to reduce image size",
                            "line": None,
                        }
                    )

                # Health check
                if not has_healthcheck:
                    recommendations.append(
                        {
                            "rule": "HEALTHCHECK_MISSING",
                            "severity": "LOW",
                            "message": "Consider adding HEALTHCHECK instruction",
                            "line": None,
                        }
                    )

                # Check for common anti-patterns
                for i, line in enumerate(lines):
                    line = line.strip()

                    if line.startswith("RUN apt-get update") and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if not next_line.startswith("RUN apt-get install"):
                            violations.append(
                                {
                                    "rule": "APT_UPDATE_INSTALL_SEPARATE",
                                    "severity": "MEDIUM",
                                    "message": "apt-get update and install should be in the same RUN instruction",
                                    "line": i + 1,
                                }
                            )

                    if "ADD" in line and not line.startswith("#"):
                        recommendations.append(
                            {
                                "rule": "PREFER_COPY",
                                "severity": "LOW",
                                "message": "Prefer COPY over ADD unless you need ADD's features",
                                "line": i + 1,
                            }
                        )

                return {
                    "violations": violations,
                    "recommendations": recommendations,
                    "score": max(
                        0, 100 - len(violations) * 10 - len(recommendations) * 5
                    ),
                    "passed": len([v for v in violations if v["severity"] == "HIGH"])
                    == 0,
                }

            def optimize_image_size(self, dockerfile_content: str) -> dict[str, Any]:
                """Suggest optimizations for image size."""
                optimizations = []

                if "apt-get update" in dockerfile_content:
                    if "rm -rf /var/lib/apt/lists/*" not in dockerfile_content:
                        optimizations.append(
                            {
                                "type": "cleanup",
                                "description": "Add 'rm -rf /var/lib/apt/lists/*' after apt-get commands",
                                "potential_savings": "50-100 MB",
                            }
                        )

                if "pip install" in dockerfile_content:
                    if "--no-cache-dir" not in dockerfile_content:
                        optimizations.append(
                            {
                                "type": "pip_cache",
                                "description": "Use --no-cache-dir with pip install",
                                "potential_savings": "10-50 MB",
                            }
                        )

                # Multi-stage build suggestion
                if (
                    dockerfile_content.count("FROM") == 1
                    and "gcc" in dockerfile_content
                ):
                    optimizations.append(
                        {
                            "type": "multi_stage",
                            "description": "Consider using multi-stage build to exclude build dependencies from final image",
                            "potential_savings": "100-500 MB",
                        }
                    )

                # Base image suggestions
                if (
                    "python:3.11" in dockerfile_content
                    and "slim" not in dockerfile_content
                ):
                    optimizations.append(
                        {
                            "type": "base_image",
                            "description": "Consider using python:3.11-slim instead of python:3.11",
                            "potential_savings": "600-800 MB",
                        }
                    )

                return {
                    "optimizations": optimizations,
                    "estimated_total_savings": sum(
                        int(opt["potential_savings"].split("-")[1].split()[0])
                        for opt in optimizations
                        if "MB" in opt["potential_savings"]
                    ),
                }

            def get_image_info(self, image_id: str) -> dict[str, Any]:
                """Get detailed image information."""
                if image_id not in self.images:
                    return {"error": "Image not found"}

                image = self.images[image_id]

                return {
                    "image_id": image_id,
                    "image_name": image["image_name"],
                    "size": image["image_size"],
                    "layers": len(image["layers"]),
                    "created": image.get("build_time", time.time()),
                    "build_args": image.get("build_args", {}),
                    "warnings": image.get("warnings", []),
                    "layer_details": image["layers"],
                }

        return MockDockerEnvironment()

    def test_dockerfile_creation(self, mock_docker_environment):
        """Test Dockerfile creation for different purposes."""
        docker_env = mock_docker_environment

        dockerfile_types = ["production", "development", "testing", "multi-stage"]

        for dockerfile_type in dockerfile_types:
            dockerfile_content = docker_env.create_dockerfile(dockerfile_type)

            assert isinstance(dockerfile_content, str)
            assert len(dockerfile_content) > 0
            assert "FROM" in dockerfile_content

            # Verify type-specific content
            if dockerfile_type == "production":
                assert "python:3.11-slim" in dockerfile_content
                assert "USER" in dockerfile_content
                assert "HEALTHCHECK" in dockerfile_content
            elif dockerfile_type == "development":
                assert "git" in dockerfile_content
                assert "vim" in dockerfile_content or "curl" in dockerfile_content
            elif dockerfile_type == "testing":
                assert "pytest" in dockerfile_content
                assert "cov" in dockerfile_content
            elif dockerfile_type == "multi-stage":
                assert dockerfile_content.count("FROM") > 1
                assert "as dependencies" in dockerfile_content
                assert "as production" in dockerfile_content

    def test_docker_image_building(self, mock_docker_environment):
        """Test Docker image building process."""
        docker_env = mock_docker_environment

        # Test production image build
        dockerfile_content = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(
            dockerfile_content=dockerfile_content, image_name="pynomaly:latest"
        )

        assert build_result["build_success"]
        assert build_result["image_name"] == "pynomaly:latest"
        assert "image_id" in build_result
        assert build_result["image_size"] > 0
        assert len(build_result["layers"]) > 0

        # Verify build warnings
        assert isinstance(build_result["warnings"], list)
        assert isinstance(build_result["errors"], list)

        # Test multi-stage build
        multistage_dockerfile = docker_env.create_dockerfile("multi-stage")
        multistage_build = docker_env.build_image(
            dockerfile_content=multistage_dockerfile,
            image_name="pynomaly:multi-stage",
            target_stage="production",
        )

        assert multistage_build["build_success"]
        assert multistage_build["target_stage"] == "production"

        # Test build with build args
        build_with_args = docker_env.build_image(
            dockerfile_content=dockerfile_content,
            image_name="pynomaly:with-args",
            build_args={"PYTHON_VERSION": "3.11", "BUILD_ENV": "production"},
        )

        assert build_with_args["build_success"]
        assert build_with_args["build_args"]["PYTHON_VERSION"] == "3.11"
        assert build_with_args["build_args"]["BUILD_ENV"] == "production"

    def test_container_execution(self, mock_docker_environment):
        """Test container execution."""
        docker_env = mock_docker_environment

        # Build image first
        dockerfile_content = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(dockerfile_content, "pynomaly:test")
        image_id = build_result["image_id"]

        # Test default command execution
        run_result = docker_env.run_container(image_id)

        assert "container_id" in run_result
        assert run_result["image_id"] == image_id
        assert run_result["status"] in ["running", "exited"]
        assert isinstance(run_result["logs"], list)

        # Test custom command execution
        test_run_result = docker_env.run_container(
            image_id=image_id,
            command=["python", "-m", "pytest", "tests/"],
            environment={"PYTHONPATH": "/app/src"},
        )

        assert test_run_result["command"] == ["python", "-m", "pytest", "tests/"]
        assert test_run_result["environment"]["PYTHONPATH"] == "/app/src"

        # Test container with port mapping
        api_run_result = docker_env.run_container(
            image_id=image_id,
            command=[
                "python",
                "-m",
                "uvicorn",
                "monorepo.api:app",
                "--host",
                "0.0.0.0",
            ],
            ports={"8000": "8000"},
        )

        assert api_run_result["ports"]["8000"] == "8000"

    def test_security_scanning(self, mock_docker_environment):
        """Test container security scanning."""
        docker_env = mock_docker_environment

        # Build image for scanning
        dockerfile_content = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(
            dockerfile_content, "pynomaly:security-test"
        )
        image_id = build_result["image_id"]

        # Test different security scanners
        scanners = ["trivy", "grype", "snyk"]

        for scanner in scanners:
            scan_result = docker_env.scan_image_security(image_id, scanner)

            assert scan_result["scanner"] == scanner
            assert scan_result["image_id"] == image_id
            assert "vulnerabilities" in scan_result
            assert "summary" in scan_result
            assert "passed" in scan_result

            # Verify summary structure
            summary = scan_result["summary"]
            assert "total" in summary
            assert "critical" in summary
            assert "high" in summary
            assert "medium" in summary
            assert "low" in summary

            # Total should match individual counts
            individual_total = (
                summary["critical"]
                + summary["high"]
                + summary["medium"]
                + summary["low"]
            )
            assert summary["total"] == individual_total

            # Verify vulnerability structure
            for vuln in scan_result["vulnerabilities"]:
                assert "id" in vuln
                assert "severity" in vuln
                assert "package" in vuln
                assert vuln["severity"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        # Test scanning non-existent image
        scan_error = docker_env.scan_image_security("non-existent-image")
        assert "error" in scan_error

    def test_image_registry_operations(self, mock_docker_environment):
        """Test image registry push/pull operations."""
        docker_env = mock_docker_environment

        # Build image for registry operations
        dockerfile_content = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(
            dockerfile_content, "pynomaly:registry-test"
        )
        image_id = build_result["image_id"]

        # Test push to registry
        push_result = docker_env.push_image_to_registry(
            image_id=image_id,
            registry_url="registry.example.com",
            image_tag="pynomaly:latest",
        )

        assert push_result["push_success"]
        assert push_result["registry_url"] == "registry.example.com"
        assert push_result["image_tag"] == "pynomaly:latest"
        assert push_result["full_image_name"] == "registry.example.com/pynomaly:latest"
        assert "manifest_digest" in push_result
        assert (
            push_result["compressed_size"] < build_result["image_size"]
        )  # Should be compressed

        # Verify image is in registry
        assert push_result["full_image_name"] in docker_env.registry_images

        # Test push with different tags
        push_dev_result = docker_env.push_image_to_registry(
            image_id=image_id,
            registry_url="registry.example.com",
            image_tag="pynomaly:dev",
        )

        assert push_dev_result["push_success"]
        assert push_dev_result["image_tag"] == "pynomaly:dev"

        # Test push of non-existent image
        push_error = docker_env.push_image_to_registry(
            image_id="non-existent",
            registry_url="registry.example.com",
            image_tag="invalid:tag",
        )

        assert not push_error["success"]
        assert "error" in push_error

    def test_docker_compose_generation(self, mock_docker_environment):
        """Test Docker Compose file generation."""
        docker_env = mock_docker_environment

        # Define services for docker-compose
        services = {
            "pynomaly-api": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": {"PYTHONPATH": "/app/src", "LOG_LEVEL": "INFO"},
                "volumes": ["./data:/app/data"],
                "depends_on": ["redis", "postgres"],
            },
            "redis": {"image": "redis:alpine", "ports": ["6379:6379"]},
            "postgres": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": "monorepo",
                    "POSTGRES_USER": "monorepo",
                    "POSTGRES_PASSWORD": "password",
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"],
            },
        }

        compose_content = docker_env.create_docker_compose(services)

        assert isinstance(compose_content, str)
        assert "version:" in compose_content
        assert "services:" in compose_content

        # Verify all services are included
        for service_name in services.keys():
            assert service_name in compose_content

        # Verify service configurations
        assert "build: ." in compose_content
        assert "8000:8000" in compose_content
        assert "redis:alpine" in compose_content
        assert "postgres:13" in compose_content
        assert "POSTGRES_DB" in compose_content

    def test_dockerfile_best_practices_validation(self, mock_docker_environment):
        """Test Dockerfile best practices validation."""
        docker_env = mock_docker_environment

        # Test production Dockerfile (should follow best practices)
        production_dockerfile = docker_env.create_dockerfile("production")
        production_validation = docker_env.validate_docker_best_practices(
            production_dockerfile
        )

        assert "violations" in production_validation
        assert "recommendations" in production_validation
        assert "score" in production_validation
        assert "passed" in production_validation

        # Production Dockerfile should have minimal violations
        high_severity_violations = [
            v for v in production_validation["violations"] if v["severity"] == "HIGH"
        ]
        assert len(high_severity_violations) == 0  # Should pass security checks

        # Test a poorly written Dockerfile
        bad_dockerfile = """
FROM python:3.11
RUN apt-get update
RUN apt-get install -y gcc
RUN pip install numpy
ADD . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        bad_validation = docker_env.validate_docker_best_practices(bad_dockerfile)

        # Should have more violations
        assert len(bad_validation["violations"]) > len(
            production_validation["violations"]
        )
        assert len(bad_validation["recommendations"]) > 0
        assert bad_validation["score"] < production_validation["score"]

        # Should fail due to missing USER instruction
        user_violations = [
            v for v in bad_validation["violations"] if v["rule"] == "USER_REQUIRED"
        ]
        assert len(user_violations) > 0

    def test_image_size_optimization(self, mock_docker_environment):
        """Test image size optimization suggestions."""
        docker_env = mock_docker_environment

        # Test optimization suggestions for different Dockerfiles
        dockerfiles = {
            "unoptimized": """
FROM python:3.11
RUN apt-get update
RUN apt-get install -y gcc g++ git
RUN pip install numpy pandas scikit-learn
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
""",
            "partially_optimized": """
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas scikit-learn
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
""",
            "well_optimized": docker_env.create_dockerfile("multi-stage"),
        }

        optimization_results = {}

        for name, dockerfile in dockerfiles.items():
            optimization_result = docker_env.optimize_image_size(dockerfile)
            optimization_results[name] = optimization_result

            assert "optimizations" in optimization_result
            assert "estimated_total_savings" in optimization_result
            assert isinstance(optimization_result["optimizations"], list)

        # Unoptimized should have the most optimization opportunities
        unoptimized_count = len(optimization_results["unoptimized"]["optimizations"])
        partially_optimized_count = len(
            optimization_results["partially_optimized"]["optimizations"]
        )
        well_optimized_count = len(
            optimization_results["well_optimized"]["optimizations"]
        )

        assert unoptimized_count >= partially_optimized_count
        assert partially_optimized_count >= well_optimized_count

        # Check specific optimizations
        unoptimized_types = [
            opt["type"] for opt in optimization_results["unoptimized"]["optimizations"]
        ]
        assert "cleanup" in unoptimized_types
        assert "pip_cache" in unoptimized_types
        assert "base_image" in unoptimized_types

    def test_image_information_retrieval(self, mock_docker_environment):
        """Test image information retrieval."""
        docker_env = mock_docker_environment

        # Build test image
        dockerfile_content = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(
            dockerfile_content=dockerfile_content,
            image_name="pynomaly:info-test",
            build_args={"ENV": "test"},
        )
        image_id = build_result["image_id"]

        # Get image information
        image_info = docker_env.get_image_info(image_id)

        assert image_info["image_id"] == image_id
        assert image_info["image_name"] == "pynomaly:info-test"
        assert "size" in image_info
        assert "layers" in image_info
        assert "created" in image_info
        assert "build_args" in image_info
        assert "layer_details" in image_info

        # Verify build args are preserved
        assert image_info["build_args"]["ENV"] == "test"

        # Verify layer details
        layer_details = image_info["layer_details"]
        assert len(layer_details) > 0

        for layer in layer_details:
            assert "layer_id" in layer
            assert "instruction" in layer
            assert "size" in layer
            assert layer["size"] > 0

        # Test non-existent image
        non_existent_info = docker_env.get_image_info("non-existent-image")
        assert "error" in non_existent_info

    def test_container_health_checks(self, mock_docker_environment):
        """Test container health check functionality."""
        docker_env = mock_docker_environment

        # Build image with health check
        dockerfile_with_healthcheck = docker_env.create_dockerfile("production")
        build_result = docker_env.build_image(
            dockerfile_with_healthcheck, "pynomaly:health-test"
        )
        image_id = build_result["image_id"]

        # Run container
        container_result = docker_env.run_container(image_id)

        # Verify health check status
        assert "health_status" in container_result

        if container_result["status"] == "running":
            assert container_result["health_status"] in [
                "starting",
                "healthy",
                "unhealthy",
            ]

        # Test container with custom health check command
        health_container = docker_env.run_container(
            image_id=image_id,
            command=["python", "-c", "import monorepo; print('healthy')"],
        )

        # Should have health status
        assert "health_status" in health_container
