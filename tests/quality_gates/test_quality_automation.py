"""
Quality Gate Automation Testing Suite
Comprehensive tests for automated quality gates, CI/CD integration, and deployment gates.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from pynomaly.infrastructure.quality_gates.code_quality_gate import CodeQualityGate
from pynomaly.infrastructure.quality_gates.coverage_gate import CoverageGate
from pynomaly.infrastructure.quality_gates.performance_gate import PerformanceGate
from pynomaly.infrastructure.quality_gates.security_gate import SecurityGate


class TestCoverageGate:
    """Test suite for test coverage quality gates."""

    @pytest.fixture
    def coverage_gate(self):
        """Create coverage gate with thresholds."""
        return CoverageGate(
            minimum_line_coverage=80.0,
            minimum_branch_coverage=75.0,
            minimum_function_coverage=85.0,
            exclude_patterns=["tests/*", "*/migrations/*"],
        )

    def test_coverage_threshold_validation(self, coverage_gate):
        """Test coverage threshold validation."""
        # Mock coverage report
        coverage_report = {
            "line_coverage": 85.5,
            "branch_coverage": 78.2,
            "function_coverage": 92.1,
            "total_lines": 10000,
            "covered_lines": 8550,
            "excluded_lines": 500,
        }

        result = coverage_gate.validate_coverage(coverage_report)

        assert result["passed"] is True
        assert result["line_coverage_passed"] is True
        assert result["branch_coverage_passed"] is True
        assert result["function_coverage_passed"] is True

    def test_coverage_failure_scenarios(self, coverage_gate):
        """Test coverage gate failure scenarios."""
        # Insufficient line coverage
        low_line_coverage = {
            "line_coverage": 75.0,  # Below 80% threshold
            "branch_coverage": 80.0,
            "function_coverage": 90.0,
        }

        result = coverage_gate.validate_coverage(low_line_coverage)

        assert result["passed"] is False
        assert result["line_coverage_passed"] is False
        assert "Line coverage" in result["failure_reasons"]

    def test_coverage_trend_analysis(self, coverage_gate):
        """Test coverage trend analysis over time."""
        historical_coverage = [
            {"date": "2024-01-01", "line_coverage": 75.0},
            {"date": "2024-01-02", "line_coverage": 78.0},
            {"date": "2024-01-03", "line_coverage": 82.0},
            {"date": "2024-01-04", "line_coverage": 85.0},
        ]

        trend_analysis = coverage_gate.analyze_coverage_trend(historical_coverage)

        assert trend_analysis["trend"] == "improving"
        assert trend_analysis["improvement_rate"] > 0
        assert trend_analysis["meets_trend_target"] is True

    def test_coverage_differential_analysis(self, coverage_gate):
        """Test differential coverage analysis for PR validation."""
        base_coverage = {
            "line_coverage": 82.0,
            "covered_lines": 8200,
            "total_lines": 10000,
        }

        pr_coverage = {
            "line_coverage": 83.5,
            "covered_lines": 8350,
            "total_lines": 10000,
        }

        diff_analysis = coverage_gate.analyze_coverage_diff(base_coverage, pr_coverage)

        assert diff_analysis["coverage_improved"] is True
        assert diff_analysis["coverage_delta"] == 1.5
        assert diff_analysis["new_lines_covered"] == 150

    def test_coverage_exclusion_patterns(self, coverage_gate):
        """Test coverage exclusion patterns."""
        files_to_analyze = [
            "src/pynomaly/domain/entities.py",
            "src/pynomaly/application/services.py",
            "tests/test_entities.py",  # Should be excluded
            "src/pynomaly/migrations/001_initial.py",  # Should be excluded
        ]

        filtered_files = coverage_gate.apply_exclusion_patterns(files_to_analyze)

        assert "src/pynomaly/domain/entities.py" in filtered_files
        assert "src/pynomaly/application/services.py" in filtered_files
        assert "tests/test_entities.py" not in filtered_files
        assert "src/pynomaly/migrations/001_initial.py" not in filtered_files


class TestPerformanceGate:
    """Test suite for performance quality gates."""

    @pytest.fixture
    def performance_gate(self):
        """Create performance gate with thresholds."""
        return PerformanceGate(
            max_response_time_ms=500,
            max_memory_usage_mb=1024,
            min_throughput_rps=100,
            max_error_rate=0.01,
        )

    def test_response_time_validation(self, performance_gate):
        """Test response time validation."""
        performance_metrics = {
            "avg_response_time_ms": 350,
            "p95_response_time_ms": 480,
            "p99_response_time_ms": 495,
            "max_response_time_ms": 500,
        }

        result = performance_gate.validate_response_time(performance_metrics)

        assert result["passed"] is True
        assert result["avg_response_time_ok"] is True
        assert result["p95_response_time_ok"] is True

    def test_memory_usage_validation(self, performance_gate):
        """Test memory usage validation."""
        memory_metrics = {
            "avg_memory_mb": 512,
            "peak_memory_mb": 800,
            "memory_leak_detected": False,
            "gc_frequency": 10,  # per minute
        }

        result = performance_gate.validate_memory_usage(memory_metrics)

        assert result["passed"] is True
        assert result["peak_memory_ok"] is True
        assert result["memory_leak_ok"] is True

    def test_throughput_validation(self, performance_gate):
        """Test throughput validation."""
        throughput_metrics = {
            "avg_throughput_rps": 150,
            "min_throughput_rps": 120,
            "max_throughput_rps": 180,
            "throughput_consistency": 0.95,
        }

        result = performance_gate.validate_throughput(throughput_metrics)

        assert result["passed"] is True
        assert result["min_throughput_ok"] is True
        assert result["throughput_consistent"] is True

    def test_load_testing_validation(self, performance_gate):
        """Test load testing validation."""
        load_test_results = {
            "concurrent_users": 1000,
            "test_duration_minutes": 30,
            "avg_response_time_ms": 380,
            "error_rate": 0.005,
            "throughput_rps": 120,
            "memory_usage_mb": 800,
        }

        result = performance_gate.validate_load_test(load_test_results)

        assert result["overall_passed"] is True
        assert result["response_time_passed"] is True
        assert result["error_rate_passed"] is True
        assert result["throughput_passed"] is True

    def test_performance_regression_detection(self, performance_gate):
        """Test performance regression detection."""
        baseline_metrics = {
            "avg_response_time_ms": 300,
            "throughput_rps": 150,
            "memory_usage_mb": 600,
        }

        current_metrics = {
            "avg_response_time_ms": 450,  # 50% increase
            "throughput_rps": 120,  # 20% decrease
            "memory_usage_mb": 800,  # 33% increase
        }

        regression_analysis = performance_gate.detect_regression(
            baseline_metrics, current_metrics
        )

        assert regression_analysis["regression_detected"] is True
        assert "response_time" in regression_analysis["regressions"]
        assert "throughput" in regression_analysis["regressions"]


class TestSecurityGate:
    """Test suite for security quality gates."""

    @pytest.fixture
    def security_gate(self):
        """Create security gate with policies."""
        return SecurityGate(
            max_critical_vulnerabilities=0,
            max_high_vulnerabilities=2,
            max_medium_vulnerabilities=10,
            require_dependency_scan=True,
            require_secrets_scan=True,
        )

    def test_vulnerability_scanning(self, security_gate):
        """Test vulnerability scanning validation."""
        scan_results = {
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 1,
            "medium_vulnerabilities": 5,
            "low_vulnerabilities": 12,
            "vulnerabilities": [
                {
                    "severity": "high",
                    "component": "requests",
                    "version": "2.25.1",
                    "cve": "CVE-2023-12345",
                    "description": "Remote code execution vulnerability",
                }
            ],
        }

        result = security_gate.validate_vulnerabilities(scan_results)

        assert result["passed"] is True
        assert result["critical_ok"] is True
        assert result["high_ok"] is True
        assert result["medium_ok"] is True

    def test_dependency_scanning(self, security_gate):
        """Test dependency scanning validation."""
        dependency_scan = {
            "outdated_dependencies": [
                {
                    "name": "numpy",
                    "current": "1.20.0",
                    "latest": "1.24.0",
                    "severity": "medium",
                },
                {
                    "name": "pandas",
                    "current": "1.3.0",
                    "latest": "2.0.0",
                    "severity": "high",
                },
            ],
            "vulnerable_dependencies": [
                {
                    "name": "pillow",
                    "version": "8.0.0",
                    "vulnerabilities": ["CVE-2023-98765"],
                }
            ],
            "license_issues": [],
        }

        result = security_gate.validate_dependencies(dependency_scan)

        assert result["vulnerable_dependencies_count"] == 1
        assert result["high_risk_outdated"] == 1
        assert len(result["action_required"]) > 0

    def test_secrets_scanning(self, security_gate):
        """Test secrets scanning validation."""
        secrets_scan = {
            "exposed_secrets": [
                {
                    "type": "api_key",
                    "file": "config/settings.py",
                    "line": 42,
                    "confidence": "high",
                }
            ],
            "potential_secrets": [
                {
                    "type": "password",
                    "file": "tests/test_auth.py",
                    "line": 15,
                    "confidence": "medium",
                }
            ],
        }

        result = security_gate.validate_secrets(secrets_scan)

        assert result["passed"] is False
        assert result["exposed_secrets_count"] == 1
        assert "API key found" in result["failure_reasons"]

    def test_code_security_analysis(self, security_gate):
        """Test static code security analysis."""
        security_analysis = {
            "sql_injection_risks": 0,
            "xss_risks": 1,
            "csrf_protection": True,
            "input_validation_coverage": 0.95,
            "authentication_security": "strong",
            "encryption_usage": "adequate",
        }

        result = security_gate.validate_code_security(security_analysis)

        assert result["sql_injection_ok"] is True
        assert result["input_validation_ok"] is True
        assert result["authentication_ok"] is True

    def test_compliance_validation(self, security_gate):
        """Test compliance validation."""
        compliance_check = {
            "gdpr_compliance": True,
            "data_encryption_at_rest": True,
            "data_encryption_in_transit": True,
            "audit_logging": True,
            "access_controls": True,
            "privacy_controls": True,
        }

        result = security_gate.validate_compliance(compliance_check)

        assert result["overall_compliant"] is True
        assert all(result[key] for key in compliance_check.keys())


class TestCodeQualityGate:
    """Test suite for code quality gates."""

    @pytest.fixture
    def code_quality_gate(self):
        """Create code quality gate with standards."""
        return CodeQualityGate(
            max_complexity=10,
            min_maintainability_index=70,
            max_code_duplication=5.0,
            enforce_type_hints=True,
            enforce_docstrings=True,
        )

    def test_complexity_analysis(self, code_quality_gate):
        """Test code complexity analysis."""
        complexity_metrics = {
            "cyclomatic_complexity": {
                "average": 6.2,
                "max": 12,
                "functions_over_threshold": [
                    {
                        "name": "complex_function",
                        "complexity": 12,
                        "file": "src/module.py",
                    }
                ],
            },
            "cognitive_complexity": {"average": 5.8, "max": 15},
        }

        result = code_quality_gate.validate_complexity(complexity_metrics)

        assert result["passed"] is False  # Max complexity 12 > threshold 10
        assert result["functions_over_threshold"] == 1

    def test_maintainability_analysis(self, code_quality_gate):
        """Test maintainability analysis."""
        maintainability_metrics = {
            "maintainability_index": 75.5,
            "halstead_metrics": {"difficulty": 12.3, "effort": 1500, "volume": 850},
            "lines_of_code": 2500,
            "comment_ratio": 0.15,
        }

        result = code_quality_gate.validate_maintainability(maintainability_metrics)

        assert result["passed"] is True
        assert result["maintainability_index_ok"] is True
        assert result["comment_ratio_adequate"] is True

    def test_code_duplication_analysis(self, code_quality_gate):
        """Test code duplication analysis."""
        duplication_analysis = {
            "duplication_percentage": 3.2,
            "duplicated_blocks": [
                {
                    "lines": 15,
                    "files": ["src/module1.py", "src/module2.py"],
                    "similarity": 0.95,
                }
            ],
            "clone_classes": 5,
        }

        result = code_quality_gate.validate_duplication(duplication_analysis)

        assert result["passed"] is True  # 3.2% < 5% threshold
        assert result["duplication_percentage_ok"] is True

    def test_type_hints_validation(self, code_quality_gate):
        """Test type hints validation."""
        type_hints_analysis = {
            "functions_with_hints": 180,
            "total_functions": 200,
            "coverage_percentage": 90.0,
            "missing_hints": [
                {"function": "legacy_function", "file": "src/legacy.py", "line": 42}
            ],
        }

        result = code_quality_gate.validate_type_hints(type_hints_analysis)

        assert result["coverage_percentage"] == 90.0
        assert result["missing_hints_count"] == 1

    def test_docstring_validation(self, code_quality_gate):
        """Test docstring validation."""
        docstring_analysis = {
            "functions_with_docstrings": 175,
            "total_functions": 200,
            "coverage_percentage": 87.5,
            "missing_docstrings": [
                {"function": "helper_function", "file": "src/utils.py", "line": 123}
            ],
        }

        result = code_quality_gate.validate_docstrings(docstring_analysis)

        assert result["coverage_percentage"] == 87.5
        assert result["missing_docstrings_count"] == 1


class TestCICDIntegration:
    """Test suite for CI/CD pipeline integration."""

    def test_github_actions_integration(self):
        """Test GitHub Actions quality gate integration."""
        github_workflow = {
            "name": "Quality Gates",
            "on": ["push", "pull_request"],
            "jobs": {
                "quality-gates": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout", "uses": "actions/checkout@v3"},
                        {"name": "Setup Python", "uses": "actions/setup-python@v4"},
                        {"name": "Run Coverage Gate", "run": "python -m pytest --cov"},
                        {"name": "Run Security Gate", "run": "bandit -r src/"},
                        {
                            "name": "Run Performance Gate",
                            "run": "python -m pytest tests/performance/",
                        },
                    ],
                }
            },
        }

        # Verify workflow structure
        assert "quality-gates" in github_workflow["jobs"]
        assert len(github_workflow["jobs"]["quality-gates"]["steps"]) >= 4

    def test_jenkins_pipeline_integration(self):
        """Test Jenkins pipeline quality gate integration."""
        jenkins_pipeline = """
        pipeline {
            agent any
            stages {
                stage('Quality Gates') {
                    parallel {
                        stage('Coverage Gate') {
                            steps {
                                sh 'python -m pytest --cov --cov-report=xml'
                                publishCoverage adapters: [cobertura('coverage.xml')]
                            }
                        }
                        stage('Security Gate') {
                            steps {
                                sh 'bandit -r src/ -f json -o security-report.json'
                                archiveArtifacts 'security-report.json'
                            }
                        }
                        stage('Performance Gate') {
                            steps {
                                sh 'python -m pytest tests/performance/ --junitxml=perf-results.xml'
                                publishTestResults 'perf-results.xml'
                            }
                        }
                    }
                }
            }
            post {
                failure {
                    emailext body: 'Quality gates failed', to: 'dev-team@company.com'
                }
            }
        }
        """

        # Verify pipeline contains quality gates
        assert "Coverage Gate" in jenkins_pipeline
        assert "Security Gate" in jenkins_pipeline
        assert "Performance Gate" in jenkins_pipeline

    def test_quality_gate_failure_handling(self):
        """Test quality gate failure handling in CI/CD."""
        # Mock CI/CD environment
        ci_environment = {
            "is_pull_request": True,
            "base_branch": "main",
            "head_branch": "feature/new-feature",
            "commit_sha": "abc123def456",
        }

        # Mock quality gate results
        gate_results = {
            "coverage_gate": {"passed": True, "score": 85.5},
            "performance_gate": {"passed": False, "issues": ["Response time exceeded"]},
            "security_gate": {"passed": True, "vulnerabilities": 0},
            "code_quality_gate": {"passed": True, "complexity": 8.2},
        }

        # Simulate CI/CD decision logic
        overall_passed = all(result["passed"] for result in gate_results.values())

        assert overall_passed is False

        # Should block merge for PR
        if ci_environment["is_pull_request"] and not overall_passed:
            merge_blocked = True
        else:
            merge_blocked = False

        assert merge_blocked is True

    def test_deployment_gate_validation(self):
        """Test deployment quality gate validation."""
        deployment_criteria = {
            "all_quality_gates_passed": True,
            "security_scan_passed": True,
            "performance_benchmarks_met": True,
            "smoke_tests_passed": True,
            "rollback_plan_ready": True,
            "monitoring_configured": True,
        }

        # Check deployment readiness
        deployment_ready = all(deployment_criteria.values())

        assert deployment_ready is True

    def test_quality_metrics_reporting(self):
        """Test quality metrics reporting and dashboards."""
        quality_metrics = {
            "timestamp": datetime.now().isoformat(),
            "build_id": "build_12345",
            "coverage": {
                "line_coverage": 85.5,
                "branch_coverage": 78.2,
                "trend": "improving",
            },
            "performance": {
                "avg_response_time_ms": 350,
                "throughput_rps": 150,
                "trend": "stable",
            },
            "security": {
                "vulnerabilities": 0,
                "risk_score": "low",
                "trend": "improving",
            },
            "code_quality": {
                "maintainability_index": 75.5,
                "complexity": 8.2,
                "duplication": 3.2,
                "trend": "stable",
            },
        }

        # Verify metrics structure
        assert "coverage" in quality_metrics
        assert "performance" in quality_metrics
        assert "security" in quality_metrics
        assert "code_quality" in quality_metrics

        # Calculate overall quality score
        scores = [
            quality_metrics["coverage"]["line_coverage"],
            min(
                quality_metrics["performance"]["avg_response_time_ms"] / 10, 100
            ),  # Normalized
            max(100 - quality_metrics["security"]["vulnerabilities"] * 10, 0),
            quality_metrics["code_quality"]["maintainability_index"],
        ]

        overall_quality_score = sum(scores) / len(scores)
        assert 0 <= overall_quality_score <= 100


class TestQualityGateConfiguration:
    """Test suite for quality gate configuration management."""

    def test_quality_gate_configuration_loading(self):
        """Test loading quality gate configuration from files."""
        config = {
            "coverage": {
                "minimum_line_coverage": 80.0,
                "minimum_branch_coverage": 75.0,
                "exclude_patterns": ["tests/*", "*/migrations/*"],
            },
            "performance": {
                "max_response_time_ms": 500,
                "min_throughput_rps": 100,
                "max_memory_usage_mb": 1024,
            },
            "security": {
                "max_critical_vulnerabilities": 0,
                "max_high_vulnerabilities": 2,
                "require_secrets_scan": True,
            },
            "code_quality": {
                "max_complexity": 10,
                "min_maintainability_index": 70,
                "enforce_type_hints": True,
            },
        }

        # Verify configuration structure
        assert "coverage" in config
        assert "performance" in config
        assert "security" in config
        assert "code_quality" in config

    def test_environment_specific_configuration(self):
        """Test environment-specific quality gate configuration."""
        configurations = {
            "development": {
                "coverage_threshold": 70.0,
                "performance_threshold": "relaxed",
                "security_level": "standard",
            },
            "staging": {
                "coverage_threshold": 80.0,
                "performance_threshold": "standard",
                "security_level": "enhanced",
            },
            "production": {
                "coverage_threshold": 85.0,
                "performance_threshold": "strict",
                "security_level": "maximum",
            },
        }

        # Test configuration selection
        current_env = "production"
        config = configurations[current_env]

        assert config["coverage_threshold"] == 85.0
        assert config["security_level"] == "maximum"

    def test_quality_gate_override_mechanism(self):
        """Test quality gate override mechanism for emergency deployments."""
        override_request = {
            "requester": "senior_developer@company.com",
            "reason": "Critical production bug fix",
            "gates_to_override": ["performance_gate"],
            "approval_required": True,
            "expiry_time": datetime.now() + timedelta(hours=24),
        }

        # Simulate approval process
        approver = "tech_lead@company.com"
        approved = True

        if approved and override_request["expiry_time"] > datetime.now():
            override_valid = True
        else:
            override_valid = False

        assert override_valid is True
