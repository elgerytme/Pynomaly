#!/usr/bin/env python3
"""
Setup configuration for Best Practices Framework
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from __init__.py
init_file = Path(__file__).parent / "__init__.py"
version = "1.0.0"
with open(init_file) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="best-practices-framework",
    version=version,
    author="Best Practices Framework Team",
    author_email="team@bestpractices.dev",
    description="Comprehensive automated framework for enforcing software engineering best practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/best-practices-framework/best-practices-framework",
    project_urls={
        "Bug Reports": "https://github.com/best-practices-framework/best-practices-framework/issues",
        "Source": "https://github.com/best-practices-framework/best-practices-framework",
        "Documentation": "https://docs.bestpractices.dev",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "best_practices_framework": [
            "configs/*.yml",
            "configs/**/*.yml",
            "templates/*.html",
            "templates/*.md",
            "templates/*.xml",
            "integrations/**/*.yml",
            "integrations/**/*.json",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        "asyncio-throttle>=1.0.0",
        "jinja2>=3.1.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "tabulate>=0.9.0",
        
        # Code analysis
        "ast-grep>=0.5.0",
        "gitpython>=3.1.0",
        "pathspec>=0.11.0",
        
        # Security scanning
        "bandit>=1.7.0",
        "safety>=2.3.0",
        "semgrep>=1.0.0",
        
        # Documentation
        "pydantic>=2.0.0",
        "jsonschema>=4.0.0",
        
        # Reporting
        "matplotlib>=3.6.0",
        "plotly>=5.0.0",
        "weasyprint>=59.0",  # For PDF generation
        
        # Testing framework integration
        "pytest>=7.0.0",
        "coverage>=7.0.0",
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        # Full feature set
        "full": [
            # Container scanning
            "docker>=6.0.0",
            "kubernetes>=26.0.0",
            
            # Infrastructure as Code
            "checkov>=2.3.0",
            "terraform-compliance>=1.3.0",
            
            # Advanced security
            "cyclonedx-bom>=3.0.0",
            "pip-audit>=2.5.0",
            
            # Performance testing
            "locust>=2.14.0",
            "py-spy>=0.3.0",
            
            # Cloud integrations
            "boto3>=1.26.0",  # AWS
            "google-cloud-core>=2.3.0",  # GCP
            "azure-identity>=1.12.0",  # Azure
        ],
        # Language-specific analyzers
        "javascript": [
            "eslint-config-security>=1.0.0",
        ],
        "java": [
            "spotbugs-python>=1.0.0",
        ],
        "go": [
            "gosec-python>=1.0.0",
        ],
        # CI/CD integrations
        "github": [
            "pygithub>=1.58.0",
        ],
        "gitlab": [
            "python-gitlab>=3.14.0",
        ],
        "jenkins": [
            "jenkinsapi>=1.7.0",
        ],
        "azure": [
            "azure-devops>=7.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "best-practices=best_practices_framework.cli:main",
            "bp-validate=best_practices_framework.cli:validate_command",
            "bp-report=best_practices_framework.cli:report_command",
            "bp-init=best_practices_framework.cli:init_command",
        ],
        # Plugin system
        "best_practices_framework.validators": [
            # Core validators will be auto-discovered
        ],
        "best_practices_framework.reporters": [
            "html=best_practices_framework.reporting.formats:HTMLReporter",
            "markdown=best_practices_framework.reporting.formats:MarkdownReporter", 
            "json=best_practices_framework.reporting.formats:JSONReporter",
            "sarif=best_practices_framework.reporting.formats:SARIFReporter",
            "junit=best_practices_framework.reporting.formats:JUnitReporter",
        ],
        "best_practices_framework.integrations": [
            "github=best_practices_framework.integrations.github:GitHubIntegration",
            "gitlab=best_practices_framework.integrations.gitlab:GitLabIntegration",
            "jenkins=best_practices_framework.integrations.jenkins:JenkinsIntegration",
        ],
    },
    keywords=[
        "best-practices",
        "code-quality", 
        "security",
        "testing",
        "devops",
        "sre",
        "architecture",
        "validation",
        "compliance",
        "automation",
        "ci-cd",
        "static-analysis"
    ],
    zip_safe=False,
)