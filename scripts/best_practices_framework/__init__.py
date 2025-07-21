"""
Best Practices Framework
========================

A comprehensive, automated framework for enforcing software engineering best practices
across architecture, security, testing, DevOps, and SRE domains.

This framework provides:
- Automated validation of industry best practices
- Configurable rules and thresholds
- Multi-format reporting (HTML, JSON, SARIF, JUnit)
- CI/CD integration for all major platforms
- Extensible plugin architecture

Usage:
    from best_practices_framework import BestPracticesValidator
    
    # Basic usage
    validator = BestPracticesValidator(project_root=".")
    report = await validator.validate_all()
    
    # Category-specific validation
    security_results = await validator.validate_category("security")
    
    # Quality gate checking
    passes_quality_gate = validator.quality_gate(report)

Categories:
- architecture: Clean architecture, SOLID principles, domain boundaries
- security: OWASP compliance, secrets detection, vulnerability scanning  
- testing: Test pyramid, coverage thresholds, quality gates
- engineering: Code quality, documentation, technical debt
- devops: CI/CD pipelines, infrastructure as code, deployment
- sre: Reliability, observability, incident management
"""

__version__ = "1.0.0"
__author__ = "Best Practices Framework Team"
__license__ = "MIT"

from .core.validator_engine import ValidatorEngine as BestPracticesValidator
from .core.base_validator import BaseValidator, ValidationResult, RuleViolation, ComplianceScore
from .reporting.report_generator import ReportGenerator
from .reporting.formats import HTMLReporter, MarkdownReporter, JSONReporter, SARIFReporter

# Main API exports
__all__ = [
    'BestPracticesValidator',
    'BaseValidator', 
    'ValidationResult',
    'RuleViolation',
    'ComplianceScore',
    'ReportGenerator',
    'HTMLReporter',
    'MarkdownReporter', 
    'JSONReporter',
    'SARIFReporter'
]

# Package metadata
FRAMEWORK_VERSION = __version__
SUPPORTED_LANGUAGES = [
    'python',
    'javascript', 
    'typescript',
    'java',
    'csharp',
    'go',
    'rust',
    'php',
    'ruby'
]

SUPPORTED_CATEGORIES = [
    'architecture',
    'engineering', 
    'security',
    'testing',
    'devops',
    'sre'
]

SUPPORTED_CI_PLATFORMS = [
    'github_actions',
    'gitlab_ci',
    'jenkins',
    'azure_devops', 
    'circleci',
    'travis_ci',
    'bamboo'
]

# Convenience functions
def get_version() -> str:
    """Get framework version"""
    return __version__

def get_supported_categories() -> list:
    """Get list of supported validation categories"""
    return SUPPORTED_CATEGORIES.copy()

def get_supported_languages() -> list:
    """Get list of supported programming languages"""
    return SUPPORTED_LANGUAGES.copy()

def get_supported_ci_platforms() -> list:
    """Get list of supported CI/CD platforms"""
    return SUPPORTED_CI_PLATFORMS.copy()

# Quick start function
async def quick_validate(project_root: str = ".", categories: list = None) -> dict:
    """
    Quick validation function for simple use cases.
    
    Args:
        project_root: Path to project root directory
        categories: List of categories to validate (default: all)
    
    Returns:
        Dictionary with validation results
    """
    validator = BestPracticesValidator(project_root=project_root)
    
    if categories:
        results = {}
        for category in categories:
            results[category] = await validator.validate_category(category)
        return results
    else:
        report = await validator.validate_all()
        return {
            'overall_score': report.compliance_score.overall_score,
            'grade': report.compliance_score.grade,
            'violations': len(report.all_violations),
            'critical_violations': report.compliance_score.critical_violations,
            'recommendations': report.compliance_score.recommendations
        }