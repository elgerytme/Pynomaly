#!/usr/bin/env python3
"""
Validator Engine
================
Main orchestrator for all best practices validation
"""

import asyncio
import importlib.util
import inspect
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import yaml

from .base_validator import BaseValidator, ValidationResult, RuleViolation


@dataclass
class ComplianceScore:
    """Overall compliance scoring across all categories"""
    overall_score: float
    category_scores: Dict[str, float]
    grade: str  # A+, A, B, C, D, F
    total_violations: int
    critical_violations: int
    high_violations: int
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    project_name: str
    project_root: str
    compliance_score: ComplianceScore
    category_results: Dict[str, List[ValidationResult]]
    all_violations: List[RuleViolation]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidatorEngine:
    """
    Main orchestrator for all best practices validation.
    
    Manages validator discovery, execution, and reporting.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, project_root: Optional[Union[str, Path]] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.validators: Dict[str, List[BaseValidator]] = {}
        self.enabled_categories = self.config.get('enabled_categories', [
            'architecture', 'engineering', 'security', 'testing', 'devops', 'sre'
        ])
        
        # Load validators
        self._load_validators()
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        
        # Default configuration
        return {
            'framework_version': '1.0.0',
            'enabled_categories': ['architecture', 'engineering', 'security', 'testing', 'devops', 'sre'],
            'global': {
                'enforcement_level': 'strict',
                'fail_on_critical': True,
                'fail_on_high': False,
                'max_violations_per_category': 10
            },
            'architecture': {'enabled': True},
            'engineering': {'enabled': True},
            'security': {'enabled': True},
            'testing': {'enabled': True},
            'devops': {'enabled': True},
            'sre': {'enabled': True}
        }
    
    def _load_validators(self) -> None:
        """Dynamically load all validators"""
        validators_dir = Path(__file__).parent.parent / 'validators'
        
        for category in self.enabled_categories:
            if not self.config.get(category, {}).get('enabled', True):
                continue
            
            category_dir = validators_dir / category
            if not category_dir.exists():
                continue
            
            self.validators[category] = []
            
            # Load validators from category directory
            for validator_file in category_dir.glob('*_validator.py'):
                try:
                    module_name = f"validators.{category}.{validator_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, validator_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find validator classes in module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseValidator) and 
                            obj != BaseValidator):
                            
                            validator = obj(self.config, self.project_root)
                            if validator.is_enabled():
                                self.validators[category].append(validator)
                                self.logger.info(f"Loaded validator: {category}.{name}")
                
                except Exception as e:
                    self.logger.error(f"Failed to load validator {validator_file}: {e}")
    
    async def validate_all(self) -> ValidationReport:
        """Run all enabled validators and generate comprehensive report"""
        start_time = time.time()
        
        self.logger.info(f"🔍 Starting comprehensive validation for project: {self.project_root}")
        
        category_results: Dict[str, List[ValidationResult]] = {}
        all_violations: List[RuleViolation] = []
        
        # Run validators for each category
        for category, validators in self.validators.items():
            if not validators:
                continue
            
            self.logger.info(f"   Validating {category} ({len(validators)} validators)")
            category_results[category] = []
            
            # Run validators in parallel within category
            tasks = [validator.validate() for validator in validators]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Validator failed: {result}")
                    continue
                
                category_results[category].append(result)
                all_violations.extend(result.violations)
        
        # Calculate compliance score
        execution_time = time.time() - start_time
        compliance_score = self._calculate_compliance_score(category_results, execution_time)
        
        # Generate report
        report = ValidationReport(
            project_name=self.project_root.name,
            project_root=str(self.project_root),
            compliance_score=compliance_score,
            category_results=category_results,
            all_violations=all_violations,
            summary=self._generate_summary(category_results, compliance_score)
        )
        
        self.logger.info(f"✅ Validation completed in {execution_time:.2f}s - Overall Score: {compliance_score.overall_score:.1f}% ({compliance_score.grade})")
        
        return report
    
    async def validate_category(self, category: str) -> List[ValidationResult]:
        """Run validators for specific category"""
        if category not in self.validators:
            raise ValueError(f"Unknown category: {category}")
        
        validators = self.validators[category]
        if not validators:
            return []
        
        self.logger.info(f"🔍 Validating {category} ({len(validators)} validators)")
        
        # Run validators in parallel
        tasks = [validator.validate() for validator in validators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Validator failed: {result}")
                continue
            valid_results.append(result)
        
        return valid_results
    
    async def validate_incremental(self, changed_files: List[str]) -> ValidationReport:
        """Run validation only on changed files for fast CI/CD feedback"""
        # TODO: Implement incremental validation
        # This would filter validators to only run on relevant changed files
        return await self.validate_all()
    
    def _calculate_compliance_score(self, category_results: Dict[str, List[ValidationResult]], execution_time: float) -> ComplianceScore:
        """Calculate overall compliance score with category weighting"""
        category_weights = {
            'security': 0.30,
            'architecture': 0.25,
            'testing': 0.20,
            'devops': 0.15,
            'engineering': 0.10,
            'sre': 0.10  # Extra weight if SRE is enabled
        }
        
        # Normalize weights based on enabled categories
        enabled_weights = {k: v for k, v in category_weights.items() if k in category_results}
        total_weight = sum(enabled_weights.values())
        if total_weight > 0:
            enabled_weights = {k: v / total_weight for k, v in enabled_weights.items()}
        
        # Calculate category scores
        category_scores = {}
        weighted_score = 0.0
        total_violations = 0
        critical_violations = 0
        high_violations = 0
        
        for category, results in category_results.items():
            if not results:
                category_scores[category] = 100.0
                continue
            
            # Average score across validators in category
            category_score = sum(result.score for result in results) / len(results)
            category_scores[category] = category_score
            
            # Count violations
            for result in results:
                total_violations += len(result.violations)
                critical_violations += result.critical_count
                high_violations += result.high_count
            
            # Apply weight to overall score
            weight = enabled_weights.get(category, 0)
            weighted_score += category_score * weight
        
        # Calculate grade
        grade = self._calculate_grade(weighted_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(category_results)
        
        return ComplianceScore(
            overall_score=weighted_score,
            category_scores=category_scores,
            grade=grade,
            total_violations=total_violations,
            critical_violations=critical_violations,
            high_violations=high_violations,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'A-'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'B-'
        elif score >= 65:
            return 'C+'
        elif score >= 60:
            return 'C'
        elif score >= 55:
            return 'C-'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, category_results: Dict[str, List[ValidationResult]]) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []
        
        # Analyze violations by severity and category
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        category_violation_counts = {}
        
        for category, results in category_results.items():
            category_violation_counts[category] = 0
            for result in results:
                category_violation_counts[category] += len(result.violations)
                for violation in result.violations:
                    severity_counts[violation.severity] += 1
        
        # Priority recommendations based on violations
        if severity_counts['critical'] > 0:
            recommendations.append(f"🚨 Address {severity_counts['critical']} critical security/architecture issues immediately")
        
        if severity_counts['high'] > 0:
            recommendations.append(f"⚠️ Resolve {severity_counts['high']} high-priority issues within 48 hours")
        
        # Category-specific recommendations
        worst_category = max(category_violation_counts, key=category_violation_counts.get) if category_violation_counts else None
        if worst_category and category_violation_counts[worst_category] > 0:
            recommendations.append(f"📊 Focus improvement efforts on {worst_category} category ({category_violation_counts[worst_category]} violations)")
        
        # Best practice recommendations
        if severity_counts['medium'] + severity_counts['low'] > 10:
            recommendations.append("🔧 Consider implementing automated fixes for common violations")
        
        recommendations.append("📚 Review best practices documentation for detailed guidance")
        
        return recommendations
    
    def _generate_summary(self, category_results: Dict[str, List[ValidationResult]], compliance_score: ComplianceScore) -> Dict[str, Any]:
        """Generate validation summary statistics"""
        total_validators = sum(len(results) for results in category_results.values())
        
        return {
            'total_validators_run': total_validators,
            'categories_validated': len(category_results),
            'overall_grade': compliance_score.grade,
            'overall_score': compliance_score.overall_score,
            'total_violations': compliance_score.total_violations,
            'critical_violations': compliance_score.critical_violations,
            'high_violations': compliance_score.high_violations,
            'execution_time': compliance_score.execution_time,
            'timestamp': compliance_score.timestamp.isoformat()
        }
    
    def get_available_validators(self) -> Dict[str, List[str]]:
        """Get list of available validators by category"""
        available = {}
        for category, validators in self.validators.items():
            available[category] = [v.get_name() for v in validators]
        return available
    
    def quality_gate(self, report: ValidationReport, enforce_critical: bool = True, enforce_high: bool = False) -> bool:
        """
        Quality gate check - determines if validation passes quality requirements.
        
        Args:
            report: Validation report to check
            enforce_critical: If True, fail on any critical violations
            enforce_high: If True, fail on any high violations
        
        Returns:
            True if quality gate passes, False otherwise
        """
        if enforce_critical and report.compliance_score.critical_violations > 0:
            self.logger.error(f"❌ Quality gate FAILED: {report.compliance_score.critical_violations} critical violations")
            return False
        
        if enforce_high and report.compliance_score.high_violations > 0:
            self.logger.error(f"❌ Quality gate FAILED: {report.compliance_score.high_violations} high violations")
            return False
        
        # Check global configuration thresholds
        global_config = self.config.get('global', {})
        fail_on_critical = global_config.get('fail_on_critical', True)
        fail_on_high = global_config.get('fail_on_high', False)
        max_violations_per_category = global_config.get('max_violations_per_category', 10)
        
        if fail_on_critical and report.compliance_score.critical_violations > 0:
            return False
        
        if fail_on_high and report.compliance_score.high_violations > 0:
            return False
        
        # Check per-category violation limits
        for category, results in report.category_results.items():
            category_violations = sum(len(result.violations) for result in results)
            if category_violations > max_violations_per_category:
                self.logger.error(f"❌ Quality gate FAILED: {category} has {category_violations} violations (limit: {max_violations_per_category})")
                return False
        
        self.logger.info(f"✅ Quality gate PASSED - Score: {report.compliance_score.overall_score:.1f}% ({report.compliance_score.grade})")
        return True