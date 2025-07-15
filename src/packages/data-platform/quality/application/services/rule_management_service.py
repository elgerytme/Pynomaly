"""Rule Management Service.

Service for managing quality rules including creation, validation, versioning,
testing, and deployment of data quality rules.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json
import re
from enum import Enum

from ...domain.entities.validation_rule import (
    QualityRule, ValidationLogic, ValidationResult, ValidationError,
    RuleId, RuleType, LogicType, Severity, QualityCategory,
    SuccessCriteria, UserId
)
from ...domain.entities.quality_profile import DatasetId
from .validation_engine import ValidationEngine, ValidationEngineConfig

logger = logging.getLogger(__name__)


class RuleTestStatus(Enum):
    """Status of rule testing."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class RuleTestResult:
    """Result of rule testing."""
    rule_id: RuleId
    test_status: RuleTestStatus
    test_dataset_size: int
    validation_result: Optional[ValidationResult] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tested_at: datetime = field(default_factory=datetime.now)
    
    def is_successful(self) -> bool:
        """Check if test was successful."""
        return self.test_status == RuleTestStatus.PASSED
    
    def has_performance_issues(self) -> bool:
        """Check if test revealed performance issues."""
        return self.performance_metrics.get('execution_time_seconds', 0) > 30
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test result summary."""
        return {
            'rule_id': str(self.rule_id),
            'test_status': self.test_status.value,
            'test_dataset_size': self.test_dataset_size,
            'is_successful': self.is_successful(),
            'has_performance_issues': self.has_performance_issues(),
            'execution_time_seconds': self.performance_metrics.get('execution_time_seconds', 0),
            'memory_usage_mb': self.performance_metrics.get('memory_usage_mb', 0),
            'recommendations_count': len(self.recommendations),
            'warnings_count': len(self.warnings),
            'tested_at': self.tested_at.isoformat()
        }


@dataclass(frozen=True)
class RuleTemplate:
    """Template for creating rules."""
    template_id: str
    template_name: str
    template_description: str
    rule_type: RuleType
    logic_type: LogicType
    expression_template: str
    parameter_definitions: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    
    def create_rule(self, 
                   rule_name: str,
                   description: str,
                   parameters: Dict[str, Any],
                   created_by: UserId,
                   severity: Severity = Severity.MEDIUM,
                   category: QualityCategory = QualityCategory.DATA_INTEGRITY) -> QualityRule:
        """Create a rule from this template."""
        # Substitute parameters in expression template
        expression = self.expression_template
        for param_name, param_value in parameters.items():
            expression = expression.replace(f"{{{param_name}}}", str(param_value))
        
        # Create validation logic
        validation_logic = ValidationLogic(
            logic_type=self.logic_type,
            expression=expression,
            parameters=parameters,
            error_message=f"Rule validation failed: {rule_name}",
            success_criteria=SuccessCriteria()
        )
        
        return QualityRule(
            rule_id=RuleId(),
            rule_name=rule_name,
            rule_type=self.rule_type,
            description=description,
            validation_logic=validation_logic,
            severity=severity,
            category=category,
            is_active=True,
            created_by=created_by,
            created_at=datetime.now(),
            last_modified=datetime.now()
        )


class RuleManagementService:
    """Service for managing quality rules."""
    
    def __init__(self, validation_engine: ValidationEngine = None):
        """Initialize rule management service."""
        if validation_engine is None:
            from ...infrastructure.config.quality_config import ValidationEngineConfig
            config = ValidationEngineConfig()
            validation_engine = ValidationEngine(config)
        self.validation_engine = validation_engine
        self._rule_templates = self._initialize_rule_templates()
        
        # Rule statistics
        self._rule_stats = {
            'total_rules_created': 0,
            'total_rules_tested': 0,
            'total_rules_deployed': 0,
            'average_test_time': 0.0,
            'test_success_rate': 0.0
        }
    
    def create_rule(self, 
                   rule_name: str,
                   rule_type: RuleType,
                   description: str,
                   validation_logic: ValidationLogic,
                   severity: Severity,
                   category: QualityCategory,
                   created_by: UserId,
                   target_tables: List[str] = None,
                   target_columns: List[str] = None,
                   tags: List[str] = None) -> QualityRule:
        """Create a new quality rule."""
        try:
            # Validate rule inputs
            self._validate_rule_inputs(rule_name, validation_logic)
            
            # Create rule entity
            rule = QualityRule(
                rule_id=RuleId(),
                rule_name=rule_name,
                rule_type=rule_type,
                description=description,
                validation_logic=validation_logic,
                severity=severity,
                category=category,
                is_active=True,
                created_by=created_by,
                created_at=datetime.now(),
                last_modified=datetime.now(),
                target_tables=target_tables or [],
                target_columns=target_columns or [],
                tags=tags or []
            )
            
            # Update statistics
            self._rule_stats['total_rules_created'] += 1
            
            logger.info(f"Created rule: {rule_name} ({rule.rule_id})")
            return rule
            
        except Exception as e:
            logger.error(f"Failed to create rule {rule_name}: {str(e)}")
            raise
    
    def create_rule_from_template(self, 
                                 template_id: str,
                                 rule_name: str,
                                 description: str,
                                 parameters: Dict[str, Any],
                                 created_by: UserId,
                                 severity: Severity = Severity.MEDIUM,
                                 category: QualityCategory = QualityCategory.DATA_INTEGRITY) -> QualityRule:
        """Create rule from template."""
        try:
            template = self._rule_templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            # Validate parameters
            self._validate_template_parameters(template, parameters)
            
            # Create rule from template
            rule = template.create_rule(
                rule_name=rule_name,
                description=description,
                parameters=parameters,
                created_by=created_by,
                severity=severity,
                category=category
            )
            
            self._rule_stats['total_rules_created'] += 1
            
            logger.info(f"Created rule from template {template_id}: {rule_name}")
            return rule
            
        except Exception as e:
            logger.error(f"Failed to create rule from template {template_id}: {str(e)}")
            raise
    
    def test_rule(self, 
                 rule: QualityRule,
                 test_dataset: pd.DataFrame,
                 dataset_id: DatasetId = None) -> RuleTestResult:
        """Test a rule against a dataset."""
        try:
            start_time = datetime.now()
            
            # Create dataset ID if not provided
            if not dataset_id:
                dataset_id = DatasetId("test_dataset")
            
            # Execute validation
            validation_result = self.validation_engine.validate_single_rule(
                df=test_dataset,
                rule=rule,
                dataset_id=dataset_id
            )
            
            # Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            memory_usage = test_dataset.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            performance_metrics = {
                'execution_time_seconds': execution_time,
                'memory_usage_mb': memory_usage,
                'records_per_second': len(test_dataset) / execution_time if execution_time > 0 else 0
            }
            
            # Determine test status
            if validation_result.status.value == 'error':
                test_status = RuleTestStatus.ERROR
            elif validation_result.status.value == 'failed':
                test_status = RuleTestStatus.FAILED
            elif validation_result.status.value == 'warning':
                test_status = RuleTestStatus.WARNING
            else:
                test_status = RuleTestStatus.PASSED
            
            # Generate recommendations
            recommendations = self._generate_test_recommendations(rule, validation_result, performance_metrics)
            
            # Generate warnings
            warnings = self._generate_test_warnings(rule, validation_result, performance_metrics)
            
            test_result = RuleTestResult(
                rule_id=rule.rule_id,
                test_status=test_status,
                test_dataset_size=len(test_dataset),
                validation_result=validation_result,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                warnings=warnings
            )
            
            # Update statistics
            self._rule_stats['total_rules_tested'] += 1
            self._update_test_statistics(test_result)
            
            logger.info(f"Tested rule {rule.rule_name}: {test_status.value}")
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to test rule {rule.rule_name}: {str(e)}")
            return RuleTestResult(
                rule_id=rule.rule_id,
                test_status=RuleTestStatus.ERROR,
                test_dataset_size=len(test_dataset),
                warnings=[f"Test execution failed: {str(e)}"]
            )
    
    def batch_test_rules(self, 
                        rules: List[QualityRule],
                        test_dataset: pd.DataFrame,
                        dataset_id: DatasetId = None) -> List[RuleTestResult]:
        """Test multiple rules against a dataset."""
        results = []
        
        for rule in rules:
            result = self.test_rule(rule, test_dataset, dataset_id)
            results.append(result)
        
        return results
    
    def validate_rule_logic(self, validation_logic: ValidationLogic) -> Dict[str, Any]:
        """Validate rule logic without executing it."""
        validation_result = {
            'is_valid': True,
            'syntax_errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Validate expression syntax
            if validation_logic.logic_type == LogicType.PYTHON:
                self._validate_python_syntax(validation_logic.expression, validation_result)
            elif validation_logic.logic_type == LogicType.SQL:
                self._validate_sql_syntax(validation_logic.expression, validation_result)
            elif validation_logic.logic_type == LogicType.REGEX:
                self._validate_regex_syntax(validation_logic.expression, validation_result)
            
            # Validate parameters
            self._validate_logic_parameters(validation_logic, validation_result)
            
            # Generate recommendations
            self._generate_logic_recommendations(validation_logic, validation_result)
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['syntax_errors'].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    def get_rule_templates(self) -> List[RuleTemplate]:
        """Get available rule templates."""
        return list(self._rule_templates.values())
    
    def get_rule_template(self, template_id: str) -> Optional[RuleTemplate]:
        """Get specific rule template."""
        return self._rule_templates.get(template_id)
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule management statistics."""
        return self._rule_stats.copy()
    
    def _validate_rule_inputs(self, rule_name: str, validation_logic: ValidationLogic) -> None:
        """Validate rule creation inputs."""
        if not rule_name.strip():
            raise ValueError("Rule name cannot be empty")
        
        if len(rule_name) > 100:
            raise ValueError("Rule name cannot exceed 100 characters")
        
        if not validation_logic.expression.strip():
            raise ValueError("Validation expression cannot be empty")
        
        if len(validation_logic.expression) > 10000:
            raise ValueError("Validation expression cannot exceed 10000 characters")
    
    def _validate_template_parameters(self, template: RuleTemplate, parameters: Dict[str, Any]) -> None:
        """Validate template parameters."""
        required_params = template.parameter_definitions.get('required', [])
        
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Required parameter missing: {param}")
        
        # Validate parameter types
        param_types = template.parameter_definitions.get('types', {})
        for param, value in parameters.items():
            expected_type = param_types.get(param)
            if expected_type and not isinstance(value, expected_type):
                raise ValueError(f"Parameter {param} must be of type {expected_type}")
    
    def _validate_python_syntax(self, expression: str, result: Dict[str, Any]) -> None:
        """Validate Python expression syntax."""
        try:
            compile(expression, '<string>', 'eval')
        except SyntaxError as e:
            result['is_valid'] = False
            result['syntax_errors'].append(f"Python syntax error: {str(e)}")
        except Exception as e:
            result['warnings'].append(f"Python validation warning: {str(e)}")
    
    def _validate_sql_syntax(self, expression: str, result: Dict[str, Any]) -> None:
        """Validate SQL expression syntax."""
        # Basic SQL validation - in production, use proper SQL parser
        sql_keywords = ['select', 'from', 'where', 'and', 'or', 'not', 'in', 'like']
        
        expr_lower = expression.lower()
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter']
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                result['warnings'].append(f"SQL contains potentially dangerous keyword: {keyword}")
        
        # Basic syntax checks
        if 'select' in expr_lower and 'from' not in expr_lower:
            result['warnings'].append("SQL SELECT statement missing FROM clause")
    
    def _validate_regex_syntax(self, expression: str, result: Dict[str, Any]) -> None:
        """Validate regex expression syntax."""
        try:
            re.compile(expression)
        except re.error as e:
            result['is_valid'] = False
            result['syntax_errors'].append(f"Regex syntax error: {str(e)}")
    
    def _validate_logic_parameters(self, logic: ValidationLogic, result: Dict[str, Any]) -> None:
        """Validate logic parameters."""
        # Check for required parameters based on logic type
        if logic.logic_type == LogicType.REGEX:
            if 'column_name' not in logic.parameters:
                result['warnings'].append("Regex validation should specify column_name parameter")
        
        elif logic.logic_type == LogicType.STATISTICAL:
            if 'column_name' not in logic.parameters:
                result['warnings'].append("Statistical validation should specify column_name parameter")
            
            stat_type = logic.parameters.get('stat_type')
            if stat_type == 'range':
                if 'min_value' not in logic.parameters or 'max_value' not in logic.parameters:
                    result['warnings'].append("Range validation should specify min_value and max_value parameters")
        
        elif logic.logic_type == LogicType.COMPARISON:
            if 'column1' not in logic.parameters or 'column2' not in logic.parameters:
                result['warnings'].append("Comparison validation should specify column1 and column2 parameters")
    
    def _generate_logic_recommendations(self, logic: ValidationLogic, result: Dict[str, Any]) -> None:
        """Generate recommendations for logic optimization."""
        # Performance recommendations
        if logic.logic_type == LogicType.PYTHON and 'df.' in logic.expression:
            result['recommendations'].append("Consider using vectorized operations instead of DataFrame operations for better performance")
        
        if logic.logic_type == LogicType.SQL and 'like' in logic.expression.lower():
            result['recommendations'].append("Consider using regex validation for better performance than SQL LIKE operations")
        
        # Error message recommendations
        if not logic.error_message or len(logic.error_message) < 10:
            result['recommendations'].append("Consider providing more descriptive error messages")
    
    def _generate_test_recommendations(self, 
                                     rule: QualityRule,
                                     validation_result: ValidationResult,
                                     performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        if performance_metrics.get('execution_time_seconds', 0) > 30:
            recommendations.append("Consider optimizing rule logic for better performance")
        
        # Failure rate recommendations
        if validation_result.failure_rate > 0.5:
            recommendations.append("High failure rate detected - consider adjusting rule criteria")
        
        # Success criteria recommendations
        if validation_result.failure_rate > 0.1 and rule.validation_logic.success_criteria.min_pass_rate == 1.0:
            recommendations.append("Consider adjusting success criteria to allow for some acceptable failures")
        
        return recommendations
    
    def _generate_test_warnings(self, 
                              rule: QualityRule,
                              validation_result: ValidationResult,
                              performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings based on test results."""
        warnings = []
        
        # Memory usage warnings
        if performance_metrics.get('memory_usage_mb', 0) > 1000:
            warnings.append("High memory usage detected - may cause issues with large datasets")
        
        # Error count warnings
        if len(validation_result.error_details) > 1000:
            warnings.append("Large number of validation errors - consider optimizing rule logic")
        
        return warnings
    
    def _initialize_rule_templates(self) -> Dict[str, RuleTemplate]:
        """Initialize built-in rule templates."""
        templates = {}
        
        # Completeness template
        templates['completeness_check'] = RuleTemplate(
            template_id='completeness_check',
            template_name='Completeness Check',
            template_description='Check for missing values in columns',
            rule_type=RuleType.COMPLETENESS,
            logic_type=LogicType.EXPRESSION,
            expression_template="df['{column_name}'].notna()",
            parameter_definitions={
                'required': ['column_name'],
                'types': {'column_name': str}
            },
            examples=['Check if customer_id is not null', 'Validate email addresses are present']
        )
        
        # Range validation template
        templates['range_validation'] = RuleTemplate(
            template_id='range_validation',
            template_name='Range Validation',
            template_description='Validate numeric values are within specified range',
            rule_type=RuleType.VALIDITY,
            logic_type=LogicType.STATISTICAL,
            expression_template='range_check',
            parameter_definitions={
                'required': ['column_name', 'min_value', 'max_value'],
                'types': {'column_name': str, 'min_value': (int, float), 'max_value': (int, float)}
            },
            examples=['Age between 0 and 150', 'Price between 0 and 10000']
        )
        
        # Pattern validation template
        templates['pattern_validation'] = RuleTemplate(
            template_id='pattern_validation',
            template_name='Pattern Validation',
            template_description='Validate text values match specified pattern',
            rule_type=RuleType.FORMAT,
            logic_type=LogicType.REGEX,
            expression_template='{pattern}',
            parameter_definitions={
                'required': ['column_name', 'pattern'],
                'types': {'column_name': str, 'pattern': str}
            },
            examples=['Email format validation', 'Phone number format validation']
        )
        
        # Uniqueness template
        templates['uniqueness_check'] = RuleTemplate(
            template_id='uniqueness_check',
            template_name='Uniqueness Check',
            template_description='Check for duplicate values in columns',
            rule_type=RuleType.UNIQUENESS,
            logic_type=LogicType.EXPRESSION,
            expression_template="~df['{column_name}'].duplicated()",
            parameter_definitions={
                'required': ['column_name'],
                'types': {'column_name': str}
            },
            examples=['Check unique customer IDs', 'Validate unique email addresses']
        )
        
        return templates
    
    def _update_test_statistics(self, test_result: RuleTestResult) -> None:
        """Update test statistics."""
        # Update average test time
        current_avg = self._rule_stats['average_test_time']
        total_tests = self._rule_stats['total_rules_tested']
        
        if test_result.performance_metrics.get('execution_time_seconds'):
            new_time = test_result.performance_metrics['execution_time_seconds']
            self._rule_stats['average_test_time'] = ((current_avg * (total_tests - 1)) + new_time) / total_tests
        
        # Update success rate
        current_success_rate = self._rule_stats['test_success_rate']
        success_count = int(current_success_rate * (total_tests - 1))
        
        if test_result.is_successful():
            success_count += 1
        
        self._rule_stats['test_success_rate'] = success_count / total_tests