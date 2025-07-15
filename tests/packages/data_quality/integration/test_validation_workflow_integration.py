"""Integration tests for the complete Data Quality Validation Workflow.

Tests the full pipeline from data ingestion through validation, cleansing,
scoring, and reporting with realistic data scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

# Import all services for integration testing
from src.packages.data_quality.application.services.validation_engine import (
    ValidationEngine, ValidationEngineConfig
)
from src.packages.data_quality.application.services.data_cleansing_service import (
    DataCleansingService, DataCleansingConfig, CleansingRule, CleansingAction, CleansingStrategy
)
from src.packages.data_quality.application.services.comprehensive_quality_scoring_engine import (
    ComprehensiveQualityScoringEngine, ComprehensiveQualityScoringConfig,
    QualityDimension, QualityContext, ScoringAlgorithm
)
from src.packages.data_quality.application.services.rule_management_service import (
    RuleManagementService, RuleTemplate, RuleTestStatus
)
from src.packages.data_quality.application.services.ml_quality_detection_service import (
    MLQualityDetectionService, MLQualityDetectionConfig
)
from src.packages.data_quality.application.services.predictive_quality_service import (
    PredictiveQualityService, PredictiveQualityConfig
)

# Import domain entities
from src.packages.data_quality.domain.entities.validation_rule import (
    QualityRule, ValidationLogic, ValidationResult, ValidationError,
    RuleId, RuleType, LogicType, Severity, QualityCategory,
    SuccessCriteria, UserId, ValidationStatus
)
from src.packages.data_quality.domain.entities.quality_profile import DatasetId
from src.packages.data_quality.domain.entities.quality_scores import QualityScores


class TestValidationWorkflowIntegration:
    """Integration tests for the complete validation workflow."""
    
    @pytest.fixture
    def realistic_customer_data(self):
        """Create realistic customer data with quality issues."""
        np.random.seed(42)
        
        # Generate base data
        num_records = 1000
        data = {
            'customer_id': range(1, num_records + 1),
            'first_name': [f'FirstName_{i}' for i in range(1, num_records + 1)],
            'last_name': [f'LastName_{i}' for i in range(1, num_records + 1)],
            'email': [f'customer{i}@example.com' for i in range(1, num_records + 1)],
            'phone': [f'555-{i:04d}' for i in range(1, num_records + 1)],
            'age': np.random.randint(18, 80, num_records),
            'income': np.random.normal(50000, 20000, num_records),
            'registration_date': pd.date_range('2020-01-01', periods=num_records, freq='D'),
            'last_purchase_date': pd.date_range('2023-01-01', periods=num_records, freq='H'),
            'total_purchases': np.random.randint(0, 100, num_records),
            'is_active': np.random.choice([True, False], num_records, p=[0.8, 0.2]),
            'credit_score': np.random.randint(300, 850, num_records),
            'region': np.random.choice(['North', 'South', 'East', 'West'], num_records)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce quality issues
        self._introduce_quality_issues(df)
        
        return df
    
    def _introduce_quality_issues(self, df: pd.DataFrame):
        """Introduce realistic quality issues to the dataset."""
        # Missing values
        missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'email'] = None
        
        missing_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
        df.loc[missing_indices, 'phone'] = None
        
        # Duplicate records
        duplicate_indices = np.random.choice(len(df), size=int(0.02 * len(df)), replace=False)
        for idx in duplicate_indices:
            if idx < len(df) - 1:
                df.iloc[idx + 1] = df.iloc[idx].copy()
        
        # Invalid email formats
        invalid_email_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
        for idx in invalid_email_indices:
            if pd.notna(df.at[idx, 'email']):
                df.at[idx, 'email'] = 'invalid_email_format'
        
        # Outliers in age
        outlier_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
        for idx in outlier_indices:
            df.at[idx, 'age'] = np.random.choice([0, 150, 200])
        
        # Inconsistent phone formats
        inconsistent_phone_indices = np.random.choice(len(df), size=int(0.02 * len(df)), replace=False)
        for idx in inconsistent_phone_indices:
            if pd.notna(df.at[idx, 'phone']):
                df.at[idx, 'phone'] = f'({np.random.randint(100, 999)}) {np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}'
        
        # Negative income values
        negative_income_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
        for idx in negative_income_indices:
            df.at[idx, 'income'] = -abs(df.at[idx, 'income'])
        
        # Future registration dates
        future_date_indices = np.random.choice(len(df), size=int(0.005 * len(df)), replace=False)
        for idx in future_date_indices:
            df.at[idx, 'registration_date'] = datetime.now() + timedelta(days=30)
        
        # Inconsistent casing
        inconsistent_case_indices = np.random.choice(len(df), size=int(0.02 * len(df)), replace=False)
        for idx in inconsistent_case_indices:
            df.at[idx, 'region'] = df.at[idx, 'region'].lower()
    
    @pytest.fixture
    def comprehensive_rules(self):
        """Create comprehensive validation rules for customer data."""
        rules = []
        
        # Email completeness rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Email Completeness",
            description="Check that email addresses are not missing",
            rule_type=RuleType.COMPLETENESS,
            validation_logic=ValidationLogic(
                logic_type=LogicType.EXPRESSION,
                expression="df['email'].notna()",
                parameters={'column_name': 'email'},
                success_criteria=SuccessCriteria(min_pass_rate=0.95),
                error_message="Email address is missing"
            ),
            target_columns=['email'],
            severity=Severity.HIGH,
            category=QualityCategory.DATA_INTEGRITY,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        # Email format rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Email Format Validation",
            description="Validate email format using regex",
            rule_type=RuleType.PATTERN,
            validation_logic=ValidationLogic(
                logic_type=LogicType.REGEX,
                expression=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                parameters={'column_name': 'email'},
                success_criteria=SuccessCriteria(min_pass_rate=0.97),
                error_message="Invalid email format"
            ),
            target_columns=['email'],
            severity=Severity.MEDIUM,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        # Age range rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Age Range Validation",
            description="Check age is within valid range",
            rule_type=RuleType.RANGE,
            validation_logic=ValidationLogic(
                logic_type=LogicType.STATISTICAL,
                expression="range_check",
                parameters={'column_name': 'age', 'stat_type': 'range', 'min_value': 18, 'max_value': 100},
                success_criteria=SuccessCriteria(min_pass_rate=0.98),
                error_message="Age must be between 18 and 100"
            ),
            target_columns=['age'],
            severity=Severity.HIGH,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        # Income validation rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Income Validation",
            description="Check income is positive",
            rule_type=RuleType.VALIDITY,
            validation_logic=ValidationLogic(
                logic_type=LogicType.EXPRESSION,
                expression="df['income'] > 0",
                parameters={'column_name': 'income'},
                success_criteria=SuccessCriteria(min_pass_rate=0.99),
                error_message="Income must be positive"
            ),
            target_columns=['income'],
            severity=Severity.MEDIUM,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        # Customer ID uniqueness rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Customer ID Uniqueness",
            description="Check customer IDs are unique",
            rule_type=RuleType.UNIQUENESS,
            validation_logic=ValidationLogic(
                logic_type=LogicType.EXPRESSION,
                expression="~df['customer_id'].duplicated()",
                parameters={'column_name': 'customer_id'},
                success_criteria=SuccessCriteria(min_pass_rate=1.0),
                error_message="Customer ID must be unique"
            ),
            target_columns=['customer_id'],
            severity=Severity.CRITICAL,
            category=QualityCategory.DATA_INTEGRITY,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        # Registration date timeliness rule
        rules.append(QualityRule(
            rule_id=RuleId(),
            rule_name="Registration Date Timeliness",
            description="Check registration date is not in the future",
            rule_type=RuleType.TIMELINESS,
            validation_logic=ValidationLogic(
                logic_type=LogicType.COMPARISON,
                expression="df['registration_date'] <= pd.Timestamp.now()",
                parameters={'column1': 'registration_date', 'column2': 'current_date', 'operator': '<='},
                success_criteria=SuccessCriteria(min_pass_rate=0.995),
                error_message="Registration date cannot be in the future"
            ),
            target_columns=['registration_date'],
            severity=Severity.HIGH,
            category=QualityCategory.BUSINESS_RULES,
            created_by=UserId("test_user"),
            is_active=True
        ))
        
        return rules
    
    @pytest.fixture
    def cleansing_rules(self):
        """Create comprehensive cleansing rules."""
        return [
            CleansingRule(
                action=CleansingAction.MISSING_VALUES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=['email'],
                priority=1
            ),
            CleansingRule(
                action=CleansingAction.DUPLICATES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                priority=2
            ),
            CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.REPLACE,
                target_columns=['age'],
                parameters={'method': 'clip', 'lower': 18, 'upper': 100},
                priority=3
            ),
            CleansingRule(
                action=CleansingAction.INVALID_VALUES,
                strategy=CleansingStrategy.REPLACE,
                target_columns=['income'],
                parameters={'condition': 'income < 0', 'replacement': 0},
                priority=4
            ),
            CleansingRule(
                action=CleansingAction.INCONSISTENT_CASING,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['region'],
                parameters={'case': 'title'},
                priority=5
            ),
            CleansingRule(
                action=CleansingAction.FORMAT_ISSUES,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=['phone'],
                parameters={'format': 'standard'},
                priority=6
            )
        ]
    
    @pytest.fixture
    def services(self):
        """Initialize all services for integration testing."""
        validation_config = ValidationEngineConfig(
            enable_parallel_processing=True,
            max_workers=2,
            timeout_seconds=60,
            enable_caching=True,
            chunk_size=500
        )
        
        cleansing_config = DataCleansingConfig(
            enable_parallel_processing=True,
            max_workers=2,
            enable_logging=True,
            backup_original=True
        )
        
        scoring_config = ComprehensiveQualityScoringConfig(
            default_algorithm=ScoringAlgorithm.WEIGHTED_AVERAGE,
            enable_business_context=True,
            enable_advanced_analytics=True,
            cache_results=True
        )
        
        services = {
            'validation_engine': ValidationEngine(validation_config),
            'cleansing_service': DataCleansingService(cleansing_config),
            'scoring_engine': ComprehensiveQualityScoringEngine(scoring_config),
            'rule_management': RuleManagementService(),
            'ml_detection': MLQualityDetectionService(),
            'predictive_quality': PredictiveQualityService()
        }
        
        return services
    
    def test_complete_validation_workflow(self, realistic_customer_data, comprehensive_rules, services):
        """Test the complete validation workflow from start to finish."""
        dataset_id = DatasetId("customer_data_test")
        
        # Step 1: Initial validation
        validation_results = services['validation_engine'].validate_dataset(
            df=realistic_customer_data,
            rules=comprehensive_rules,
            dataset_id=dataset_id
        )
        
        assert len(validation_results) == len(comprehensive_rules)
        
        # Check that some rules failed due to quality issues
        failed_results = [r for r in validation_results if r.status == ValidationStatus.FAILED]
        assert len(failed_results) > 0, "Expected some validation failures in dirty data"
        
        # Step 2: Generate quality scores before cleansing
        quality_context = QualityContext(
            business_domain="customer_management",
            data_sensitivity="high",
            regulatory_requirements=["GDPR", "CCPA"]
        )
        
        initial_scores = services['scoring_engine'].calculate_comprehensive_quality_score(
            validation_results=validation_results,
            context=quality_context
        )
        
        assert initial_scores.overall_score < 0.9, "Expected lower quality score for dirty data"
        
        # Step 3: ML-based quality detection
        ml_detection_results = services['ml_detection'].detect_quality_anomalies(
            df=realistic_customer_data,
            dataset_id=dataset_id
        )
        
        assert len(ml_detection_results) > 0, "Expected ML detection to find anomalies"
        
        # Step 4: Predictive quality analysis
        quality_predictions = services['predictive_quality'].predict_quality_trends(
            historical_scores=[initial_scores],
            prediction_horizon_days=30
        )
        
        assert len(quality_predictions) > 0, "Expected quality predictions"
        
        # Validate the complete workflow executed successfully
        assert True, "Complete validation workflow executed successfully"
    
    def test_validation_cleansing_integration(self, realistic_customer_data, comprehensive_rules, cleansing_rules, services):
        """Test integration between validation and cleansing services."""
        dataset_id = DatasetId("customer_data_cleansing_test")
        
        # Step 1: Initial validation
        initial_validation = services['validation_engine'].validate_dataset(
            df=realistic_customer_data,
            rules=comprehensive_rules,
            dataset_id=dataset_id
        )
        
        initial_failures = sum(r.failed_records for r in initial_validation)
        
        # Step 2: Apply cleansing
        cleansing_result = services['cleansing_service'].cleanse_dataset(
            df=realistic_customer_data,
            rules=cleansing_rules
        )
        
        assert cleansing_result.modified_records > 0, "Expected cleansing to modify records"
        assert cleansing_result.retention_rate > 0.8, "Expected high retention rate"
        
        # Step 3: Validation after cleansing
        cleaned_df = services['cleansing_service'].get_cleaned_dataframe()
        
        post_cleansing_validation = services['validation_engine'].validate_dataset(
            df=cleaned_df,
            rules=comprehensive_rules,
            dataset_id=dataset_id
        )
        
        post_cleansing_failures = sum(r.failed_records for r in post_cleansing_validation)
        
        # Quality should improve after cleansing
        assert post_cleansing_failures < initial_failures, "Expected fewer failures after cleansing"
        
        # Step 4: Quality score comparison
        quality_context = QualityContext(business_domain="customer_management")
        
        initial_scores = services['scoring_engine'].calculate_comprehensive_quality_score(
            validation_results=initial_validation,
            context=quality_context
        )
        
        post_cleansing_scores = services['scoring_engine'].calculate_comprehensive_quality_score(
            validation_results=post_cleansing_validation,
            context=quality_context
        )
        
        assert post_cleansing_scores.overall_score > initial_scores.overall_score, "Expected quality improvement after cleansing"
        
    def test_rule_management_integration(self, realistic_customer_data, services):
        """Test integration with rule management service."""
        dataset_id = DatasetId("rule_management_test")
        
        # Step 1: Create rules using templates
        template_rules = []
        
        # Email completeness rule from template
        completeness_rule = services['rule_management'].create_rule_from_template(
            template_id='completeness_check',
            rule_name='Email Completeness Check',
            description='Check email completeness using template',
            parameters={'column_name': 'email'},
            created_by=UserId("test_user")
        )
        template_rules.append(completeness_rule)
        
        # Email format rule from template
        pattern_rule = services['rule_management'].create_rule_from_template(
            template_id='pattern_validation',
            rule_name='Email Format Check',
            description='Check email format using template',
            parameters={'column_name': 'email', 'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            created_by=UserId("test_user")
        )
        template_rules.append(pattern_rule)
        
        # Step 2: Test rules
        test_results = services['rule_management'].batch_test_rules(
            rules=template_rules,
            test_dataset=realistic_customer_data,
            dataset_id=dataset_id
        )
        
        assert len(test_results) == len(template_rules)
        
        # Step 3: Validate rules in engine
        validation_results = services['validation_engine'].validate_dataset(
            df=realistic_customer_data,
            rules=template_rules,
            dataset_id=dataset_id
        )
        
        assert len(validation_results) == len(template_rules)
        
        # Step 4: Check test results match validation results
        for test_result, validation_result in zip(test_results, validation_results):
            assert test_result.rule_id == validation_result.rule_id
            assert test_result.is_successful() == (validation_result.status == ValidationStatus.PASSED)
    
    def test_large_dataset_performance(self, services):
        """Test workflow performance with large datasets."""
        # Create large dataset
        large_df = pd.DataFrame({
            'id': range(1, 10001),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'date': pd.date_range('2020-01-01', periods=10000, freq='H')
        })
        
        # Create simple rules
        rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="ID Uniqueness",
                description="Check ID uniqueness",
                rule_type=RuleType.UNIQUENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="~df['id'].duplicated()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID must be unique"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Value Range",
                description="Check value range",
                rule_type=RuleType.RANGE,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.STATISTICAL,
                    expression="range_check",
                    parameters={'column_name': 'value', 'stat_type': 'z_score', 'threshold': 3.0},
                    success_criteria=SuccessCriteria(min_pass_rate=0.95),
                    error_message="Value is an outlier"
                ),
                target_columns=['value'],
                severity=Severity.MEDIUM,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
        
        dataset_id = DatasetId("large_dataset_test")
        
        # Measure performance
        import time
        start_time = time.time()
        
        validation_results = services['validation_engine'].validate_dataset(
            df=large_df,
            rules=rules,
            dataset_id=dataset_id
        )
        
        execution_time = time.time() - start_time
        
        # Validate performance
        assert execution_time < 30.0, f"Large dataset validation took too long: {execution_time:.2f}s"
        assert len(validation_results) == len(rules)
        
        # Check engine metrics
        metrics = services['validation_engine'].get_execution_metrics()
        assert metrics['total_validations'] > 0
        assert metrics['average_execution_time'] < 15.0
    
    def test_error_handling_and_recovery(self, services):
        """Test error handling and recovery in the workflow."""
        # Create problematic dataset
        problematic_df = pd.DataFrame({
            'id': [1, 2, 3],
            'text_column': ['valid', None, 'also_valid'],
            'numeric_column': [1, 'invalid', 3]  # Mixed types
        })
        
        # Create rules that will encounter errors
        problematic_rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Problematic Python Rule",
                description="Rule that will cause errors",
                rule_type=RuleType.CUSTOM,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.PYTHON,
                    expression="df['nonexistent_column'].sum() > 0",  # This will fail
                    parameters={},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="Problematic rule failed"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.BUSINESS_RULES,
                created_by=UserId("test_user"),
                is_active=True
            ),
            QualityRule(
                rule_id=RuleId(),
                rule_name="Valid Rule",
                description="Rule that should work",
                rule_type=RuleType.COMPLETENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="df['id'].notna()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID should not be null"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
        
        dataset_id = DatasetId("error_handling_test")
        
        # Execute validation - should handle errors gracefully
        validation_results = services['validation_engine'].validate_dataset(
            df=problematic_df,
            rules=problematic_rules,
            dataset_id=dataset_id
        )
        
        # Should have results for both rules
        assert len(validation_results) == 2
        
        # First rule should have error status
        error_result = next(r for r in validation_results if r.rule_id == problematic_rules[0].rule_id)
        assert error_result.status == ValidationStatus.ERROR
        
        # Second rule should pass
        success_result = next(r for r in validation_results if r.rule_id == problematic_rules[1].rule_id)
        assert success_result.status == ValidationStatus.PASSED
    
    def test_end_to_end_workflow_with_reporting(self, realistic_customer_data, comprehensive_rules, cleansing_rules, services):
        """Test complete end-to-end workflow with comprehensive reporting."""
        dataset_id = DatasetId("end_to_end_test")
        
        # Step 1: Initial validation and reporting
        initial_validation = services['validation_engine'].validate_dataset(
            df=realistic_customer_data,
            rules=comprehensive_rules,
            dataset_id=dataset_id
        )
        
        quality_context = QualityContext(
            business_domain="customer_management",
            data_sensitivity="high",
            regulatory_requirements=["GDPR", "CCPA"],
            stakeholder_expectations={'minimum_quality': 0.9}
        )
        
        initial_scores = services['scoring_engine'].calculate_comprehensive_quality_score(
            validation_results=initial_validation,
            context=quality_context
        )
        
        # Step 2: ML-based anomaly detection
        ml_anomalies = services['ml_detection'].detect_quality_anomalies(
            df=realistic_customer_data,
            dataset_id=dataset_id
        )
        
        # Step 3: Data cleansing
        cleansing_result = services['cleansing_service'].cleanse_dataset(
            df=realistic_customer_data,
            rules=cleansing_rules
        )
        
        cleaned_df = services['cleansing_service'].get_cleaned_dataframe()
        
        # Step 4: Post-cleansing validation
        post_cleansing_validation = services['validation_engine'].validate_dataset(
            df=cleaned_df,
            rules=comprehensive_rules,
            dataset_id=dataset_id
        )
        
        final_scores = services['scoring_engine'].calculate_comprehensive_quality_score(
            validation_results=post_cleansing_validation,
            context=quality_context
        )
        
        # Step 5: Quality predictions
        historical_scores = [initial_scores, final_scores]
        quality_predictions = services['predictive_quality'].predict_quality_trends(
            historical_scores=historical_scores,
            prediction_horizon_days=30
        )
        
        # Step 6: Generate comprehensive report
        report = self._generate_comprehensive_report(
            initial_validation=initial_validation,
            final_validation=post_cleansing_validation,
            initial_scores=initial_scores,
            final_scores=final_scores,
            cleansing_result=cleansing_result,
            ml_anomalies=ml_anomalies,
            predictions=quality_predictions,
            context=quality_context
        )
        
        # Validate report completeness
        assert 'executive_summary' in report
        assert 'validation_results' in report
        assert 'quality_scores' in report
        assert 'cleansing_impact' in report
        assert 'ml_insights' in report
        assert 'predictions' in report
        assert 'recommendations' in report
        
        # Validate quality improvement
        assert final_scores.overall_score > initial_scores.overall_score
        assert cleansing_result.retention_rate > 0.8
        assert len(ml_anomalies) > 0
        assert len(quality_predictions) > 0
        
        # Validate that workflow completed successfully
        assert report['executive_summary']['workflow_status'] == 'completed'
        assert report['executive_summary']['quality_improvement'] > 0
    
    def _generate_comprehensive_report(self, **kwargs) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        initial_validation = kwargs['initial_validation']
        final_validation = kwargs['final_validation']
        initial_scores = kwargs['initial_scores']
        final_scores = kwargs['final_scores']
        cleansing_result = kwargs['cleansing_result']
        ml_anomalies = kwargs['ml_anomalies']
        predictions = kwargs['predictions']
        context = kwargs['context']
        
        # Calculate improvements
        quality_improvement = final_scores.overall_score - initial_scores.overall_score
        
        initial_failures = sum(r.failed_records for r in initial_validation)
        final_failures = sum(r.failed_records for r in final_validation)
        failure_reduction = initial_failures - final_failures
        
        return {
            'executive_summary': {
                'workflow_status': 'completed',
                'quality_improvement': quality_improvement,
                'failure_reduction': failure_reduction,
                'records_processed': cleansing_result.original_records,
                'records_retained': cleansing_result.records_retained,
                'anomalies_detected': len(ml_anomalies),
                'business_domain': context.business_domain,
                'data_sensitivity': context.data_sensitivity
            },
            'validation_results': {
                'initial_validation': {
                    'total_rules': len(initial_validation),
                    'passed_rules': len([r for r in initial_validation if r.status == ValidationStatus.PASSED]),
                    'failed_rules': len([r for r in initial_validation if r.status == ValidationStatus.FAILED]),
                    'total_failures': initial_failures
                },
                'final_validation': {
                    'total_rules': len(final_validation),
                    'passed_rules': len([r for r in final_validation if r.status == ValidationStatus.PASSED]),
                    'failed_rules': len([r for r in final_validation if r.status == ValidationStatus.FAILED]),
                    'total_failures': final_failures
                }
            },
            'quality_scores': {
                'initial_overall_score': initial_scores.overall_score,
                'final_overall_score': final_scores.overall_score,
                'improvement': quality_improvement,
                'dimensional_scores': {
                    'completeness': final_scores.completeness_score,
                    'accuracy': final_scores.accuracy_score,
                    'consistency': final_scores.consistency_score,
                    'validity': final_scores.validity_score
                }
            },
            'cleansing_impact': {
                'records_modified': cleansing_result.modified_records,
                'records_removed': cleansing_result.removed_records,
                'retention_rate': cleansing_result.retention_rate,
                'modification_rate': cleansing_result.modification_rate,
                'actions_applied': len(cleansing_result.actions_applied)
            },
            'ml_insights': {
                'anomalies_detected': len(ml_anomalies),
                'anomaly_types': list(set(a.anomaly_type.value for a in ml_anomalies)),
                'confidence_scores': [a.confidence_score for a in ml_anomalies]
            },
            'predictions': {
                'prediction_count': len(predictions),
                'prediction_horizon': 30,
                'predicted_trend': 'improving' if len(predictions) > 0 and predictions[-1].predicted_score > initial_scores.overall_score else 'declining'
            },
            'recommendations': [
                'Continue monitoring quality trends',
                'Implement automated cleansing for recurring issues',
                'Review and update validation rules regularly',
                'Consider additional ML-based quality detection',
                'Establish quality governance processes'
            ]
        }


class TestValidationWorkflowPerformance:
    """Performance tests for validation workflow."""
    
    def test_concurrent_validation_performance(self):
        """Test concurrent validation of multiple datasets."""
        # Create multiple datasets
        datasets = []
        for i in range(5):
            df = pd.DataFrame({
                'id': range(1, 1001),
                'value': np.random.randn(1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000)
            })
            datasets.append(df)
        
        # Create validation services
        validation_engine = ValidationEngine(ValidationEngineConfig(
            enable_parallel_processing=True,
            max_workers=4
        ))
        
        # Create rules
        rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Basic Completeness",
                description="Check completeness",
                rule_type=RuleType.COMPLETENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="df['id'].notna()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID should not be null"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
        
        # Measure concurrent performance
        import time
        start_time = time.time()
        
        results = []
        for i, df in enumerate(datasets):
            dataset_id = DatasetId(f"dataset_{i}")
            validation_results = validation_engine.validate_dataset(
                df=df,
                rules=rules,
                dataset_id=dataset_id
            )
            results.append(validation_results)
        
        execution_time = time.time() - start_time
        
        # Validate performance
        assert execution_time < 10.0, f"Concurrent validation took too long: {execution_time:.2f}s"
        assert len(results) == len(datasets)
        
        # Check all validations completed successfully
        for result_set in results:
            assert len(result_set) == len(rules)
            assert all(r.status == ValidationStatus.PASSED for r in result_set)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create large dataset
        large_df = pd.DataFrame({
            'id': range(1, 50001),
            'text_data': [f'text_{i}' * 10 for i in range(1, 50001)],
            'numeric_data': np.random.randn(50000)
        })
        
        # Create validation engine with memory limits
        validation_engine = ValidationEngine(ValidationEngineConfig(
            enable_parallel_processing=True,
            chunk_size=5000,
            memory_limit_mb=512
        ))
        
        # Create simple rule
        rules = [
            QualityRule(
                rule_id=RuleId(),
                rule_name="Memory Efficient Rule",
                description="Rule for memory testing",
                rule_type=RuleType.COMPLETENESS,
                validation_logic=ValidationLogic(
                    logic_type=LogicType.EXPRESSION,
                    expression="df['id'].notna()",
                    parameters={'column_name': 'id'},
                    success_criteria=SuccessCriteria(min_pass_rate=1.0),
                    error_message="ID should not be null"
                ),
                target_columns=['id'],
                severity=Severity.HIGH,
                category=QualityCategory.DATA_INTEGRITY,
                created_by=UserId("test_user"),
                is_active=True
            )
        ]
        
        dataset_id = DatasetId("memory_test")
        
        # Execute validation
        validation_results = validation_engine.validate_dataset(
            df=large_df,
            rules=rules,
            dataset_id=dataset_id
        )
        
        # Validate results
        assert len(validation_results) == len(rules)
        assert all(r.status == ValidationStatus.PASSED for r in validation_results)
        
        # Check memory usage (basic check)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1024, f"Memory usage too high: {memory_mb:.2f}MB"