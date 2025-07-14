"""Quality Assessment Service for orchestrating data quality evaluations."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from ...domain.entities.quality_rule import QualityRule, ValidationResult, DatasetId, RuleId
from ...domain.entities.data_quality_profile import DataQualityProfile, QualityIssue, QualityIssueType, IssueStatus
from ...domain.entities.quality_job import QualityJob, QualityJobType, JobStatus, JobMetrics, DataSource
from ...domain.repositories.quality_rule_repository import QualityRuleRepository
from ...domain.services.validation_engine import ValidationEngine
from ...domain.value_objects.quality_scores import QualityScores, ScoringMethod
from ...domain.value_objects.business_impact import BusinessImpact, ImpactLevel


logger = logging.getLogger(__name__)


class QualityAssessmentService:
    """Service for orchestrating comprehensive data quality assessments."""
    
    def __init__(
        self,
        rule_repository: QualityRuleRepository,
        validation_engine: ValidationEngine
    ):
        self.rule_repository = rule_repository
        self.validation_engine = validation_engine
    
    async def execute_full_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        job_config: Optional[Dict[str, Any]] = None
    ) -> DataQualityProfile:
        """Execute a comprehensive quality assessment for a dataset."""
        logger.info(f"Starting full quality assessment for dataset {dataset_id.value}")
        
        try:
            # Get all active rules for the dataset
            rules = await self._get_applicable_rules(dataset_id)
            
            if not rules:
                logger.warning(f"No active rules found for dataset {dataset_id.value}")
                return self._create_empty_profile(dataset_id)
            
            # Execute validation rules
            validation_results = await self._execute_validation_rules(rules, data_source)
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(validation_results)
            
            # Generate quality issues
            quality_issues = self._generate_quality_issues(validation_results)
            
            # Create quality profile
            profile = DataQualityProfile(
                dataset_id=dataset_id,
                quality_scores=quality_scores,
                validation_results=validation_results,
                quality_issues=quality_issues,
                rules_executed=len(rules),
                total_records_assessed=self._calculate_total_records(validation_results),
                created_by=None  # TODO: Get from context
            )
            
            logger.info(f"Quality assessment completed. Overall score: {quality_scores.overall_score:.2f}")
            return profile
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    async def execute_incremental_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        rule_ids: Optional[List[RuleId]] = None
    ) -> DataQualityProfile:
        """Execute incremental quality assessment with specific rules."""
        logger.info(f"Starting incremental quality assessment for dataset {dataset_id.value}")
        
        if rule_ids:
            rules = []
            for rule_id in rule_ids:
                rule = await self.rule_repository.get_by_id(rule_id)
                if rule and rule.is_active():
                    rules.append(rule)
        else:
            rules = await self._get_applicable_rules(dataset_id)
        
        validation_results = await self._execute_validation_rules(rules, data_source)
        quality_scores = self._calculate_quality_scores(validation_results)
        quality_issues = self._generate_quality_issues(validation_results)
        
        return DataQualityProfile(
            dataset_id=dataset_id,
            quality_scores=quality_scores,
            validation_results=validation_results,
            quality_issues=quality_issues,
            rules_executed=len(rules),
            total_records_assessed=self._calculate_total_records(validation_results),
            created_by=None  # TODO: Get from context
        )
    
    async def validate_single_rule(
        self,
        rule: QualityRule,
        data_source: DataSource
    ) -> ValidationResult:
        """Validate a single rule against a data source."""
        logger.info(f"Validating rule {rule.rule_id.value}")
        
        # Convert rule to validation engine format
        records = await self._load_data_from_source(data_source)
        
        # Execute validation
        validation_result = await self._execute_single_rule(rule, records)
        
        # Update rule with result
        rule.add_validation_result(validation_result)
        await self.rule_repository.save(rule)
        
        return validation_result
    
    async def _get_applicable_rules(self, dataset_id: DatasetId) -> List[QualityRule]:
        """Get all rules applicable to a dataset."""
        # Get rules specifically for this dataset
        dataset_rules = await self.rule_repository.get_by_dataset_id(dataset_id)
        
        # Get global active rules (rules that apply to all datasets)
        active_rules = await self.rule_repository.get_active_rules()
        global_rules = [rule for rule in active_rules if not rule.target_datasets]
        
        # Combine and deduplicate
        all_rules = dataset_rules + global_rules
        seen_rule_ids = set()
        unique_rules = []
        
        for rule in all_rules:
            if rule.rule_id.value not in seen_rule_ids:
                unique_rules.append(rule)
                seen_rule_ids.add(rule.rule_id.value)
        
        return unique_rules
    
    async def _execute_validation_rules(
        self,
        rules: List[QualityRule],
        data_source: DataSource
    ) -> List[ValidationResult]:
        """Execute multiple validation rules."""
        results = []
        
        # Load data once for all rules
        records = await self._load_data_from_source(data_source)
        
        # Execute rules in parallel (configurable)
        tasks = [self._execute_single_rule(rule, records) for rule in rules]
        validation_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"Rule {rules[i].rule_id.value} failed: {str(result)}")
                # Create error result
                error_result = ValidationResult(
                    rule_id=rules[i].rule_id,
                    dataset_id=rules[i].target_datasets[0] if rules[i].target_datasets else None,
                    status="error",
                    total_records=len(records),
                    records_passed=0,
                    records_failed=len(records),
                    pass_rate=0.0,
                    execution_time_seconds=0.0
                )
                results.append(error_result)
            else:
                results.append(result)
        
        return results
    
    async def _execute_single_rule(
        self,
        rule: QualityRule,
        records: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Execute a single validation rule."""
        start_time = datetime.utcnow()
        
        try:
            # Convert domain rule to validation engine rule
            engine_rule = self._convert_to_engine_rule(rule)
            
            # Execute validation
            engine_results = self.validation_engine.run([engine_rule], records)
            engine_result = engine_results[0] if engine_results else None
            
            # Calculate metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            total_records = len(records)
            
            if engine_result:
                records_failed = engine_result.failed_records
                records_passed = total_records - records_failed
                pass_rate = records_passed / total_records if total_records > 0 else 1.0
                status = "passed" if engine_result.passed else "failed"
            else:
                records_failed = total_records
                records_passed = 0
                pass_rate = 0.0
                status = "error"
            
            return ValidationResult(
                rule_id=rule.rule_id,
                dataset_id=rule.target_datasets[0] if rule.target_datasets else None,
                status=status,
                total_records=total_records,
                records_passed=records_passed,
                records_failed=records_failed,
                pass_rate=pass_rate,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error executing rule {rule.rule_id.value}: {str(e)}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ValidationResult(
                rule_id=rule.rule_id,
                dataset_id=rule.target_datasets[0] if rule.target_datasets else None,
                status="error",
                total_records=len(records),
                records_passed=0,
                records_failed=len(records),
                pass_rate=0.0,
                execution_time_seconds=execution_time
            )
    
    def _convert_to_engine_rule(self, rule: QualityRule):
        """Convert domain rule to validation engine rule."""
        # This is a simplified conversion - in practice, this would be more sophisticated
        from ...domain.services.validation_engine import CompletenessRule, RangeRule, FormatRule
        
        logic = rule.validation_logic
        
        if rule.rule_type.value == "completeness":
            return CompletenessRule(
                rule_id=str(rule.rule_id.value),
                field=rule.target_columns[0] if rule.target_columns else "id",
                description=rule.metadata.description
            )
        elif rule.rule_type.value == "validity" and logic.logic_type.value == "range":
            params = logic.parameters
            return RangeRule(
                rule_id=str(rule.rule_id.value),
                field=rule.target_columns[0] if rule.target_columns else "value",
                min_value=params.get("min_value"),
                max_value=params.get("max_value"),
                description=rule.metadata.description
            )
        elif rule.rule_type.value == "validity" and logic.logic_type.value == "regex":
            return FormatRule(
                rule_id=str(rule.rule_id.value),
                field=rule.target_columns[0] if rule.target_columns else "value",
                pattern=logic.expression,
                description=rule.metadata.description
            )
        else:
            # Default to completeness rule for unsupported types
            return CompletenessRule(
                rule_id=str(rule.rule_id.value),
                field=rule.target_columns[0] if rule.target_columns else "id",
                description=rule.metadata.description
            )
    
    async def _load_data_from_source(self, data_source: DataSource) -> List[Dict[str, Any]]:
        """Load data from the specified data source."""
        # This is a placeholder implementation
        # In practice, this would connect to various data sources
        
        if data_source.source_type == "file":
            # Load from file
            return self._load_from_file(data_source.file_path)
        elif data_source.source_type == "database":
            # Load from database
            return await self._load_from_database(data_source)
        else:
            # Return sample data for testing
            return [
                {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"},
                {"id": 3, "name": "", "age": 35, "email": "invalid-email"},
                {"id": 4, "name": "Bob Johnson", "age": -5, "email": "bob@example.com"}
            ]
    
    def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a file."""
        # Placeholder implementation
        return []
    
    async def _load_from_database(self, data_source: DataSource) -> List[Dict[str, Any]]:
        """Load data from a database."""
        # Placeholder implementation
        return []
    
    def _calculate_quality_scores(self, validation_results: List[ValidationResult]) -> QualityScores:
        """Calculate composite quality scores from validation results."""
        if not validation_results:
            return QualityScores(
                overall_score=1.0,
                completeness_score=1.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                validity_score=1.0,
                uniqueness_score=1.0,
                timeliness_score=1.0
            )
        
        # Group results by rule type
        score_by_type = {}
        
        for result in validation_results:
            # Get rule to determine type (simplified)
            rule_type = "validity"  # Default
            
            if rule_type not in score_by_type:
                score_by_type[rule_type] = []
            
            score_by_type[rule_type].append(result.pass_rate)
        
        # Calculate average score for each dimension
        completeness_score = self._average_scores(score_by_type.get("completeness", [1.0]))
        accuracy_score = self._average_scores(score_by_type.get("accuracy", [1.0]))
        consistency_score = self._average_scores(score_by_type.get("consistency", [1.0]))
        validity_score = self._average_scores(score_by_type.get("validity", [1.0]))
        uniqueness_score = self._average_scores(score_by_type.get("uniqueness", [1.0]))
        timeliness_score = self._average_scores(score_by_type.get("timeliness", [1.0]))
        
        # Calculate overall weighted score
        scores = QualityScores(
            overall_score=0.0,  # Will be calculated
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            timeliness_score=timeliness_score,
            scoring_method=ScoringMethod.WEIGHTED_AVERAGE
        )
        
        # Calculate weighted overall score
        overall_score = scores.get_weighted_score()
        
        return QualityScores(
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            timeliness_score=timeliness_score,
            scoring_method=ScoringMethod.WEIGHTED_AVERAGE
        )
    
    def _average_scores(self, scores: List[float]) -> float:
        """Calculate average of scores."""
        if not scores:
            return 1.0
        return sum(scores) / len(scores)
    
    def _generate_quality_issues(self, validation_results: List[ValidationResult]) -> List[QualityIssue]:
        """Generate quality issues from validation results."""
        issues = []
        
        for result in validation_results:
            if result.status == "failed" and result.records_failed > 0:
                # Determine issue type based on rule
                issue_type = self._determine_issue_type(result)
                
                # Determine severity based on failure rate
                if result.pass_rate < 0.5:
                    severity = "critical"
                elif result.pass_rate < 0.7:
                    severity = "high"
                elif result.pass_rate < 0.9:
                    severity = "medium"
                else:
                    severity = "low"
                
                issue = QualityIssue(
                    issue_type=issue_type,
                    severity=severity,
                    description=f"Rule validation failed with {result.records_failed} failed records out of {result.total_records}",
                    affected_records=result.records_failed,
                    affected_columns=[],  # TODO: Extract from rule
                    status=IssueStatus.OPEN,
                    source_rule_id=str(result.rule_id.value),
                    source_validation_id=result.validation_id
                )
                
                issues.append(issue)
        
        return issues
    
    def _determine_issue_type(self, result: ValidationResult) -> QualityIssueType:
        """Determine issue type from validation result."""
        # Simplified mapping - in practice, this would analyze the rule type
        return QualityIssueType.VALIDITY_VIOLATION
    
    def _calculate_total_records(self, validation_results: List[ValidationResult]) -> int:
        """Calculate total records assessed."""
        if not validation_results:
            return 0
        return max(result.total_records for result in validation_results)
    
    def _create_empty_profile(self, dataset_id: DatasetId) -> DataQualityProfile:
        """Create an empty quality profile when no rules are found."""
        return DataQualityProfile(
            dataset_id=dataset_id,
            quality_scores=QualityScores(
                overall_score=1.0,
                completeness_score=1.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                validity_score=1.0,
                uniqueness_score=1.0,
                timeliness_score=1.0
            ),
            validation_results=[],
            quality_issues=[],
            rules_executed=0,
            total_records_assessed=0,
            created_by=None  # TODO: Get from context
        )