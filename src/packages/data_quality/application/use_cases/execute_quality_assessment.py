"""Use case for executing comprehensive data quality assessments."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from ...domain.entities.quality_rule import DatasetId, UserId, RuleId
from ...domain.entities.data_quality_profile import DataQualityProfile
from ...domain.entities.quality_job import QualityJob, QualityJobType, JobStatus, DataSource, QualityJobConfig
from ..services.quality_assessment_service import QualityAssessmentService
from ..services.rule_management_service import RuleManagementService


logger = logging.getLogger(__name__)


class ExecuteQualityAssessmentUseCase:
    """Use case for executing data quality assessments by data stewards."""
    
    def __init__(
        self,
        quality_assessment_service: QualityAssessmentService,
        rule_management_service: RuleManagementService
    ):
        self.quality_assessment_service = quality_assessment_service
        self.rule_management_service = rule_management_service
    
    async def execute_comprehensive_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        assessment_name: Optional[str] = None,
        executed_by: Optional[UserId] = None,
        job_config: Optional[QualityJobConfig] = None
    ) -> DataQualityProfile:
        """Execute a comprehensive quality assessment for a dataset."""
        
        job_name = assessment_name or f"Quality Assessment - {dataset_id.value}"
        logger.info(f"Starting comprehensive quality assessment: {job_name}")
        
        # Create quality job for tracking
        job = QualityJob(
            job_type=QualityJobType.FULL_ASSESSMENT,
            dataset_source=data_source,
            job_config=job_config or QualityJobConfig(),
            job_name=job_name,
            job_description=f"Comprehensive quality assessment for dataset {dataset_id.value}",
            created_by=executed_by or UserId()
        )
        
        try:
            # Start the job
            job.start()
            
            # Execute the assessment
            profile = await self.quality_assessment_service.execute_full_assessment(
                dataset_id=dataset_id,
                data_source=data_source,
                job_config=job_config.__dict__ if job_config else None
            )
            
            # Complete the job with results
            metrics = self._create_job_metrics(profile, job)
            job.complete(profile, metrics)
            
            logger.info(f"Assessment completed. Overall score: {profile.quality_scores.overall_score:.2f}")
            
            # Log critical issues if any
            critical_issues = profile.get_critical_issues()
            if critical_issues:
                logger.warning(f"Found {len(critical_issues)} critical quality issues")
            
            return profile
            
        except Exception as e:
            job.fail(f"Assessment failed: {str(e)}")
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    async def execute_rule_specific_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        rule_ids: List[RuleId],
        assessment_name: Optional[str] = None,
        executed_by: Optional[UserId] = None
    ) -> DataQualityProfile:
        """Execute quality assessment with specific rules only."""
        
        job_name = assessment_name or f"Rule-Specific Assessment - {len(rule_ids)} rules"
        logger.info(f"Starting rule-specific assessment: {job_name}")
        
        # Validate that all rules exist and are active
        valid_rules = []
        for rule_id in rule_ids:
            rule = await self.rule_management_service.rule_repository.get_by_id(rule_id)
            if rule and rule.is_active():
                valid_rules.append(rule)
            else:
                logger.warning(f"Rule {rule_id.value} not found or inactive")
        
        if not valid_rules:
            raise ValueError("No valid active rules found for assessment")
        
        # Create quality job
        job = QualityJob(
            job_type=QualityJobType.RULE_VALIDATION,
            dataset_source=data_source,
            rules_applied=rule_ids,
            job_name=job_name,
            job_description=f"Rule-specific assessment with {len(valid_rules)} rules",
            created_by=executed_by or UserId()
        )
        
        try:
            job.start()
            
            # Execute incremental assessment
            profile = await self.quality_assessment_service.execute_incremental_assessment(
                dataset_id=dataset_id,
                data_source=data_source,
                rule_ids=rule_ids
            )
            
            # Complete the job
            metrics = self._create_job_metrics(profile, job)
            job.complete(profile, metrics)
            
            logger.info(f"Rule-specific assessment completed with {len(valid_rules)} rules")
            return profile
            
        except Exception as e:
            job.fail(f"Rule-specific assessment failed: {str(e)}")
            logger.error(f"Rule-specific assessment failed: {str(e)}")
            raise
    
    async def execute_scheduled_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        executed_by: Optional[UserId] = None
    ) -> DataQualityProfile:
        """Execute assessment using only scheduled rules."""
        
        logger.info(f"Starting scheduled assessment for dataset {dataset_id.value}")
        
        # Get all scheduled rules that are due for execution
        scheduled_rules = await self.rule_management_service.get_rules_due_for_execution()
        
        # Filter rules applicable to this dataset
        applicable_rules = []
        for rule in scheduled_rules:
            if not rule.target_datasets or dataset_id in rule.target_datasets:
                applicable_rules.append(rule)
        
        if not applicable_rules:
            logger.info("No scheduled rules due for execution")
            # Return empty assessment
            return await self._create_empty_assessment_result(dataset_id, executed_by)
        
        # Execute assessment with scheduled rules
        rule_ids = [rule.rule_id for rule in applicable_rules]
        return await self.execute_rule_specific_assessment(
            dataset_id=dataset_id,
            data_source=data_source,
            rule_ids=rule_ids,
            assessment_name=f"Scheduled Assessment - {len(rule_ids)} rules",
            executed_by=executed_by
        )
    
    async def execute_monitoring_check(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        critical_rules_only: bool = True,
        executed_by: Optional[UserId] = None
    ) -> DataQualityProfile:
        """Execute a quick monitoring check with critical rules only."""
        
        logger.info(f"Starting monitoring check for dataset {dataset_id.value}")
        
        # Get applicable rules
        all_rules = await self.rule_management_service.get_rules_by_dataset(dataset_id)
        
        # Filter for critical rules if specified
        if critical_rules_only:
            critical_rules = [rule for rule in all_rules if rule.severity.value == "critical"]
            rules_to_execute = critical_rules
        else:
            rules_to_execute = all_rules
        
        if not rules_to_execute:
            logger.info("No critical rules found for monitoring check")
            return await self._create_empty_assessment_result(dataset_id, executed_by)
        
        # Create lightweight job config for monitoring
        monitoring_config = QualityJobConfig(
            max_execution_time_minutes=10,  # Short timeout for monitoring
            enable_sampling=True,
            sample_percentage=0.1,  # Sample 10% for quick check
            store_detailed_results=False,
            generate_report=False,
            send_notifications=True
        )
        
        return await self.execute_rule_specific_assessment(
            dataset_id=dataset_id,
            data_source=data_source,
            rule_ids=[rule.rule_id for rule in rules_to_execute],
            assessment_name=f"Monitoring Check - {len(rules_to_execute)} critical rules",
            executed_by=executed_by
        )
    
    async def execute_incremental_assessment(
        self,
        dataset_id: DatasetId,
        data_source: DataSource,
        previous_profile: DataQualityProfile,
        executed_by: Optional[UserId] = None
    ) -> DataQualityProfile:
        """Execute incremental assessment comparing against previous results."""
        
        logger.info(f"Starting incremental assessment for dataset {dataset_id.value}")
        
        # Execute new assessment
        current_profile = await self.execute_comprehensive_assessment(
            dataset_id=dataset_id,
            data_source=data_source,
            assessment_name=f"Incremental Assessment - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            executed_by=executed_by
        )
        
        # Compare with previous profile
        comparison_results = self._compare_profiles(previous_profile, current_profile)
        
        # Add comparison metadata to current profile
        current_profile.quality_trends = self._calculate_quality_trends(
            previous_profile.quality_scores,
            current_profile.quality_scores
        )
        
        logger.info(f"Incremental assessment completed. Score change: {comparison_results['score_change']:.3f}")
        
        return current_profile
    
    async def validate_data_source_connectivity(self, data_source: DataSource) -> bool:
        """Validate that the data source is accessible and contains data."""
        
        logger.info(f"Validating connectivity to {data_source.source_type} data source")
        
        try:
            # Attempt to load a small sample of data
            sample_data = await self.quality_assessment_service._load_data_from_source(data_source)
            
            if not sample_data:
                logger.warning("Data source is accessible but contains no data")
                return False
            
            logger.info(f"Data source validation successful. Found {len(sample_data)} sample records")
            return True
            
        except Exception as e:
            logger.error(f"Data source validation failed: {str(e)}")
            return False
    
    async def get_assessment_recommendations(
        self,
        dataset_id: DatasetId,
        include_rule_suggestions: bool = True
    ) -> Dict[str, Any]:
        """Get recommendations for improving data quality assessment."""
        
        # Get existing rules for the dataset
        existing_rules = await self.rule_management_service.get_rules_by_dataset(dataset_id)
        
        # Get rule statistics
        rule_stats = await self.rule_management_service.get_rule_statistics()
        
        recommendations = {
            "dataset_id": str(dataset_id.value),
            "current_rule_count": len(existing_rules),
            "rule_coverage": self._analyze_rule_coverage(existing_rules),
            "recommendations": []
        }
        
        # Analyze rule coverage gaps
        coverage_gaps = self._identify_coverage_gaps(existing_rules)
        
        for gap in coverage_gaps:
            recommendations["recommendations"].append({
                "type": "coverage_gap",
                "priority": "medium",
                "description": f"Consider adding {gap} validation rules",
                "action": f"Create {gap} rules to improve quality coverage"
            })
        
        # Check for missing critical validations
        if not any(rule.rule_type.value == "completeness" for rule in existing_rules):
            recommendations["recommendations"].append({
                "type": "missing_critical",
                "priority": "high",
                "description": "No completeness validation rules found",
                "action": "Add completeness rules for critical fields"
            })
        
        # Check for outdated rules
        outdated_rules = [rule for rule in existing_rules if rule.version == 1 and 
                         (datetime.utcnow() - rule.created_at).days > 365]
        
        if outdated_rules:
            recommendations["recommendations"].append({
                "type": "outdated_rules",
                "priority": "low",
                "description": f"{len(outdated_rules)} rules haven't been updated in over a year",
                "action": "Review and update outdated validation rules"
            })
        
        # Suggest rule templates if enabled
        if include_rule_suggestions:
            from .define_quality_rules import DefineQualityRulesUseCase
            define_rules_uc = DefineQualityRulesUseCase(self.rule_management_service)
            templates = await define_rules_uc.get_rule_templates()
            
            recommendations["suggested_templates"] = templates[:5]  # Top 5 templates
        
        return recommendations
    
    def _create_job_metrics(self, profile: DataQualityProfile, job: QualityJob):
        """Create job metrics from assessment results."""
        from ...domain.entities.quality_job import JobMetrics
        
        return JobMetrics(
            execution_time_seconds=job.get_duration_seconds() or 0.0,
            records_processed=profile.total_records_assessed,
            rules_executed=profile.rules_executed,
            rules_passed=len([r for r in profile.validation_results if r.status == "passed"]),
            rules_failed=len([r for r in profile.validation_results if r.status == "failed"]),
            rules_with_errors=len([r for r in profile.validation_results if r.status == "error"]),
            overall_pass_rate=profile.get_overall_pass_rate(),
            critical_issues_found=len(profile.get_critical_issues()),
            total_issues_found=len(profile.quality_issues)
        )
    
    async def _create_empty_assessment_result(
        self,
        dataset_id: DatasetId,
        executed_by: Optional[UserId]
    ) -> DataQualityProfile:
        """Create an empty assessment result when no rules are applicable."""
        
        from ...domain.value_objects.quality_scores import QualityScores
        
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
            created_by=executed_by
        )
    
    def _compare_profiles(
        self,
        previous: DataQualityProfile,
        current: DataQualityProfile
    ) -> Dict[str, Any]:
        """Compare two quality profiles and return differences."""
        
        return {
            "score_change": current.quality_scores.overall_score - previous.quality_scores.overall_score,
            "dimension_changes": current.quality_scores.compare_with(previous.quality_scores),
            "new_issues": len(current.quality_issues) - len(previous.quality_issues),
            "resolution_rate": self._calculate_resolution_rate(previous, current),
            "trend": "improving" if current.quality_scores.overall_score > previous.quality_scores.overall_score else "declining"
        }
    
    def _calculate_quality_trends(self, previous_scores, current_scores):
        """Calculate quality trends from score comparison."""
        from ...domain.entities.data_quality_profile import QualityTrends
        
        trends = QualityTrends(
            current_period_score=current_scores.overall_score,
            previous_period_score=previous_scores.overall_score,
            periods_analyzed=2
        )
        
        trends.calculate_trend()
        return trends
    
    def _calculate_resolution_rate(
        self,
        previous: DataQualityProfile,
        current: DataQualityProfile
    ) -> float:
        """Calculate the rate of issue resolution between assessments."""
        
        if not previous.quality_issues:
            return 1.0
        
        previous_issue_count = len(previous.quality_issues)
        current_issue_count = len(current.quality_issues)
        
        if current_issue_count >= previous_issue_count:
            return 0.0
        
        return (previous_issue_count - current_issue_count) / previous_issue_count
    
    def _analyze_rule_coverage(self, rules: List) -> Dict[str, int]:
        """Analyze rule coverage by type and category."""
        coverage = {
            "by_type": {},
            "by_category": {},
            "total_rules": len(rules)
        }
        
        for rule in rules:
            rule_type = rule.rule_type.value
            category = rule.category.value
            
            coverage["by_type"][rule_type] = coverage["by_type"].get(rule_type, 0) + 1
            coverage["by_category"][category] = coverage["by_category"].get(category, 0) + 1
        
        return coverage
    
    def _identify_coverage_gaps(self, existing_rules: List) -> List[str]:
        """Identify gaps in rule coverage."""
        existing_types = {rule.rule_type.value for rule in existing_rules}
        
        essential_types = {
            "completeness", "validity", "uniqueness", "accuracy"
        }
        
        gaps = essential_types - existing_types
        return list(gaps)