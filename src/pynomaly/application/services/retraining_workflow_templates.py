"""
Retraining Workflow Templates

This module provides predefined workflow templates for common automated retraining scenarios
as part of Issue #9 (A-001) implementation.
"""

from typing import Dict, List, Optional, Any
from datetime import timedelta
from dataclasses import dataclass

from .automated_retraining_orchestrator import WorkflowConfig, WorkflowType


@dataclass
class WorkflowTemplate:
    """Template for creating retraining workflows"""
    name: str
    description: str
    workflow_type: WorkflowType
    default_config: WorkflowConfig
    use_cases: List[str]
    prerequisites: List[str]
    estimated_duration: str


class RetrainingWorkflowTemplates:
    """Collection of predefined retraining workflow templates"""
    
    @staticmethod
    def performance_degradation_template() -> WorkflowTemplate:
        """Template for performance degradation triggered retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=[],  # To be filled by user
            monitoring_interval_minutes=15,
            accuracy_threshold=0.05,
            f1_score_threshold=0.03,
            precision_threshold=0.03,
            recall_threshold=0.03,
            max_concurrent_retraining=2,
            validation_dataset_size=1000,
            champion_challenger_duration_hours=24,
            auto_rollback_enabled=True,
            rollback_threshold=0.02,
            business_rules={
                "max_data_age": 90,
                "min_samples": 1000,
                "max_training_time": 6,
                "improvement_threshold": 0.02
            }
        )
        
        return WorkflowTemplate(
            name="Performance Degradation Response",
            description="Automatically retrain models when performance degrades below thresholds",
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            default_config=config,
            use_cases=[
                "Production model performance monitoring",
                "Real-time anomaly detection systems",
                "Critical business applications",
                "High-frequency trading models"
            ],
            prerequisites=[
                "Performance baseline established",
                "Monitoring infrastructure configured",
                "Training data pipeline available",
                "Validation dataset prepared"
            ],
            estimated_duration="2-6 hours"
        )
    
    @staticmethod
    def scheduled_maintenance_template() -> WorkflowTemplate:
        """Template for scheduled maintenance retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.SCHEDULED_MAINTENANCE,
            model_ids=[],
            monitoring_interval_minutes=60,
            schedule_cron="0 2 * * 0",  # Weekly at 2 AM on Sunday
            max_concurrent_retraining=3,
            validation_dataset_size=1500,
            champion_challenger_duration_hours=72,
            auto_rollback_enabled=True,
            rollback_threshold=0.01,
            business_rules={
                "max_data_age": 30,
                "min_samples": 2000,
                "max_training_time": 8,
                "improvement_threshold": 0.01,
                "maintenance_window": "weekend"
            }
        )
        
        return WorkflowTemplate(
            name="Scheduled Maintenance Retraining",
            description="Regularly scheduled retraining to maintain model freshness",
            workflow_type=WorkflowType.SCHEDULED_MAINTENANCE,
            default_config=config,
            use_cases=[
                "Batch processing systems",
                "Weekly/monthly model updates",
                "Seasonal adjustment models",
                "Compliance-driven retraining"
            ],
            prerequisites=[
                "Scheduled maintenance windows defined",
                "Historical data accumulation",
                "Automated data pipeline",
                "Quality assurance process"
            ],
            estimated_duration="4-8 hours"
        )
    
    @staticmethod
    def data_drift_response_template() -> WorkflowTemplate:
        """Template for data drift triggered retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.DATA_DRIFT_RESPONSE,
            model_ids=[],
            monitoring_interval_minutes=30,
            drift_threshold=0.1,
            max_concurrent_retraining=2,
            validation_dataset_size=1200,
            champion_challenger_duration_hours=48,
            auto_rollback_enabled=True,
            rollback_threshold=0.015,
            business_rules={
                "max_data_age": 60,
                "min_samples": 1500,
                "max_training_time": 4,
                "improvement_threshold": 0.015,
                "drift_detection_window": 7,
                "statistical_tests": ["ks_test", "psi"]
            }
        )
        
        return WorkflowTemplate(
            name="Data Drift Response",
            description="Retrain models when significant data drift is detected",
            workflow_type=WorkflowType.DATA_DRIFT_RESPONSE,
            default_config=config,
            use_cases=[
                "Customer behavior models",
                "Market prediction models",
                "Fraud detection systems",
                "Dynamic pricing models"
            ],
            prerequisites=[
                "Data drift detection configured",
                "Reference dataset established",
                "Statistical test thresholds tuned",
                "Drift monitoring dashboard"
            ],
            estimated_duration="3-5 hours"
        )
    
    @staticmethod
    def concept_drift_response_template() -> WorkflowTemplate:
        """Template for concept drift triggered retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.CONCEPT_DRIFT_RESPONSE,
            model_ids=[],
            monitoring_interval_minutes=60,
            concept_drift_threshold=0.15,
            max_concurrent_retraining=1,
            validation_dataset_size=2000,
            champion_challenger_duration_hours=96,
            auto_rollback_enabled=True,
            rollback_threshold=0.01,
            business_rules={
                "max_data_age": 45,
                "min_samples": 3000,
                "max_training_time": 10,
                "improvement_threshold": 0.01,
                "concept_drift_window": 14,
                "adaptation_strategy": "gradual"
            }
        )
        
        return WorkflowTemplate(
            name="Concept Drift Response",
            description="Retrain models when underlying relationships change",
            workflow_type=WorkflowType.CONCEPT_DRIFT_RESPONSE,
            default_config=config,
            use_cases=[
                "Long-term prediction models",
                "Economic forecasting",
                "Medical diagnosis systems",
                "Climate prediction models"
            ],
            prerequisites=[
                "Concept drift detection implemented",
                "Long-term data history",
                "Domain expertise validation",
                "Gradual adaptation framework"
            ],
            estimated_duration="6-10 hours"
        )
    
    @staticmethod
    def business_rule_triggered_template() -> WorkflowTemplate:
        """Template for business rule triggered retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.BUSINESS_RULE_TRIGGERED,
            model_ids=[],
            monitoring_interval_minutes=120,
            max_concurrent_retraining=2,
            validation_dataset_size=1000,
            champion_challenger_duration_hours=24,
            auto_rollback_enabled=True,
            rollback_threshold=0.02,
            business_rules={
                "max_data_age": 30,
                "min_samples": 1000,
                "max_training_time": 6,
                "improvement_threshold": 0.02,
                "business_metrics": {
                    "revenue_impact_threshold": 0.05,
                    "customer_satisfaction_threshold": 0.03,
                    "error_rate_threshold": 0.02
                },
                "approval_required": True,
                "stakeholder_notification": True
            }
        )
        
        return WorkflowTemplate(
            name="Business Rule Triggered",
            description="Retrain models based on business-specific conditions",
            workflow_type=WorkflowType.BUSINESS_RULE_TRIGGERED,
            default_config=config,
            use_cases=[
                "Revenue optimization models",
                "Customer churn prevention",
                "Inventory management",
                "Risk assessment models"
            ],
            prerequisites=[
                "Business rules defined",
                "Stakeholder approval process",
                "Business metrics tracking",
                "Impact assessment framework"
            ],
            estimated_duration="3-6 hours"
        )
    
    @staticmethod
    def feedback_accumulation_template() -> WorkflowTemplate:
        """Template for feedback accumulation triggered retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.FEEDBACK_ACCUMULATION,
            model_ids=[],
            monitoring_interval_minutes=240,
            max_concurrent_retraining=1,
            validation_dataset_size=1500,
            champion_challenger_duration_hours=48,
            auto_rollback_enabled=True,
            rollback_threshold=0.01,
            business_rules={
                "max_data_age": 60,
                "min_samples": 2000,
                "max_training_time": 8,
                "improvement_threshold": 0.01,
                "feedback_threshold": 100,
                "feedback_quality_threshold": 0.8,
                "feedback_types": ["explicit", "implicit", "correction"],
                "feedback_weighting": {
                    "explicit": 1.0,
                    "implicit": 0.5,
                    "correction": 1.5
                }
            }
        )
        
        return WorkflowTemplate(
            name="Feedback Accumulation",
            description="Retrain models when sufficient user feedback accumulates",
            workflow_type=WorkflowType.FEEDBACK_ACCUMULATION,
            default_config=config,
            use_cases=[
                "Recommendation systems",
                "Search ranking models",
                "Content classification",
                "Personalization engines"
            ],
            prerequisites=[
                "Feedback collection system",
                "Feedback quality assessment",
                "User interaction tracking",
                "Feedback incorporation mechanism"
            ],
            estimated_duration="4-8 hours"
        )
    
    @staticmethod
    def multi_model_coordination_template() -> WorkflowTemplate:
        """Template for coordinated multi-model retraining"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.MULTI_MODEL_COORDINATION,
            model_ids=[],
            monitoring_interval_minutes=60,
            max_concurrent_retraining=1,  # Sequential for dependencies
            validation_dataset_size=2000,
            champion_challenger_duration_hours=72,
            auto_rollback_enabled=True,
            rollback_threshold=0.01,
            business_rules={
                "max_data_age": 30,
                "min_samples": 2000,
                "max_training_time": 12,
                "improvement_threshold": 0.01,
                "coordination_strategy": "sequential",
                "dependency_order": [],  # To be defined
                "cross_model_validation": True,
                "ensemble_performance_check": True
            }
        )
        
        return WorkflowTemplate(
            name="Multi-Model Coordination",
            description="Coordinate retraining across multiple interdependent models",
            workflow_type=WorkflowType.MULTI_MODEL_COORDINATION,
            default_config=config,
            use_cases=[
                "Model ensembles",
                "Multi-stage pipelines",
                "Hierarchical models",
                "Dependent model systems"
            ],
            prerequisites=[
                "Model dependencies mapped",
                "Cross-model validation framework",
                "Ensemble performance metrics",
                "Coordinated deployment strategy"
            ],
            estimated_duration="8-12 hours"
        )
    
    @staticmethod
    def get_all_templates() -> List[WorkflowTemplate]:
        """Get all available workflow templates"""
        return [
            RetrainingWorkflowTemplates.performance_degradation_template(),
            RetrainingWorkflowTemplates.scheduled_maintenance_template(),
            RetrainingWorkflowTemplates.data_drift_response_template(),
            RetrainingWorkflowTemplates.concept_drift_response_template(),
            RetrainingWorkflowTemplates.business_rule_triggered_template(),
            RetrainingWorkflowTemplates.feedback_accumulation_template(),
            RetrainingWorkflowTemplates.multi_model_coordination_template()
        ]
    
    @staticmethod
    def get_template_by_name(name: str) -> Optional[WorkflowTemplate]:
        """Get a specific template by name"""
        templates = RetrainingWorkflowTemplates.get_all_templates()
        for template in templates:
            if template.name == name:
                return template
        return None
    
    @staticmethod
    def get_templates_by_use_case(use_case: str) -> List[WorkflowTemplate]:
        """Get templates that match a specific use case"""
        templates = RetrainingWorkflowTemplates.get_all_templates()
        matching_templates = []
        
        for template in templates:
            for template_use_case in template.use_cases:
                if use_case.lower() in template_use_case.lower():
                    matching_templates.append(template)
                    break
        
        return matching_templates
    
    @staticmethod
    def customize_template(
        template: WorkflowTemplate,
        model_ids: List[str],
        custom_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowConfig:
        """Customize a template for specific models and configuration"""
        config = template.default_config
        config.model_ids = model_ids
        
        if custom_config:
            # Apply custom configuration overrides
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                elif key == "business_rules":
                    config.business_rules.update(value)
        
        return config


# Example usage and configuration helpers
class WorkflowConfigurationHelper:
    """Helper class for workflow configuration"""
    
    @staticmethod
    def validate_configuration(config: WorkflowConfig) -> List[str]:
        """Validate workflow configuration and return issues"""
        issues = []
        
        if not config.model_ids:
            issues.append("Model IDs cannot be empty")
        
        if config.monitoring_interval_minutes < 1:
            issues.append("Monitoring interval must be at least 1 minute")
        
        if config.accuracy_threshold < 0 or config.accuracy_threshold > 1:
            issues.append("Accuracy threshold must be between 0 and 1")
        
        if config.max_concurrent_retraining < 1:
            issues.append("Max concurrent retraining must be at least 1")
        
        if config.validation_dataset_size < 100:
            issues.append("Validation dataset size should be at least 100")
        
        return issues
    
    @staticmethod
    def estimate_resource_requirements(config: WorkflowConfig) -> Dict[str, Any]:
        """Estimate resource requirements for workflow"""
        num_models = len(config.model_ids)
        concurrent_training = min(config.max_concurrent_retraining, num_models)
        
        # Rough estimates based on configuration
        estimated_cpu_hours = num_models * config.business_rules.get("max_training_time", 6)
        estimated_memory_gb = concurrent_training * 8  # 8GB per concurrent training
        estimated_storage_gb = num_models * 10  # 10GB per model
        
        return {
            "estimated_cpu_hours": estimated_cpu_hours,
            "estimated_memory_gb": estimated_memory_gb,
            "estimated_storage_gb": estimated_storage_gb,
            "concurrent_training_slots": concurrent_training,
            "estimated_duration_hours": estimated_cpu_hours / concurrent_training
        }
    
    @staticmethod
    def generate_configuration_summary(config: WorkflowConfig) -> str:
        """Generate a human-readable configuration summary"""
        summary = f"""
Workflow Configuration Summary:
- Type: {config.workflow_type.value}
- Models: {len(config.model_ids)} model(s)
- Monitoring: Every {config.monitoring_interval_minutes} minutes
- Concurrent Retraining: {config.max_concurrent_retraining} model(s)
- Validation Size: {config.validation_dataset_size} samples
- Champion/Challenger: {config.champion_challenger_duration_hours} hours
- Auto Rollback: {'Enabled' if config.auto_rollback_enabled else 'Disabled'}
- Rollback Threshold: {config.rollback_threshold}
"""
        
        if config.schedule_cron:
            summary += f"- Schedule: {config.schedule_cron}\n"
        
        return summary