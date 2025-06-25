"""Workflow simplification service for enhanced user experience.

This service provides intelligent workflow automation, guided detection processes,
and simplified interfaces for common anomaly detection tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from ...domain.entities import Dataset
from ...domain.entities.simple_detector import SimpleDetector
from ...infrastructure.config.feature_flags import require_feature


class WorkflowComplexity(Enum):
    """Complexity levels for workflow automation."""

    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class UserExperience(Enum):
    """User experience levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    step_id: str
    name: str
    description: str
    required: bool = True
    estimated_time_seconds: int = 30
    complexity: WorkflowComplexity = WorkflowComplexity.SIMPLE
    prerequisites: list[str] = field(default_factory=list)
    validation_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "estimated_time_seconds": self.estimated_time_seconds,
            "complexity": self.complexity.value,
            "prerequisites": self.prerequisites,
            "validation_rules": self.validation_rules,
        }


@dataclass
class WorkflowRecommendation:
    """A workflow recommendation for the user."""

    workflow_id: str
    name: str
    description: str
    estimated_duration_minutes: int
    confidence_score: float
    steps: list[WorkflowStep]
    reasoning: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "confidence_score": self.confidence_score,
            "steps": [step.to_dict() for step in self.steps],
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
        }


class WorkflowSimplificationService:
    """Service for simplifying user workflows and providing guided experiences."""

    def __init__(self):
        """Initialize workflow simplification service."""
        self.workflow_templates = self._initialize_workflow_templates()
        self.user_session_data = {}
        self.workflow_analytics = {
            "successful_completions": 0,
            "abandoned_workflows": 0,
            "most_used_workflows": {},
            "average_completion_time": {},
        }

    @require_feature("cli_simplification")
    def recommend_workflow(
        self,
        user_context: dict[str, Any],
        dataset_info: dict[str, Any] | None = None,
        user_experience: UserExperience = UserExperience.BEGINNER,
    ) -> WorkflowRecommendation:
        """Recommend optimal workflow based on user context and data.

        Args:
            user_context: Information about user's goals and constraints
            dataset_info: Information about the dataset to analyze
            user_experience: User's experience level

        Returns:
            Recommended workflow with guided steps
        """
        # Analyze user intent
        intent = self._analyze_user_intent(user_context)

        # Analyze dataset characteristics if provided
        dataset_profile = (
            self._analyze_dataset_for_workflow(dataset_info) if dataset_info else {}
        )

        # Select appropriate workflow template
        workflow_template = self._select_workflow_template(
            intent, dataset_profile, user_experience
        )

        # Customize workflow for user
        customized_workflow = self._customize_workflow(
            workflow_template, user_context, dataset_profile, user_experience
        )

        return customized_workflow

    @require_feature("interactive_guidance")
    def get_next_step_guidance(
        self,
        workflow_id: str,
        current_step: str,
        user_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get guidance for the next step in a workflow.

        Args:
            workflow_id: ID of the current workflow
            current_step: Current step ID
            user_input: User's input or progress on current step

        Returns:
            Guidance for next step including instructions and validation
        """
        workflow = self._get_workflow_by_id(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}

        current_step_index = self._get_step_index(workflow, current_step)
        if current_step_index == -1:
            return {"error": "Step not found in workflow"}

        # Validate current step if user provided input
        validation_result = (
            self._validate_step_completion(
                workflow.steps[current_step_index], user_input
            )
            if user_input
            else {"valid": True}
        )

        if not validation_result["valid"]:
            return {
                "step_valid": False,
                "validation_errors": validation_result.get("errors", []),
                "suggestions": validation_result.get("suggestions", []),
                "current_step": workflow.steps[current_step_index].to_dict(),
            }

        # Get next step
        next_step_index = current_step_index + 1
        if next_step_index >= len(workflow.steps):
            return {
                "workflow_complete": True,
                "completion_summary": self._generate_completion_summary(workflow_id),
                "next_actions": self._suggest_next_actions(workflow),
            }

        next_step = workflow.steps[next_step_index]

        return {
            "step_valid": True,
            "next_step": next_step.to_dict(),
            "progress": {
                "completed_steps": current_step_index + 1,
                "total_steps": len(workflow.steps),
                "percentage": ((current_step_index + 1) / len(workflow.steps)) * 100,
            },
            "guidance": self._generate_step_guidance(next_step, workflow),
            "tips": self._generate_contextual_tips(next_step, workflow_id),
        }

    @require_feature("cli_simplification")
    def simplify_detection_workflow(
        self,
        dataset: Dataset,
        detection_goals: dict[str, Any],
        automation_level: str = "balanced",  # "minimal", "balanced", "maximum"
    ) -> dict[str, Any]:
        """Simplify the anomaly detection workflow with intelligent automation.

        Args:
            dataset: Dataset to analyze
            detection_goals: User's detection goals and requirements
            automation_level: Level of automation to apply

        Returns:
            Simplified workflow execution results
        """
        # Analyze dataset for automatic recommendations
        dataset_analysis = self._analyze_dataset_comprehensively(dataset)

        # Generate intelligent recommendations
        recommendations = self._generate_intelligent_recommendations(
            dataset_analysis, detection_goals, automation_level
        )

        # Execute simplified workflow based on automation level
        if automation_level == "maximum":
            return self._execute_fully_automated_workflow(
                dataset, recommendations, detection_goals
            )
        elif automation_level == "balanced":
            return self._execute_guided_workflow(
                dataset, recommendations, detection_goals
            )
        else:  # minimal
            return self._provide_manual_guidance(
                dataset, recommendations, detection_goals
            )

    @require_feature("error_recovery")
    def handle_workflow_error(
        self,
        workflow_id: str,
        error_context: dict[str, Any],
        user_experience: UserExperience = UserExperience.BEGINNER,
    ) -> dict[str, Any]:
        """Handle errors in workflow execution with recovery suggestions.

        Args:
            workflow_id: ID of the workflow that encountered an error
            error_context: Context about the error that occurred
            user_experience: User's experience level for appropriate guidance

        Returns:
            Error recovery guidance and alternative approaches
        """
        error_type = error_context.get("error_type", "unknown")
        error_step = error_context.get("step_id", "unknown")

        # Analyze error and determine recovery strategy
        recovery_strategy = self._analyze_error_for_recovery(
            error_type, error_step, workflow_id, user_experience
        )

        # Generate specific recovery actions
        recovery_actions = self._generate_recovery_actions(
            recovery_strategy, error_context, user_experience
        )

        # Suggest workflow modifications if needed
        workflow_modifications = self._suggest_workflow_modifications(
            workflow_id, error_context, recovery_strategy
        )

        return {
            "error_analysis": recovery_strategy,
            "recovery_actions": recovery_actions,
            "workflow_modifications": workflow_modifications,
            "alternative_approaches": self._suggest_alternative_approaches(
                workflow_id, error_context
            ),
            "prevention_tips": self._generate_prevention_tips(
                error_type, user_experience
            ),
        }

    @require_feature("interactive_guidance")
    def get_contextual_help(
        self, context: dict[str, Any], user_question: str | None = None
    ) -> dict[str, Any]:
        """Provide contextual help based on current workflow state.

        Args:
            context: Current workflow context
            user_question: Specific question from user

        Returns:
            Contextual help and guidance
        """
        help_type = self._determine_help_type(context, user_question)

        if help_type == "concept_explanation":
            return self._provide_concept_explanation(context, user_question)
        elif help_type == "parameter_guidance":
            return self._provide_parameter_guidance(context)
        elif help_type == "troubleshooting":
            return self._provide_troubleshooting_help(context)
        elif help_type == "best_practices":
            return self._provide_best_practices(context)
        else:
            return self._provide_general_help(context)

    def get_workflow_analytics(self) -> dict[str, Any]:
        """Get analytics about workflow usage and effectiveness."""
        return {
            "analytics": self.workflow_analytics.copy(),
            "workflow_templates": list(self.workflow_templates.keys()),
            "success_rate": self._calculate_success_rate(),
            "popular_workflows": self._get_popular_workflows(),
            "improvement_suggestions": self._generate_improvement_suggestions(),
        }

    def _initialize_workflow_templates(self) -> dict[str, WorkflowRecommendation]:
        """Initialize predefined workflow templates."""
        templates = {}

        # Quick Start Workflow
        templates["quick_start"] = WorkflowRecommendation(
            workflow_id="quick_start",
            name="Quick Anomaly Detection",
            description="Fast anomaly detection for beginners with automatic algorithm selection",
            estimated_duration_minutes=5,
            confidence_score=0.9,
            steps=[
                WorkflowStep(
                    step_id="data_upload",
                    name="Upload Data",
                    description="Upload your dataset (CSV, Excel, or JSON format)",
                    estimated_time_seconds=60,
                ),
                WorkflowStep(
                    step_id="auto_analysis",
                    name="Automatic Analysis",
                    description="Let Pynomaly automatically analyze your data and select the best algorithm",
                    estimated_time_seconds=30,
                ),
                WorkflowStep(
                    step_id="view_results",
                    name="View Results",
                    description="Review detected anomalies and download results",
                    estimated_time_seconds=120,
                ),
            ],
            reasoning=[
                "Minimal setup required",
                "Automatic algorithm selection",
                "Fast results",
            ],
        )

        # Comprehensive Analysis Workflow
        templates["comprehensive"] = WorkflowRecommendation(
            workflow_id="comprehensive",
            name="Comprehensive Anomaly Analysis",
            description="Detailed analysis with multiple algorithms and explanations",
            estimated_duration_minutes=15,
            confidence_score=0.85,
            steps=[
                WorkflowStep(
                    step_id="data_preparation",
                    name="Data Preparation",
                    description="Upload and preprocess your data",
                    estimated_time_seconds=180,
                    complexity=WorkflowComplexity.INTERMEDIATE,
                ),
                WorkflowStep(
                    step_id="algorithm_selection",
                    name="Algorithm Selection",
                    description="Choose algorithms or let Pynomaly recommend the best combination",
                    estimated_time_seconds=120,
                    complexity=WorkflowComplexity.INTERMEDIATE,
                ),
                WorkflowStep(
                    step_id="parameter_tuning",
                    name="Parameter Optimization",
                    description="Optimize algorithm parameters for your dataset",
                    estimated_time_seconds=300,
                    complexity=WorkflowComplexity.ADVANCED,
                ),
                WorkflowStep(
                    step_id="detection_execution",
                    name="Run Detection",
                    description="Execute anomaly detection with selected algorithms",
                    estimated_time_seconds=180,
                ),
                WorkflowStep(
                    step_id="results_analysis",
                    name="Analyze Results",
                    description="Review results, explanations, and performance metrics",
                    estimated_time_seconds=300,
                    complexity=WorkflowComplexity.INTERMEDIATE,
                ),
            ],
            reasoning=[
                "Comprehensive analysis",
                "Multiple algorithms",
                "Detailed explanations",
            ],
        )

        # Production Deployment Workflow
        templates["production"] = WorkflowRecommendation(
            workflow_id="production",
            name="Production Deployment",
            description="Deploy anomaly detection models to production environment",
            estimated_duration_minutes=30,
            confidence_score=0.8,
            steps=[
                WorkflowStep(
                    step_id="model_validation",
                    name="Model Validation",
                    description="Validate model performance on test data",
                    estimated_time_seconds=300,
                    complexity=WorkflowComplexity.ADVANCED,
                ),
                WorkflowStep(
                    step_id="production_config",
                    name="Production Configuration",
                    description="Configure deployment settings and monitoring",
                    estimated_time_seconds=600,
                    complexity=WorkflowComplexity.EXPERT,
                ),
                WorkflowStep(
                    step_id="deployment",
                    name="Deploy Model",
                    description="Deploy model to production environment",
                    estimated_time_seconds=300,
                    complexity=WorkflowComplexity.EXPERT,
                ),
                WorkflowStep(
                    step_id="monitoring_setup",
                    name="Setup Monitoring",
                    description="Configure performance monitoring and alerts",
                    estimated_time_seconds=480,
                    complexity=WorkflowComplexity.EXPERT,
                ),
            ],
            reasoning=[
                "Production-ready deployment",
                "Performance monitoring",
                "Enterprise features",
            ],
        )

        return templates

    def _analyze_user_intent(self, user_context: dict[str, Any]) -> dict[str, Any]:
        """Analyze user intent from context."""
        intent = {
            "primary_goal": user_context.get("goal", "detect_anomalies"),
            "urgency": user_context.get("urgency", "normal"),
            "experience_needed": user_context.get("experience_level", "beginner"),
            "time_constraint": user_context.get("time_limit_minutes", 30),
            "automation_preference": user_context.get("automation_level", "balanced"),
        }

        # Infer additional intent from context
        if "production" in str(user_context).lower():
            intent["environment"] = "production"
        elif (
            "test" in str(user_context).lower()
            or "experiment" in str(user_context).lower()
        ):
            intent["environment"] = "testing"
        else:
            intent["environment"] = "development"

        return intent

    def _analyze_dataset_for_workflow(
        self, dataset_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze dataset characteristics for workflow selection."""
        profile = {
            "size_category": "unknown",
            "complexity": "unknown",
            "data_quality": "unknown",
            "preprocessing_needed": False,
        }

        # Analyze size
        n_rows = dataset_info.get("n_rows", 0)
        n_cols = dataset_info.get("n_columns", 0)

        if n_rows < 1000:
            profile["size_category"] = "small"
        elif n_rows < 100000:
            profile["size_category"] = "medium"
        else:
            profile["size_category"] = "large"

        # Analyze complexity
        if n_cols < 10:
            profile["complexity"] = "simple"
        elif n_cols < 100:
            profile["complexity"] = "moderate"
        else:
            profile["complexity"] = "complex"

        # Check for preprocessing needs
        missing_ratio = dataset_info.get("missing_values_ratio", 0)
        if missing_ratio > 0.1:
            profile["preprocessing_needed"] = True

        return profile

    def _select_workflow_template(
        self,
        intent: dict[str, Any],
        dataset_profile: dict[str, Any],
        user_experience: UserExperience,
    ) -> WorkflowRecommendation:
        """Select appropriate workflow template."""
        # Simple selection logic based on intent and experience
        if (
            user_experience == UserExperience.BEGINNER
            or intent.get("urgency") == "high"
        ):
            return self.workflow_templates["quick_start"]
        elif intent.get("environment") == "production":
            return self.workflow_templates["production"]
        else:
            return self.workflow_templates["comprehensive"]

    def _customize_workflow(
        self,
        workflow: WorkflowRecommendation,
        user_context: dict[str, Any],
        dataset_profile: dict[str, Any],
        user_experience: UserExperience,
    ) -> WorkflowRecommendation:
        """Customize workflow based on specific context."""
        customized = WorkflowRecommendation(
            workflow_id=f"{workflow.workflow_id}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=workflow.name,
            description=workflow.description,
            estimated_duration_minutes=workflow.estimated_duration_minutes,
            confidence_score=workflow.confidence_score,
            steps=workflow.steps.copy(),
        )

        # Adjust based on dataset size
        if dataset_profile.get("size_category") == "large":
            customized.estimated_duration_minutes = int(
                workflow.estimated_duration_minutes * 1.5
            )
            customized.reasoning.append("Extended time for large dataset processing")

        # Adjust based on user experience
        if user_experience == UserExperience.BEGINNER:
            # Add more detailed descriptions
            for step in customized.steps:
                step.description = f"{step.description} (Guided assistance available)"

        return customized

    def _analyze_dataset_comprehensively(self, dataset: Dataset) -> dict[str, Any]:
        """Perform comprehensive dataset analysis for workflow automation."""
        data = dataset.data

        # Handle pandas DataFrame vs numpy array
        if hasattr(data, "values"):
            data_array = data.values
            n_rows, n_cols = data.shape
            missing_count = data.isnull().sum().sum() if hasattr(data, "isnull") else 0
        else:
            data_array = data
            n_rows = len(data)
            n_cols = data.shape[1] if len(data.shape) > 1 else 1
            missing_count = 0  # Assume no missing values in raw arrays

        analysis = {
            "basic_stats": {
                "n_rows": int(n_rows),
                "n_columns": int(n_cols),
                "memory_usage_mb": (
                    data_array.nbytes / 1024 / 1024
                    if hasattr(data_array, "nbytes")
                    else 0
                ),
            },
            "data_quality": {
                "missing_values": int(missing_count),
                "duplicate_rows": 0,
                "data_types": "numeric",  # Simplified
            },
            "anomaly_indicators": {
                "potential_outliers": self._estimate_outlier_count(data_array),
                "data_distribution": "normal",  # Simplified
                "recommended_contamination": 0.1,
            },
        }

        return analysis

    def _generate_intelligent_recommendations(
        self,
        dataset_analysis: dict[str, Any],
        detection_goals: dict[str, Any],
        automation_level: str,
    ) -> dict[str, Any]:
        """Generate intelligent recommendations based on analysis."""
        recommendations = {
            "algorithms": ["IsolationForest"],  # Simplified recommendation
            "parameters": {
                "contamination": dataset_analysis["anomaly_indicators"][
                    "recommended_contamination"
                ],
                "n_estimators": 100,
            },
            "preprocessing": [],
            "confidence": 0.8,
        }

        # Add preprocessing if needed
        if dataset_analysis["data_quality"]["missing_values"] > 0:
            recommendations["preprocessing"].append("handle_missing_values")

        return recommendations

    def _execute_fully_automated_workflow(
        self, dataset: Dataset, recommendations: dict[str, Any], goals: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute fully automated workflow."""
        # Create detector with recommended settings
        detector = SimpleDetector(
            name="auto_detector",
            algorithm_name=recommendations["algorithms"][0],
            parameters=recommendations["parameters"],
        )

        # Fit and detect
        detector.fit(dataset)
        result = detector.detect(dataset)

        return {
            "execution_type": "fully_automated",
            "detector_used": recommendations["algorithms"][0],
            "parameters_used": recommendations["parameters"],
            "anomalies_detected": (
                np.sum(result.labels) if hasattr(result, "labels") else 0
            ),
            "confidence_score": recommendations["confidence"],
            "processing_time_seconds": 2.5,  # Mock timing
            "recommendations_applied": recommendations,
        }

    def _execute_guided_workflow(
        self, dataset: Dataset, recommendations: dict[str, Any], goals: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute guided workflow with user interaction points."""
        return {
            "execution_type": "guided",
            "recommendations": recommendations,
            "interaction_points": [
                {
                    "step": "algorithm_confirmation",
                    "message": f"Recommended algorithm: {recommendations['algorithms'][0]}",
                    "options": ["accept", "modify", "explain"],
                },
                {
                    "step": "parameter_review",
                    "message": "Review recommended parameters",
                    "parameters": recommendations["parameters"],
                    "options": ["accept", "modify", "auto_tune"],
                },
            ],
            "next_action": "await_user_confirmation",
        }

    def _provide_manual_guidance(
        self, dataset: Dataset, recommendations: dict[str, Any], goals: dict[str, Any]
    ) -> dict[str, Any]:
        """Provide manual guidance without automation."""
        return {
            "execution_type": "manual_guidance",
            "guidance": {
                "step_1": "Choose an algorithm from: IsolationForest, LocalOutlierFactor, OneClassSVM",
                "step_2": "Set contamination rate (typically 0.05-0.15)",
                "step_3": "Configure algorithm-specific parameters",
                "step_4": "Run detection and analyze results",
            },
            "recommendations": recommendations,
            "helpful_hints": [
                "Start with IsolationForest for general use",
                "Use LocalOutlierFactor for density-based detection",
                "Consider ensemble methods for robust detection",
            ],
        }

    def _estimate_outlier_count(self, data) -> int:
        """Estimate number of potential outliers in data."""
        # Simple IQR-based estimation
        try:
            if len(data.shape) == 1:
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                outliers = np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))
            else:
                # For multivariate data, estimate based on first dimension
                Q1 = np.percentile(data[:, 0], 25)
                Q3 = np.percentile(data[:, 0], 75)
                IQR = Q3 - Q1
                outliers = np.sum(
                    (data[:, 0] < Q1 - 1.5 * IQR) | (data[:, 0] > Q3 + 1.5 * IQR)
                )
            return int(outliers)
        except:
            return int(len(data) * 0.05)  # Default 5% estimate

    def _get_workflow_by_id(self, workflow_id: str) -> WorkflowRecommendation | None:
        """Get workflow by ID."""
        return self.workflow_templates.get(workflow_id)

    def _get_step_index(self, workflow: WorkflowRecommendation, step_id: str) -> int:
        """Get index of step in workflow."""
        for i, step in enumerate(workflow.steps):
            if step.step_id == step_id:
                return i
        return -1

    def _validate_step_completion(
        self, step: WorkflowStep, user_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate if step was completed correctly."""
        # Simplified validation
        return {"valid": True}

    def _generate_completion_summary(self, workflow_id: str) -> dict[str, Any]:
        """Generate workflow completion summary."""
        return {
            "workflow_completed": workflow_id,
            "completion_time": datetime.now().isoformat(),
            "success": True,
            "summary": "Workflow completed successfully",
        }

    def _suggest_next_actions(self, workflow: WorkflowRecommendation) -> list[str]:
        """Suggest next actions after workflow completion."""
        return [
            "Review and export results",
            "Set up monitoring for production use",
            "Try alternative algorithms for comparison",
            "Create automated reports",
        ]

    def _generate_step_guidance(
        self, step: WorkflowStep, workflow: WorkflowRecommendation
    ) -> dict[str, Any]:
        """Generate detailed guidance for a step."""
        return {
            "instructions": step.description,
            "estimated_time": f"{step.estimated_time_seconds} seconds",
            "complexity": step.complexity.value,
            "tips": [f"Take your time with {step.name}", "Ask for help if needed"],
        }

    def _generate_contextual_tips(
        self, step: WorkflowStep, workflow_id: str
    ) -> list[str]:
        """Generate contextual tips for a step."""
        tips = [
            f"This step typically takes {step.estimated_time_seconds} seconds",
            "You can always go back to previous steps if needed",
        ]

        if step.complexity != WorkflowComplexity.SIMPLE:
            tips.append(
                "This is an advanced step - consider seeking help if you're unsure"
            )

        return tips

    def _analyze_error_for_recovery(
        self,
        error_type: str,
        error_step: str,
        workflow_id: str,
        user_experience: UserExperience,
    ) -> dict[str, Any]:
        """Analyze error for recovery strategy."""
        return {
            "error_category": "recoverable",
            "complexity": "medium",
            "recovery_confidence": 0.8,
            "estimated_recovery_time": 300,
        }

    def _generate_recovery_actions(
        self,
        strategy: dict[str, Any],
        error_context: dict[str, Any],
        user_experience: UserExperience,
    ) -> list[dict[str, Any]]:
        """Generate specific recovery actions."""
        return [
            {
                "action": "retry_step",
                "description": "Retry the failed step with corrected input",
                "confidence": 0.9,
            },
            {
                "action": "skip_step",
                "description": "Skip this step and continue (if optional)",
                "confidence": 0.6,
            },
        ]

    def _suggest_workflow_modifications(
        self, workflow_id: str, error_context: dict[str, Any], strategy: dict[str, Any]
    ) -> list[str]:
        """Suggest modifications to workflow."""
        return [
            "Add data validation step before processing",
            "Include error handling mechanisms",
            "Provide alternative algorithm options",
        ]

    def _suggest_alternative_approaches(
        self, workflow_id: str, error_context: dict[str, Any]
    ) -> list[str]:
        """Suggest alternative approaches."""
        return [
            "Try a different algorithm",
            "Use automated parameter selection",
            "Switch to simplified workflow",
        ]

    def _generate_prevention_tips(
        self, error_type: str, user_experience: UserExperience
    ) -> list[str]:
        """Generate tips to prevent similar errors."""
        return [
            "Validate data format before upload",
            "Check data quality indicators",
            "Review parameter ranges and constraints",
        ]

    def _determine_help_type(
        self, context: dict[str, Any], user_question: str | None
    ) -> str:
        """Determine type of help needed."""
        if user_question and any(
            word in user_question.lower() for word in ["what", "explain", "meaning"]
        ):
            return "concept_explanation"
        elif "parameter" in str(context).lower():
            return "parameter_guidance"
        elif "error" in str(context).lower():
            return "troubleshooting"
        else:
            return "general_help"

    def _provide_concept_explanation(
        self, context: dict[str, Any], question: str
    ) -> dict[str, Any]:
        """Provide concept explanations."""
        return {
            "type": "concept_explanation",
            "explanation": "Anomaly detection identifies unusual patterns in data",
            "examples": ["Fraud detection", "System monitoring", "Quality control"],
            "related_concepts": [
                "Machine learning",
                "Statistical analysis",
                "Pattern recognition",
            ],
        }

    def _provide_parameter_guidance(self, context: dict[str, Any]) -> dict[str, Any]:
        """Provide parameter guidance."""
        return {
            "type": "parameter_guidance",
            "parameters": {
                "contamination": "Expected proportion of anomalies (0.05-0.15)",
                "n_estimators": "Number of trees for ensemble methods (50-200)",
                "n_neighbors": "Number of neighbors for LOF (10-30)",
            },
            "recommendations": "Start with default values and adjust based on results",
        }

    def _provide_troubleshooting_help(self, context: dict[str, Any]) -> dict[str, Any]:
        """Provide troubleshooting help."""
        return {
            "type": "troubleshooting",
            "common_issues": [
                "Data format not supported",
                "Too many missing values",
                "Dataset too large for available memory",
            ],
            "solutions": [
                "Convert to supported format (CSV, Excel, JSON)",
                "Clean data or handle missing values",
                "Use streaming processing for large datasets",
            ],
        }

    def _provide_best_practices(self, context: dict[str, Any]) -> dict[str, Any]:
        """Provide best practices."""
        return {
            "type": "best_practices",
            "practices": [
                "Always validate your data first",
                "Start with simple algorithms",
                "Compare multiple algorithms",
                "Validate results with domain knowledge",
            ],
        }

    def _provide_general_help(self, context: dict[str, Any]) -> dict[str, Any]:
        """Provide general help."""
        return {
            "type": "general_help",
            "help_topics": [
                "Getting started with anomaly detection",
                "Choosing the right algorithm",
                "Understanding results",
                "Troubleshooting common issues",
            ],
            "contact_info": "For more help, consult the documentation or community forums",
        }

    def _calculate_success_rate(self) -> float:
        """Calculate workflow success rate."""
        total = (
            self.workflow_analytics["successful_completions"]
            + self.workflow_analytics["abandoned_workflows"]
        )
        if total == 0:
            return 0.0
        return self.workflow_analytics["successful_completions"] / total

    def _get_popular_workflows(self) -> list[str]:
        """Get most popular workflows."""
        return sorted(
            self.workflow_analytics["most_used_workflows"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

    def _generate_improvement_suggestions(self) -> list[str]:
        """Generate suggestions for workflow improvements."""
        return [
            "Add more interactive guidance steps",
            "Improve error messages and recovery options",
            "Provide more algorithm recommendations",
            "Add workflow customization options",
        ]
