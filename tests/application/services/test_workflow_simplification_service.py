"""Tests for workflow simplification service."""

from unittest.mock import patch

import numpy as np
from pynomaly.application.services.workflow_simplification_service import (
    UserExperience,
    WorkflowComplexity,
    WorkflowRecommendation,
    WorkflowSimplificationService,
    WorkflowStep,
)
from pynomaly.domain.entities import Dataset


class TestWorkflowSimplificationService:
    """Test workflow simplification service functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = WorkflowSimplificationService()

        # Create test dataset
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 5))
        outliers = np.random.normal(3, 1, (50, 5))
        self.test_data = np.vstack([normal_data, outliers])
        self.test_dataset = Dataset(name="test_dataset", data=self.test_data)

    def test_workflow_service_initialization(self):
        """Test workflow service initialization."""
        assert self.service is not None
        assert len(self.service.workflow_templates) > 0
        assert "quick_start" in self.service.workflow_templates
        assert "comprehensive" in self.service.workflow_templates
        assert "production" in self.service.workflow_templates

        # Check analytics initialization
        assert "successful_completions" in self.service.workflow_analytics
        assert "abandoned_workflows" in self.service.workflow_analytics

    def test_workflow_step_creation(self):
        """Test workflow step data class."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step",
            description="Test description",
            estimated_time_seconds=120,
            complexity=WorkflowComplexity.INTERMEDIATE,
        )

        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.complexity == WorkflowComplexity.INTERMEDIATE
        assert step.required is True  # Default value

        # Test serialization
        step_dict = step.to_dict()
        assert step_dict["step_id"] == "test_step"
        assert step_dict["complexity"] == "intermediate"

    def test_workflow_recommendation_creation(self):
        """Test workflow recommendation data class."""
        steps = [
            WorkflowStep("step1", "Step 1", "First step"),
            WorkflowStep("step2", "Step 2", "Second step"),
        ]

        recommendation = WorkflowRecommendation(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Test workflow description",
            estimated_duration_minutes=10,
            confidence_score=0.85,
            steps=steps,
        )

        assert recommendation.workflow_id == "test_workflow"
        assert len(recommendation.steps) == 2
        assert recommendation.confidence_score == 0.85

        # Test serialization
        rec_dict = recommendation.to_dict()
        assert rec_dict["workflow_id"] == "test_workflow"
        assert len(rec_dict["steps"]) == 2

    def test_recommend_workflow_basic(self):
        """Test basic workflow recommendation."""

        user_context = {
            "goal": "detect_anomalies",
            "urgency": "normal",
            "experience_level": "beginner",
            "time_limit_minutes": 15,
        }

        dataset_info = {"n_rows": 1000, "n_columns": 5, "missing_values_ratio": 0.05}

        recommendation = self.service.recommend_workflow(
            user_context, dataset_info, UserExperience.BEGINNER
        )

        assert recommendation is not None
        assert recommendation.workflow_id.startswith(
            "quick_start"
        )  # Should recommend quick start for beginners
        assert len(recommendation.steps) > 0
        assert recommendation.confidence_score > 0

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_recommend_workflow_advanced_user(self, mock_feature_enabled):
        """Test workflow recommendation for advanced users."""
        mock_feature_enabled.return_value = True

        user_context = {
            "goal": "comprehensive_analysis",
            "urgency": "low",
            "experience_level": "advanced",
            "time_limit_minutes": 60,
        }

        dataset_info = {"n_rows": 5000, "n_columns": 20, "missing_values_ratio": 0.02}

        recommendation = self.service.recommend_workflow(
            user_context, dataset_info, UserExperience.ADVANCED
        )

        assert recommendation is not None
        # Advanced users with time should get comprehensive workflow
        assert "comprehensive" in recommendation.workflow_id
        assert (
            recommendation.estimated_duration_minutes > 10
        )  # Should be longer for comprehensive

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_recommend_workflow_production(self, mock_feature_enabled):
        """Test workflow recommendation for production environment."""
        mock_feature_enabled.return_value = True

        user_context = {
            "goal": "production_deployment",
            "urgency": "normal",
            "experience_level": "expert",
            "environment": "production",
        }

        recommendation = self.service.recommend_workflow(
            user_context, user_experience=UserExperience.EXPERT
        )

        assert recommendation is not None
        assert "production" in recommendation.workflow_id
        assert (
            recommendation.estimated_duration_minutes > 20
        )  # Production should take longer

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_get_next_step_guidance(self, mock_feature_enabled):
        """Test next step guidance functionality."""
        mock_feature_enabled.return_value = True

        # Get a workflow first
        user_context = {"goal": "detect_anomalies"}
        recommendation = self.service.recommend_workflow(user_context)
        workflow_id = recommendation.workflow_id

        # Test getting guidance for first step
        first_step = recommendation.steps[0]
        guidance = self.service.get_next_step_guidance(workflow_id, first_step.step_id)

        assert guidance is not None
        if not guidance.get("error"):
            assert "next_step" in guidance or "workflow_complete" in guidance
            if "next_step" in guidance:
                assert "progress" in guidance
                assert "guidance" in guidance

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_simplify_detection_workflow_automated(self, mock_feature_enabled):
        """Test simplified detection workflow with maximum automation."""
        mock_feature_enabled.return_value = True

        detection_goals = {
            "target": "fraud_detection",
            "accuracy_requirement": "high",
            "speed_requirement": "medium",
        }

        result = self.service.simplify_detection_workflow(
            self.test_dataset, detection_goals, automation_level="maximum"
        )

        assert result is not None
        assert result["execution_type"] == "fully_automated"
        assert "detector_used" in result
        assert "anomalies_detected" in result
        assert "confidence_score" in result

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_simplify_detection_workflow_guided(self, mock_feature_enabled):
        """Test simplified detection workflow with guided automation."""
        mock_feature_enabled.return_value = True

        detection_goals = {
            "target": "quality_control",
            "accuracy_requirement": "medium",
        }

        result = self.service.simplify_detection_workflow(
            self.test_dataset, detection_goals, automation_level="balanced"
        )

        assert result is not None
        assert result["execution_type"] == "guided"
        assert "recommendations" in result
        assert "interaction_points" in result
        assert len(result["interaction_points"]) > 0

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_simplify_detection_workflow_manual(self, mock_feature_enabled):
        """Test simplified detection workflow with manual guidance."""
        mock_feature_enabled.return_value = True

        detection_goals = {"target": "exploration"}

        result = self.service.simplify_detection_workflow(
            self.test_dataset, detection_goals, automation_level="minimal"
        )

        assert result is not None
        assert result["execution_type"] == "manual_guidance"
        assert "guidance" in result
        assert "helpful_hints" in result
        assert len(result["helpful_hints"]) > 0

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_handle_workflow_error(self, mock_feature_enabled):
        """Test workflow error handling."""
        mock_feature_enabled.return_value = True

        error_context = {
            "error_type": "data_format_error",
            "step_id": "data_upload",
            "error_message": "Invalid CSV format",
        }

        recovery_info = self.service.handle_workflow_error(
            "quick_start_workflow", error_context, UserExperience.BEGINNER
        )

        assert recovery_info is not None
        assert "error_analysis" in recovery_info
        assert "recovery_actions" in recovery_info
        assert "alternative_approaches" in recovery_info
        assert "prevention_tips" in recovery_info
        assert len(recovery_info["recovery_actions"]) > 0

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_get_contextual_help(self, mock_feature_enabled):
        """Test contextual help functionality."""
        mock_feature_enabled.return_value = True

        # Test concept explanation
        context = {"current_step": "algorithm_selection"}
        help_info = self.service.get_contextual_help(
            context, "What is anomaly detection?"
        )

        assert help_info is not None
        assert help_info["type"] == "concept_explanation"
        assert "explanation" in help_info

        # Test parameter guidance
        context = {"current_step": "parameter_tuning", "algorithm": "isolation_forest"}
        help_info = self.service.get_contextual_help(context)

        assert help_info is not None
        if help_info["type"] == "parameter_guidance":
            assert "parameters" in help_info

    def test_workflow_analytics(self):
        """Test workflow analytics functionality."""
        analytics = self.service.get_workflow_analytics()

        assert analytics is not None
        assert "analytics" in analytics
        assert "workflow_templates" in analytics
        assert "success_rate" in analytics
        assert "popular_workflows" in analytics
        assert "improvement_suggestions" in analytics

        # Check template list
        templates = analytics["workflow_templates"]
        assert "quick_start" in templates
        assert "comprehensive" in templates
        assert "production" in templates

    def test_dataset_analysis_for_workflow(self):
        """Test dataset analysis for workflow selection."""
        dataset_info = {"n_rows": 5000, "n_columns": 25, "missing_values_ratio": 0.15}

        profile = self.service._analyze_dataset_for_workflow(dataset_info)

        assert profile is not None
        assert profile["size_category"] == "medium"  # 5000 rows
        assert profile["complexity"] == "moderate"  # 25 columns
        assert profile["preprocessing_needed"] is True  # 15% missing values

    def test_dataset_comprehensive_analysis(self):
        """Test comprehensive dataset analysis."""
        analysis = self.service._analyze_dataset_comprehensively(self.test_dataset)

        assert analysis is not None
        assert "basic_stats" in analysis
        assert "data_quality" in analysis
        assert "anomaly_indicators" in analysis

        # Check basic stats
        basic_stats = analysis["basic_stats"]
        assert basic_stats["n_rows"] == len(self.test_data)
        assert basic_stats["n_columns"] == self.test_data.shape[1]

        # Check anomaly indicators
        anomaly_indicators = analysis["anomaly_indicators"]
        assert "potential_outliers" in anomaly_indicators
        assert "recommended_contamination" in anomaly_indicators

    def test_outlier_estimation(self):
        """Test outlier estimation functionality."""
        # Test with 1D data
        data_1d = np.random.normal(0, 1, 100)
        # Add some clear outliers
        data_1d = np.append(data_1d, [5, -5, 6])

        outlier_count = self.service._estimate_outlier_count(data_1d)
        assert outlier_count >= 2  # Should detect at least the clear outliers

        # Test with 2D data
        outlier_count_2d = self.service._estimate_outlier_count(self.test_data)
        assert outlier_count_2d > 0  # Should detect some outliers in mixed data

    def test_workflow_template_customization(self):
        """Test workflow template customization."""
        base_workflow = self.service.workflow_templates["quick_start"]

        user_context = {"time_limit_minutes": 60}
        dataset_profile = {"size_category": "large"}

        customized = self.service._customize_workflow(
            base_workflow, user_context, dataset_profile, UserExperience.BEGINNER
        )

        assert customized is not None
        assert (
            customized.workflow_id != base_workflow.workflow_id
        )  # Should be different
        assert customized.name == base_workflow.name  # Name should be same

        # For large datasets, duration should be extended
        if dataset_profile["size_category"] == "large":
            assert (
                customized.estimated_duration_minutes
                >= base_workflow.estimated_duration_minutes
            )

    def test_intelligent_recommendations(self):
        """Test intelligent recommendation generation."""
        dataset_analysis = {
            "basic_stats": {"n_rows": 1000, "n_columns": 5},
            "data_quality": {"missing_values": 0},
            "anomaly_indicators": {"recommended_contamination": 0.1},
        }

        detection_goals = {"target": "general", "speed": "high"}

        recommendations = self.service._generate_intelligent_recommendations(
            dataset_analysis, detection_goals, "balanced"
        )

        assert recommendations is not None
        assert "algorithms" in recommendations
        assert "parameters" in recommendations
        assert "preprocessing" in recommendations
        assert "confidence" in recommendations

        # Should recommend at least one algorithm
        assert len(recommendations["algorithms"]) > 0

        # Parameters should include contamination
        assert "contamination" in recommendations["parameters"]

    def test_error_recovery_suggestions(self):
        """Test error recovery suggestion generation."""
        strategy = {
            "error_category": "recoverable",
            "complexity": "medium",
            "recovery_confidence": 0.8,
        }

        error_context = {"error_type": "memory_error", "step_id": "data_processing"}

        recovery_actions = self.service._generate_recovery_actions(
            strategy, error_context, UserExperience.BEGINNER
        )

        assert recovery_actions is not None
        assert len(recovery_actions) > 0

        # Each action should have required fields
        for action in recovery_actions:
            assert "action" in action
            assert "description" in action
            assert "confidence" in action

    def test_help_type_determination(self):
        """Test help type determination logic."""
        # Test concept explanation
        context = {}
        question = "What is anomaly detection?"
        help_type = self.service._determine_help_type(context, question)
        assert help_type == "concept_explanation"

        # Test parameter guidance
        context = {"current_step": "parameter_tuning"}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == "parameter_guidance"

        # Test troubleshooting
        context = {"error": "data_loading_failed"}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == "troubleshooting"

        # Test general help
        context = {}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == "general_help"

    def test_workflow_step_validation(self):
        """Test workflow step completion validation."""
        step = WorkflowStep(
            step_id="test_step", name="Test Step", description="Test step description"
        )

        user_input = {"completed": True, "data": {"algorithm": "IsolationForest"}}

        validation_result = self.service._validate_step_completion(step, user_input)

        assert validation_result is not None
        assert "valid" in validation_result
        # Current implementation returns True by default
        assert validation_result["valid"] is True

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Initially should be 0 (no completions)
        success_rate = self.service._calculate_success_rate()
        assert success_rate == 0.0

        # Add some analytics data
        self.service.workflow_analytics["successful_completions"] = 8
        self.service.workflow_analytics["abandoned_workflows"] = 2

        success_rate = self.service._calculate_success_rate()
        assert success_rate == 0.8  # 8/(8+2) = 0.8

    def test_popular_workflows_tracking(self):
        """Test popular workflows tracking."""
        # Add some usage data
        self.service.workflow_analytics["most_used_workflows"] = {
            "quick_start": 100,
            "comprehensive": 50,
            "production": 25,
        }

        popular = self.service._get_popular_workflows()
        assert len(popular) <= 3  # Should return top 3
        if popular:
            # Should be sorted by usage (descending)
            assert popular[0][0] == "quick_start"  # Most popular
            assert popular[0][1] == 100

    @patch("pynomaly.infrastructure.config.feature_flags.is_feature_enabled")
    def test_feature_flag_integration(self, mock_feature_enabled):
        """Test feature flag integration."""
        # Test with feature disabled
        mock_feature_enabled.return_value = False

        user_context = {"goal": "detect_anomalies"}

        # Should raise or handle gracefully when feature is disabled
        try:
            recommendation = self.service.recommend_workflow(user_context)
            # If no exception, check that it handles gracefully
            assert recommendation is not None
        except Exception:
            # Feature flags might prevent execution
            pass

        # Test with feature enabled
        mock_feature_enabled.return_value = True

        recommendation = self.service.recommend_workflow(user_context)
        assert recommendation is not None


class TestWorkflowDataClasses:
    """Test workflow data classes in isolation."""

    def test_workflow_complexity_enum(self):
        """Test workflow complexity enumeration."""
        assert WorkflowComplexity.SIMPLE.value == "simple"
        assert WorkflowComplexity.INTERMEDIATE.value == "intermediate"
        assert WorkflowComplexity.ADVANCED.value == "advanced"
        assert WorkflowComplexity.EXPERT.value == "expert"

    def test_user_experience_enum(self):
        """Test user experience enumeration."""
        assert UserExperience.BEGINNER.value == "beginner"
        assert UserExperience.INTERMEDIATE.value == "intermediate"
        assert UserExperience.ADVANCED.value == "advanced"
        assert UserExperience.EXPERT.value == "expert"

    def test_workflow_step_defaults(self):
        """Test workflow step default values."""
        step = WorkflowStep(step_id="test", name="Test", description="Test description")

        assert step.required is True
        assert step.estimated_time_seconds == 30
        assert step.complexity == WorkflowComplexity.SIMPLE
        assert step.prerequisites == []
        assert step.validation_rules == []

    def test_workflow_recommendation_defaults(self):
        """Test workflow recommendation default values."""
        recommendation = WorkflowRecommendation(
            workflow_id="test",
            name="Test",
            description="Test description",
            estimated_duration_minutes=5,
            confidence_score=0.8,
            steps=[],
        )

        assert recommendation.reasoning == []
        assert recommendation.alternatives == []
