"""Basic tests for workflow simplification service without feature flag mocking."""

import pytest
import numpy as np
import pandas as pd

from pynomaly.application.services.workflow_simplification_service import (
    WorkflowSimplificationService, WorkflowComplexity, UserExperience,
    WorkflowStep, WorkflowRecommendation
)
from pynomaly.domain.entities import Dataset
from pynomaly.domain.entities.simple_detector import SimpleDetector


class TestWorkflowSimplificationBasic:
    """Basic test workflow simplification service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = WorkflowSimplificationService()
        
        # Create test dataset
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 5))
        outliers = np.random.normal(3, 1, (50, 5))
        self.test_data = np.vstack([normal_data, outliers])
        self.test_dataset = Dataset(name="test_dataset", data=self.test_data)
    
    def test_service_initialization(self):
        """Test workflow service initialization."""
        assert self.service is not None
        assert len(self.service.workflow_templates) > 0
        assert 'quick_start' in self.service.workflow_templates
        assert 'comprehensive' in self.service.workflow_templates
        assert 'production' in self.service.workflow_templates
    
    def test_workflow_step_creation(self):
        """Test workflow step data class."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step",
            description="Test description",
            estimated_time_seconds=120,
            complexity=WorkflowComplexity.INTERMEDIATE
        )
        
        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.complexity == WorkflowComplexity.INTERMEDIATE
        assert step.required is True
        
        step_dict = step.to_dict()
        assert step_dict['step_id'] == "test_step"
        assert step_dict['complexity'] == 'intermediate'
    
    def test_workflow_recommendation_creation(self):
        """Test workflow recommendation data class."""
        steps = [
            WorkflowStep("step1", "Step 1", "First step"),
            WorkflowStep("step2", "Step 2", "Second step")
        ]
        
        recommendation = WorkflowRecommendation(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="Test workflow description",
            estimated_duration_minutes=10,
            confidence_score=0.85,
            steps=steps
        )
        
        assert recommendation.workflow_id == "test_workflow"
        assert len(recommendation.steps) == 2
        assert recommendation.confidence_score == 0.85
    
    def test_workflow_analytics(self):
        """Test workflow analytics functionality."""
        analytics = self.service.get_workflow_analytics()
        
        assert analytics is not None
        assert 'analytics' in analytics
        assert 'workflow_templates' in analytics
        assert 'success_rate' in analytics
        assert 'popular_workflows' in analytics
        assert 'improvement_suggestions' in analytics
        
        templates = analytics['workflow_templates']
        assert 'quick_start' in templates
        assert 'comprehensive' in templates
        assert 'production' in templates
    
    def test_dataset_analysis_for_workflow(self):
        """Test dataset analysis for workflow selection."""
        dataset_info = {
            'n_rows': 5000,
            'n_columns': 25,
            'missing_values_ratio': 0.15
        }
        
        profile = self.service._analyze_dataset_for_workflow(dataset_info)
        
        assert profile is not None
        assert profile['size_category'] == 'medium'
        assert profile['complexity'] == 'moderate'
        assert profile['preprocessing_needed'] is True
    
    def test_dataset_comprehensive_analysis(self):
        """Test comprehensive dataset analysis."""
        analysis = self.service._analyze_dataset_comprehensively(self.test_dataset)
        
        assert analysis is not None
        assert 'basic_stats' in analysis
        assert 'data_quality' in analysis
        assert 'anomaly_indicators' in analysis
        
        basic_stats = analysis['basic_stats']
        assert basic_stats['n_rows'] == len(self.test_data)
        assert basic_stats['n_columns'] == self.test_data.shape[1]
        
        anomaly_indicators = analysis['anomaly_indicators']
        assert 'potential_outliers' in anomaly_indicators
        assert 'recommended_contamination' in anomaly_indicators
    
    def test_outlier_estimation(self):
        """Test outlier estimation functionality."""
        data_1d = np.random.normal(0, 1, 100)
        data_1d = np.append(data_1d, [5, -5, 6])
        
        outlier_count = self.service._estimate_outlier_count(data_1d)
        assert outlier_count >= 2
        
        outlier_count_2d = self.service._estimate_outlier_count(self.test_data)
        assert outlier_count_2d > 0
    
    def test_workflow_template_customization(self):
        """Test workflow template customization."""
        base_workflow = self.service.workflow_templates['quick_start']
        
        user_context = {'time_limit_minutes': 60}
        dataset_profile = {'size_category': 'large'}
        
        customized = self.service._customize_workflow(
            base_workflow, user_context, dataset_profile, UserExperience.BEGINNER
        )
        
        assert customized is not None
        assert customized.workflow_id != base_workflow.workflow_id
        assert customized.name == base_workflow.name
        
        if dataset_profile['size_category'] == 'large':
            assert customized.estimated_duration_minutes >= base_workflow.estimated_duration_minutes
    
    def test_intelligent_recommendations(self):
        """Test intelligent recommendation generation."""
        dataset_analysis = {
            'basic_stats': {'n_rows': 1000, 'n_columns': 5},
            'data_quality': {'missing_values': 0},
            'anomaly_indicators': {'recommended_contamination': 0.1}
        }
        
        detection_goals = {'target': 'general', 'speed': 'high'}
        
        recommendations = self.service._generate_intelligent_recommendations(
            dataset_analysis, detection_goals, "balanced"
        )
        
        assert recommendations is not None
        assert 'algorithms' in recommendations
        assert 'parameters' in recommendations
        assert 'preprocessing' in recommendations
        assert 'confidence' in recommendations
        assert len(recommendations['algorithms']) > 0
        assert 'contamination' in recommendations['parameters']
    
    def test_help_type_determination(self):
        """Test help type determination logic."""
        context = {}
        question = "What is anomaly detection?"
        help_type = self.service._determine_help_type(context, question)
        assert help_type == 'concept_explanation'
        
        context = {'current_step': 'parameter_tuning'}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == 'parameter_guidance'
        
        context = {'error': 'data_loading_failed'}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == 'troubleshooting'
        
        context = {}
        help_type = self.service._determine_help_type(context, None)
        assert help_type == 'general_help'
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        success_rate = self.service._calculate_success_rate()
        assert success_rate == 0.0
        
        self.service.workflow_analytics['successful_completions'] = 8
        self.service.workflow_analytics['abandoned_workflows'] = 2
        
        success_rate = self.service._calculate_success_rate()
        assert success_rate == 0.8
    
    def test_popular_workflows_tracking(self):
        """Test popular workflows tracking."""
        self.service.workflow_analytics['most_used_workflows'] = {
            'quick_start': 100,
            'comprehensive': 50,
            'production': 25
        }
        
        popular = self.service._get_popular_workflows()
        assert len(popular) <= 3
        if popular:
            assert popular[0][0] == 'quick_start'
            assert popular[0][1] == 100


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
        step = WorkflowStep(
            step_id="test",
            name="Test",
            description="Test description"
        )
        
        assert step.required is True
        assert step.estimated_time_seconds == 30
        assert step.complexity == WorkflowComplexity.SIMPLE
        assert step.prerequisites == []
        assert step.validation_rules == []