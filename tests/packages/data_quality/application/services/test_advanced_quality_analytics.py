"""Comprehensive test suite for advanced quality analytics services.

Tests for ML-enhanced quality detection, intelligent rule discovery,
quality lineage, predictive analytics, advanced metrics, and optimization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the services we're testing
from src.packages.data_quality.application.services.ml_quality_detection_service import (
    MLQualityDetectionService, MLDetectionConfig, DetectionMethod
)
from src.packages.data_quality.application.services.intelligent_rule_discovery_service import (
    IntelligentRuleDiscoveryService, RuleDiscoveryConfig, RuleDiscoveryMethod
)
from src.packages.data_quality.application.services.quality_lineage_service import (
    QualityLineageService, LineageServiceConfig, LineageAnalysisType
)
from src.packages.data_quality.application.services.predictive_quality_service import (
    PredictiveQualityService, PredictiveQualityConfig, PredictionHorizon, PredictionModel
)
from src.packages.data_quality.application.services.advanced_quality_metrics_service import (
    AdvancedQualityMetricsService, AdvancedMetricsConfig, MetricType, ScoringMethod
)
from src.packages.data_quality.application.services.quality_optimization_service import (
    QualityOptimizationService, QualityOptimizationConfig, OptimizationType, OptimizationPriority
)

# Import domain entities
from src.packages.data_quality.domain.entities.quality_profile import DataQualityProfile
from src.packages.data_quality.domain.entities.quality_scores import QualityScores
from src.packages.data_quality.domain.entities.quality_anomaly import (
    QualityAnomaly, QualityAnomalyId, AnomalyType, AnomalySeverity, AnomalyStatus, 
    AnomalyDetectionResult, AnomalyPattern
)
from src.packages.data_quality.domain.entities.quality_lineage import (
    QualityLineageGraph, LineageNode, LineageEdge, LineageNodeType, 
    LineageNodeId, LineageEdgeId, ImpactType, ImpactSeverity
)


class TestMLQualityDetectionService:
    """Test suite for ML Quality Detection Service."""
    
    @pytest.fixture
    def ml_config(self):
        """ML detection service configuration."""
        return MLDetectionConfig(
            enabled_methods=[
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.LOCAL_OUTLIER_FACTOR,
                DetectionMethod.STATISTICAL_OUTLIER
            ],
            contamination_rate=0.1,
            n_estimators=50,  # Reduced for faster testing
            enable_temporal_analysis=True
        )
    
    @pytest.fixture
    def ml_service(self, ml_config):
        """ML quality detection service instance."""
        return MLQualityDetectionService(ml_config)
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for ML models."""
        np.random.seed(42)
        data = []
        
        # Generate normal quality data
        for i in range(100):
            data.append({
                'timestamp': datetime.now() - timedelta(days=i),
                'completeness_score': np.random.normal(0.9, 0.05),
                'accuracy_score': np.random.normal(0.85, 0.05),
                'consistency_score': np.random.normal(0.8, 0.05),
                'row_count': np.random.randint(1000, 2000),
                'null_count': np.random.randint(10, 50)
            })
        
        # Add some anomalous data
        for i in range(10):
            data.append({
                'timestamp': datetime.now() - timedelta(days=i*5),
                'completeness_score': np.random.normal(0.3, 0.1),  # Anomalous
                'accuracy_score': np.random.normal(0.4, 0.1),      # Anomalous
                'consistency_score': np.random.normal(0.2, 0.1),   # Anomalous
                'row_count': np.random.randint(100, 300),           # Anomalous
                'null_count': np.random.randint(200, 500)           # Anomalous
            })
        
        return data
    
    def test_ml_service_initialization(self, ml_service):
        """Test ML service initialization."""
        assert ml_service is not None
        assert ml_service.config.contamination_rate == 0.1
        assert DetectionMethod.ISOLATION_FOREST in ml_service.config.enabled_methods
    
    def test_train_anomaly_detection_models(self, ml_service, sample_training_data):
        """Test training anomaly detection models."""
        dataset_id = "test_dataset_001"
        
        # Train models
        training_results = ml_service.train_anomaly_detection_models(
            dataset_id, sample_training_data
        )
        
        assert training_results is not None
        assert 'models_trained' in training_results
        assert 'training_accuracy' in training_results
        assert training_results['models_trained'] > 0
        assert 0 <= training_results['training_accuracy'] <= 1
    
    def test_detect_quality_anomalies(self, ml_service, sample_training_data):
        """Test quality anomaly detection."""
        dataset_id = "test_dataset_001"
        
        # Train models first
        ml_service.train_anomaly_detection_models(dataset_id, sample_training_data)
        
        # Create current quality data (anomalous)
        current_data = {
            'timestamp': datetime.now(),
            'completeness_score': 0.2,  # Very low
            'accuracy_score': 0.3,      # Very low
            'consistency_score': 0.1,   # Very low
            'row_count': 50,             # Very low
            'null_count': 800            # Very high
        }
        
        # Detect anomalies
        anomalies = ml_service.detect_quality_anomalies(dataset_id, current_data)
        
        assert anomalies is not None
        assert len(anomalies) > 0
        
        # Check anomaly properties
        anomaly = anomalies[0]
        assert isinstance(anomaly, QualityAnomaly)
        assert anomaly.dataset_id == dataset_id
        assert anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        assert anomaly.detection_result.anomaly_score > 0.5
    
    def test_ensemble_detection(self, ml_service, sample_training_data):
        """Test ensemble anomaly detection."""
        dataset_id = "test_dataset_002"
        
        # Use ensemble method
        config = MLDetectionConfig(
            enabled_methods=[
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.LOCAL_OUTLIER_FACTOR,
                DetectionMethod.ENSEMBLE_DETECTION
            ]
        )
        ensemble_service = MLQualityDetectionService(config)
        
        # Train and detect
        ensemble_service.train_anomaly_detection_models(dataset_id, sample_training_data)
        
        current_data = {
            'timestamp': datetime.now(),
            'completeness_score': 0.15,
            'accuracy_score': 0.25,
            'consistency_score': 0.05
        }
        
        anomalies = ensemble_service.detect_quality_anomalies(dataset_id, current_data)
        
        assert anomalies is not None
        # Ensemble detection should be more confident
        if anomalies:
            assert anomalies[0].detection_result.confidence_score >= 0.5


class TestIntelligentRuleDiscoveryService:
    """Test suite for Intelligent Rule Discovery Service."""
    
    @pytest.fixture
    def rule_config(self):
        """Rule discovery service configuration."""
        return RuleDiscoveryConfig(
            enabled_methods=[
                RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
                RuleDiscoveryMethod.PATTERN_RECOGNITION,
                RuleDiscoveryMethod.CORRELATION_RULES
            ],
            min_confidence=0.7,
            min_support=0.1
        )
    
    @pytest.fixture
    def rule_service(self, rule_config):
        """Rule discovery service instance."""
        return IntelligentRuleDiscoveryService(rule_config)
    
    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for rule discovery."""
        np.random.seed(42)
        
        # Create structured data with patterns
        data = {
            'email': [f"user{i}@example.com" for i in range(100)] + 
                    [f"invalid_email_{i}" for i in range(10)],  # 10% invalid
            'age': list(np.random.randint(18, 80, 100)) + 
                  [150, 200, -5, -10, 999],  # Some outliers
            'salary': list(np.random.randint(30000, 120000, 100)) + 
                     [0, -1000, 500000, 1000000, 2000000],  # Some outliers
            'phone': [f"555-{i:04d}" for i in range(100)] + 
                    ["invalid", "123", "", "abc-defg", None],  # Invalid phones
            'status': ['active'] * 80 + ['inactive'] * 20 + ['unknown'] * 10
        }
        
        return pd.DataFrame(data)
    
    def test_rule_service_initialization(self, rule_service):
        """Test rule discovery service initialization."""
        assert rule_service is not None
        assert rule_service.config.min_confidence == 0.7
        assert RuleDiscoveryMethod.STATISTICAL_ANALYSIS in rule_service.config.enabled_methods
    
    def test_discover_validation_rules(self, rule_service, sample_dataset):
        """Test validation rule discovery."""
        dataset_id = "test_dataset_rules"
        
        # Discover rules
        discovered_rules = rule_service.discover_validation_rules(
            dataset_id, sample_dataset
        )
        
        assert discovered_rules is not None
        assert len(discovered_rules) > 0
        
        # Check rule properties
        rule = discovered_rules[0]
        assert hasattr(rule, 'rule_id')
        assert hasattr(rule, 'description')
        assert hasattr(rule, 'confidence_score')
        assert 0 <= rule.confidence_score <= 1
    
    def test_pattern_recognition_rules(self, rule_service, sample_dataset):
        """Test pattern recognition rule discovery."""
        dataset_id = "test_pattern_rules"
        
        # Focus on pattern recognition
        config = RuleDiscoveryConfig(
            enabled_methods=[RuleDiscoveryMethod.PATTERN_RECOGNITION],
            pattern_min_frequency=5
        )
        pattern_service = IntelligentRuleDiscoveryService(config)
        
        rules = pattern_service.discover_validation_rules(dataset_id, sample_dataset)
        
        # Should discover email pattern rules
        email_rules = [r for r in rules if 'email' in r.description.lower()]
        assert len(email_rules) > 0
    
    def test_statistical_analysis_rules(self, rule_service, sample_dataset):
        """Test statistical analysis rule discovery."""
        dataset_id = "test_statistical_rules"
        
        # Focus on statistical analysis
        config = RuleDiscoveryConfig(
            enabled_methods=[RuleDiscoveryMethod.STATISTICAL_ANALYSIS],
            outlier_threshold=2.5
        )
        stats_service = IntelligentRuleDiscoveryService(config)
        
        rules = stats_service.discover_validation_rules(dataset_id, sample_dataset)
        
        # Should discover range rules for numeric columns
        range_rules = [r for r in rules if 'range' in r.description.lower()]
        assert len(range_rules) > 0


class TestQualityLineageService:
    """Test suite for Quality Lineage Service."""
    
    @pytest.fixture
    def lineage_config(self):
        """Lineage service configuration."""
        return LineageServiceConfig(
            max_propagation_depth=5,
            enable_impact_scoring=True,
            auto_discover_relationships=True
        )
    
    @pytest.fixture
    def lineage_service(self, lineage_config):
        """Lineage service instance."""
        return QualityLineageService(lineage_config)
    
    @pytest.fixture
    def sample_lineage_graph(self, lineage_service):
        """Sample lineage graph."""
        graph = lineage_service.create_lineage_graph(
            "Test Lineage Graph",
            "Sample graph for testing"
        )
        
        # Create nodes
        source_node = lineage_service.add_dataset_node(
            graph, "source_db", "Source Database"
        )
        
        transform_node = LineageNode(
            node_id=LineageNodeId(),
            node_type=LineageNodeType.TRANSFORMATION,
            name="Data Transformation",
            description="ETL Process"
        )
        graph.add_node(transform_node)
        
        target_node = lineage_service.add_dataset_node(
            graph, "target_warehouse", "Data Warehouse"
        )
        
        ml_model_node = lineage_service.add_downstream_system_node(
            graph, "prediction_model", "ml_model", [target_node]
        )
        
        # Create edges
        edge1 = LineageEdge(
            edge_id=LineageEdgeId(),
            source_node_id=source_node.node_id,
            target_node_id=transform_node.node_id,
            relationship_type="feeds",
            impact_type=ImpactType.DIRECT,
            quality_impact_factor=0.9
        )
        graph.add_edge(edge1)
        
        edge2 = LineageEdge(
            edge_id=LineageEdgeId(),
            source_node_id=transform_node.node_id,
            target_node_id=target_node.node_id,
            relationship_type="produces",
            impact_type=ImpactType.CASCADING,
            quality_impact_factor=0.8
        )
        graph.add_edge(edge2)
        
        return graph, source_node, transform_node, target_node, ml_model_node
    
    def test_lineage_service_initialization(self, lineage_service):
        """Test lineage service initialization."""
        assert lineage_service is not None
        assert lineage_service.config.max_propagation_depth == 5
        assert lineage_service.config.enable_impact_scoring is True
    
    def test_create_lineage_graph(self, lineage_service):
        """Test lineage graph creation."""
        graph = lineage_service.create_lineage_graph(
            "Test Graph", "Test Description"
        )
        
        assert graph is not None
        assert graph.name == "Test Graph"
        assert graph.description == "Test Description"
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_dataset_nodes(self, lineage_service, sample_lineage_graph):
        """Test adding dataset nodes."""
        graph, source_node, _, target_node, _ = sample_lineage_graph
        
        assert len(graph.nodes) >= 2
        assert source_node.node_type == LineageNodeType.DATASET
        assert target_node.node_type == LineageNodeType.DATASET
        assert source_node.name == "Source Database"
        assert target_node.name == "Data Warehouse"
    
    def test_quality_impact_analysis(self, lineage_service, sample_lineage_graph):
        """Test quality impact analysis."""
        graph, source_node, _, _, _ = sample_lineage_graph
        
        # Simulate quality issue at source
        source_node.quality_score = 0.3  # Poor quality
        
        # Analyze downstream impact
        impact_analysis = lineage_service.analyze_quality_impact(
            graph, source_node.node_id, "downstream"
        )
        
        assert impact_analysis is not None
        assert len(impact_analysis.impacted_nodes) > 0
        assert impact_analysis.total_nodes_analyzed > 0
        
        # Check for critical impacts
        critical_impacts = impact_analysis.get_critical_impacts()
        assert isinstance(critical_impacts, list)
    
    def test_find_quality_dependencies(self, lineage_service, sample_lineage_graph):
        """Test finding quality dependencies."""
        graph, _, transform_node, _, _ = sample_lineage_graph
        
        dependencies = lineage_service.find_quality_dependencies(
            graph, transform_node.node_id, "all"
        )
        
        assert dependencies is not None
        assert 'upstream' in dependencies
        assert 'downstream' in dependencies
        assert len(dependencies['upstream']) > 0
        assert len(dependencies['downstream']) > 0
    
    def test_lineage_summary(self, lineage_service, sample_lineage_graph):
        """Test lineage summary generation."""
        graph, _, _, _, _ = sample_lineage_graph
        
        summary = lineage_service.get_quality_lineage_summary(graph)
        
        assert summary is not None
        assert 'graph_statistics' in summary
        assert 'quality_metrics' in summary
        assert 'critical_paths' in summary
        assert 'risk_assessment' in summary


class TestPredictiveQualityService:
    """Test suite for Predictive Quality Service."""
    
    @pytest.fixture
    def predictive_config(self):
        """Predictive service configuration."""
        return PredictiveQualityConfig(
            default_models=[PredictionModel.RANDOM_FOREST, PredictionModel.LINEAR_REGRESSION],
            min_historical_points=10,  # Reduced for testing
            enable_anomaly_prediction=True
        )
    
    @pytest.fixture
    def predictive_service(self, predictive_config):
        """Predictive service instance."""
        return PredictiveQualityService(predictive_config)
    
    @pytest.fixture
    def historical_quality_data(self):
        """Historical quality data for prediction."""
        data = []
        base_date = datetime.now() - timedelta(days=60)
        
        for i in range(50):
            # Simulate declining quality over time
            trend_factor = 0.9 - (i * 0.01)  # Gradual decline
            noise = np.random.normal(0, 0.05)
            
            data.append({
                'timestamp': base_date + timedelta(days=i),
                'quality_score': max(0.1, min(1.0, trend_factor + noise)),
                'completeness': max(0.1, min(1.0, trend_factor + noise + 0.1)),
                'accuracy': max(0.1, min(1.0, trend_factor + noise - 0.05)),
                'consistency': max(0.1, min(1.0, trend_factor + noise))
            })
        
        return data
    
    def test_predictive_service_initialization(self, predictive_service):
        """Test predictive service initialization."""
        assert predictive_service is not None
        assert predictive_service.config.min_historical_points == 10
        assert PredictionModel.RANDOM_FOREST in predictive_service.config.default_models
    
    def test_add_historical_data(self, predictive_service, historical_quality_data):
        """Test adding historical data."""
        dataset_id = "test_prediction_dataset"
        
        predictive_service.add_historical_data(dataset_id, historical_quality_data)
        
        assert dataset_id in predictive_service._historical_data
        df = predictive_service._historical_data[dataset_id]
        assert len(df) == len(historical_quality_data)
        assert 'timestamp' in df.columns
        assert 'quality_score' in df.columns
    
    def test_predict_quality(self, predictive_service, historical_quality_data):
        """Test quality prediction."""
        dataset_id = "test_prediction_dataset"
        
        # Add historical data
        predictive_service.add_historical_data(dataset_id, historical_quality_data)
        
        # Make prediction
        prediction = predictive_service.predict_quality(
            dataset_id, PredictionHorizon.SHORT_TERM
        )
        
        assert prediction is not None
        assert prediction.dataset_id == dataset_id
        assert 0 <= prediction.predicted_value <= 1
        assert prediction.prediction_horizon == PredictionHorizon.SHORT_TERM
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] <= prediction.predicted_value <= prediction.confidence_interval[1]
    
    def test_generate_forecast(self, predictive_service, historical_quality_data):
        """Test quality forecasting."""
        dataset_id = "test_forecast_dataset"
        
        # Add historical data
        predictive_service.add_historical_data(dataset_id, historical_quality_data)
        
        # Generate forecast
        forecast = predictive_service.generate_quality_forecast(
            dataset_id, forecast_days=14
        )
        
        assert forecast is not None
        assert forecast.dataset_id == dataset_id
        assert len(forecast.forecast_dates) == 14
        assert len(forecast.forecasted_values) == 14
        assert len(forecast.confidence_bands) == 14
        assert all(0 <= val <= 1 for val in forecast.forecasted_values)
    
    def test_anomaly_likelihood_prediction(self, predictive_service, historical_quality_data):
        """Test anomaly likelihood prediction."""
        dataset_id = "test_anomaly_prediction"
        
        # Add historical data
        predictive_service.add_historical_data(dataset_id, historical_quality_data)
        
        # Predict anomaly likelihood
        anomaly_prediction = predictive_service.predict_anomaly_likelihood(
            dataset_id, PredictionHorizon.SHORT_TERM
        )
        
        assert anomaly_prediction is not None
        assert 'anomaly_likelihood' in anomaly_prediction
        assert 'risk_level' in anomaly_prediction
        assert 0 <= anomaly_prediction['anomaly_likelihood'] <= 1
        assert anomaly_prediction['risk_level'] in ['minimal', 'low', 'medium', 'high']


class TestAdvancedQualityMetricsService:
    """Test suite for Advanced Quality Metrics Service."""
    
    @pytest.fixture
    def metrics_config(self):
        """Advanced metrics service configuration."""
        return AdvancedMetricsConfig(
            default_scoring_method=ScoringMethod.WEIGHTED_AVERAGE,
            enable_benchmarking=True,
            enable_temporal_scoring=True
        )
    
    @pytest.fixture
    def metrics_service(self, metrics_config):
        """Advanced metrics service instance."""
        return AdvancedQualityMetricsService(metrics_config)
    
    @pytest.fixture
    def sample_quality_profile(self):
        """Sample quality profile for testing."""
        # Mock quality profile with realistic values
        profile = Mock(spec=DataQualityProfile)
        profile.dataset_id = "test_metrics_dataset"
        profile.row_count = 10000
        profile.column_count = 15
        profile.completeness_score = 0.85
        profile.accuracy_score = 0.78
        profile.consistency_score = 0.82
        profile.validity_score = 0.88
        profile.uniqueness_score = 0.95
        profile.overall_quality_score = 0.83
        
        # Mock column profiles
        profile.column_profiles = {
            'email': Mock(completeness=0.9, data_type_accuracy=0.85),
            'age': Mock(completeness=0.95, data_type_accuracy=0.98),
            'salary': Mock(completeness=0.88, data_type_accuracy=0.92)
        }
        
        return profile
    
    def test_metrics_service_initialization(self, metrics_service):
        """Test metrics service initialization."""
        assert metrics_service is not None
        assert metrics_service.config.default_scoring_method == ScoringMethod.WEIGHTED_AVERAGE
        assert metrics_service.config.enable_benchmarking is True
    
    def test_calculate_advanced_score(self, metrics_service, sample_quality_profile):
        """Test advanced score calculation."""
        dataset_id = "test_advanced_metrics"
        
        # Calculate advanced score
        advanced_score = metrics_service.calculate_advanced_score(
            dataset_id, sample_quality_profile
        )
        
        assert advanced_score is not None
        assert advanced_score.dataset_id == dataset_id
        assert 0 <= advanced_score.overall_score <= 1
        assert 0 <= advanced_score.normalized_score <= 1
        assert advanced_score.quality_tier is not None
        assert len(advanced_score.confidence_interval) == 2
        assert len(advanced_score.metrics) > 0
    
    def test_individual_metrics_calculation(self, metrics_service, sample_quality_profile):
        """Test individual metrics calculation."""
        dataset_id = "test_individual_metrics"
        
        advanced_score = metrics_service.calculate_advanced_score(
            dataset_id, sample_quality_profile
        )
        
        # Check that key metrics are calculated
        assert MetricType.COMPLETENESS in advanced_score.metrics
        assert MetricType.ACCURACY in advanced_score.metrics
        assert MetricType.CONSISTENCY in advanced_score.metrics
        
        # Check metric properties
        completeness_metric = advanced_score.metrics[MetricType.COMPLETENESS]
        assert completeness_metric.metric_type == MetricType.COMPLETENESS
        assert 0 <= completeness_metric.value <= 1
        assert 0 <= completeness_metric.confidence <= 1
        assert completeness_metric.weight > 0
    
    def test_scoring_methods(self, metrics_service, sample_quality_profile):
        """Test different scoring methods."""
        dataset_id = "test_scoring_methods"
        
        # Test weighted average
        score_weighted = metrics_service.calculate_advanced_score(
            dataset_id, sample_quality_profile, scoring_method=ScoringMethod.WEIGHTED_AVERAGE
        )
        
        # Test geometric mean
        score_geometric = metrics_service.calculate_advanced_score(
            dataset_id, sample_quality_profile, scoring_method=ScoringMethod.GEOMETRIC_MEAN
        )
        
        # Test harmonic mean
        score_harmonic = metrics_service.calculate_advanced_score(
            dataset_id, sample_quality_profile, scoring_method=ScoringMethod.HARMONIC_MEAN
        )
        
        # All should produce valid scores
        assert 0 <= score_weighted.overall_score <= 1
        assert 0 <= score_geometric.overall_score <= 1
        assert 0 <= score_harmonic.overall_score <= 1
        
        # Harmonic mean should typically be lowest (most conservative)
        assert score_harmonic.overall_score <= score_weighted.overall_score
    
    def test_quality_dashboard_data(self, metrics_service, sample_quality_profile):
        """Test quality dashboard data generation."""
        # Generate some historical scores
        for i in range(5):
            dataset_id = f"test_dataset_{i}"
            metrics_service.calculate_advanced_score(dataset_id, sample_quality_profile)
        
        # Get dashboard data
        dashboard_data = metrics_service.get_quality_dashboard_data()
        
        assert dashboard_data is not None
        assert 'summary' in dashboard_data
        assert 'datasets' in dashboard_data
        assert 'benchmarks' in dashboard_data
        assert dashboard_data['summary']['total_datasets'] >= 5


class TestQualityOptimizationService:
    """Test suite for Quality Optimization Service."""
    
    @pytest.fixture
    def optimization_config(self):
        """Optimization service configuration."""
        return QualityOptimizationConfig(
            enable_plan_generation=True,
            min_quality_improvement=0.05,
            include_code_examples=True,
            include_tool_recommendations=True
        )
    
    @pytest.fixture
    def optimization_service(self, optimization_config):
        """Optimization service instance."""
        return QualityOptimizationService(optimization_config)
    
    @pytest.fixture
    def low_quality_profile(self):
        """Low quality profile for optimization testing."""
        profile = Mock(spec=DataQualityProfile)
        profile.dataset_id = "low_quality_dataset"
        profile.row_count = 5000
        profile.column_count = 10
        profile.completeness_score = 0.45  # Poor
        profile.accuracy_score = 0.52      # Poor
        profile.consistency_score = 0.38   # Poor
        profile.validity_score = 0.41      # Poor
        profile.uniqueness_score = 0.67    # Fair
        profile.overall_quality_score = 0.48  # Poor
        
        return profile
    
    @pytest.fixture
    def low_quality_advanced_score(self, metrics_service, low_quality_profile):
        """Low quality advanced score for optimization testing."""
        return metrics_service.calculate_advanced_score(
            "low_quality_dataset", low_quality_profile
        )
    
    def test_optimization_service_initialization(self, optimization_service):
        """Test optimization service initialization."""
        assert optimization_service is not None
        assert optimization_service.config.enable_plan_generation is True
        assert optimization_service.config.min_quality_improvement == 0.05
        assert len(optimization_service._strategies) > 0
    
    def test_generate_optimization_recommendations(self, optimization_service, 
                                                 low_quality_profile, low_quality_advanced_score):
        """Test optimization recommendations generation."""
        dataset_id = "test_optimization_dataset"
        
        recommendations = optimization_service.generate_optimization_recommendations(
            dataset_id, low_quality_profile, low_quality_advanced_score
        )
        
        assert recommendations is not None
        assert len(recommendations) > 0
        
        # Check recommendation properties
        rec = recommendations[0]
        assert hasattr(rec, 'recommendation_id')
        assert hasattr(rec, 'title')
        assert hasattr(rec, 'optimization_type')
        assert hasattr(rec, 'priority')
        assert hasattr(rec, 'estimated_quality_improvement')
        assert rec.estimated_quality_improvement > 0
    
    def test_create_optimization_plan(self, optimization_service, 
                                    low_quality_profile, low_quality_advanced_score):
        """Test optimization plan creation."""
        dataset_id = "test_plan_dataset"
        
        # Generate recommendations first
        recommendations = optimization_service.generate_optimization_recommendations(
            dataset_id, low_quality_profile, low_quality_advanced_score
        )
        
        # Create optimization plan
        plan = optimization_service.create_optimization_plan(
            dataset_id, recommendations, target_quality_score=0.8
        )
        
        assert plan is not None
        assert plan.dataset_id == dataset_id
        assert plan.target_quality_score == 0.8
        assert len(plan.recommendations) == len(recommendations)
        assert plan.total_estimated_improvement > 0
        assert plan.estimated_completion_weeks > 0
        assert len(plan.implementation_phases) > 0
    
    def test_quick_wins_identification(self, optimization_service,
                                     low_quality_profile, low_quality_advanced_score):
        """Test quick wins identification."""
        dataset_id = "test_quick_wins"
        
        recommendations = optimization_service.generate_optimization_recommendations(
            dataset_id, low_quality_profile, low_quality_advanced_score
        )
        
        quick_wins = optimization_service.get_quick_wins(recommendations)
        
        assert isinstance(quick_wins, list)
        # Quick wins should have simple/moderate complexity and good improvement
        for qw in quick_wins:
            assert qw.implementation_complexity.value in ['simple', 'moderate']
            assert qw.estimated_quality_improvement >= 0.1
    
    def test_impact_estimation(self, optimization_service,
                             low_quality_profile, low_quality_advanced_score):
        """Test optimization impact estimation."""
        dataset_id = "test_impact_estimation"
        
        recommendations = optimization_service.generate_optimization_recommendations(
            dataset_id, low_quality_profile, low_quality_advanced_score
        )
        
        current_score = low_quality_advanced_score.overall_score
        impact = optimization_service.estimate_optimization_impact(
            current_score, recommendations
        )
        
        assert impact is not None
        assert 'current_score' in impact
        assert 'projected_score' in impact
        assert 'total_improvement' in impact
        assert 'total_effort_hours' in impact
        assert impact['current_score'] == current_score
        assert impact['projected_score'] > current_score
        assert impact['total_improvement'] > 0
    
    def test_optimization_report_generation(self, optimization_service,
                                          low_quality_profile, low_quality_advanced_score):
        """Test optimization report generation."""
        dataset_id = "test_optimization_report"
        
        recommendations = optimization_service.generate_optimization_recommendations(
            dataset_id, low_quality_profile, low_quality_advanced_score
        )
        
        current_score = low_quality_advanced_score.overall_score
        report = optimization_service.generate_optimization_report(
            dataset_id, recommendations, current_score
        )
        
        assert report is not None
        assert 'dataset_id' in report
        assert 'current_quality_assessment' in report
        assert 'optimization_summary' in report
        assert 'impact_analysis' in report
        assert 'quick_wins' in report
        assert 'implementation_roadmap' in report
        assert report['dataset_id'] == dataset_id


class TestIntegratedQualityAnalytics:
    """Test suite for integrated quality analytics functionality."""
    
    @pytest.fixture
    def integrated_services(self):
        """Set up integrated services for end-to-end testing."""
        services = {
            'ml_detection': MLQualityDetectionService(),
            'rule_discovery': IntelligentRuleDiscoveryService(),
            'lineage': QualityLineageService(),
            'predictive': PredictiveQualityService(),
            'metrics': AdvancedQualityMetricsService(),
            'optimization': QualityOptimizationService()
        }
        return services
    
    @pytest.fixture
    def sample_quality_scenario(self):
        """Create a realistic quality scenario for testing."""
        # Mock quality profile with various issues
        profile = Mock(spec=DataQualityProfile)
        profile.dataset_id = "integrated_test_dataset"
        profile.row_count = 15000
        profile.column_count = 20
        profile.completeness_score = 0.72
        profile.accuracy_score = 0.68
        profile.consistency_score = 0.75
        profile.validity_score = 0.71
        profile.uniqueness_score = 0.89
        profile.overall_quality_score = 0.73
        
        # Historical quality data showing decline
        historical_data = []
        base_date = datetime.now() - timedelta(days=90)
        
        for i in range(90):
            # Simulate gradual quality decline
            decline_factor = 1.0 - (i * 0.003)  # 0.3% decline per day
            noise = np.random.normal(0, 0.02)
            
            historical_data.append({
                'timestamp': base_date + timedelta(days=i),
                'quality_score': max(0.5, min(1.0, 0.9 * decline_factor + noise)),
                'completeness': max(0.5, min(1.0, 0.92 * decline_factor + noise)),
                'accuracy': max(0.5, min(1.0, 0.88 * decline_factor + noise))
            })
        
        return profile, historical_data
    
    def test_end_to_end_quality_analysis(self, integrated_services, sample_quality_scenario):
        """Test end-to-end quality analysis workflow."""
        profile, historical_data = sample_quality_scenario
        dataset_id = profile.dataset_id
        
        # 1. Calculate advanced metrics
        advanced_score = integrated_services['metrics'].calculate_advanced_score(
            dataset_id, profile, historical_data
        )
        
        assert advanced_score is not None
        assert advanced_score.overall_score > 0
        
        # 2. Generate optimization recommendations
        recommendations = integrated_services['optimization'].generate_optimization_recommendations(
            dataset_id, profile, advanced_score, historical_data
        )
        
        assert len(recommendations) > 0
        
        # 3. Create lineage graph and analyze impact
        lineage_graph = integrated_services['lineage'].create_lineage_graph(
            "Test Lineage", "End-to-end test lineage"
        )
        
        dataset_node = integrated_services['lineage'].add_dataset_node(
            lineage_graph, dataset_id, "Test Dataset", profile
        )
        
        # Add downstream system
        ml_model_node = integrated_services['lineage'].add_downstream_system_node(
            lineage_graph, "prediction_model", "ml_model", [dataset_node]
        )
        
        # Analyze impact
        impact_analysis = integrated_services['lineage'].analyze_quality_impact(
            lineage_graph, dataset_node.node_id, "downstream"
        )
        
        assert impact_analysis is not None
        assert len(impact_analysis.impacted_nodes) > 0
        
        # 4. Add predictive analysis
        integrated_services['predictive'].add_historical_data(dataset_id, historical_data)
        
        prediction = integrated_services['predictive'].predict_quality(
            dataset_id, PredictionHorizon.MEDIUM_TERM
        )
        
        assert prediction is not None
        
        # 5. Generate comprehensive report
        optimization_report = integrated_services['optimization'].generate_optimization_report(
            dataset_id, recommendations, advanced_score.overall_score
        )
        
        assert optimization_report is not None
        assert 'impact_analysis' in optimization_report
        assert 'recommendations_by_priority' in optimization_report
    
    def test_anomaly_detection_and_optimization_integration(self, integrated_services):
        """Test integration between anomaly detection and optimization."""
        dataset_id = "anomaly_optimization_test"
        
        # Generate training data with known patterns
        training_data = []
        for i in range(50):
            training_data.append({
                'timestamp': datetime.now() - timedelta(days=i),
                'completeness_score': np.random.normal(0.85, 0.05),
                'accuracy_score': np.random.normal(0.8, 0.05),
                'row_count': np.random.randint(9000, 11000)
            })
        
        # Train ML models
        training_results = integrated_services['ml_detection'].train_anomaly_detection_models(
            dataset_id, training_data
        )
        
        assert training_results is not None
        
        # Detect anomaly with poor quality data
        anomalous_data = {
            'timestamp': datetime.now(),
            'completeness_score': 0.3,  # Very low
            'accuracy_score': 0.25,     # Very low
            'row_count': 2000           # Much lower than normal
        }
        
        anomalies = integrated_services['ml_detection'].detect_quality_anomalies(
            dataset_id, anomalous_data
        )
        
        assert len(anomalies) > 0
        
        # Use anomaly information to generate targeted optimization recommendations
        anomaly = anomalies[0]
        
        # Mock quality profile based on anomaly
        poor_profile = Mock(spec=DataQualityProfile)
        poor_profile.dataset_id = dataset_id
        poor_profile.completeness_score = anomalous_data['completeness_score']
        poor_profile.accuracy_score = anomalous_data['accuracy_score']
        poor_profile.consistency_score = 0.4
        poor_profile.validity_score = 0.35
        poor_profile.uniqueness_score = 0.8
        poor_profile.overall_quality_score = 0.42
        
        advanced_score = integrated_services['metrics'].calculate_advanced_score(
            dataset_id, poor_profile
        )
        
        recommendations = integrated_services['optimization'].generate_optimization_recommendations(
            dataset_id, poor_profile, advanced_score
        )
        
        # Should generate critical priority recommendations
        critical_recs = [r for r in recommendations 
                        if r.priority == OptimizationPriority.CRITICAL]
        assert len(critical_recs) > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])