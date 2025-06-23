"""Comprehensive tests for Explainability services."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, List, Any
import uuid
import numpy as np

from tests.conftest_dependencies import requires_dependency, requires_dependencies

from pynomaly.application.services.explainability_service import ExplainabilityService
from pynomaly.application.dto.explainability_dto import (
    ExplanationRequestDTO,
    ExplanationResponseDTO,
    FeatureImportanceRequestDTO,
    FeatureImportanceResponseDTO,
    CohortExplanationRequestDTO,
    CohortExplanationResponseDTO,
    ExplanationComparisonRequestDTO,
    ExplanationComparisonResponseDTO
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.exceptions import ProcessingError, ValidationError


@requires_dependencies('shap', 'lime')
class TestExplainabilityService:
    """Test Explainability service functionality."""
    
    @pytest.fixture
    def explainability_service(self):
        """Create an explainability service instance."""
        return ExplainabilityService(
            shap_explainer=Mock(),
            lime_explainer=Mock(),
            feature_analyzer=Mock(),
            explanation_aggregator=Mock()
        )
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return Dataset(
            id=str(uuid.uuid4()),
            name="test_dataset",
            data=Mock(),
            features=["temperature", "humidity", "pressure", "vibration"],
            metadata={
                "rows": 1000,
                "columns": 4,
                "feature_types": {
                    "temperature": "numerical",
                    "humidity": "numerical", 
                    "pressure": "numerical",
                    "vibration": "numerical"
                }
            }
        )
    
    @pytest.fixture
    def sample_detector(self):
        """Create a sample detector for testing."""
        return Detector(
            id=str(uuid.uuid4()),
            name="test_detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100, "contamination": 0.1},
            is_fitted=True
        )
    
    @pytest.fixture
    def sample_anomaly(self):
        """Create a sample anomaly for testing."""
        return Anomaly(
            id=str(uuid.uuid4()),
            score=0.85,
            data_point={"temperature": 95.2, "humidity": 30.1, "pressure": 1013.2, "vibration": 0.8},
            features=["temperature", "humidity", "pressure", "vibration"],
            metadata={"index": 42, "timestamp": "2024-01-15T10:30:00Z"}
        )
    
    @pytest.mark.asyncio
    async def test_shap_explanation_generation(self, explainability_service, sample_detector, sample_anomaly):
        """Test SHAP explanation generation for anomalies."""
        request = ExplanationRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=sample_anomaly.data_point,
            explanation_method="shap",
            explanation_type="local",
            feature_names=sample_anomaly.features,
            model_type="tree_based"
        )
        
        # Mock SHAP explainer
        explainability_service.shap_explainer.explain_instance.return_value = {
            "feature_contributions": {
                "temperature": 0.45,
                "humidity": -0.12,
                "pressure": 0.08,
                "vibration": 0.23
            },
            "base_value": 0.1,
            "predicted_value": 0.85,
            "explanation_metadata": {
                "explainer_type": "TreeExplainer",
                "model_output": "anomaly_score"
            }
        }
        
        response = await explainability_service.explain_anomaly(request)
        
        assert isinstance(response, ExplanationResponseDTO)
        assert response.explanation_method == "shap"
        assert response.explanation_type == "local"
        assert "temperature" in response.feature_contributions
        assert response.feature_contributions["temperature"] == 0.45
        assert response.overall_confidence > 0
    
    @pytest.mark.asyncio
    async def test_lime_explanation_generation(self, explainability_service, sample_detector, sample_anomaly):
        """Test LIME explanation generation for anomalies."""
        request = ExplanationRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=sample_anomaly.data_point,
            explanation_method="lime",
            explanation_type="local",
            feature_names=sample_anomaly.features,
            lime_config={
                "num_features": 4,
                "num_samples": 5000,
                "distance_metric": "euclidean"
            }
        )
        
        # Mock LIME explainer
        explainability_service.lime_explainer.explain_instance.return_value = {
            "feature_contributions": {
                "temperature": 0.42,
                "humidity": -0.08,
                "pressure": 0.05,
                "vibration": 0.31
            },
            "local_score": 0.83,
            "explanation_metadata": {
                "num_features_used": 4,
                "num_samples_generated": 5000,
                "r2_score": 0.95
            }
        }
        
        response = await explainability_service.explain_anomaly(request)
        
        assert isinstance(response, ExplanationResponseDTO)
        assert response.explanation_method == "lime"
        assert response.explanation_type == "local"
        assert response.feature_contributions["temperature"] == 0.42
        assert response.explanation_metadata["r2_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_global_feature_importance(self, explainability_service, sample_detector, sample_dataset):
        """Test global feature importance analysis."""
        request = FeatureImportanceRequestDTO(
            detector_id=sample_detector.id,
            dataset_id=sample_dataset.id,
            importance_method="shap_global",
            aggregation_method="mean_absolute",
            sample_size=1000
        )
        
        # Mock feature analyzer
        explainability_service.feature_analyzer.compute_global_importance.return_value = {
            "feature_importance": {
                "temperature": 0.35,
                "vibration": 0.28,
                "pressure": 0.22,
                "humidity": 0.15
            },
            "importance_statistics": {
                "total_importance": 1.0,
                "top_feature": "temperature",
                "importance_distribution": "moderate_concentration"
            },
            "confidence_intervals": {
                "temperature": {"lower": 0.32, "upper": 0.38},
                "vibration": {"lower": 0.24, "upper": 0.32},
                "pressure": {"lower": 0.18, "upper": 0.26},
                "humidity": {"lower": 0.12, "upper": 0.18}
            }
        }
        
        response = await explainability_service.analyze_feature_importance(request)
        
        assert isinstance(response, FeatureImportanceResponseDTO)
        assert response.feature_importance["temperature"] == 0.35
        assert response.top_features[0] == "temperature"
        assert "confidence_intervals" in response.analysis_metadata
    
    @pytest.mark.asyncio
    async def test_cohort_explanation_analysis(self, explainability_service, sample_detector, sample_dataset):
        """Test cohort-based explanation analysis."""
        request = CohortExplanationRequestDTO(
            detector_id=sample_detector.id,
            dataset_id=sample_dataset.id,
            cohort_definitions=[
                {
                    "name": "high_temperature",
                    "condition": "temperature > 80",
                    "description": "High temperature conditions"
                },
                {
                    "name": "high_vibration",
                    "condition": "vibration > 0.5",
                    "description": "High vibration conditions"
                }
            ],
            explanation_method="shap",
            comparison_baseline="global"
        )
        
        # Mock cohort analysis
        explainability_service.explanation_aggregator.analyze_cohorts.return_value = {
            "cohort_explanations": {
                "high_temperature": {
                    "feature_importance": {
                        "temperature": 0.55,
                        "pressure": 0.25,
                        "humidity": 0.12,
                        "vibration": 0.08
                    },
                    "anomaly_patterns": ["temperature_spikes", "pressure_drops"],
                    "cohort_size": 156
                },
                "high_vibration": {
                    "feature_importance": {
                        "vibration": 0.48,
                        "temperature": 0.22,
                        "pressure": 0.20,
                        "humidity": 0.10
                    },
                    "anomaly_patterns": ["vibration_spikes", "temperature_correlation"],
                    "cohort_size": 203
                }
            },
            "cohort_comparisons": {
                "high_temperature_vs_global": {
                    "temperature_diff": 0.20,
                    "significance": "high"
                },
                "high_vibration_vs_global": {
                    "vibration_diff": 0.20,
                    "significance": "high"
                }
            }
        }
        
        response = await explainability_service.explain_cohorts(request)
        
        assert isinstance(response, CohortExplanationResponseDTO)
        assert "high_temperature" in response.cohort_explanations
        assert "high_vibration" in response.cohort_explanations
        assert response.cohort_explanations["high_temperature"]["cohort_size"] == 156
        assert "temperature_spikes" in response.cohort_explanations["high_temperature"]["anomaly_patterns"]
    
    @pytest.mark.asyncio
    async def test_explanation_method_comparison(self, explainability_service, sample_detector, sample_anomaly):
        """Test comparison between different explanation methods."""
        request = ExplanationComparisonRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=sample_anomaly.data_point,
            explanation_methods=["shap", "lime"],
            feature_names=sample_anomaly.features,
            comparison_metrics=["correlation", "rank_similarity", "magnitude_difference"]
        )
        
        # Mock explanation comparison
        shap_result = {
            "temperature": 0.45, "humidity": -0.12, "pressure": 0.08, "vibration": 0.23
        }
        lime_result = {
            "temperature": 0.42, "humidity": -0.08, "pressure": 0.05, "vibration": 0.31
        }
        
        explainability_service.shap_explainer.explain_instance.return_value = {
            "feature_contributions": shap_result
        }
        explainability_service.lime_explainer.explain_instance.return_value = {
            "feature_contributions": lime_result
        }
        
        # Mock comparison metrics
        with patch.object(explainability_service, '_compare_explanations') as mock_compare:
            mock_compare.return_value = {
                "correlation": 0.89,
                "rank_similarity": 0.75,
                "magnitude_difference": 0.15,
                "agreement_score": 0.82,
                "disagreement_features": ["vibration"],
                "consensus_features": ["temperature", "humidity", "pressure"]
            }
            
            response = await explainability_service.compare_explanations(request)
            
            assert isinstance(response, ExplanationComparisonResponseDTO)
            assert "shap" in response.method_explanations
            assert "lime" in response.method_explanations
            assert response.comparison_metrics["correlation"] == 0.89
            assert "vibration" in response.disagreement_analysis["disagreement_features"]
    
    @pytest.mark.asyncio
    async def test_temporal_explanation_analysis(self, explainability_service, sample_detector):
        """Test temporal explanation analysis for time-series anomalies."""
        temporal_data = [
            {"timestamp": "2024-01-15T10:00:00Z", "temperature": 75.2, "humidity": 45.1},
            {"timestamp": "2024-01-15T10:05:00Z", "temperature": 78.1, "humidity": 43.8},
            {"timestamp": "2024-01-15T10:10:00Z", "temperature": 95.2, "humidity": 30.1},  # Anomaly
            {"timestamp": "2024-01-15T10:15:00Z", "temperature": 76.8, "humidity": 44.2}
        ]
        
        request = ExplanationRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=temporal_data[2],
            explanation_method="shap",
            explanation_type="temporal",
            temporal_context=temporal_data,
            temporal_config={
                "window_size": 3,
                "include_trends": True,
                "include_seasonality": False
            }
        )
        
        # Mock temporal explanation
        explainability_service.shap_explainer.explain_temporal.return_value = {
            "temporal_contributions": {
                "temperature_current": 0.35,
                "temperature_trend": 0.25,
                "humidity_current": -0.15,
                "humidity_trend": -0.08
            },
            "temporal_patterns": {
                "sudden_temperature_spike": 0.40,
                "humidity_drop_correlation": 0.20
            },
            "context_importance": {
                "previous_1_step": 0.30,
                "previous_2_steps": 0.15,
                "trend_component": 0.25
            }
        }
        
        response = await explainability_service.explain_anomaly(request)
        
        assert response.explanation_type == "temporal"
        assert "temperature_trend" in response.feature_contributions
        assert "temporal_patterns" in response.explanation_metadata
    
    @pytest.mark.asyncio
    async def test_model_agnostic_explanations(self, explainability_service):
        """Test model-agnostic explanation generation."""
        # Test with different model types
        model_types = ["tree_based", "neural_network", "ensemble", "distance_based"]
        
        for model_type in model_types:
            detector_id = str(uuid.uuid4())
            request = ExplanationRequestDTO(
                detector_id=detector_id,
                anomaly_data={"feature1": 1.5, "feature2": -0.8},
                explanation_method="shap",
                explanation_type="local",
                model_type=model_type
            )
            
            # Mock model-specific explanations
            if model_type == "tree_based":
                explainer_result = {"explainer_type": "TreeExplainer"}
            elif model_type == "neural_network":
                explainer_result = {"explainer_type": "DeepExplainer"}
            elif model_type == "ensemble":
                explainer_result = {"explainer_type": "EnsembleExplainer"}
            else:
                explainer_result = {"explainer_type": "KernelExplainer"}
            
            explainability_service.shap_explainer.get_explainer_for_model.return_value = explainer_result
            explainability_service.shap_explainer.explain_instance.return_value = {
                "feature_contributions": {"feature1": 0.6, "feature2": -0.3},
                "explanation_metadata": explainer_result
            }
            
            response = await explainability_service.explain_anomaly(request)
            
            assert explainer_result["explainer_type"] in response.explanation_metadata["explainer_type"]
    
    @pytest.mark.asyncio
    async def test_explanation_confidence_assessment(self, explainability_service, sample_detector, sample_anomaly):
        """Test explanation confidence and reliability assessment."""
        request = ExplanationRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=sample_anomaly.data_point,
            explanation_method="shap",
            explanation_type="local",
            confidence_assessment=True,
            bootstrap_samples=100
        )
        
        # Mock confidence assessment
        explainability_service.shap_explainer.assess_explanation_confidence.return_value = {
            "confidence_score": 0.87,
            "stability_metrics": {
                "feature_contribution_variance": 0.05,
                "rank_stability": 0.92,
                "magnitude_stability": 0.89
            },
            "reliability_indicators": {
                "model_fidelity": 0.95,
                "local_approximation_quality": 0.88,
                "feature_correlation_impact": 0.12
            },
            "confidence_intervals": {
                "temperature": {"lower": 0.40, "upper": 0.50},
                "humidity": {"lower": -0.15, "upper": -0.09}
            }
        }
        
        response = await explainability_service.explain_anomaly(request)
        
        assert response.overall_confidence == 0.87
        assert "stability_metrics" in response.explanation_metadata
        assert "confidence_intervals" in response.explanation_metadata
    
    @pytest.mark.asyncio
    async def test_explanation_validation_and_errors(self, explainability_service):
        """Test explanation validation and error handling."""
        # Test invalid explanation method
        invalid_request = ExplanationRequestDTO(
            detector_id=str(uuid.uuid4()),
            anomaly_data={"feature1": 1.0},
            explanation_method="invalid_method",
            explanation_type="local"
        )
        
        with pytest.raises(ValidationError):
            await explainability_service.explain_anomaly(invalid_request)
        
        # Test missing detector
        explainability_service.shap_explainer.get_detector.return_value = None
        
        valid_request = ExplanationRequestDTO(
            detector_id=str(uuid.uuid4()),
            anomaly_data={"feature1": 1.0},
            explanation_method="shap",
            explanation_type="local"
        )
        
        with pytest.raises(ProcessingError, match="Detector not found"):
            await explainability_service.explain_anomaly(valid_request)
    
    @pytest.mark.asyncio
    async def test_explanation_caching(self, explainability_service, sample_detector, sample_anomaly):
        """Test explanation result caching."""
        request = ExplanationRequestDTO(
            detector_id=sample_detector.id,
            anomaly_data=sample_anomaly.data_point,
            explanation_method="shap",
            explanation_type="local",
            use_cache=True
        )
        
        # Mock caching behavior
        cached_result = {
            "feature_contributions": {"temperature": 0.45, "humidity": -0.12},
            "cached": True,
            "cache_timestamp": datetime.now().isoformat()
        }
        
        with patch.object(explainability_service, '_get_cached_explanation') as mock_cache:
            mock_cache.return_value = cached_result
            
            response = await explainability_service.explain_anomaly(request)
            
            assert response.explanation_metadata.get("cached") is True
            assert "cache_timestamp" in response.explanation_metadata


class TestExplainabilityIntegration:
    """Integration tests for explainability services."""
    
    @pytest.fixture
    def explainability_system(self):
        """Create a complete explainability system setup."""
        return {
            'explainability_service': ExplainabilityService(
                shap_explainer=Mock(),
                lime_explainer=Mock(),
                feature_analyzer=Mock(),
                explanation_aggregator=Mock()
            ),
            'detector': Detector(
                id=str(uuid.uuid4()),
                name="integration_detector",
                algorithm_name="IsolationForest",
                parameters={"n_estimators": 100},
                is_fitted=True
            ),
            'dataset': Dataset(
                id=str(uuid.uuid4()),
                name="integration_dataset",
                data=Mock(),
                features=["f1", "f2", "f3", "f4"]
            )
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_explanation_workflow(self, explainability_system):
        """Test complete end-to-end explanation workflow."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        anomaly_data = {"f1": 2.5, "f2": -1.8, "f3": 0.9, "f4": 3.2}
        
        # 1. Generate local explanation
        local_request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data=anomaly_data,
            explanation_method="shap",
            explanation_type="local"
        )
        local_response = await service.explain_anomaly(local_request)
        assert local_response.explanation_type == "local"
        
        # 2. Analyze global feature importance
        importance_request = FeatureImportanceRequestDTO(
            detector_id=detector.id,
            dataset_id=explainability_system['dataset'].id,
            importance_method="shap_global"
        )
        importance_response = await service.analyze_feature_importance(importance_request)
        assert len(importance_response.feature_importance) == 4
        
        # 3. Compare explanation methods
        comparison_request = ExplanationComparisonRequestDTO(
            detector_id=detector.id,
            anomaly_data=anomaly_data,
            explanation_methods=["shap", "lime"],
            feature_names=["f1", "f2", "f3", "f4"]
        )
        comparison_response = await service.compare_explanations(comparison_request)
        assert len(comparison_response.method_explanations) == 2
    
    @pytest.mark.asyncio
    async def test_explanation_consistency_validation(self, explainability_system):
        """Test explanation consistency across multiple runs."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        anomaly_data = {"f1": 1.5, "f2": -0.8, "f3": 2.1, "f4": 0.3}
        
        # Generate multiple explanations
        explanations = []
        for i in range(5):
            request = ExplanationRequestDTO(
                detector_id=detector.id,
                anomaly_data=anomaly_data,
                explanation_method="shap",
                explanation_type="local",
                random_seed=42  # Fixed seed for consistency
            )
            response = await service.explain_anomaly(request)
            explanations.append(response.feature_contributions)
        
        # Check consistency (all explanations should be identical with fixed seed)
        for explanation in explanations[1:]:
            assert explanation == explanations[0]
    
    @pytest.mark.asyncio
    async def test_explainability_performance_optimization(self, explainability_system):
        """Test explainability performance optimization."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        # Test with performance optimization enabled
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data={"f1": 1.0, "f2": 2.0, "f3": -1.0, "f4": 0.5},
            explanation_method="shap",
            explanation_type="local",
            performance_mode="fast",
            max_explanation_time=5.0
        )
        
        # Mock fast explanation
        service.shap_explainer.explain_instance_fast.return_value = {
            "feature_contributions": {"f1": 0.3, "f2": 0.4, "f3": -0.2, "f4": 0.1},
            "performance_metrics": {
                "explanation_time": 2.1,
                "approximation_quality": 0.92
            }
        }
        
        response = await service.explain_anomaly(request)
        
        assert response.explanation_metadata["explanation_time"] < 5.0
        assert response.explanation_metadata["approximation_quality"] > 0.9
    
    @pytest.mark.asyncio
    async def test_multimodal_explanation_analysis(self, explainability_system):
        """Test multimodal explanation for different data types."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        # Test with mixed data types
        multimodal_data = {
            "numerical": {"temperature": 95.2, "pressure": 1013.2},
            "categorical": {"location": "zone_A", "sensor_type": "thermal"},
            "text": {"description": "High temperature anomaly detected"},
            "image": {"features": [0.1, 0.8, 0.3, 0.9]}  # Extracted features
        }
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data=multimodal_data,
            explanation_method="multimodal_shap",
            explanation_type="local",
            data_modalities=["numerical", "categorical", "text", "image"]
        )
        
        response = await service.explain_multimodal_anomaly(request)
        assert len(response.modality_explanations) == 4
        assert "numerical" in response.modality_explanations
    
    @pytest.mark.asyncio
    async def test_explanation_drift_detection(self, explainability_system):
        """Test explanation drift detection over time."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        # Historical explanations
        historical_explanations = [
            {"timestamp": "2024-01-01", "feature_importance": {"f1": 0.4, "f2": 0.3, "f3": 0.3}},
            {"timestamp": "2024-01-02", "feature_importance": {"f1": 0.42, "f2": 0.28, "f3": 0.3}},
            {"timestamp": "2024-01-03", "feature_importance": {"f1": 0.6, "f2": 0.2, "f3": 0.2}}  # Drift
        ]
        
        drift_analysis = await service.analyze_explanation_drift(
            detector_id=detector.id,
            historical_explanations=historical_explanations,
            drift_threshold=0.15
        )
        
        assert drift_analysis["drift_detected"] is True
        assert drift_analysis["drift_magnitude"] > 0.15
        assert "f1" in drift_analysis["drifted_features"]
    
    @pytest.mark.asyncio
    async def test_contrastive_explanations(self, explainability_system):
        """Test contrastive explanations comparing normal vs anomalous instances."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        anomalous_instance = {"temperature": 95.2, "humidity": 30.1, "pressure": 1013.2}
        normal_instance = {"temperature": 75.2, "humidity": 45.1, "pressure": 1013.5}
        
        request = ExplanationComparisonRequestDTO(
            detector_id=detector.id,
            anomaly_data=anomalous_instance,
            comparison_data=normal_instance,
            explanation_methods=["contrastive_shap"],
            contrastive_config={"focus_on_differences": True}
        )
        
        response = await service.generate_contrastive_explanation(request)
        assert response.contrastive_analysis["key_differences"] is not None
        assert "temperature" in response.contrastive_analysis["discriminative_features"]
    
    @pytest.mark.asyncio
    async def test_hierarchical_explanations(self, explainability_system):
        """Test hierarchical explanations at different levels of granularity."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        hierarchical_data = {
            "system_level": {"total_power": 1500, "efficiency": 0.85},
            "component_level": {"motor_temp": 95.2, "bearing_vibration": 0.8},
            "sensor_level": {"temp_sensor_1": 94.8, "temp_sensor_2": 95.6}
        }
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data=hierarchical_data,
            explanation_method="hierarchical_shap",
            explanation_type="hierarchical",
            hierarchy_levels=["system", "component", "sensor"]
        )
        
        response = await service.explain_hierarchical_anomaly(request)
        assert len(response.hierarchical_explanations) == 3
        assert "system" in response.hierarchical_explanations
    
    @pytest.mark.asyncio
    async def test_causal_explanation_analysis(self, explainability_system):
        """Test causal explanation analysis."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        causal_data = {
            "features": {"temperature": 95.2, "pressure": 990.1, "flow_rate": 15.2},
            "causal_graph": {
                "temperature": ["pressure", "flow_rate"],
                "pressure": ["flow_rate"]
            }
        }
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data=causal_data["features"],
            explanation_method="causal_shap",
            explanation_type="causal",
            causal_config={"causal_graph": causal_data["causal_graph"]}
        )
        
        response = await service.explain_causal_relationships(request)
        assert response.causal_explanations["direct_causes"] is not None
        assert response.causal_explanations["indirect_effects"] is not None
    
    @pytest.mark.asyncio
    async def test_explanation_aggregation_strategies(self, explainability_system):
        """Test different explanation aggregation strategies."""
        service = explainability_system['explainability_service']
        
        individual_explanations = [
            {"instance_1": {"f1": 0.4, "f2": 0.3, "f3": 0.3}},
            {"instance_2": {"f1": 0.5, "f2": 0.2, "f3": 0.3}},
            {"instance_3": {"f1": 0.3, "f2": 0.4, "f3": 0.3}}
        ]
        
        aggregation_strategies = ["mean", "median", "weighted_mean", "consensus"]
        
        for strategy in aggregation_strategies:
            aggregated = await service.aggregate_explanations(
                explanations=individual_explanations,
                aggregation_method=strategy,
                weights=[0.4, 0.4, 0.2] if strategy == "weighted_mean" else None
            )
            
            assert "f1" in aggregated.aggregated_importance
            assert aggregated.aggregation_method == strategy
    
    @pytest.mark.asyncio
    async def test_explanation_quality_assessment(self, explainability_system):
        """Test explanation quality assessment and validation."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data={"f1": 2.5, "f2": -1.8, "f3": 0.9},
            explanation_method="shap",
            explanation_type="local",
            quality_assessment=True
        )
        
        response = await service.explain_with_quality_assessment(request)
        
        quality_metrics = response.quality_assessment
        assert "fidelity" in quality_metrics
        assert "stability" in quality_metrics
        assert "consistency" in quality_metrics
        assert quality_metrics["overall_quality_score"] > 0
    
    @pytest.mark.asyncio
    async def test_real_time_explanation_generation(self, explainability_system):
        """Test real-time explanation generation for streaming data."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        streaming_config = {
            "batch_size": 10,
            "max_latency": 100,  # milliseconds
            "explanation_cache": True,
            "incremental_updates": True
        }
        
        data_stream = [
            {"f1": 1.0, "f2": 2.0, "f3": -1.0},
            {"f1": 1.2, "f2": 2.1, "f3": -0.9},
            {"f1": 3.5, "f2": 1.8, "f3": -2.1}  # Anomaly
        ]
        
        for data_point in data_stream:
            request = ExplanationRequestDTO(
                detector_id=detector.id,
                anomaly_data=data_point,
                explanation_method="fast_shap",
                explanation_type="local",
                streaming_config=streaming_config
            )
            
            response = await service.explain_streaming_anomaly(request)
            assert response.explanation_metadata["latency"] <= 100
    
    @pytest.mark.asyncio
    async def test_cross_model_explanation_comparison(self, explainability_system):
        """Test explanation comparison across different models."""
        service = explainability_system['explainability_service']
        
        models = [
            {"id": "model_1", "type": "IsolationForest"},
            {"id": "model_2", "type": "LocalOutlierFactor"},
            {"id": "model_3", "type": "OneClassSVM"}
        ]
        
        anomaly_data = {"f1": 2.5, "f2": -1.8, "f3": 0.9, "f4": 3.2}
        
        cross_model_explanations = await service.compare_explanations_across_models(
            models=models,
            anomaly_data=anomaly_data,
            explanation_method="shap"
        )
        
        assert len(cross_model_explanations.model_explanations) == 3
        assert cross_model_explanations.consensus_explanation is not None
        assert cross_model_explanations.model_agreement_score > 0
    
    @pytest.mark.asyncio
    async def test_explanation_personalization(self, explainability_system):
        """Test personalized explanations based on user preferences."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        user_preferences = {
            "expertise_level": "expert",
            "explanation_style": "technical",
            "focus_areas": ["feature_importance", "confidence_intervals"],
            "visualization_preferences": ["bar_chart", "waterfall"],
            "detail_level": "high"
        }
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data={"f1": 2.5, "f2": -1.8},
            explanation_method="shap",
            explanation_type="local",
            personalization=user_preferences
        )
        
        response = await service.generate_personalized_explanation(request)
        assert response.personalization_info["expertise_level"] == "expert"
        assert "confidence_intervals" in response.explanation_components
    
    @pytest.mark.asyncio
    async def test_explanation_natural_language_generation(self, explainability_system):
        """Test natural language explanation generation."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        request = ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data={"temperature": 95.2, "humidity": 30.1, "pressure": 1013.2},
            explanation_method="shap",
            explanation_type="local",
            natural_language_config={
                "generate_text": True,
                "language": "english",
                "explanation_template": "technical"
            }
        )
        
        response = await service.explain_with_natural_language(request)
        assert response.natural_language_explanation is not None
        assert len(response.natural_language_explanation) > 50  # Meaningful explanation
        assert "temperature" in response.natural_language_explanation.lower()
    
    @pytest.mark.asyncio
    async def test_explanation_interactive_exploration(self, explainability_system):
        """Test interactive explanation exploration capabilities."""
        service = explainability_system['explainability_service']
        detector = explainability_system['detector']
        
        base_explanation = await service.explain_anomaly(ExplanationRequestDTO(
            detector_id=detector.id,
            anomaly_data={"f1": 2.5, "f2": -1.8, "f3": 0.9},
            explanation_method="shap",
            explanation_type="local"
        ))
        
        # Interactive what-if analysis
        what_if_scenarios = [
            {"f1": 1.5, "f2": -1.8, "f3": 0.9},  # Reduce f1
            {"f1": 2.5, "f2": -0.5, "f3": 0.9},  # Increase f2
        ]
        
        interactive_analysis = await service.explore_counterfactuals(
            base_explanation=base_explanation,
            what_if_scenarios=what_if_scenarios,
            detector_id=detector.id
        )
        
        assert len(interactive_analysis.counterfactual_explanations) == 2
        assert interactive_analysis.feature_sensitivity_analysis is not None