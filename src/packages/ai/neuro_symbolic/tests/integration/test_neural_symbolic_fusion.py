"""Integration tests for neural-symbolic reasoning workflows."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from neuro_symbolic.application.services.neuro_symbolic_reasoning_service import NeuroSymbolicReasoningService
from neuro_symbolic.application.services.explainable_reasoning_service import ExplainableReasoningService
from neuro_symbolic.infrastructure.neural_adapters import TransformerBackbone, CNNBackbone
from neuro_symbolic.infrastructure.symbolic_adapters import PropositionalReasoner, FirstOrderReasoner
from neuro_symbolic.infrastructure.persistence.model_repository import ModelRepository
from neuro_symbolic.domain.entities.neuro_symbolic_model import NeuroSymbolicModel
from neuro_symbolic.domain.value_objects.reasoning_result import (
    NeuroSymbolicReasoningResult, FusionStrategy
)


@pytest.fixture
def sample_neural_data():
    """Generate sample neural network input data."""
    return np.random.randn(10, 64)  # 10 samples, 64 features


@pytest.fixture
def sample_symbolic_knowledge():
    """Generate sample symbolic knowledge base."""
    return {
        "rules": [
            "temperature > 80 -> high_class",
            "pressure < 10 -> low_class", 
            "cpu_usage > 90 -> critical_class",
            "memory_usage > 85 AND disk_usage > 95 -> overload_class"
        ],
        "facts": [
            "sensor_type(temp_sensor_1, temperature)",
            "sensor_type(pressure_sensor_1, pressure)",
            "threshold(temperature, 75)",
            "threshold(pressure, 15)"
        ],
        "constraints": [
            "0 <= temperature <= 150",
            "0 <= pressure <= 100",
            "0 <= cpu_usage <= 100",
            "0 <= memory_usage <= 100"
        ]
    }


@pytest.fixture
def mock_trained_model():
    """Create a mock trained neuro-symbolic model."""
    model = Mock(spec=NeuroSymbolicModel)
    model.id = "test_model_001"
    model.name = "Integration Test Model"
    model.is_trained = True
    model.neural_backbone = "transformer"
    model.symbolic_reasoner = "first_order_logic"
    
    # Mock neural predictions
    model.neural_predict.return_value = {
        "predictions": np.array([0.8, 0.2, 0.9, 0.1, 0.95]),
        "confidence": np.array([0.92, 0.88, 0.94, 0.85, 0.96]),
        "embeddings": np.random.randn(5, 32),
        "attention_weights": np.random.randn(5, 8, 10, 10)
    }
    
    # Mock symbolic reasoning
    model.symbolic_reason.return_value = {
        "conclusions": ["positive", "negative", "positive", "negative", "critical_positive"],
        "confidence": [0.9, 0.85, 0.88, 0.82, 0.95],
        "applied_rules": [
            "Rule 1: temperature > 80",
            "No activations",
            "Rule 1: temperature > 80",
            "No activations", 
            "Rule 3: cpu_usage > 90"
        ],
        "symbolic_proof": [
            ["temperature=85", "85 > 80", "positive"],
            [],
            ["temperature=82", "82 > 80", "positive"],
            [],
            ["cpu_usage=95", "95 > 90", "critical_positive"]
        ]
    }
    
    return model


class TestNeuralSymbolicReasoningWorkflows:
    """Integration tests for complete neural-symbolic reasoning workflows."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.neural_adapter = Mock()
        self.symbolic_adapter = Mock()
        self.model_repository = Mock(spec=ModelRepository)
        
        self.reasoning_service = NeuroSymbolicReasoningService(
            neural_adapter=self.neural_adapter,
            symbolic_adapter=self.symbolic_adapter,
            model_repository=self.model_repository
        )
    
    def test_end_to_end_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test complete end-to-end reasoning workflow."""
        # Setup model repository
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Setup neural adapter
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.8, 0.2, 0.9, 0.1, 0.95, 0.3, 0.7, 0.85, 0.15, 0.92]),
            "confidence": np.array([0.92, 0.88, 0.94, 0.85, 0.96, 0.89, 0.91, 0.93, 0.87, 0.95]),
            "embeddings": np.random.randn(10, 32),
            "feature_importance": np.random.randn(10, 64)
        }
        
        # Setup symbolic adapter
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive", "negative", "critical_positive", 
                          "negative", "positive", "positive", "negative", "positive"],
            "confidence": [0.9, 0.85, 0.88, 0.82, 0.95, 0.83, 0.89, 0.91, 0.84, 0.93],
            "explanations": [
                "Temperature threshold exceeded",
                "All parameters within range",
                "Pressure below threshold", 
                "All parameters within range",
                "CPU usage critical",
                "All parameters within range",
                "Multiple thresholds exceeded",
                "Temperature and pressure conditions",
                "All parameters within range",
                "System overload identified"
            ]
        }
        
        # Execute workflow
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=sample_neural_data,
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE
        )
        
        # Verify results
        assert isinstance(result, NeuroSymbolicReasoningResult)
        assert result.model_id == "test_model_001"
        assert len(result.predictions) == 10
        assert len(result.confidence_scores) == 10
        assert result.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE
        
        # Check that both neural and symbolic components were used
        self.neural_adapter.predict.assert_called_once()
        self.symbolic_adapter.reason.assert_called_once()
        
        # Verify fusion logic worked
        assert all(pred in ["positive", "negative", "critical_positive"] for pred in result.predictions)
        assert all(0 <= conf <= 1 for conf in result.confidence_scores)
    
    def test_attention_based_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test attention-based reasoning strategy workflow."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Neural predictions with attention weights
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.7, 0.3, 0.8]),
            "confidence": np.array([0.9, 0.85, 0.92]),
            "embeddings": np.random.randn(3, 32),
            "attention_weights": np.random.randn(3, 8, 10, 10)  # attention mechanism output
        }
        
        # Symbolic reasoning with varying confidence
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive"],
            "confidence": [0.95, 0.8, 0.88],
            "rule_activations": [0.9, 0.1, 0.8]  # How strongly rules fire
        }
        
        test_data = sample_neural_data[:3]
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=test_data,
            fusion_strategy=FusionStrategy.ATTENTION_BASED
        )
        
        assert result.fusion_strategy == FusionStrategy.ATTENTION_BASED
        assert len(result.fusion_weights) == 3
        
        # Attention-based fusion should consider both neural attention and symbolic rule strength
        for i, weights in enumerate(result.fusion_weights):
            assert "neural_attention" in weights
            assert "symbolic_activation" in weights
    
    def test_gating_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test gating-based reasoning strategy workflow."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Neural predictions
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.6, 0.4, 0.9, 0.2]),
            "confidence": np.array([0.88, 0.82, 0.95, 0.79]),
            "uncertainty": np.array([0.12, 0.18, 0.05, 0.21])  # Neural uncertainty
        }
        
        # Symbolic reasoning with rule coverage
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["negative", "negative", "positive", "negative"],
            "confidence": [0.85, 0.87, 0.93, 0.81],
            "rule_coverage": [0.9, 0.95, 0.8, 0.7]  # How well rules cover the input
        }
        
        test_data = sample_neural_data[:4]
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=test_data,
            fusion_strategy=FusionStrategy.GATING
        )
        
        assert result.fusion_strategy == FusionStrategy.GATING
        
        # Gating should adaptively choose between neural and symbolic based on confidence/coverage
        for i, decision in enumerate(result.gating_decisions):
            assert decision["primary_component"] in ["neural", "symbolic", "hybrid"]
            assert "confidence_ratio" in decision
            assert "rule_coverage" in decision
    
    def test_multi_modal_reasoning_workflow(self, mock_trained_model):
        """Test reasoning with multi-modal data (text + numerical)."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Multi-modal data
        numerical_data = np.random.randn(5, 32)
        text_embeddings = np.random.randn(5, 128)
        
        multimodal_data = {
            "numerical": numerical_data,
            "text": text_embeddings,
            "metadata": {
                "timestamp": ["2024-01-01T10:00:00Z"] * 5,
                "source": ["sensor_A", "sensor_B", "sensor_A", "sensor_C", "sensor_B"]
            }
        }
        
        # Neural adapter handles multi-modal input
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.75, 0.25, 0.85, 0.15, 0.9]),
            "confidence": np.array([0.9, 0.88, 0.92, 0.85, 0.94]),
            "modal_contributions": {
                "numerical": [0.6, 0.8, 0.7, 0.9, 0.65],
                "text": [0.4, 0.2, 0.3, 0.1, 0.35]
            }
        }
        
        # Symbolic reasoning incorporates metadata
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive", "negative", "positive"],
            "confidence": [0.88, 0.83, 0.9, 0.81, 0.92],
            "temporal_consistency": [0.95, 0.9, 0.92, 0.88, 0.94]
        }
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=multimodal_data,
            fusion_strategy=FusionStrategy.ATTENTION_BASED
        )
        
        assert len(result.predictions) == 5
        assert "modal_contributions" in result.additional_info
        assert "temporal_consistency" in result.additional_info
    
    def test_hierarchical_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test hierarchical neural-symbolic reasoning workflow."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Neural predictions at different abstraction levels
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.8, 0.3, 0.9]),
            "confidence": np.array([0.92, 0.86, 0.94]),
            "hierarchical_features": {
                "low_level": np.random.randn(3, 16),   # Raw sensor features
                "mid_level": np.random.randn(3, 8),    # Aggregated patterns  
                "high_level": np.random.randn(3, 4)    # Abstract concepts
            }
        }
        
        # Symbolic reasoning at multiple levels
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive"],
            "confidence": [0.9, 0.84, 0.91],
            "reasoning_levels": {
                "sensor_level": [
                    "temp_sensor_1 > threshold",
                    "all_sensors_baseline", 
                    "pressure_sensor_2 < threshold"
                ],
                "system_level": [
                    "thermal_condition_identified",
                    "system_operating_normally",
                    "hydraulic_system_condition"
                ],
                "domain_level": [
                    "potential_equipment_issue",
                    "nominal_operation",
                    "maintenance_recommended"
                ]
            }
        }
        
        test_data = sample_neural_data[:3]
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=test_data,
            enable_hierarchical_reasoning=True
        )
        
        assert "reasoning_levels" in result.additional_info
        assert len(result.additional_info["reasoning_levels"]["sensor_level"]) == 3
        assert len(result.additional_info["reasoning_levels"]["system_level"]) == 3
        assert len(result.additional_info["reasoning_levels"]["domain_level"]) == 3
    
    def test_uncertainty_aware_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test reasoning workflow with uncertainty quantification."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Neural predictions with uncertainty estimates
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.7, 0.4, 0.8, 0.2]),
            "confidence": np.array([0.88, 0.82, 0.91, 0.79]),
            "aleatoric_uncertainty": np.array([0.1, 0.15, 0.08, 0.18]),  # Data noise
            "epistemic_uncertainty": np.array([0.05, 0.12, 0.04, 0.16])  # Model uncertainty
        }
        
        # Symbolic reasoning with confidence intervals
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive", "negative"],
            "confidence": [0.85, 0.89, 0.87, 0.83],
            "confidence_intervals": [
                (0.8, 0.9), (0.85, 0.93), (0.82, 0.92), (0.78, 0.88)
            ],
            "rule_uncertainty": [0.08, 0.06, 0.09, 0.11]
        }
        
        test_data = sample_neural_data[:4]
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=test_data,
            quantify_uncertainty=True
        )
        
        assert "uncertainty_breakdown" in result.additional_info
        uncertainty = result.additional_info["uncertainty_breakdown"]
        
        for i in range(4):
            assert "total_uncertainty" in uncertainty[i]
            assert "neural_uncertainty" in uncertainty[i]
            assert "symbolic_uncertainty" in uncertainty[i]
            assert "fusion_uncertainty" in uncertainty[i]
    
    def test_adaptive_reasoning_workflow(self, mock_trained_model, sample_neural_data):
        """Test adaptive reasoning that changes strategy based on data characteristics."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Diverse data requiring different fusion approaches
        self.neural_adapter.predict.return_value = {
            "predictions": np.array([0.9, 0.1, 0.8, 0.3, 0.95]),
            "confidence": np.array([0.95, 0.89, 0.87, 0.84, 0.97]),
            "data_complexity": [0.8, 0.3, 0.9, 0.6, 0.95],  # How complex each sample is
            "out_of_distribution": [False, True, False, False, True]  # OOD analysis
        }
        
        self.symbolic_adapter.reason.return_value = {
            "conclusions": ["positive", "negative", "positive", "negative", "critical_positive"],
            "confidence": [0.92, 0.81, 0.86, 0.88, 0.94],
            "rule_applicability": [0.9, 0.4, 0.85, 0.7, 0.95]  # How well rules apply
        }
        
        test_data = sample_neural_data[:5]
        
        result = self.reasoning_service.perform_reasoning(
            model_id="test_model_001",
            data=test_data,
            fusion_strategy=FusionStrategy.ADAPTIVE
        )
        
        assert result.fusion_strategy == FusionStrategy.ADAPTIVE
        assert "adaptive_decisions" in result.additional_info
        
        # Should use different strategies for different samples
        decisions = result.additional_info["adaptive_decisions"]
        strategies_used = set(decision["chosen_strategy"] for decision in decisions)
        assert len(strategies_used) > 1  # Multiple strategies should be used
    
    def test_continual_learning_integration(self, mock_trained_model, sample_neural_data):
        """Test integration with continual learning mechanisms."""
        self.model_repository.get_by_id.return_value = mock_trained_model
        
        # Simulate evolving data distribution
        batch_1 = sample_neural_data[:3]  # Original distribution
        batch_2 = sample_neural_data[3:6] + 2.0  # Distribution shift
        batch_3 = sample_neural_data[6:9] + 5.0  # Larger shift
        
        results = []
        
        for i, batch in enumerate([batch_1, batch_2, batch_3]):
            # Neural adapter adapts to distribution
            self.neural_adapter.predict.return_value = {
                "predictions": np.array([0.7 + i*0.1, 0.3, 0.8 + i*0.05]),
                "confidence": np.array([0.9 - i*0.05, 0.85, 0.88 - i*0.02]),
                "distribution_shift": i * 0.3,
                "adaptation_score": max(0, 1 - i*0.2)
            }
            
            # Symbolic reasoning maintains consistency
            self.symbolic_adapter.reason.return_value = {
                "conclusions": ["positive", "negative", "positive"],
                "confidence": [0.87 + i*0.02, 0.84, 0.89 + i*0.01],
                "rule_stability": [0.95 - i*0.05, 0.96, 0.94 - i*0.03]
            }
            
            result = self.reasoning_service.perform_reasoning(
                model_id="test_model_001",
                data=batch,
                enable_continual_learning=True
            )
            
            results.append(result)
        
        # Verify adaptation over batches
        assert len(results) == 3
        for i, result in enumerate(results):
            assert "adaptation_metrics" in result.additional_info
            adaptation = result.additional_info["adaptation_metrics"]
            assert adaptation["batch_number"] == i
            assert "distribution_shift" in adaptation
            assert "rule_stability" in adaptation


class TestExplainableReasoningIntegrationWorkflows:
    """Integration tests for explainable reasoning workflows."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up explainable reasoning service."""
        self.reasoning_service = Mock(spec=NeuroSymbolicReasoningService)
        self.causal_engine = Mock()
        self.counterfactual_engine = Mock()
        
        self.explainable_service = ExplainableReasoningService(
            reasoning_service=self.reasoning_service,
            causal_engine=self.causal_engine,
            counterfactual_engine=self.counterfactual_engine
        )
    
    def test_complete_explainable_reasoning_workflow(self, sample_neural_data):
        """Test complete explainable reasoning workflow from raw data to business recommendations."""
        # Mock reasoning result
        mock_result = Mock()
        mock_result.model_id = "test_model"
        mock_result.predictions = ["positive", "negative", "critical_positive"]
        mock_result.confidence_scores = [0.92, 0.84, 0.96]
        mock_result.neural_scores = [0.88, 0.25, 0.94]
        mock_result.symbolic_conclusions = [
            "Temperature rule activated",
            "All systems within range",
            "Multiple critical thresholds exceeded"
        ]
        
        self.reasoning_service.perform_reasoning.return_value = mock_result
        
        # Mock causal analysis
        self.causal_engine.analyze_causal_factors.return_value = {
            "primary_causes": [
                {"factor": "CPU_temperature", "contribution": 0.8, "evidence": "Strong correlation"},
                {"factor": "System_load", "contribution": 0.6, "evidence": "Causal relationship"}
            ],
            "causal_chains": [
                "High_load → CPU_heating → Thermal_throttling → Performance_issue"
            ]
        }
        
        # Mock counterfactual analysis
        self.counterfactual_engine.generate_counterfactuals.return_value = {
            "scenarios": [
                {
                    "description": "Reduce system load to 70%",
                    "feasibility": 0.8,
                    "expected_outcome": "negative",
                    "required_changes": {"system_load": -20}
                }
            ]
        }
        
        # Business context
        business_context = {
            "system_type": "production_server",
            "business_hours": True,
            "sla_requirements": {"uptime": 99.9},
            "financial_impact_per_hour": 25000
        }
        
        test_data = sample_neural_data[:3]
        
        result = self.explainable_service.analyze_and_explain_reasoning(
            model_id="test_model",
            data=test_data,
            business_context=business_context
        )
        
        # Verify complete workflow execution
        assert result.reasoning_result == mock_result
        assert len(result.causal_explanations) > 0
        assert len(result.counterfactual_scenarios) > 0
        assert result.business_context == business_context
        assert hasattr(result, 'business_risk_assessment')
        assert hasattr(result, 'recommended_actions')
        
        # Verify business integration
        assert result.business_risk_assessment.severity in ["low", "medium", "high", "critical"]
        assert result.business_risk_assessment.financial_impact > 0
        assert len(result.recommended_actions) > 0
    
    def test_real_time_reasoning_explanation_workflow(self, sample_neural_data):
        """Test real-time reasoning explanation generation workflow."""
        # Simulate streaming data
        streaming_batches = [
            sample_neural_data[:2],
            sample_neural_data[2:4], 
            sample_neural_data[4:6]
        ]
        
        explanations = []
        
        for i, batch in enumerate(streaming_batches):
            # Mock reasoning for each batch
            mock_result = Mock()
            mock_result.predictions = ["positive", "negative"]
            mock_result.confidence_scores = [0.9 - i*0.05, 0.8 + i*0.02]
            mock_result.timestamp = f"2024-01-01T{10+i}:00:00Z"
            
            self.reasoning_service.perform_reasoning.return_value = mock_result
            
            # Mock rapid causal analysis
            self.causal_engine.analyze_causal_factors.return_value = {
                "primary_causes": [
                    {"factor": f"feature_{i}", "contribution": 0.8 - i*0.1}
                ],
                "processing_time_ms": 50 + i*10  # Increasing with complexity
            }
            
            result = self.explainable_service.analyze_and_explain_reasoning(
                model_id="streaming_model",
                data=batch,
                real_time_mode=True
            )
            
            explanations.append(result)
        
        # Verify real-time processing
        assert len(explanations) == 3
        for explanation in explanations:
            assert hasattr(explanation, 'processing_time')
            assert explanation.processing_time < 1.0  # Under 1 second
    
    def test_multi_stakeholder_reasoning_explanation_workflow(self, sample_neural_data):
        """Test generating reasoning explanations for different stakeholder types."""
        mock_result = Mock()
        mock_result.predictions = ["positive"]
        mock_result.confidence_scores = [0.94]
        
        self.reasoning_service.perform_reasoning.return_value = mock_result
        
        # Mock explanations for different audiences
        self.causal_engine.analyze_causal_factors.return_value = {
            "technical_explanation": {
                "primary_causes": ["Feature vector deviation", "Rule engine activation"],
                "statistical_details": {"p_value": 0.001, "confidence_interval": (0.9, 0.98)}
            },
            "business_explanation": {
                "primary_causes": ["System overload", "Performance degradation"],
                "impact_assessment": "High risk to operations"
            },
            "executive_explanation": {
                "summary": "Critical system condition requiring immediate attention",
                "business_impact": "$50K potential loss",
                "recommendation": "Deploy emergency response team"
            }
        }
        
        stakeholder_types = ["technical", "business", "executive"]
        
        for stakeholder in stakeholder_types:
            result = self.explainable_service.analyze_and_explain_reasoning(
                model_id="test_model",
                data=sample_neural_data[:1],
                explanation_audience=stakeholder
            )
            
            assert f"{stakeholder}_explanation" in result.audience_specific_explanations
            explanation = result.audience_specific_explanations[f"{stakeholder}_explanation"]
            assert "primary_causes" in explanation or "summary" in explanation
    
    def test_regulatory_compliance_reasoning_workflow(self, sample_neural_data):
        """Test reasoning workflow for regulatory compliance and auditability."""
        mock_result = Mock()
        mock_result.predictions = ["positive"]
        mock_result.confidence_scores = [0.93]
        mock_result.model_version = "v2.1.0"
        mock_result.data_lineage = {"source": "sensor_network", "preprocessing": "standard"}
        
        self.reasoning_service.perform_reasoning.return_value = mock_result
        
        # Mock audit trail generation
        self.causal_engine.analyze_causal_factors.return_value = {
            "audit_trail": {
                "decision_path": [
                    "Input validation passed",
                    "Neural network inference executed", 
                    "Symbolic rules evaluated",
                    "Fusion algorithm applied",
                    "Classification threshold exceeded"
                ],
                "model_artifacts": {
                    "neural_weights_hash": "abc123",
                    "symbolic_rules_hash": "def456", 
                    "fusion_config_hash": "ghi789"
                },
                "data_integrity": {
                    "input_hash": "jkl012",
                    "preprocessing_log": "applied_standard_normalization"
                }
            }
        }
        
        regulatory_context = {
            "regulation": "GDPR",
            "audit_requirements": ["explainability", "data_lineage", "model_transparency"],
            "retention_period": "7_years"
        }
        
        result = self.explainable_service.analyze_and_explain_reasoning(
            model_id="regulated_model",
            data=sample_neural_data[:1],
            regulatory_context=regulatory_context
        )
        
        # Verify regulatory compliance features
        assert "audit_trail" in result.compliance_documentation
        assert "model_transparency" in result.compliance_documentation
        assert "data_lineage" in result.compliance_documentation
        
        audit_trail = result.compliance_documentation["audit_trail"]
        assert "decision_path" in audit_trail
        assert "model_artifacts" in audit_trail
        assert len(audit_trail["decision_path"]) >= 3