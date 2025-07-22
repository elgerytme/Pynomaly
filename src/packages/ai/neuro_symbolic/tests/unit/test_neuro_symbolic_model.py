"""Unit tests for NeuroSymbolicModel entity."""

import pytest
from neuro_symbolic.domain.entities.neuro_symbolic_model import NeuroSymbolicModel
from neuro_symbolic.domain.entities.knowledge_graph import KnowledgeGraph
from neuro_symbolic.domain.value_objects.reasoning_result import ReasoningResult


class TestNeuroSymbolicModel:
    """Test cases for NeuroSymbolicModel."""
    
    def test_create_model(self):
        """Test model creation."""
        model = NeuroSymbolicModel.create(
            name="test_model",
            neural_backbone="transformer",
            symbolic_reasoner="first_order_logic"
        )
        
        assert model.name == "test_model"
        assert model.neural_backbone == "transformer"
        assert model.symbolic_reasoner == "first_order_logic"
        assert model.id is not None
        assert not model.is_trained
        assert len(model.knowledge_graphs) == 0
        assert len(model.symbolic_constraints) == 0
    
    def test_add_knowledge_graph(self):
        """Test adding knowledge graph to model."""
        model = NeuroSymbolicModel.create("test_model")
        kg = KnowledgeGraph.create("test_kg")
        
        model.add_knowledge_graph(kg)
        
        assert len(model.knowledge_graphs) == 1
        assert model.knowledge_graphs[0] == kg
        
        # Adding the same KG again should not duplicate
        model.add_knowledge_graph(kg)
        assert len(model.knowledge_graphs) == 1
    
    def test_add_symbolic_constraint(self):
        """Test adding symbolic constraints."""
        model = NeuroSymbolicModel.create("test_model")
        constraint = {"type": "logical", "rule": "all_positive"}
        
        model.add_symbolic_constraint(constraint)
        
        assert len(model.symbolic_constraints) == 1
        assert model.symbolic_constraints[0] == constraint
    
    def test_train_model(self):
        """Test model training."""
        model = NeuroSymbolicModel.create("test_model")
        training_data = {"data": "test"}
        
        assert not model.is_trained
        
        model.train(training_data)
        
        assert model.is_trained
    
    def test_predict_untrained_model_raises_error(self):
        """Test that prediction on untrained model raises error."""
        model = NeuroSymbolicModel.create("test_model")
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict_with_explanation({"input": "test"})
    
    def test_predict_with_explanation(self):
        """Test prediction with explanation on trained model."""
        model = NeuroSymbolicModel.create("test_model")
        model.train({"data": "test"})  # Train first
        
        result = model.predict_with_explanation({"input": "test"})
        
        assert isinstance(result, ReasoningResult)
        assert result.prediction is not None
        assert result.confidence > 0
        assert len(result.symbolic_explanation) > 0