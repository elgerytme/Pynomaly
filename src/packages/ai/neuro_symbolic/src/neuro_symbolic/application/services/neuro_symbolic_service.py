"""High-level service for neuro-symbolic AI operations."""

from typing import Any, Dict, List, Optional
import structlog

from ...domain.entities.neuro_symbolic_model import NeuroSymbolicModel
from ...domain.entities.knowledge_graph import KnowledgeGraph
from ...domain.value_objects.reasoning_result import ReasoningResult


logger = structlog.get_logger(__name__)


class NeuroSymbolicService:
    """
    Application service that orchestrates neuro-symbolic AI workflows.
    Provides high-level interface for training, inference, and explanation.
    """
    
    def __init__(self):
        self.models: Dict[str, NeuroSymbolicModel] = {}
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
    
    def create_model(
        self,
        name: str,
        neural_backbone: str = "transformer",
        symbolic_reasoner: str = "first_order_logic"
    ) -> NeuroSymbolicModel:
        """Create and register a new neuro-symbolic model."""
        model = NeuroSymbolicModel.create(
            name=name,
            neural_backbone=neural_backbone,
            symbolic_reasoner=symbolic_reasoner
        )
        
        self.models[model.id] = model
        
        logger.info(
            "Created neuro-symbolic model",
            model_id=model.id,
            name=name,
            neural_backbone=neural_backbone,
            symbolic_reasoner=symbolic_reasoner
        )
        
        return model
    
    def load_knowledge_graph(self, name: str, file_path: str) -> KnowledgeGraph:
        """Load and register a knowledge graph from file."""
        kg = KnowledgeGraph.from_file(file_path)
        kg.name = name
        
        self.knowledge_graphs[kg.id] = kg
        
        logger.info(
            "Loaded knowledge graph",
            kg_id=kg.id,
            name=name,
            file_path=file_path,
            num_triples=len(kg.triples)
        )
        
        return kg
    
    def attach_knowledge_to_model(
        self,
        model_id: str,
        knowledge_graph_id: str
    ) -> None:
        """Attach a knowledge graph to a model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if knowledge_graph_id not in self.knowledge_graphs:
            raise ValueError(f"Knowledge graph {knowledge_graph_id} not found")
        
        model = self.models[model_id]
        kg = self.knowledge_graphs[knowledge_graph_id]
        
        model.add_knowledge_graph(kg)
        
        logger.info(
            "Attached knowledge graph to model",
            model_id=model_id,
            kg_id=knowledge_graph_id
        )
    
    def train_model(
        self,
        model_id: str,
        training_data: Any,
        symbolic_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Train a neuro-symbolic model with optional constraints."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Add symbolic constraints if provided
        if symbolic_constraints:
            for constraint in symbolic_constraints:
                model.add_symbolic_constraint(constraint)
        
        # Train the model
        model.train(training_data, **kwargs)
        
        logger.info(
            "Trained neuro-symbolic model",
            model_id=model_id,
            num_constraints=len(model.symbolic_constraints),
            num_knowledge_graphs=len(model.knowledge_graphs)
        )
    
    def predict_with_explanation(
        self,
        model_id: str,
        input_data: Any
    ) -> ReasoningResult:
        """Make predictions with symbolic explanations."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        result = model.predict_with_explanation(input_data)
        
        logger.info(
            "Generated prediction with explanation",
            model_id=model_id,
            confidence=result.confidence,
            num_explanation_steps=len(result.symbolic_explanation)
        )
        
        return result
    
    def get_model(self, model_id: str) -> NeuroSymbolicModel:
        """Retrieve a model by ID."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        return self.models[model_id]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return [
            {
                "id": model.id,
                "name": model.name,
                "neural_backbone": model.neural_backbone,
                "symbolic_reasoner": model.symbolic_reasoner,
                "is_trained": model.is_trained,
                "num_knowledge_graphs": len(model.knowledge_graphs)
            }
            for model in self.models.values()
        ]