"""Core neuro-symbolic model entity."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import uuid

from ..value_objects.reasoning_result import ReasoningResult


@dataclass
class NeuroSymbolicModel:
    """
    Core entity representing a neuro-symbolic AI model that combines
    neural networks with symbolic reasoning capabilities.
    """
    
    id: str
    name: str
    neural_backbone: str
    symbolic_reasoner: str
    knowledge_graphs: List["KnowledgeGraph"]
    symbolic_constraints: List[Dict[str, Any]]
    is_trained: bool = False
    version: str = "0.1.0"
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @classmethod
    def create(
        cls,
        name: str,
        neural_backbone: str = "transformer",
        symbolic_reasoner: str = "first_order_logic"
    ) -> "NeuroSymbolicModel":
        """Create a new neuro-symbolic model."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            neural_backbone=neural_backbone,
            symbolic_reasoner=symbolic_reasoner,
            knowledge_graphs=[],
            symbolic_constraints=[]
        )
    
    def add_knowledge_graph(self, knowledge_graph: "KnowledgeGraph") -> None:
        """Add a knowledge graph to the model."""
        if knowledge_graph not in self.knowledge_graphs:
            self.knowledge_graphs.append(knowledge_graph)
    
    def add_symbolic_constraint(self, constraint: Dict[str, Any]) -> None:
        """Add a symbolic constraint to guide neural training."""
        self.symbolic_constraints.append(constraint)
    
    def train(self, data: Any, **kwargs) -> None:
        """Train the neuro-symbolic model."""
        # Training logic would be implemented here
        self.is_trained = True
    
    def predict_with_explanation(self, input_data: Any) -> ReasoningResult:
        """Make predictions with symbolic explanations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prediction and reasoning logic would be implemented here
        return ReasoningResult.create(
            prediction="example_prediction",
            confidence=0.95,
            symbolic_explanation=["Rule 1 applied", "Neural output processed"]
        )