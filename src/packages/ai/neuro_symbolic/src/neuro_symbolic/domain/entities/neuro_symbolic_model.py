"""Core neuro-symbolic model entity."""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import warnings

from ..value_objects.reasoning_result import ReasoningResult


class ModelStatus(Enum):
    """Status of the neuro-symbolic model."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FINE_TUNING = "fine_tuning"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ERROR = "error"


class FusionStrategy(Enum):
    """Strategies for combining neural and symbolic components."""
    EARLY_FUSION = "early_fusion"  # Combine at input level
    LATE_FUSION = "late_fusion"    # Combine at output level
    INTERMEDIATE_FUSION = "intermediate_fusion"  # Combine at hidden layers
    ATTENTION_FUSION = "attention_fusion"  # Use attention mechanisms
    HIERARCHICAL_FUSION = "hierarchical_fusion"  # Multi-level fusion
    DYNAMIC_FUSION = "dynamic_fusion"  # Adaptive fusion strategy


@dataclass
class TrainingMetrics:
    """Training metrics for neuro-symbolic model."""
    epoch: int
    neural_loss: float
    symbolic_loss: float
    combined_loss: float
    neural_accuracy: float
    symbolic_accuracy: float
    combined_accuracy: float
    reasoning_quality: float
    explanation_coherence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SymbolicConstraint:
    """Symbolic constraint for neural training."""
    id: str
    name: str
    constraint_type: str  # logical, causal, temporal, etc.
    rule: str
    confidence: float
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


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
    symbolic_constraints: List[SymbolicConstraint]
    fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION
    status: ModelStatus = ModelStatus.UNTRAINED
    version: str = "0.1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Training and performance metrics
    training_history: List[TrainingMetrics] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Neural component information
    neural_architecture: Optional[Dict[str, Any]] = None
    neural_parameters_count: Optional[int] = None
    
    # Symbolic component information
    reasoning_depth: int = 3
    max_inference_steps: int = 100
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Convert dict constraints to SymbolicConstraint objects if needed
        converted_constraints = []
        for constraint in self.symbolic_constraints:
            if isinstance(constraint, dict):
                converted_constraints.append(
                    SymbolicConstraint(
                        id=constraint.get('id', str(uuid.uuid4())),
                        name=constraint.get('name', 'Unnamed Constraint'),
                        constraint_type=constraint.get('type', 'logical'),
                        rule=constraint.get('rule', ''),
                        confidence=constraint.get('confidence', 1.0),
                        active=constraint.get('active', True),
                        metadata=constraint.get('metadata', {})
                    )
                )
            else:
                converted_constraints.append(constraint)
        
        self.symbolic_constraints = converted_constraints
    
    @classmethod
    def create(
        cls,
        name: str,
        neural_backbone: str = "transformer",
        symbolic_reasoner: str = "first_order_logic",
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> "NeuroSymbolicModel":
        """Create a new neuro-symbolic model."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            neural_backbone=neural_backbone,
            symbolic_reasoner=symbolic_reasoner,
            fusion_strategy=fusion_strategy,
            symbolic_constraints=[],
            description=description,
            tags=tags or []
        )
    
    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self.status in [ModelStatus.TRAINED, ModelStatus.DEPLOYED]
    
    @property
    def total_constraints(self) -> int:
        """Get total number of symbolic constraints."""
        return len([c for c in self.symbolic_constraints if c.active])
    
    
    def update_status(self, status: ModelStatus) -> None:
        """Update model status and timestamp."""
        self.status = status
        self.updated_at = datetime.now()
    
    
    def add_symbolic_constraint(
        self, 
        constraint: Union[Dict[str, Any], SymbolicConstraint]
    ) -> None:
        """Add a symbolic constraint to guide neural training."""
        if isinstance(constraint, dict):
            constraint = SymbolicConstraint(
                id=constraint.get('id', str(uuid.uuid4())),
                name=constraint.get('name', 'Unnamed Constraint'),
                constraint_type=constraint.get('type', 'logical'),
                rule=constraint.get('rule', ''),
                confidence=constraint.get('confidence', 1.0),
                active=constraint.get('active', True),
                metadata=constraint.get('metadata', {})
            )
        
        self.symbolic_constraints.append(constraint)
        self.updated_at = datetime.now()
    
    def deactivate_constraint(self, constraint_id: str) -> bool:
        """Deactivate a symbolic constraint."""
        for constraint in self.symbolic_constraints:
            if constraint.id == constraint_id:
                constraint.active = False
                self.updated_at = datetime.now()
                return True
        return False
    
    def get_active_constraints(self) -> List[SymbolicConstraint]:
        """Get all active symbolic constraints."""
        return [c for c in self.symbolic_constraints if c.active]
    
    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Set model hyperparameters."""
        self.hyperparameters.update(hyperparams)
        self.updated_at = datetime.now()
    
    def add_training_metric(
        self,
        epoch: int,
        neural_loss: float,
        symbolic_loss: float, 
        combined_loss: float,
        neural_accuracy: float,
        symbolic_accuracy: float,
        combined_accuracy: float,
        reasoning_quality: float = 0.0,
        explanation_coherence: float = 0.0
    ) -> None:
        """Add training metrics for an epoch."""
        metric = TrainingMetrics(
            epoch=epoch,
            neural_loss=neural_loss,
            symbolic_loss=symbolic_loss,
            combined_loss=combined_loss,
            neural_accuracy=neural_accuracy,
            symbolic_accuracy=symbolic_accuracy,
            combined_accuracy=combined_accuracy,
            reasoning_quality=reasoning_quality,
            explanation_coherence=explanation_coherence
        )
        self.training_history.append(metric)
        self.updated_at = datetime.now()
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the latest training metrics."""
        return self.training_history[-1] if self.training_history else None
    
    def train(
        self, 
        data: Any, 
        neural_data: Optional[Any] = None,
        symbolic_data: Optional[Any] = None,
        **kwargs
    ) -> None:
        """Train the neuro-symbolic model."""
        self.update_status(ModelStatus.TRAINING)
        
        try:
            # Training logic would be implemented here
            # This is a placeholder that simulates training
            
            # Extract hyperparameters
            epochs = kwargs.get('epochs', self.hyperparameters.get('epochs', 10))
            
            # Simulate training progress
            for epoch in range(epochs):
                # Placeholder metrics - real implementation would compute these
                neural_loss = max(0.1, 1.0 - (epoch / epochs) * 0.8)
                symbolic_loss = max(0.05, 0.5 - (epoch / epochs) * 0.4) 
                combined_loss = neural_loss + symbolic_loss
                
                neural_acc = min(0.95, 0.5 + (epoch / epochs) * 0.4)
                symbolic_acc = min(0.98, 0.6 + (epoch / epochs) * 0.3)
                combined_acc = (neural_acc + symbolic_acc) / 2
                
                reasoning_quality = min(0.9, 0.3 + (epoch / epochs) * 0.6)
                explanation_coherence = min(0.85, 0.4 + (epoch / epochs) * 0.45)
                
                self.add_training_metric(
                    epoch=epoch + 1,
                    neural_loss=neural_loss,
                    symbolic_loss=symbolic_loss,
                    combined_loss=combined_loss,
                    neural_accuracy=neural_acc,
                    symbolic_accuracy=symbolic_acc,
                    combined_accuracy=combined_acc,
                    reasoning_quality=reasoning_quality,
                    explanation_coherence=explanation_coherence
                )
            
            self.update_status(ModelStatus.TRAINED)
            
            # Update performance metrics
            final_metrics = self.get_latest_metrics()
            if final_metrics:
                self.performance_metrics.update({
                    'final_accuracy': final_metrics.combined_accuracy,
                    'final_loss': final_metrics.combined_loss,
                    'reasoning_quality': final_metrics.reasoning_quality,
                    'explanation_coherence': final_metrics.explanation_coherence
                })
        
        except Exception as e:
            self.update_status(ModelStatus.ERROR)
            raise RuntimeError(f"Training failed: {e}")
    
    def fine_tune(
        self,
        data: Any,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """Fine-tune the pre-trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before fine-tuning")
        
        self.update_status(ModelStatus.FINE_TUNING)
        
        try:
            # Fine-tuning logic would be implemented here
            # For now, just update status and add minimal metrics
            
            self.add_training_metric(
                epoch=len(self.training_history) + 1,
                neural_loss=0.05,
                symbolic_loss=0.02,
                combined_loss=0.07,
                neural_accuracy=0.97,
                symbolic_accuracy=0.99,
                combined_accuracy=0.98,
                reasoning_quality=0.92,
                explanation_coherence=0.89
            )
            
            self.update_status(ModelStatus.TRAINED)
            
        except Exception as e:
            self.update_status(ModelStatus.ERROR)
            raise RuntimeError(f"Fine-tuning failed: {e}")
    
    def predict_with_explanation(self, input_data: Any, **kwargs) -> ReasoningResult:
        """Make predictions with symbolic explanations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prediction and reasoning logic would be implemented here
            # This is a sophisticated placeholder
            
            # Simulate neural processing
            neural_confidence = kwargs.get('neural_confidence', 0.85)
            
            # Simulate symbolic reasoning
            symbolic_steps = []
            for i, constraint in enumerate(self.get_active_constraints()[:3]):
                symbolic_steps.append(f"Applied constraint '{constraint.name}': {constraint.rule}")
            
            
            if not symbolic_steps:
                symbolic_steps = ["Neural prediction without symbolic reasoning"]
            
            # Combine neural and symbolic confidences
            symbolic_confidence = min([c.confidence for c in self.get_active_constraints()], default=1.0)
            
            if self.fusion_strategy == FusionStrategy.LATE_FUSION:
                combined_confidence = (neural_confidence + symbolic_confidence) / 2
            elif self.fusion_strategy == FusionStrategy.EARLY_FUSION:
                combined_confidence = neural_confidence * symbolic_confidence
            else:
                # Default to late fusion
                combined_confidence = (neural_confidence + symbolic_confidence) / 2
            
            return ReasoningResult.create(
                prediction=f"prediction_based_on_{self.fusion_strategy.value}",
                confidence=combined_confidence,
                symbolic_explanation=symbolic_steps,
                neural_features={'confidence': neural_confidence},
                reasoning_trace=symbolic_steps
            )
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(
        self, 
        batch_data: List[Any],
        **kwargs
    ) -> List[ReasoningResult]:
        """Make batch predictions with explanations."""
        results = []
        for data in batch_data:
            try:
                result = self.predict_with_explanation(data, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = ReasoningResult.create(
                    prediction=None,
                    confidence=0.0,
                    symbolic_explanation=[f"Prediction failed: {e}"]
                )
                results.append(error_result)
        
        return results
    
    def validate(self, validation_data: Any) -> Dict[str, float]:
        """Validate the model on validation data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        # Validation logic would be implemented here
        # This is a placeholder
        return {
            'accuracy': 0.89,
            'precision': 0.91,
            'recall': 0.87,
            'f1_score': 0.89,
            'reasoning_quality': 0.85,
            'explanation_coherence': 0.82
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the model."""
        latest_metrics = self.get_latest_metrics()
        
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'neural_backbone': self.neural_backbone,
            'symbolic_reasoner': self.symbolic_reasoner,
            'fusion_strategy': self.fusion_strategy.value,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'active_constraints': self.total_constraints,
            'training_epochs': len(self.training_history),
            'performance': {
                'latest_accuracy': latest_metrics.combined_accuracy if latest_metrics else None,
                'latest_loss': latest_metrics.combined_loss if latest_metrics else None,
                'reasoning_quality': latest_metrics.reasoning_quality if latest_metrics else None,
                'explanation_coherence': latest_metrics.explanation_coherence if latest_metrics else None
            },
            'hyperparameters': self.hyperparameters,
            'tags': self.tags,
            'description': self.description
        }
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
            return True
        return False