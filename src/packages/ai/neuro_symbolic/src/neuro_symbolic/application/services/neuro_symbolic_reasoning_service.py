"""Neuro-symbolic reasoning service that combines neural networks with symbolic reasoning."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime

from ...domain.entities.neuro_symbolic_model import NeuroSymbolicModel, ModelStatus, FusionStrategy
from ...domain.value_objects.reasoning_result import ReasoningResult
from ...domain.value_objects.causal_explanation import (
    CausalExplanation, CausalFactor, CausalLink, CausalChain, 
    CausalRelationType, TemporalRelation
)
from ...domain.value_objects.counterfactual_result import (
    CounterfactualResult, CounterfactualScenario, FeatureChange,
    CounterfactualType, ChangeDirection
)
from ...infrastructure.adapters.neural_adapter import NeuralAdapter
from ...infrastructure.adapters.symbolic_adapter import SymbolicAdapter, LogicalRule, LogicType
from ...infrastructure.config.settings import get_config
from ...infrastructure.error_handling import (
    NeuroSymbolicError, ValidationError, ModelError, InferenceError, DataError,
    InputValidator, ErrorRecovery, error_handler, setup_error_logging
)


@dataclass
class NeuroSymbolicResult:
    """Result of neuro-symbolic reasoning with enhanced explanations."""
    
    # Basic reasoning results
    predictions: NDArray[np.floating]  # Continuous predictions/scores
    confidence_scores: NDArray[np.floating]
    neural_outputs: NDArray[np.floating] 
    symbolic_outputs: NDArray[np.floating]
    
    # Enhanced explanations
    explanations: List[ReasoningResult]
    causal_explanations: Optional[List[CausalExplanation]] = None
    counterfactual_explanations: Optional[List[CounterfactualResult]] = None
    
    # Metadata
    algorithm: str = "neuro_symbolic"
    fusion_strategy: str = "late_fusion"
    constraints_applied: List[str] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.constraints_applied is None:
            self.constraints_applied = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def num_samples(self) -> int:
        """Number of samples processed."""
        return len(self.predictions)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence across all predictions."""
        return float(np.mean(self.confidence_scores))
    
    @property
    def success(self) -> bool:
        """Whether reasoning was successful."""
        return len(self.predictions) > 0 and len(self.explanations) > 0


class NeuroSymbolicReasoningService:
    """
    Advanced reasoning service that combines neural networks with symbolic reasoning.
    Provides general-purpose neuro-symbolic capabilities for various reasoning tasks.
    """
    
    def __init__(self):
        """Initialize the neuro-symbolic reasoning service."""
        try:
            self.logger = setup_error_logging()
            self.neural_adapter = NeuralAdapter()
            self.symbolic_adapter = SymbolicAdapter()
            self.models: Dict[str, NeuroSymbolicModel] = {}
            self.config = get_config()
            self.validator = InputValidator()
            
            # Initialize default reasoning components
            self._initialize_default_reasoners()
            self.logger.info("NeuroSymbolicReasoningService initialized successfully")
            
        except Exception as e:
            raise NeuroSymbolicError(
                "Failed to initialize NeuroSymbolicReasoningService",
                cause=e,
                remediation="Check system dependencies and configuration"
            )
    
    def _initialize_default_reasoners(self):
        """Initialize default symbolic reasoners."""
        try:
            # Create propositional reasoner for simple logical rules
            self.symbolic_adapter.create_reasoner(
                "propositional",
                LogicType.PROPOSITIONAL
            )
            
            # Create first-order reasoner for complex domain knowledge
            self.symbolic_adapter.create_reasoner(
                "first_order", 
                LogicType.FIRST_ORDER
            )
            
            # Create SMT solver for constraint satisfaction
            self.symbolic_adapter.create_reasoner(
                "smt",
                "smt"
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize some symbolic reasoners: {e}")
    
    @error_handler(reraise=True)
    def create_reasoning_model(
        self,
        model_id: str,
        name: str,
        neural_backbone: str = "transformer",
        symbolic_reasoner: str = "first_order_logic",
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION,
        description: Optional[str] = None
    ) -> NeuroSymbolicModel:
        """Create a new neuro-symbolic reasoning model."""
        
        # Validate inputs
        model_id = self.validator.validate_model_id(model_id, "model_id")
        
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "Model name must be a non-empty string",
                field_name="name",
                remediation="Provide a valid model name"
            )
        
        # Validate neural backbone
        valid_backbones = ["transformer", "cnn", "lstm", "autoencoder"]
        if neural_backbone not in valid_backbones:
            raise ValidationError(
                f"Neural backbone must be one of {valid_backbones}, got '{neural_backbone}'",
                field_name="neural_backbone",
                remediation=f"Use one of: {valid_backbones}"
            )
        
        # Validate symbolic reasoner
        valid_reasoners = ["propositional", "first_order_logic", "smt", "prolog"]
        if symbolic_reasoner not in valid_reasoners:
            raise ValidationError(
                f"Symbolic reasoner must be one of {valid_reasoners}, got '{symbolic_reasoner}'",
                field_name="symbolic_reasoner",
                remediation=f"Use one of: {valid_reasoners}"
            )
        
        # Check if model ID already exists
        if model_id in self.models:
            raise ModelError(
                f"Model with ID '{model_id}' already exists",
                model_id=model_id,
                remediation="Use a different model ID or delete the existing model"
            )
        
        try:
            model = NeuroSymbolicModel.create(
                name=name.strip(),
                neural_backbone=neural_backbone,
                symbolic_reasoner=symbolic_reasoner,
                fusion_strategy=fusion_strategy,
                description=description or f"Neuro-symbolic reasoning model: {name}",
                tags=["reasoning", "neuro_symbolic"]
            )
            
            model.id = model_id  # Override with specified ID
            self.models[model_id] = model
            
            self.logger.info(f"Created reasoning model '{model_id}' with backbone '{neural_backbone}'")
            return model
            
        except Exception as e:
            raise ModelError(
                f"Failed to create reasoning model '{model_id}'",
                model_id=model_id,
                cause=e,
                remediation="Check model parameters and system resources"
            )
    
    # Knowledge graph functionality has been moved to the knowledge_graph package
    
    # Reasoning rules functionality has been moved to the knowledge_graph package
    
    def train_reasoning_model(
        self,
        model_id: str,
        training_data: NDArray[np.floating],
        labels: Optional[NDArray[np.floating]] = None,
        neural_config: Optional[Dict[str, Any]] = None,
        symbolic_constraints: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        """Train the neuro-symbolic reasoning model."""
        if model_id not in self.models:
            raise ModelError(
                f"Model '{model_id}' not found",
                model_id=model_id,
                remediation="Create the model first"
            )
        
        model = self.models[model_id]
        model.update_status(ModelStatus.TRAINING)
        
        try:
            # Create neural backbone for reasoning
            neural_config = neural_config or {}
            
            if model.neural_backbone == "transformer":
                backbone = self.neural_adapter.create_backbone(
                    "transformer",
                    f"{model_id}_neural",
                    input_dim=training_data.shape[1],
                    **neural_config
                )
            elif model.neural_backbone == "cnn":
                backbone = self.neural_adapter.create_backbone(
                    "cnn", 
                    f"{model_id}_neural",
                    **neural_config
                )
            elif model.neural_backbone == "lstm":
                backbone = self.neural_adapter.create_backbone(
                    "lstm",
                    f"{model_id}_neural", 
                    input_dim=training_data.shape[1],
                    **neural_config
                )
            else:  # autoencoder
                backbone = self.neural_adapter.create_backbone(
                    "autoencoder",
                    f"{model_id}_neural",
                    input_dim=training_data.shape[1],
                    **neural_config
                )
            
            # Train neural component
            if labels is not None:
                # Supervised training
                training_history = self.neural_adapter.train_backbone(
                    f"{model_id}_neural",
                    (training_data, labels),
                    epochs=kwargs.get('epochs', 50)
                )
            else:
                # Unsupervised training
                dummy_labels = np.zeros(len(training_data))
                training_history = self.neural_adapter.train_backbone(
                    f"{model_id}_neural", 
                    (training_data, dummy_labels),
                    epochs=kwargs.get('epochs', 50)
                )
            
            # Add symbolic constraints
            if symbolic_constraints:
                for constraint in symbolic_constraints:
                    model.add_symbolic_constraint(constraint)
            
            # Train the model (updates status internally)
            model.train(
                training_data,
                neural_data=(training_data, labels),
                epochs=kwargs.get('epochs', 50),
                **kwargs
            )
            
            # Store neural architecture info
            model.neural_architecture = {
                'type': model.neural_backbone,
                'input_dim': training_data.shape[1],
                'training_samples': len(training_data),
                'training_history': training_history
            }
            
        except Exception as e:
            model.update_status(ModelStatus.ERROR)
            raise ModelError(
                f"Failed to train model '{model_id}'",
                model_id=model_id,
                cause=e
            )
    
    @error_handler(reraise=True)
    def perform_reasoning(
        self,
        model_id: str,
        data: NDArray[np.floating],
        include_causal_explanations: bool = True,
        include_counterfactuals: bool = True,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> NeuroSymbolicResult:
        """Perform neuro-symbolic reasoning on input data."""
        
        # Validate inputs
        model_id = self.validator.validate_model_id(model_id, "model_id")
        
        if model_id not in self.models:
            raise ModelError(
                f"Model '{model_id}' not found",
                model_id=model_id,
                remediation="Create the model first or check the model ID"
            )
        
        model = self.models[model_id]
        
        if not model.is_trained:
            raise ModelError(
                f"Model '{model_id}' must be trained before reasoning",
                model_id=model_id,
                remediation="Train the model using train_reasoning_model()"
            )
        
        # Validate data input
        try:
            data = self.validator.validate_array_input(
                data,
                name="data",
                min_dimensions=2,
                max_dimensions=3,
                min_samples=1,
                max_samples=10000,
                min_features=1,
                allow_nan=False,
                allow_inf=False
            )
        except ValidationError as e:
            # Try data recovery for common issues
            if "NaN" in str(e) or "infinite" in str(e):
                self.logger.warning(f"Data contains invalid values, attempting recovery")
                try:
                    data = ErrorRecovery.handle_data_corruption(data, strategy="interpolate")
                    data = self.validator.validate_array_input(data, name="data", min_dimensions=2)
                    self.logger.info("Data recovery successful")
                except Exception:
                    raise e
            else:
                raise e
        
        # Validate feature names if provided
        if feature_names is not None:
            feature_names = self.validator.validate_feature_names(
                feature_names,
                expected_count=data.shape[1],
                name="feature_names"
            )
        
        start_time = datetime.now()
        self.logger.info(f"Starting reasoning for model '{model_id}' with {len(data)} samples")
        
        try:
            # Neural reasoning
            try:
                neural_outputs = self._compute_neural_outputs(model_id, data)
            except MemoryError:
                neural_outputs = ErrorRecovery.handle_memory_error(
                    self._compute_neural_outputs, model_id, data
                )
            
            # Symbolic reasoning
            symbolic_outputs, symbolic_explanations = self._compute_symbolic_outputs(
                model, data, neural_outputs, feature_names
            )
            
            # Fusion of neural and symbolic outputs
            combined_outputs, predictions = self._fuse_outputs(
                neural_outputs, symbolic_outputs, model.fusion_strategy
            )
            
            # Generate explanations
            explanations = self._generate_explanations(
                model, data, predictions, neural_outputs, symbolic_outputs,
                symbolic_explanations, feature_names
            )
            
            # Generate causal explanations if requested
            causal_explanations = None
            if include_causal_explanations:
                try:
                    causal_explanations = self._generate_causal_explanations(
                        model, data, predictions, neural_outputs, symbolic_outputs, feature_names
                    )
                except Exception as e:
                    self.logger.warning(f"Causal explanation generation failed: {e}")
                    causal_explanations = []
            
            # Generate counterfactual explanations if requested
            counterfactual_explanations = None
            if include_counterfactuals:
                try:
                    counterfactual_explanations = self._generate_counterfactual_explanations(
                        model, data, predictions, feature_names
                    )
                except Exception as e:
                    self.logger.warning(f"Counterfactual explanation generation failed: {e}")
                    counterfactual_explanations = []
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Validate results before returning
            if len(predictions) != len(data):
                raise InferenceError(
                    f"Prediction count ({len(predictions)}) doesn't match input count ({len(data)})",
                    remediation="Check data preprocessing and model implementation"
                )
            
            result = NeuroSymbolicResult(
                predictions=predictions,
                confidence_scores=combined_outputs,
                neural_outputs=neural_outputs,
                symbolic_outputs=symbolic_outputs,
                explanations=explanations,
                causal_explanations=causal_explanations,
                counterfactual_explanations=counterfactual_explanations,
                algorithm="neuro_symbolic",
                fusion_strategy=model.fusion_strategy.value,
                constraints_applied=[c.name for c in model.get_active_constraints()],
                processing_time=processing_time,
                metadata={
                    'model_id': model_id,
                    'data_shape': data.shape,
                    'num_constraints': model.total_constraints,
                    'neural_backbone': model.neural_backbone,
                    'symbolic_reasoner': model.symbolic_reasoner
                }
            )
            
            self.logger.info(f"Reasoning completed for model '{model_id}': processed {len(data)} samples")
            return result
            
        except MemoryError as e:
            raise InferenceError(
                f"Memory error during reasoning for model '{model_id}'",
                model_id=model_id,
                cause=e,
                remediation="Reduce batch size, use smaller model, or increase available memory"
            )
        except (ValidationError, ModelError, InferenceError):
            # Re-raise our custom errors
            raise
        except Exception as e:
            raise InferenceError(
                f"Reasoning failed for model '{model_id}': {str(e)}",
                model_id=model_id,
                cause=e,
                remediation="Check model state, input data, and system resources"
            )
    
    def _compute_neural_outputs(self, model_id: str, data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute neural network outputs."""
        try:
            # Get neural predictions
            neural_outputs = self.neural_adapter.predict(f"{model_id}_neural", data)
            
            # Convert to numpy if needed
            if hasattr(neural_outputs, 'detach'):
                neural_outputs = neural_outputs.detach().cpu().numpy()
            
            # Handle different output shapes
            if len(neural_outputs.shape) > 1 and neural_outputs.shape[1] > 1:
                # Multi-output case - use max or mean
                neural_scores = np.max(neural_outputs, axis=1)
            else:
                # Single output or single column
                neural_scores = neural_outputs.flatten()
            
            # Normalize outputs to [0, 1] if needed
            if neural_scores.std() > 0:
                neural_scores = (neural_scores - neural_scores.min()) / (neural_scores.max() - neural_scores.min())
            
            return neural_scores
            
        except Exception as e:
            warnings.warn(f"Neural computation failed: {e}")
            return np.zeros(len(data))
    
    def _compute_symbolic_outputs(
        self,
        model: NeuroSymbolicModel,
        data: NDArray[np.floating],
        neural_outputs: NDArray[np.floating],
        feature_names: Optional[List[str]]
    ) -> Tuple[NDArray[np.floating], List[Dict[str, Any]]]:
        """Compute symbolic reasoning outputs."""
        symbolic_outputs = np.zeros(len(data))
        symbolic_explanations = []
        
        try:
            # Apply knowledge graph reasoning
            for i, sample in enumerate(data):
                sample_explanations = []
                
                # Rule-based reasoning
                rule_score = 0.0
                for constraint in model.get_active_constraints():
                    # Generic rule evaluation
                    if "confidence" in constraint.rule.lower():
                        # Confidence-based reasoning
                        if neural_outputs[i] > 0.7:
                            rule_score += constraint.confidence
                            sample_explanations.append(f"High confidence rule: {constraint.rule}")
                    
                    elif "consistency" in constraint.rule.lower():
                        # Consistency checking
                        sample_variance = np.var(sample)
                        if sample_variance < np.mean([np.var(row) for row in data]) * 0.5:
                            rule_score += constraint.confidence
                            sample_explanations.append(f"Consistency rule satisfied: {constraint.rule}")
                
                # Use rule score directly
                symbolic_outputs[i] = min(1.0, rule_score)
                symbolic_explanations.append({
                    'sample_index': i,
                    'rule_score': rule_score,
                    'explanations': sample_explanations
                })
            
            return symbolic_outputs, symbolic_explanations
            
        except Exception as e:
            warnings.warn(f"Symbolic reasoning failed: {e}")
            return np.zeros(len(data)), []
    
    def _fuse_outputs(
        self,
        neural_outputs: NDArray[np.floating],
        symbolic_outputs: NDArray[np.floating], 
        fusion_strategy: FusionStrategy
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Fuse neural and symbolic outputs based on strategy."""
        
        if fusion_strategy == FusionStrategy.EARLY_FUSION:
            # Multiplicative fusion
            combined_outputs = neural_outputs * symbolic_outputs
        elif fusion_strategy == FusionStrategy.LATE_FUSION:
            # Additive fusion with equal weights
            combined_outputs = (neural_outputs + symbolic_outputs) / 2
        elif fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # Attention-weighted fusion
            attention_weights = neural_outputs / (neural_outputs + symbolic_outputs + 1e-6)
            combined_outputs = attention_weights * neural_outputs + (1 - attention_weights) * symbolic_outputs
        else:
            # Default to late fusion
            combined_outputs = (neural_outputs + symbolic_outputs) / 2
        
        # Return both combined outputs and the same as predictions (continuous)
        return combined_outputs, combined_outputs
    
    def _generate_explanations(
        self,
        model: NeuroSymbolicModel,
        data: NDArray[np.floating],
        predictions: NDArray[np.floating],
        neural_outputs: NDArray[np.floating],
        symbolic_outputs: NDArray[np.floating],
        symbolic_explanations: List[Dict[str, Any]],
        feature_names: Optional[List[str]]
    ) -> List[ReasoningResult]:
        """Generate reasoning explanations for predictions."""
        explanations = []
        
        for i in range(len(predictions)):
            explanation_steps = []
            
            # Neural reasoning step
            explanation_steps.append(
                f"Neural {model.neural_backbone} output: {neural_outputs[i]:.3f}"
            )
            
            # Symbolic reasoning steps
            if i < len(symbolic_explanations):
                for exp in symbolic_explanations[i]['explanations']:
                    explanation_steps.append(f"Symbolic reasoning: {exp}")
            
            # Fusion explanation
            explanation_steps.append(
                f"Fused using {model.fusion_strategy.value} strategy"
            )
            
            # Final prediction
            explanation_steps.append(f"Final prediction: {predictions[i]:.3f}")
            
            reasoning_result = ReasoningResult.create(
                prediction=float(predictions[i]),
                confidence=max(neural_outputs[i], symbolic_outputs[i]),
                symbolic_explanation=explanation_steps,
                neural_features={
                    'neural_output': neural_outputs[i],
                    'symbolic_output': symbolic_outputs[i]
                }
            )
            
            explanations.append(reasoning_result)
        
        return explanations
    
    def _generate_causal_explanations(
        self,
        model: NeuroSymbolicModel,
        data: NDArray[np.floating],
        predictions: NDArray[np.floating],
        neural_outputs: NDArray[np.floating],
        symbolic_outputs: NDArray[np.floating],
        feature_names: Optional[List[str]]
    ) -> List[CausalExplanation]:
        """Generate causal explanations for reasoning results."""
        causal_explanations = []
        
        # Generate for top predictions
        top_indices = np.argsort(predictions)[-10:]  # Top 10 predictions
        
        for idx in top_indices:
            sample = data[idx]
            
            # Create causal factors
            target_outcome = CausalFactor(
                id=f"reasoning_{idx}",
                name="Reasoning Result",
                value=predictions[idx],
                confidence=max(neural_outputs[idx], symbolic_outputs[idx]),
                evidence=[f"Neural output: {neural_outputs[idx]:.3f}", f"Symbolic output: {symbolic_outputs[idx]:.3f}"]
            )
            
            # Identify primary causes based on feature values
            primary_causes = []
            if feature_names:
                # Find influential features
                feature_stats = {
                    name: {'mean': np.mean(data[:, i]), 'std': np.std(data[:, i])}
                    for i, name in enumerate(feature_names[:min(len(feature_names), sample.shape[0])])
                }
                
                for i, (name, stats) in enumerate(feature_stats.items()):
                    z_score = abs((sample[i] - stats['mean']) / (stats['std'] + 1e-6))
                    if z_score > 1.5:  # Influential feature threshold
                        cause = CausalFactor(
                            id=f"feature_{name}_{idx}",
                            name=f"Feature {name}",
                            value=sample[i],
                            confidence=min(1.0, z_score / 3.0),
                            evidence=[f"Z-score: {z_score:.2f}", f"Value: {sample[i]:.3f}"]
                        )
                        primary_causes.append(cause)
            
            if not primary_causes:
                # Fallback: create generic cause
                primary_causes.append(CausalFactor(
                    id=f"data_pattern_{idx}",
                    name="Data Pattern",
                    value="influential",
                    confidence=0.7,
                    evidence=["Pattern identified in neural-symbolic reasoning"]
                ))
            
            # Create causal chain
            causal_links = []
            for cause in primary_causes[:3]:  # Limit to top 3 causes
                link = CausalLink(
                    cause=cause,
                    effect=target_outcome,
                    relation_type=CausalRelationType.CONTRIBUTING_CAUSE,
                    strength=cause.confidence,
                    temporal_relation=TemporalRelation.SIMULTANEOUS,
                    evidence=[f"Correlation with reasoning output"]
                )
                causal_links.append(link)
            
            if causal_links:
                chain = CausalChain(
                    links=causal_links,
                    total_strength=sum(link.strength for link in causal_links) / len(causal_links),
                    confidence=target_outcome.confidence
                )
                
                explanation = CausalExplanation.create(
                    target_outcome=target_outcome,
                    primary_causes=primary_causes,
                    causal_chains=[chain],
                    confidence=target_outcome.confidence,
                    methodology="neuro_symbolic_causal_analysis",
                    assumptions=["Independence of features", "Linear relationships"],
                    limitations=["Limited to available features", "Correlation-based causality"]
                )
                
                causal_explanations.append(explanation)
        
        return causal_explanations
    
    def _generate_counterfactual_explanations(
        self,
        model: NeuroSymbolicModel,
        data: NDArray[np.floating],
        predictions: NDArray[np.floating],
        feature_names: Optional[List[str]]
    ) -> List[CounterfactualResult]:
        """Generate counterfactual explanations for reasoning results."""
        counterfactual_explanations = []
        
        # Generate for samples with extreme predictions
        extreme_indices = []
        extreme_indices.extend(np.argsort(predictions)[-3:])  # Top 3
        extreme_indices.extend(np.argsort(predictions)[:3])   # Bottom 3
        
        for idx in extreme_indices:
            sample = data[idx]
            
            # Generate scenarios
            scenarios = []
            
            # Scenario 1: Move towards median values
            changes = []
            if feature_names:
                for i, name in enumerate(feature_names[:min(len(feature_names), len(sample))]):
                    feature_median = np.median(data[:, i])
                    feature_std = np.std(data[:, i])
                    
                    if abs(sample[i] - feature_median) > feature_std:  # Feature is extreme
                        change = FeatureChange(
                            feature_name=name,
                            original_value=sample[i],
                            counterfactual_value=feature_median,
                            change_direction=ChangeDirection.REPLACE,
                            change_magnitude=abs(sample[i] - feature_median),
                            confidence=0.8,
                            feasibility=0.9
                        )
                        changes.append(change)
            
            if changes:
                scenario = CounterfactualScenario(
                    id=f"median_scenario_{idx}",
                    name="Move to Median Values",
                    changes=changes,
                    original_prediction=f"{predictions[idx]:.3f}",
                    counterfactual_prediction="median_value",
                    prediction_change_magnitude=0.5,
                    scenario_probability=0.7,
                    explanation="Moving extreme feature values to population median"
                )
                scenarios.append(scenario)
            
            if scenarios:
                # Calculate feature importance
                feature_importance = []
                if feature_names:
                    for i, name in enumerate(feature_names[:min(len(feature_names), len(sample))]):
                        feature_median = np.median(data[:, i])
                        feature_std = np.std(data[:, i])
                        importance = abs(sample[i] - feature_median) / (feature_std + 1e-8)
                        feature_importance.append((name, min(1.0, importance)))
                
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                counterfactual = CounterfactualResult.create(
                    query=f"What changes would affect reasoning for sample {idx}?",
                    counterfactual_type=CounterfactualType.WHAT_IF,
                    original_input={f"feature_{i}": float(val) for i, val in enumerate(sample)},
                    original_prediction=f"{predictions[idx]:.3f}",
                    original_confidence=0.8,
                    scenarios=scenarios,
                    feature_importance_ranking=feature_importance,
                    stability_score=0.7,
                    robustness_score=0.6,
                    assumptions=["Feature independence", "Continuous prediction space"],
                    limitations=["Based on statistical analysis only", "Limited feature interactions"]
                )
                
                counterfactual_explanations.append(counterfactual)
        
        return counterfactual_explanations
    
    def get_model(self, model_id: str) -> NeuroSymbolicModel:
        """Get a model by ID."""
        if model_id not in self.models:
            raise ModelError(
                f"Model '{model_id}' not found",
                model_id=model_id
            )
        return self.models[model_id]
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all reasoning models."""
        return [model.get_model_summary() for model in self.models.values()]
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a reasoning model."""
        if model_id in self.models:
            del self.models[model_id]
            
            # Clean up neural adapter
            try:
                neural_backbone_id = f"{model_id}_neural"
                if neural_backbone_id in self.neural_adapter.backbones:
                    del self.neural_adapter.backbones[neural_backbone_id]
            except:
                pass
            
            self.logger.info(f"Deleted reasoning model '{model_id}'")
            return True
        return False