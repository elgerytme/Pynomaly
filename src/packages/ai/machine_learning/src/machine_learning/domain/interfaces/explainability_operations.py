"""Explainability Operations Interfaces (Ports).

This module defines the abstract interfaces for explainability operations
that the machine learning domain requires. These interfaces represent the
"ports" in hexagonal architecture, defining contracts for external
explainability libraries without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external explainability capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..entities.model import Model
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult


class ExplanationMethod(Enum):
    """Available explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_ABLATION = "feature_ablation"
    GRADIENT_BASED = "gradient_based"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class ExplanationScope(Enum):
    """Scope of explanation (local vs global)."""
    LOCAL = "local"  # Individual prediction explanation
    GLOBAL = "global"  # Model behavior explanation
    COHORT = "cohort"  # Group-level explanation


@dataclass
class ExplanationRequest:
    """Request for generating model explanations."""
    model: Model
    data: Dataset
    method: ExplanationMethod
    scope: ExplanationScope
    instance_index: Optional[int] = None  # For local explanations
    feature_names: Optional[List[str]] = None
    target_class: Optional[str] = None
    num_features: int = 10  # Number of features to explain
    background_data: Optional[Dataset] = None
    created_by: str = "system"


@dataclass
class FeatureContribution:
    """Feature contribution to a prediction."""
    feature_name: str
    contribution_value: float
    importance_rank: int
    confidence: float
    description: Optional[str] = None


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    method: ExplanationMethod
    scope: ExplanationScope
    feature_contributions: List[FeatureContribution]
    explanation_metadata: Dict[str, Any]
    confidence_score: float
    processing_time_seconds: float
    visualization_data: Optional[Dict[str, Any]] = None
    textual_explanation: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class GlobalExplanationResult:
    """Result of global model explanation."""
    feature_importance: Dict[str, float]
    feature_interactions: Optional[Dict[str, Dict[str, float]]]
    model_performance_by_feature: Dict[str, float]
    decision_rules: Optional[List[str]]
    explanation_metadata: Dict[str, Any]
    confidence_score: float
    processing_time_seconds: float
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ExplainabilityPort(ABC):
    """Port for explainability operations.
    
    This interface defines the contract for generating explanations of
    machine learning model predictions and behaviors using various
    explainability techniques.
    """

    @abstractmethod
    async def explain_prediction(
        self, 
        request: ExplanationRequest
    ) -> ExplanationResult:
        """Generate explanation for a specific prediction.
        
        Args:
            request: Explanation request with model, data, and configuration
            
        Returns:
            Explanation result with feature contributions and metadata
            
        Raises:
            ExplanationError: If explanation generation fails
            UnsupportedMethodError: If explanation method is not supported
        """
        pass

    @abstractmethod
    async def explain_model_behavior(
        self, 
        model: Model, 
        representative_data: Dataset,
        method: ExplanationMethod = ExplanationMethod.SHAP
    ) -> GlobalExplanationResult:
        """Generate global explanation of model behavior.
        
        Args:
            model: Trained model to explain
            representative_data: Representative sample for global explanation
            method: Explanation method to use
            
        Returns:
            Global explanation with feature importance and interactions
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass

    @abstractmethod
    async def compare_explanations(
        self,
        explanations: List[ExplanationResult]
    ) -> Dict[str, Any]:
        """Compare multiple explanations to identify patterns.
        
        Args:
            explanations: List of explanation results to compare
            
        Returns:
            Comparison analysis with similarities and differences
            
        Raises:
            ComparisonError: If comparison fails
        """
        pass

    @abstractmethod
    async def get_supported_methods(self) -> List[ExplanationMethod]:
        """Get list of supported explanation methods.
        
        Returns:
            List of explanation methods supported by this implementation
        """
        pass

    @abstractmethod
    async def validate_explanation_request(
        self, 
        request: ExplanationRequest
    ) -> bool:
        """Validate if explanation request can be processed.
        
        Args:
            request: Explanation request to validate
            
        Returns:
            True if request is valid, False otherwise
            
        Raises:
            ValidationError: If validation fails due to configuration issues
        """
        pass


class ModelInterpretabilityPort(ABC):
    """Port for model interpretability operations.
    
    This interface defines the contract for analyzing model structure,
    decision boundaries, and internal representations.
    """

    @abstractmethod
    async def analyze_decision_boundaries(
        self,
        model: Model,
        data: Dataset,
        feature_pairs: Optional[List[tuple]] = None
    ) -> Dict[str, Any]:
        """Analyze model decision boundaries.
        
        Args:
            model: Trained model to analyze
            data: Data for boundary analysis
            feature_pairs: Specific feature pairs to analyze
            
        Returns:
            Decision boundary analysis results
            
        Raises:
            AnalysisError: If boundary analysis fails
        """
        pass

    @abstractmethod
    async def extract_decision_rules(
        self,
        model: Model,
        max_rules: int = 50
    ) -> List[str]:
        """Extract human-readable decision rules from model.
        
        Args:
            model: Trained model to extract rules from
            max_rules: Maximum number of rules to extract
            
        Returns:
            List of decision rules in natural language
            
        Raises:
            RuleExtractionError: If rule extraction fails
        """
        pass

    @abstractmethod
    async def analyze_feature_interactions(
        self,
        model: Model,
        data: Dataset,
        top_k_interactions: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Analyze interactions between features.
        
        Args:
            model: Trained model for interaction analysis
            data: Data for interaction analysis
            top_k_interactions: Number of top interactions to return
            
        Returns:
            Feature interaction strengths
            
        Raises:
            InteractionAnalysisError: If interaction analysis fails
        """
        pass

    @abstractmethod
    async def generate_counterfactual_examples(
        self,
        model: Model,
        instance: Dict[str, Any],
        num_examples: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual examples for given instance.
        
        Args:
            model: Trained model for counterfactual generation
            instance: Input instance to generate counterexamples for
            num_examples: Number of counterfactual examples to generate
            
        Returns:
            List of counterfactual examples
            
        Raises:
            CounterfactualGenerationError: If generation fails
        """
        pass


# Custom exceptions for explainability operations
class ExplainabilityOperationError(Exception):
    """Base exception for explainability operation errors."""
    pass


class ExplanationError(ExplainabilityOperationError):
    """Exception raised during explanation generation."""
    pass


class UnsupportedMethodError(ExplainabilityOperationError):
    """Exception raised for unsupported explanation methods."""
    pass


class ComparisonError(ExplainabilityOperationError):
    """Exception raised during explanation comparison."""
    pass


class ValidationError(ExplainabilityOperationError):
    """Exception raised during request validation."""
    pass


class AnalysisError(ExplainabilityOperationError):
    """Exception raised during model analysis."""
    pass


class RuleExtractionError(ExplainabilityOperationError):
    """Exception raised during rule extraction."""
    pass


class InteractionAnalysisError(ExplainabilityOperationError):
    """Exception raised during interaction analysis."""
    pass


class CounterfactualGenerationError(ExplainabilityOperationError):
    """Exception raised during counterfactual generation."""
    pass