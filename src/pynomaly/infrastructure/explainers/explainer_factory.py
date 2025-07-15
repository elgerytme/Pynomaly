"""
Factory for creating and configuring explainers.

This completes Issue #98: Complete Explainability Service Implementation
by providing proper explainer instantiation and configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pynomaly.domain.services.explainability_service import (
    ExplanationMethod,
    ExplainerProtocol
)
from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger

# Import concrete explainers
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer


class ExplainerConfiguration:
    """Configuration for explainers."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize explainer configuration."""
        self.settings = settings or Settings()
        self.logger = StructuredLogger(__name__)
        
        # Default configurations for each explainer type
        self.shap_config = {
            "feature_perturbation": self.settings.explainability.shap_feature_perturbation,
            "algorithm": self.settings.explainability.shap_algorithm,
            "max_evals": 2 * self.settings.explainability.max_features + 1,
            "batch_size": self.settings.explainability.batch_size,
            "n_background_samples": self.settings.explainability.background_samples,
            "random_state": 42,
            "check_additivity": self.settings.explainability.shap_check_additivity,
            "linearize_link": self.settings.explainability.shap_linearize_link
        }
        
        self.lime_config = {
            "feature_selection": self.settings.explainability.lime_feature_selection,
            "num_features": self.settings.explainability.max_features,
            "num_samples": self.settings.explainability.lime_num_samples,
            "distance_metric": self.settings.explainability.lime_distance_metric,
            "kernel_width": None,  # Auto-determined
            "discretize_continuous": self.settings.explainability.lime_discretize_continuous,
            "discretizer": self.settings.explainability.lime_discretizer,
            "sample_around_instance": True,
            "random_state": 42
        }
        
        # Performance and caching settings
        self.performance_config = {
            "enable_caching": self.settings.explainability.enable_caching,
            "cache_ttl": self.settings.explainability.cache_ttl,
            "max_cache_size": self.settings.explainability.max_cache_size,
            "enable_parallel": True,
            "n_jobs": self.settings.explainability.n_jobs,
            "chunk_size": 1000
        }


class ExplainerFactory:
    """Factory for creating configured explainers."""
    
    def __init__(self, configuration: Optional[ExplainerConfiguration] = None):
        """Initialize explainer factory."""
        self.config = configuration or ExplainerConfiguration()
        self.logger = StructuredLogger(__name__)
        
        # Registry of available explainers
        self._explainer_classes = {
            ExplanationMethod.SHAP: SHAPExplainer,
            ExplanationMethod.LIME: LIMEExplainer,
        }
        
        # Cache for configured explainers
        self._explainer_cache: Dict[ExplanationMethod, ExplainerProtocol] = {}
    
    def create_explainer(
        self,
        method: ExplanationMethod,
        model_type: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ExplainerProtocol:
        """
        Create and configure an explainer for the specified method.
        
        Args:
            method: The explanation method to create explainer for
            model_type: Optional model type for method-specific configuration
            custom_config: Optional custom configuration overrides
            
        Returns:
            Configured explainer instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in self._explainer_classes:
            available_methods = list(self._explainer_classes.keys())
            raise ValueError(
                f"Unsupported explanation method: {method}. "
                f"Available methods: {[m.value for m in available_methods]}"
            )
        
        # Check cache first
        cache_key = method
        if cache_key in self._explainer_cache and not custom_config:
            self.logger.debug(f"Returning cached explainer for {method.value}")
            return self._explainer_cache[cache_key]
        
        # Get base configuration for the method
        base_config = self._get_base_config(method, model_type)
        
        # Apply custom configuration overrides
        if custom_config:
            base_config.update(custom_config)
        
        # Create explainer instance
        explainer_class = self._explainer_classes[method]
        
        try:
            explainer = explainer_class(**base_config)
            
            # Cache the explainer if no custom config
            if not custom_config:
                self._explainer_cache[cache_key] = explainer
            
            self.logger.info(
                f"Created explainer for method {method.value}",
                model_type=model_type,
                config_keys=list(base_config.keys())
            )
            
            return explainer
            
        except Exception as e:
            self.logger.error(
                f"Failed to create explainer for method {method.value}",
                error=str(e),
                config=base_config,
                exc_info=True
            )
            raise
    
    def create_all_explainers(
        self,
        model_type: Optional[str] = None
    ) -> Dict[ExplanationMethod, ExplainerProtocol]:
        """
        Create all available explainers.
        
        Args:
            model_type: Optional model type for method-specific configuration
            
        Returns:
            Dictionary mapping methods to configured explainers
        """
        explainers = {}
        
        for method in self._explainer_classes.keys():
            try:
                explainer = self.create_explainer(method, model_type)
                explainers[method] = explainer
            except Exception as e:
                self.logger.warning(
                    f"Failed to create explainer for {method.value}",
                    error=str(e)
                )
        
        self.logger.info(
            f"Created {len(explainers)} explainers",
            methods=[m.value for m in explainers.keys()]
        )
        
        return explainers
    
    def _get_base_config(
        self,
        method: ExplanationMethod,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get base configuration for an explainer method."""
        base_config = {}
        
        if method == ExplanationMethod.SHAP:
            base_config = self.config.shap_config.copy()
            
            # Model-specific SHAP configuration
            if model_type:
                if model_type in ["tree", "ensemble", "random_forest", "gradient_boosting"]:
                    base_config["algorithm"] = "tree"
                elif model_type in ["linear", "logistic_regression", "svm"]:
                    base_config["algorithm"] = "linear"
                elif model_type in ["neural_network", "deep_learning"]:
                    base_config["algorithm"] = "deep"
                else:
                    base_config["algorithm"] = "kernel"
        
        elif method == ExplanationMethod.LIME:
            base_config = self.config.lime_config.copy()
            
            # Model-specific LIME configuration
            if model_type:
                if model_type in ["tree", "ensemble"]:
                    base_config["discretize_continuous"] = False
                elif model_type in ["neural_network", "deep_learning"]:
                    base_config["num_samples"] = 10000  # More samples for complex models
        
        # Add performance configuration
        base_config.update(self.config.performance_config)
        
        return base_config
    
    def register_explainer_class(
        self,
        method: ExplanationMethod,
        explainer_class: type
    ) -> None:
        """Register a custom explainer class for a method."""
        self._explainer_classes[method] = explainer_class
        
        # Clear cache for this method if it exists
        if method in self._explainer_cache:
            del self._explainer_cache[method]
        
        self.logger.info(
            f"Registered custom explainer class for {method.value}",
            explainer_class=explainer_class.__name__
        )
    
    def clear_cache(self) -> None:
        """Clear the explainer cache."""
        self._explainer_cache.clear()
        self.logger.debug("Cleared explainer cache")
    
    def get_supported_methods(self) -> list[ExplanationMethod]:
        """Get list of supported explanation methods."""
        return list(self._explainer_classes.keys())


class ExplainerManager:
    """Manager for explainer lifecycle and configuration."""
    
    def __init__(
        self,
        factory: Optional[ExplainerFactory] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize explainer manager."""
        self.factory = factory or ExplainerFactory()
        self.settings = settings or Settings()
        self.logger = StructuredLogger(__name__)
        
        # Active explainers
        self._active_explainers: Dict[str, Dict[ExplanationMethod, ExplainerProtocol]] = {}
    
    def get_explainer_for_model(
        self,
        model_id: str,
        method: ExplanationMethod,
        model_type: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ExplainerProtocol:
        """
        Get or create an explainer for a specific model.
        
        Args:
            model_id: Unique identifier for the model
            method: Explanation method
            model_type: Optional model type for configuration
            custom_config: Optional custom configuration
            
        Returns:
            Configured explainer instance
        """
        # Check if we have a cached explainer for this model
        if model_id in self._active_explainers:
            if method in self._active_explainers[model_id]:
                if not custom_config:  # Only use cache if no custom config
                    return self._active_explainers[model_id][method]
        
        # Create new explainer
        explainer = self.factory.create_explainer(
            method=method,
            model_type=model_type,
            custom_config=custom_config
        )
        
        # Cache the explainer for this model (if no custom config)
        if not custom_config:
            if model_id not in self._active_explainers:
                self._active_explainers[model_id] = {}
            self._active_explainers[model_id][method] = explainer
        
        return explainer
    
    def initialize_explainers_for_model(
        self,
        model_id: str,
        model_type: Optional[str] = None,
        methods: Optional[list[ExplanationMethod]] = None
    ) -> Dict[ExplanationMethod, ExplainerProtocol]:
        """
        Initialize all explainers for a model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Optional model type for configuration
            methods: Optional list of methods to initialize (default: all)
            
        Returns:
            Dictionary of initialized explainers
        """
        if methods is None:
            methods = self.factory.get_supported_methods()
        
        explainers = {}
        
        for method in methods:
            try:
                explainer = self.get_explainer_for_model(
                    model_id=model_id,
                    method=method,
                    model_type=model_type
                )
                explainers[method] = explainer
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize explainer {method.value} for model {model_id}",
                    error=str(e)
                )
        
        self.logger.info(
            f"Initialized {len(explainers)} explainers for model {model_id}",
            methods=[m.value for m in explainers.keys()]
        )
        
        return explainers
    
    def cleanup_model_explainers(self, model_id: str) -> None:
        """Clean up explainers for a specific model."""
        if model_id in self._active_explainers:
            del self._active_explainers[model_id]
            self.logger.debug(f"Cleaned up explainers for model {model_id}")
    
    def get_explainer_stats(self) -> Dict[str, Any]:
        """Get statistics about active explainers."""
        total_explainers = sum(
            len(explainers) for explainers in self._active_explainers.values()
        )
        
        method_counts = {}
        for explainers in self._active_explainers.values():
            for method in explainers.keys():
                method_counts[method.value] = method_counts.get(method.value, 0) + 1
        
        return {
            "total_models": len(self._active_explainers),
            "total_explainers": total_explainers,
            "method_distribution": method_counts,
            "supported_methods": [m.value for m in self.factory.get_supported_methods()]
        }


# Global explainer manager instance
_explainer_manager: Optional[ExplainerManager] = None


def get_explainer_manager() -> ExplainerManager:
    """Get the global explainer manager instance."""
    global _explainer_manager
    
    if _explainer_manager is None:
        _explainer_manager = ExplainerManager()
    
    return _explainer_manager


def configure_explainers(settings: Settings) -> ExplainerManager:
    """Configure explainers with custom settings."""
    global _explainer_manager
    
    configuration = ExplainerConfiguration(settings)
    factory = ExplainerFactory(configuration)
    _explainer_manager = ExplainerManager(factory, settings)
    
    return _explainer_manager