"""
Service initialization for explainability components.

This completes Issue #98: Complete Explainability Service Implementation
by providing proper service wiring and initialization.
"""

from __future__ import annotations

import logging
from typing import Optional

from pynomaly.domain.services.explainability_service import (
    ExplainabilityService,
    ExplanationMethod
)
from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger

from .explainer_factory import ExplainerFactory, ExplainerManager, configure_explainers
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer


def initialize_explainability_service(
    settings: Optional[Settings] = None
) -> ExplainabilityService:
    """
    Initialize the explainability service with all configured explainers.
    
    Args:
        settings: Optional settings instance. If None, will create from environment.
        
    Returns:
        Fully configured ExplainabilityService instance
    """
    logger = StructuredLogger(__name__)
    
    # Get or create settings
    if settings is None:
        from pynomaly.infrastructure.config import get_settings
        settings = get_settings()
    
    logger.info("Initializing explainability service")
    
    try:
        # Configure explainer manager with settings
        explainer_manager = configure_explainers(settings)
        
        # Create domain service
        explainability_service = ExplainabilityService()
        
        # Register available explainers
        _register_explainers(explainability_service, explainer_manager, settings)
        
        logger.info(
            "Successfully initialized explainability service",
            available_methods=[m.value for m in explainability_service.get_available_methods()]
        )
        
        return explainability_service
        
    except Exception as e:
        logger.error(
            "Failed to initialize explainability service",
            error=str(e),
            exc_info=True
        )
        raise


def _register_explainers(
    service: ExplainabilityService,
    manager: ExplainerManager,
    settings: Settings
) -> None:
    """Register all configured explainers with the service."""
    logger = StructuredLogger(__name__)
    
    # Check which explainers should be enabled based on settings
    enabled_methods = []
    
    if settings.explainability.enable_explainability:
        # SHAP explainer
        try:
            shap_explainer = manager.factory.create_explainer(ExplanationMethod.SHAP)
            service.register_explainer(ExplanationMethod.SHAP, shap_explainer)
            enabled_methods.append(ExplanationMethod.SHAP)
            logger.debug("Registered SHAP explainer")
        except Exception as e:
            logger.warning(f"Failed to register SHAP explainer: {e}")
        
        # LIME explainer
        try:
            lime_explainer = manager.factory.create_explainer(ExplanationMethod.LIME)
            service.register_explainer(ExplanationMethod.LIME, lime_explainer)
            enabled_methods.append(ExplanationMethod.LIME)
            logger.debug("Registered LIME explainer")
        except Exception as e:
            logger.warning(f"Failed to register LIME explainer: {e}")
        
        # Feature importance explainer (if available)
        try:
            # Note: Feature importance could be a simple explainer
            # For now, we'll use SHAP as it provides feature importance
            feature_importance_explainer = manager.factory.create_explainer(
                ExplanationMethod.SHAP,
                custom_config={"algorithm": "permutation"}
            )
            service.register_explainer(
                ExplanationMethod.FEATURE_IMPORTANCE, 
                feature_importance_explainer
            )
            enabled_methods.append(ExplanationMethod.FEATURE_IMPORTANCE)
            logger.debug("Registered Feature Importance explainer")
        except Exception as e:
            logger.warning(f"Failed to register Feature Importance explainer: {e}")
    
    logger.info(
        f"Registered {len(enabled_methods)} explainers",
        methods=[m.value for m in enabled_methods]
    )


def create_explainer_for_model(
    model: any,
    model_type: Optional[str] = None,
    method: ExplanationMethod = ExplanationMethod.SHAP,
    settings: Optional[Settings] = None
) -> any:
    """
    Create a configured explainer for a specific model.
    
    Args:
        model: The trained model to create explainer for
        model_type: Optional model type hint for optimization
        method: Explanation method to use
        settings: Optional settings override
        
    Returns:
        Configured explainer instance
    """
    logger = StructuredLogger(__name__)
    
    if settings is None:
        from pynomaly.infrastructure.config import get_settings
        settings = get_settings()
    
    if not settings.explainability.enable_explainability:
        raise ValueError("Explainability is disabled in settings")
    
    # Get explainer manager
    manager = configure_explainers(settings)
    
    # Generate unique model ID (in practice, this would come from model registry)
    model_id = f"model_{id(model)}"
    
    try:
        explainer = manager.get_explainer_for_model(
            model_id=model_id,
            method=method,
            model_type=model_type
        )
        
        logger.info(
            f"Created {method.value} explainer for model",
            model_id=model_id,
            model_type=model_type
        )
        
        return explainer
        
    except Exception as e:
        logger.error(
            f"Failed to create explainer for model",
            model_id=model_id,
            method=method.value,
            error=str(e),
            exc_info=True
        )
        raise


def cleanup_explainers_for_model(model_id: str) -> None:
    """Clean up explainers for a specific model."""
    logger = StructuredLogger(__name__)
    
    try:
        from .explainer_factory import get_explainer_manager
        manager = get_explainer_manager()
        manager.cleanup_model_explainers(model_id)
        
        logger.debug(f"Cleaned up explainers for model {model_id}")
        
    except Exception as e:
        logger.warning(
            f"Failed to cleanup explainers for model {model_id}",
            error=str(e)
        )


def get_explainer_statistics() -> dict:
    """Get statistics about the explainer system."""
    try:
        from .explainer_factory import get_explainer_manager
        manager = get_explainer_manager()
        return manager.get_explainer_stats()
    except Exception:
        return {"error": "Failed to get explainer statistics"}


# Global service instance for dependency injection
_explainability_service: Optional[ExplainabilityService] = None


def get_explainability_service() -> ExplainabilityService:
    """Get the global explainability service instance."""
    global _explainability_service
    
    if _explainability_service is None:
        _explainability_service = initialize_explainability_service()
    
    return _explainability_service


def reset_explainability_service() -> None:
    """Reset the global explainability service (useful for testing)."""
    global _explainability_service
    _explainability_service = None