#!/usr/bin/env python3
"""
Model Optimization Service - Handles model selection and hyperparameter optimization
"""

import logging
from typing import Any

import pandas as pd

from pynomaly.application.services.advanced_model_optimization_service import (
    AdvancedModelOptimizationService,
    AdvancedOptimizationConfig,
)
from pynomaly.domain.models.pipeline_models import PipelineConfig, PipelineMode

logger = logging.getLogger(__name__)


class ModelOptimizationService:
    """Service responsible for model selection and optimization"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.optimization_service = self._initialize_optimization_service()

    def _initialize_optimization_service(self) -> AdvancedModelOptimizationService:
        """Initialize the optimization service based on pipeline mode"""

        if self.config.mode == PipelineMode.FAST:
            optimization_config = AdvancedOptimizationConfig(
                n_trials=20, timeout_seconds=300, cv_folds=3
            )
        elif self.config.mode == PipelineMode.THOROUGH:
            optimization_config = AdvancedOptimizationConfig(
                n_trials=500, timeout_seconds=3600, cv_folds=10
            )
        else:  # BALANCED
            optimization_config = AdvancedOptimizationConfig(
                n_trials=100, timeout_seconds=1800, cv_folds=5
            )

        return AdvancedModelOptimizationService(optimization_config)

    async def optimize_models(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """
        Run complete model optimization pipeline
        
        Returns:
            Dictionary containing optimization results
        """

        # Select candidate models
        model_selection_result = await self.select_models(X, y)
        selected_models = model_selection_result["selected_models"]

        # Optimize hyperparameters
        optimization_result = await self.optimize_hyperparameters(X, y, selected_models)

        return {
            "model_selection": model_selection_result,
            "optimization": optimization_result,
            "best_model": optimization_result.get("best_model"),
            "best_params": optimization_result.get("best_params", {}),
            "best_performance": optimization_result.get("best_performance", {}),
        }

    async def select_models(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Select candidate models based on data characteristics"""

        logger.info("🎯 Selecting candidate models")

        n_samples, n_features = X.shape
        model_candidates = []
        selection_rationale = {}

        # Size-based selection
        if n_samples < 500:
            model_candidates.extend(["one_class_svm", "local_outlier_factor"])
            selection_rationale["size_factor"] = "Small dataset: SVM and LOF preferred"
        elif n_samples < 5000:
            model_candidates.extend(
                ["isolation_forest", "random_forest", "one_class_svm"]
            )
            selection_rationale["size_factor"] = (
                "Medium dataset: Tree-based and SVM models"
            )
        else:
            model_candidates.extend(
                ["isolation_forest", "random_forest", "gradient_boosting"]
            )
            selection_rationale["size_factor"] = (
                "Large dataset: Tree-based and ensemble methods"
            )

        # Dimensionality-based selection
        if n_features > 100:
            model_candidates.extend(["random_forest", "extra_trees"])
            selection_rationale["dimensionality_factor"] = (
                "High-dimensional: Random Forest preferred"
            )
        elif n_features < 10:
            model_candidates.extend(["one_class_svm", "local_outlier_factor"])
            selection_rationale["dimensionality_factor"] = (
                "Low-dimensional: SVM and LOF suitable"
            )
        else:
            model_candidates.extend(["isolation_forest", "random_forest"])
            selection_rationale["dimensionality_factor"] = (
                "Medium-dimensional: Standard algorithms"
            )

        # Always include baseline models
        model_candidates.extend(["isolation_forest", "random_forest"])

        # Remove duplicates and limit to max models
        model_candidates = list(dict.fromkeys(model_candidates))
        model_candidates = model_candidates[: self.config.max_models_to_evaluate]

        logger.info(f"Selected {len(model_candidates)} candidate models: {model_candidates}")

        return {
            "selected_models": model_candidates,
            "selection_rationale": selection_rationale,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
            },
        }

    async def optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series | None, selected_models: list[str]
    ) -> dict[str, Any]:
        """Optimize hyperparameters for selected models"""

        logger.info("🎯 Optimizing hyperparameters")

        try:
            # Run advanced optimization
            optimization_result = await self.optimization_service.optimize_model_advanced(
                X, y, selected_models
            )

            logger.info("Optimization completed successfully")
            logger.info(
                f"Best model: {optimization_result.best_params.get('model_type', 'unknown')}"
            )

            return {
                "optimization_result": optimization_result,
                "best_model": optimization_result.best_model,
                "best_params": optimization_result.best_params,
                "best_performance": optimization_result.best_scores,
                "optimization_time": optimization_result.optimization_time,
                "total_trials": optimization_result.total_trials,
                "successful_trials": optimization_result.successful_trials,
            }

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")

            # Fallback to simple optimization
            return await self._fallback_optimization(X, y)

    async def _fallback_optimization(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Fallback optimization using simple methods"""

        logger.info("Using fallback optimization")

        from sklearn.ensemble import IsolationForest

        # Simple model with default parameters
        best_model = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42
        )
        best_model.fit(X)

        best_params = {
            "model_type": "isolation_forest",
            "n_estimators": 100,
            "contamination": 0.1,
        }

        if y is not None:
            # Calculate simple performance metric for supervised case
            y_pred = best_model.predict(X)
            from sklearn.metrics import f1_score

            try:
                best_performance = {"f1_score": f1_score(y, y_pred, average="weighted")}
            except:
                best_performance = {"fitted": True}
        else:
            best_performance = {"fitted": True}

        return {
            "optimization_result": None,
            "best_model": best_model,
            "best_params": best_params,
            "best_performance": best_performance,
            "optimization_time": 0.0,
            "total_trials": 1,
            "successful_trials": 1,
            "fallback_used": True,
        }
