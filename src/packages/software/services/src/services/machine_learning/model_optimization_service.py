#!/usr/bin/env python3
"""
Processor Optimization Service - Handles processor selection and hyperparameter optimization
"""

import logging
from typing import Any

import pandas as pd

from monorepo.application.services.advanced_model_optimization_service import (
    AdvancedModelOptimizationService,
    AdvancedOptimizationConfig,
)
from monorepo.domain.models.pipeline_models import PipelineConfig, PipelineMode

logger = logging.getLogger(__name__)


class ModelOptimizationService:
    """Service responsible for processor selection and optimization"""

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

    async def optimize_processors(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """
        Run complete processor optimization pipeline

        Returns:
            Dictionary containing optimization results
        """

        # Select candidate models
        processor_selection_result = await self.select_processors(X, y)
        selected_processors = processor_selection_result["selected_processors"]

        # Optimize hyperparameters
        optimization_result = await self.optimize_hyperparameters(X, y, selected_processors)

        return {
            "processor_selection": processor_selection_result,
            "optimization": optimization_result,
            "best_processor": optimization_result.get("best_processor"),
            "best_params": optimization_result.get("best_params", {}),
            "best_performance": optimization_result.get("best_performance", {}),
        }

    async def select_processors(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Select candidate models based on data characteristics"""

        logger.info("🎯 Selecting candidate models")

        n_samples, n_features = X.shape
        processor_candidates = []
        selection_rationale = {}

        # Size-based selection
        if n_samples < 500:
            processor_candidates.extend(["one_class_svm", "local_outlier_factor"])
            selection_rationale["size_factor"] = "Small data_collection: SVM and LOF preferred"
        elif n_samples < 5000:
            processor_candidates.extend(
                ["isolation_forest", "random_forest", "one_class_svm"]
            )
            selection_rationale["size_factor"] = (
                "Medium data_collection: Tree-based and SVM models"
            )
        else:
            processor_candidates.extend(
                ["isolation_forest", "random_forest", "gradient_boosting"]
            )
            selection_rationale["size_factor"] = (
                "Large data_collection: Tree-based and ensemble methods"
            )

        # Dimensionality-based selection
        if n_features > 100:
            processor_candidates.extend(["random_forest", "extra_trees"])
            selection_rationale["dimensionality_factor"] = (
                "High-dimensional: Random Forest preferred"
            )
        elif n_features < 10:
            processor_candidates.extend(["one_class_svm", "local_outlier_factor"])
            selection_rationale["dimensionality_factor"] = (
                "Low-dimensional: SVM and LOF suitable"
            )
        else:
            processor_candidates.extend(["isolation_forest", "random_forest"])
            selection_rationale["dimensionality_factor"] = (
                "Medium-dimensional: Standard algorithms"
            )

        # Always include baseline models
        processor_candidates.extend(["isolation_forest", "random_forest"])

        # Remove duplicates and limit to max models
        processor_candidates = list(dict.fromkeys(processor_candidates))
        processor_candidates = processor_candidates[: self.config.max_processors_to_evaluate]

        logger.info(
            f"Selected {len(processor_candidates)} candidate models: {processor_candidates}"
        )

        return {
            "selected_processors": processor_candidates,
            "selection_rationale": selection_rationale,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
            },
        }

    async def optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series | None, selected_processors: list[str]
    ) -> dict[str, Any]:
        """Optimize hyperparameters for selected models"""

        logger.info("🎯 Optimizing hyperparameters")

        try:
            # Run advanced optimization
            optimization_result = (
                await self.optimization_service.optimize_processor_advanced(
                    X, y, selected_processors
                )
            )

            logger.info("Optimization completed successfully")
            logger.info(
                f"Best processor: {optimization_result.best_params.get('processor_type', 'unknown')}"
            )

            return {
                "optimization_result": optimization_result,
                "best_processor": optimization_result.best_processor,
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

        # Simple processor with default parameters
        best_processor = IsolationForest(
            n_estimators=100, contamination=0.1, random_state=42
        )
        best_processor.fit(X)

        best_params = {
            "processor_type": "isolation_forest",
            "n_estimators": 100,
            "contamination": 0.1,
        }

        if y is not None:
            # Calculate simple performance metric for supervised case
            y_pred = best_processor.predict(X)
            from sklearn.metrics import f1_score

            try:
                best_performance = {"f1_score": f1_score(y, y_pred, average="weighted")}
            except:
                best_performance = {"fitted": True}
        else:
            best_performance = {"fitted": True}

        return {
            "optimization_result": None,
            "best_processor": best_processor,
            "best_params": best_params,
            "best_performance": best_performance,
            "optimization_time": 0.0,
            "total_trials": 1,
            "successful_trials": 1,
            "fallback_used": True,
        }
