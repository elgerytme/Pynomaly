#!/usr/bin/env python3
"""
Pipeline Configuration Service - Handles pipeline configuration and setup
"""

import logging

from monorepo.domain.models.pipeline_models import PipelineConfig, PipelineMode

logger = logging.getLogger(__name__)


class PipelineConfigService:
    """Service responsible for pipeline configuration management"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_optimization_config(self) -> dict:
        """Get optimization configuration based on pipeline mode"""

        if self.config.mode == PipelineMode.FAST:
            return {
                "n_trials": 20,
                "timeout_seconds": 300,
                "cv_folds": 3,
                "max_models": 5,
            }
        elif self.config.mode == PipelineMode.THOROUGH:
            return {
                "n_trials": 500,
                "timeout_seconds": 3600,
                "cv_folds": 10,
                "max_models": 30,
            }
        else:  # BALANCED
            return {
                "n_trials": 100,
                "timeout_seconds": 1800,
                "cv_folds": 5,
                "max_models": 20,
            }

    def get_feature_engineering_config(self) -> dict:
        """Get feature engineering configuration"""

        return {
            "enabled": self.config.enable_feature_engineering,
            "max_combinations": self.config.max_feature_combinations,
            "selection_threshold": self.config.feature_selection_threshold,
        }

    def get_resource_limits(self) -> dict:
        """Get resource limit configuration"""

        return {
            "max_memory_gb": self.config.max_memory_usage_gb,
            "max_cpu_cores": self.config.max_cpu_cores,
            "optimization_time_budget": self.config.optimization_time_budget_minutes,
        }

    def validate_config(self) -> dict:
        """Validate pipeline configuration and return issues if any"""

        issues = []
        warnings = []

        # Check resource limits
        if self.config.max_memory_usage_gb < 1.0:
            warnings.append("Low memory limit may cause performance issues")

        if self.config.max_cpu_cores < 1:
            issues.append("Invalid CPU core count")

        # Check time budgets
        if self.config.optimization_time_budget_minutes < 1:
            warnings.append("Very short optimization time budget")

        # Check model limits
        if self.config.max_models_to_evaluate < 1:
            issues.append("Must evaluate at least one model")

        # Check cross-validation folds
        if self.config.cross_validation_folds < 2:
            issues.append("Cross-validation requires at least 2 folds")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
