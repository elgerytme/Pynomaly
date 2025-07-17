"""Interfaces for pipeline services following single responsibility principle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import pandas as pd


class DataValidationResult:
    """Result of data validation containing quality measurements and statistics."""

    def __init__(
        self,
        is_valid: bool,
        statistics: dict[str, Any],
        quality_score: float,
        issues: list[str],
        recommendations: list[str],
    ):
        self.is_valid = is_valid
        self.statistics = statistics
        self.quality_score = quality_score
        self.issues = issues
        self.recommendations = recommendations


class DataProfile:
    """Profile of a data_collection containing characteristics and analysis."""

    def __init__(
        self,
        basic_stats: dict[str, Any],
        feature_analysis: dict[str, Any],
        data_quality: dict[str, Any],
        sparsity_ratio: float,
        missing_values_ratio: float,
        complexity_score: float,
    ):
        self.basic_stats = basic_stats
        self.feature_analysis = feature_analysis
        self.data_quality = data_quality
        self.sparsity_ratio = sparsity_ratio
        self.missing_values_ratio = missing_values_ratio
        self.complexity_score = complexity_score


class FeatureEngineeringResult:
    """Result of feature engineering containing transformed data and metadata."""

    def __init__(
        self,
        engineered_data: pd.DataFrame,
        selected_features: list[str],
        engineered_features: list[str],
        feature_metadata: dict[str, Any],
    ):
        self.engineered_data = engineered_data
        self.selected_features = selected_features
        self.engineered_features = engineered_features
        self.feature_metadata = feature_metadata


class IDataValidationService(Protocol):
    """Interface for data validation service."""

    async def validate_data(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> DataValidationResult:
        """Validate input data and assess quality.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Validation result with quality measurements
        """
        ...


class IDataProfilingService(Protocol):
    """Interface for data profiling service."""

    async def profile_data(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> DataProfile:
        """Profile data_collection to understand its characteristics.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Data profile with characteristics
        """
        ...


class IFeatureEngineeringService(Protocol):
    """Interface for feature engineering service."""

    async def engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> FeatureEngineeringResult:
        """Engineer features for improved processor performance.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Feature engineering result with transformed data
        """
        ...


class IModelSelectionService(Protocol):
    """Interface for processor selection service."""

    async def select_processors(
        self, data_profile: DataProfile, max_processors: int = 5
    ) -> dict[str, Any]:
        """Select candidate models based on data characteristics.

        Args:
            data_profile: Profile of the data_collection
            max_processors: Maximum number of models to select

        Returns:
            Processor selection result with candidates and rationale
        """
        ...


class IPipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this pipeline stage."""
        ...

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute this pipeline stage.

        Args:
            context: Pipeline execution context

        Returns:
            Stage execution result
        """
        ...

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Get list of stage dependencies."""
        ...


class IPipelineOrchestrator(Protocol):
    """Interface for pipeline orchestration."""

    async def execute_pipeline(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Pipeline execution result
        """
        ...
