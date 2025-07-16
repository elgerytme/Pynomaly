#!/usr/bin/env python3
"""
Data Preprocessing Service

Handles data validation, cleaning, feature engineering, and quality assessment
for AutoML pipelines. Extracted from AutoMLPipelineOrchestrator to follow
Single Responsibility Principle.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pynomaly.application.services.automl_service import DatasetProfile
from pynomaly.infrastructure.config.feature_flags import require_feature

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report of data quality assessment"""

    dataset_id: str
    assessment_time: datetime

    # Quality metrics
    missing_values_ratio: float
    duplicate_rows_ratio: float
    outlier_ratio: float
    sparsity_ratio: float

    # Data characteristics
    n_samples: int
    n_features: int
    feature_types: dict[str, str]

    # Quality issues
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Quality score (0-1)
    overall_quality_score: float = 0.0


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering process"""

    original_features: list[str]
    engineered_features: list[str]
    selected_features: list[str]
    feature_importance: dict[str, float]

    # Transformations applied
    transformations: list[str] = field(default_factory=list)
    scaling_applied: bool = False
    encoding_applied: bool = False

    # Performance impact
    dimensionality_reduction: float = 0.0  # Percentage reduction
    processing_time_seconds: float = 0.0


@dataclass
class PreprocessingResult:
    """Complete preprocessing result"""

    dataset_id: str
    start_time: datetime
    end_time: datetime

    # Input data info
    original_shape: tuple[int, int]
    original_profile: DatasetProfile

    # Preprocessing outputs
    processed_data: pd.DataFrame
    target_data: pd.Series | None = None

    # Quality assessment
    quality_report: DataQualityReport | None = None

    # Feature engineering
    feature_engineering_result: FeatureEngineeringResult | None = None

    # Processing metadata
    processing_time_seconds: float = 0.0
    transformations_applied: list[str] = field(default_factory=list)

    # Success status
    success: bool = True
    error_message: str | None = None


class DataPreprocessingService:
    """Service for data preprocessing and quality assessment"""

    def __init__(self):
        self.scalers: dict[str, Any] = {}
        self.encoders: dict[str, Any] = {}
        self.imputers: dict[str, Any] = {}

    @require_feature("automl_feature_engineering")
    async def preprocess_dataset(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        target: pd.Series | None = None,
        enable_feature_engineering: bool = True,
        enable_quality_assessment: bool = True,
        max_features: int | None = None,
    ) -> PreprocessingResult:
        """
        Complete preprocessing pipeline for a dataset

        Args:
            dataset_id: Unique identifier for the dataset
            data: Input DataFrame
            target: Optional target variable
            enable_feature_engineering: Whether to perform feature engineering
            enable_quality_assessment: Whether to assess data quality
            max_features: Maximum number of features to retain

        Returns:
            Complete preprocessing result
        """
        start_time = datetime.now()
        logger.info(f"Starting preprocessing for dataset {dataset_id}")

        try:
            # Create dataset profile
            original_profile = self._create_dataset_profile(data)
            original_shape = data.shape

            # Initialize result
            result = PreprocessingResult(
                dataset_id=dataset_id,
                start_time=start_time,
                end_time=start_time,  # Will be updated
                original_shape=original_shape,
                original_profile=original_profile,
                processed_data=data.copy(),
                target_data=target.copy() if target is not None else None,
            )

            # Step 1: Data quality assessment
            if enable_quality_assessment:
                logger.info(f"Assessing data quality for dataset {dataset_id}")
                result.quality_report = await self._assess_data_quality(
                    dataset_id, data, target
                )

                # Check if data quality is sufficient
                if result.quality_report.overall_quality_score < 0.3:
                    result.success = False
                    result.error_message = (
                        "Data quality too low for reliable processing"
                    )
                    return result

            # Step 2: Basic data cleaning
            logger.info(f"Cleaning data for dataset {dataset_id}")
            result.processed_data = await self._clean_data(result.processed_data)
            result.transformations_applied.append("data_cleaning")

            # Step 3: Handle missing values
            logger.info(f"Handling missing values for dataset {dataset_id}")
            result.processed_data = await self._handle_missing_values(
                result.processed_data
            )
            result.transformations_applied.append("missing_value_imputation")

            # Step 4: Feature engineering (if enabled)
            if enable_feature_engineering:
                logger.info(f"Performing feature engineering for dataset {dataset_id}")
                result.feature_engineering_result = await self._engineer_features(
                    result.processed_data, target, max_features
                )

                # Apply feature engineering results
                if result.feature_engineering_result.selected_features:
                    result.processed_data = result.processed_data[
                        result.feature_engineering_result.selected_features
                    ]

                result.transformations_applied.append("feature_engineering")

            # Step 5: Final data validation
            logger.info(f"Validating processed data for dataset {dataset_id}")
            await self._validate_processed_data(result.processed_data)

            # Update timing and success status
            result.end_time = datetime.now()
            result.processing_time_seconds = (
                result.end_time - result.start_time
            ).total_seconds()
            result.success = True

            logger.info(
                f"Preprocessing completed for dataset {dataset_id} in "
                f"{result.processing_time_seconds:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Preprocessing failed for dataset {dataset_id}: {str(e)}")
            end_time = datetime.now()

            return PreprocessingResult(
                dataset_id=dataset_id,
                start_time=start_time,
                end_time=end_time,
                original_shape=data.shape,
                original_profile=self._create_dataset_profile(data),
                processed_data=data,
                success=False,
                error_message=str(e),
                processing_time_seconds=(end_time - start_time).total_seconds(),
            )

    def _create_dataset_profile(self, data: pd.DataFrame) -> DatasetProfile:
        """Create a basic dataset profile"""
        n_samples, n_features = data.shape

        # Analyze feature types
        feature_types = {}
        categorical_features = []
        numerical_features = []
        time_series_features = []

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                feature_types[col] = "numerical"
                numerical_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                feature_types[col] = "datetime"
                time_series_features.append(col)
            else:
                feature_types[col] = "categorical"
                categorical_features.append(col)

        # Calculate basic metrics
        missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)

        # Sparsity for numerical features
        if numerical_features:
            numerical_data = data[numerical_features]
            sparsity = (numerical_data == 0).sum().sum() / (
                n_samples * len(numerical_features)
            )
        else:
            sparsity = 0.0

        # Basic contamination estimate
        contamination_estimate = 0.1  # Default

        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            contamination_estimate=contamination_estimate,
            feature_types=feature_types,
            missing_values_ratio=missing_ratio,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            time_series_features=time_series_features,
            sparsity_ratio=sparsity,
            dimensionality_ratio=n_features / n_samples,
            dataset_size_mb=data.memory_usage(deep=True).sum() / (1024 * 1024),
            has_temporal_structure=len(time_series_features) > 0,
        )

    async def _assess_data_quality(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> DataQualityReport:
        """Assess data quality and identify issues"""
        n_samples, n_features = data.shape

        # Calculate quality metrics
        missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
        duplicate_ratio = data.duplicated().sum() / n_samples

        # Outlier detection using IQR method for numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_count = 0

        if len(numerical_cols) > 0:
            for col in numerical_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outlier_count += outliers

        outlier_ratio = (
            outlier_count / (n_samples * len(numerical_cols))
            if numerical_cols.size > 0
            else 0
        )

        # Sparsity calculation
        if len(numerical_cols) > 0:
            sparsity = (data[numerical_cols] == 0).sum().sum() / (
                n_samples * len(numerical_cols)
            )
        else:
            sparsity = 0.0

        # Feature type analysis
        feature_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                feature_types[col] = "numerical"
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                feature_types[col] = "datetime"
            else:
                feature_types[col] = "categorical"

        # Identify issues and recommendations
        issues = []
        warnings = []
        recommendations = []

        if missing_ratio > 0.3:
            issues.append(f"High missing values ratio: {missing_ratio:.2%}")
            recommendations.append("Consider data imputation or feature removal")
        elif missing_ratio > 0.1:
            warnings.append(f"Moderate missing values: {missing_ratio:.2%}")

        if duplicate_ratio > 0.1:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
            recommendations.append("Remove duplicate rows")

        if outlier_ratio > 0.2:
            warnings.append(f"High outlier ratio: {outlier_ratio:.2%}")
            recommendations.append("Consider outlier treatment")

        if n_features / n_samples > 0.8:
            warnings.append("High dimensionality relative to sample size")
            recommendations.append("Consider dimensionality reduction")

        # Calculate overall quality score
        quality_score = 1.0
        quality_score -= missing_ratio * 0.4  # Missing values penalty
        quality_score -= duplicate_ratio * 0.3  # Duplicate penalty
        quality_score -= min(outlier_ratio, 0.3) * 0.2  # Outlier penalty
        quality_score -= min(sparsity, 0.5) * 0.1  # Sparsity penalty
        quality_score = max(quality_score, 0.0)

        return DataQualityReport(
            dataset_id=dataset_id,
            assessment_time=datetime.now(),
            missing_values_ratio=missing_ratio,
            duplicate_rows_ratio=duplicate_ratio,
            outlier_ratio=outlier_ratio,
            sparsity_ratio=sparsity,
            n_samples=n_samples,
            n_features=n_features,
            feature_types=feature_types,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            overall_quality_score=quality_score,
        )

    async def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        cleaned_data = data.copy()

        # Remove duplicate rows
        original_shape = cleaned_data.shape
        cleaned_data = cleaned_data.drop_duplicates()

        if cleaned_data.shape[0] < original_shape[0]:
            logger.info(
                f"Removed {original_shape[0] - cleaned_data.shape[0]} duplicate rows"
            )

        # Remove columns with all missing values
        cleaned_data = cleaned_data.dropna(axis=1, how="all")

        # Remove rows with all missing values
        cleaned_data = cleaned_data.dropna(axis=0, how="all")

        return cleaned_data

    async def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        if data.isnull().sum().sum() == 0:
            return data

        processed_data = data.copy()

        # Separate numerical and categorical columns
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_data.select_dtypes(exclude=[np.number]).columns

        # Handle numerical missing values
        if len(numerical_cols) > 0:
            imputer_key = "numerical"
            if imputer_key not in self.imputers:
                self.imputers[imputer_key] = SimpleImputer(strategy="median")

            processed_data[numerical_cols] = self.imputers[imputer_key].fit_transform(
                processed_data[numerical_cols]
            )

        # Handle categorical missing values
        if len(categorical_cols) > 0:
            imputer_key = "categorical"
            if imputer_key not in self.imputers:
                self.imputers[imputer_key] = SimpleImputer(strategy="most_frequent")

            processed_data[categorical_cols] = self.imputers[imputer_key].fit_transform(
                processed_data[categorical_cols]
            )

        return processed_data

    async def _engineer_features(
        self,
        data: pd.DataFrame,
        target: pd.Series | None = None,
        max_features: int | None = None,
    ) -> FeatureEngineeringResult:
        """Perform feature engineering operations"""
        start_time = datetime.now()
        original_features = list(data.columns)
        engineered_data = data.copy()
        transformations = []

        # Feature scaling for numerical features
        numerical_cols = engineered_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            scaler_key = "standard"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()

            engineered_data[numerical_cols] = self.scalers[scaler_key].fit_transform(
                engineered_data[numerical_cols]
            )
            transformations.append("standard_scaling")

        # Feature encoding for categorical features
        categorical_cols = engineered_data.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                encoder_key = f"label_{col}"
                if encoder_key not in self.encoders:
                    self.encoders[encoder_key] = LabelEncoder()

                engineered_data[col] = self.encoders[encoder_key].fit_transform(
                    engineered_data[col].astype(str)
                )
            transformations.append("label_encoding")

        # Feature selection (if target is provided and max_features is specified)
        selected_features = original_features
        feature_importance = {}

        if (
            target is not None
            and max_features is not None
            and len(original_features) > max_features
        ):
            try:
                # Use SelectKBest for feature selection
                selector = SelectKBest(
                    f_classif, k=min(max_features, len(original_features))
                )
                selected_data = selector.fit_transform(engineered_data, target)

                # Get selected feature names
                selected_mask = selector.get_support()
                selected_features = [
                    feature
                    for feature, selected in zip(
                        original_features, selected_mask, strict=False
                    )
                    if selected
                ]

                # Get feature scores as importance
                feature_scores = selector.scores_
                feature_importance = {
                    feature: score
                    for feature, score in zip(
                        original_features, feature_scores, strict=False
                    )
                    if not np.isnan(score)
                }

                transformations.append("feature_selection")

            except Exception as e:
                logger.warning(f"Feature selection failed: {str(e)}")
                selected_features = (
                    original_features[:max_features]
                    if max_features
                    else original_features
                )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Calculate dimensionality reduction
        dimensionality_reduction = (
            (len(original_features) - len(selected_features))
            / len(original_features)
            * 100
            if len(original_features) > 0
            else 0
        )

        return FeatureEngineeringResult(
            original_features=original_features,
            engineered_features=list(engineered_data.columns),
            selected_features=selected_features,
            feature_importance=feature_importance,
            transformations=transformations,
            scaling_applied="standard_scaling" in transformations,
            encoding_applied="label_encoding" in transformations,
            dimensionality_reduction=dimensionality_reduction,
            processing_time_seconds=processing_time,
        )

    async def _validate_processed_data(self, data: pd.DataFrame) -> None:
        """Validate the processed data"""
        if data.empty:
            raise ValueError("Processed data is empty")

        if data.isnull().all().any():
            raise ValueError("Some columns have only missing values after processing")

        # Check for infinite values
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if np.isinf(data[numerical_cols]).any().any():
                raise ValueError("Data contains infinite values after processing")

        logger.info(
            f"Data validation passed: {data.shape[0]} samples, {data.shape[1]} features"
        )

    def get_preprocessing_summary(self, result: PreprocessingResult) -> dict[str, Any]:
        """Get a summary of preprocessing results"""
        summary = {
            "dataset_id": result.dataset_id,
            "success": result.success,
            "processing_time_seconds": result.processing_time_seconds,
            "original_shape": result.original_shape,
            "final_shape": result.processed_data.shape,
            "transformations_applied": result.transformations_applied,
        }

        if result.quality_report:
            summary["quality_score"] = result.quality_report.overall_quality_score
            summary["quality_issues"] = len(result.quality_report.issues)
            summary["quality_warnings"] = len(result.quality_report.warnings)

        if result.feature_engineering_result:
            summary["dimensionality_reduction"] = (
                result.feature_engineering_result.dimensionality_reduction
            )
            summary["features_selected"] = len(
                result.feature_engineering_result.selected_features
            )
            summary["feature_engineering_time"] = (
                result.feature_engineering_result.processing_time_seconds
            )

        if not result.success:
            summary["error_message"] = result.error_message

        return summary
