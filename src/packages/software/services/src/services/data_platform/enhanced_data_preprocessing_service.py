#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Service

Advanced data preprocessing service that integrates the data_transformation package
with existing Pynomaly infrastructure for anomaly detection workflows.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

# Import existing Pynomaly components
from monorepo.application.services.automl_service import DatasetProfile
from monorepo.infrastructure.config.feature_flags import require_feature
from monorepo.domain.entities import Dataset

# Import data_transformation components
try:
    from data_transformation.application.use_cases.data_pipeline import DataPipelineUseCase
    from data_transformation.domain.value_objects.pipeline_config import (
        PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
    )
    from data_transformation.domain.services.data_cleaning_service import DataCleaningService
    from data_transformation.application.dto.pipeline_result import PipelineResult
    DATA_TRANSFORMATION_AVAILABLE = True
except ImportError:
    DATA_TRANSFORMATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDataQualityReport:
    """Enhanced report of data quality assessment with detailed insights."""

    dataset_id: str
    assessment_time: datetime

    # Core quality metrics
    missing_values_ratio: float
    duplicate_rows_ratio: float
    outlier_ratio: float
    sparsity_ratio: float

    # Data characteristics
    n_samples: int
    n_features: int
    feature_types: dict[str, str]

    # Quality issues and recommendations
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Enhanced metrics
    data_drift_score: float = 0.0
    feature_importance_scores: dict[str, float] = field(default_factory=dict)
    transformation_suggestions: list[str] = field(default_factory=list)
    preprocessing_pipeline: Optional[dict] = None

    # Quality score (0-1)
    quality_score: float = 0.0

    def get_overall_assessment(self) -> str:
        """Get overall data quality assessment."""
        if self.quality_score >= 0.9:
            return "Excellent"
        elif self.quality_score >= 0.7:
            return "Good"
        elif self.quality_score >= 0.5:
            return "Fair"
        else:
            return "Poor"


class EnhancedDataPreprocessingService:
    """Enhanced data preprocessing service with advanced transformation capabilities."""

    def __init__(self, enable_advanced_features: bool = True):
        """Initialize the enhanced preprocessing service.
        
        Args:
            enable_advanced_features: Whether to enable advanced transformation features
        """
        self.enable_advanced_features = enable_advanced_features and DATA_TRANSFORMATION_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if self.enable_advanced_features:
            self.cleaning_service = DataCleaningService()
            self.logger.info("Advanced data transformation features enabled")
        else:
            self.logger.warning("Advanced features disabled or data_transformation unavailable")

    async def preprocess_for_anomaly_detection(
        self,
        dataset: Union[pd.DataFrame, Dataset, str, Path],
        config: Optional[PipelineConfig] = None,
        target_column: Optional[str] = None
    ) -> tuple[pd.DataFrame, EnhancedDataQualityReport]:
        """Preprocess data specifically for anomaly detection workflows.
        
        Args:
            dataset: Input dataset (DataFrame, Dataset entity, or file path)
            config: Optional preprocessing configuration
            target_column: Optional target column for supervised preprocessing
            
        Returns:
            Tuple of (preprocessed_data, quality_report)
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(dataset, (str, Path)):
                df = pd.read_csv(dataset)
                dataset_id = str(Path(dataset).stem)
            elif isinstance(dataset, Dataset):
                df = dataset.data  # Assuming Dataset has a data attribute
                dataset_id = str(dataset.id)
            else:
                df = dataset
                dataset_id = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Generate quality report
            quality_report = await self._assess_data_quality(df, dataset_id)

            if not self.enable_advanced_features:
                # Fallback to basic preprocessing
                processed_df = self._basic_preprocessing(df)
                return processed_df, quality_report

            # Create optimal configuration if not provided
            if config is None:
                config = self._create_anomaly_detection_config(df, quality_report)

            # Apply advanced preprocessing pipeline
            pipeline = DataPipelineUseCase(config)
            result = pipeline.execute(df, target=target_column)

            if result.success:
                # Update quality report with transformation details
                quality_report.preprocessing_pipeline = {
                    "steps_executed": [step.step_type for step in result.steps_executed],
                    "execution_time": result.execution_time,
                    "transformations_applied": len(result.steps_executed)
                }
                
                self.logger.info(
                    f"Successfully preprocessed dataset {dataset_id}: "
                    f"{len(result.data)} rows, {len(result.data.columns)} columns"
                )
                
                return result.data, quality_report
            else:
                self.logger.warning(
                    f"Advanced preprocessing failed for {dataset_id}: {result.error_message}"
                )
                processed_df = self._basic_preprocessing(df)
                return processed_df, quality_report

        except Exception as e:
            self.logger.error(f"Error in preprocess_for_anomaly_detection: {e}")
            # Fallback to basic preprocessing
            processed_df = self._basic_preprocessing(df if 'df' in locals() else dataset)
            quality_report = await self._assess_data_quality(processed_df, dataset_id)
            return processed_df, quality_report

    async def get_preprocessing_recommendations(
        self,
        dataset: Union[pd.DataFrame, str, Path],
        anomaly_detection_type: str = "unsupervised"
    ) -> dict[str, Any]:
        """Get intelligent preprocessing recommendations for anomaly detection.
        
        Args:
            dataset: Input dataset
            anomaly_detection_type: Type of anomaly detection (supervised/unsupervised)
            
        Returns:
            Dictionary containing preprocessing recommendations
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(dataset, (str, Path)):
                df = pd.read_csv(dataset)
            else:
                df = dataset

            recommendations = {
                "basic_stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                    "missing_values": df.isnull().sum().sum()
                },
                "data_quality_issues": [],
                "preprocessing_steps": [],
                "optimization_suggestions": []
            }

            if self.enable_advanced_features:
                # Use advanced analysis
                quality_report = await self._assess_data_quality(df, "temp_analysis")
                
                recommendations["data_quality_issues"] = quality_report.issues
                recommendations["preprocessing_steps"] = quality_report.transformation_suggestions
                recommendations["quality_score"] = quality_report.quality_score
                
                # Add anomaly detection specific recommendations
                if anomaly_detection_type == "unsupervised":
                    recommendations["optimization_suggestions"].extend([
                        "Consider feature scaling for distance-based algorithms",
                        "Remove or transform highly correlated features",
                        "Handle categorical variables appropriately"
                    ])
                else:
                    recommendations["optimization_suggestions"].extend([
                        "Consider feature selection based on target correlation",
                        "Balance the dataset if anomalies are rare",
                        "Use appropriate encoding for categorical features"
                    ])

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {"error": f"Failed to generate recommendations: {e}"}

    def _create_anomaly_detection_config(
        self,
        df: pd.DataFrame,
        quality_report: EnhancedDataQualityReport
    ) -> PipelineConfig:
        """Create optimal configuration for anomaly detection workflows."""
        # Choose strategies based on data characteristics and quality
        if quality_report.missing_values_ratio > 0.1:
            cleaning_strategy = CleaningStrategy.STATISTICAL
        elif quality_report.quality_score < 0.7:
            cleaning_strategy = CleaningStrategy.AUTO
        else:
            cleaning_strategy = CleaningStrategy.CONSERVATIVE

        # For anomaly detection, robust scaling is often preferred
        scaling_method = ScalingMethod.ROBUST

        # Choose encoding based on categorical ratio
        categorical_ratio = len(df.select_dtypes(include=['object']).columns) / len(df.columns)
        if categorical_ratio > 0.3:
            encoding_strategy = EncodingStrategy.ONEHOT
        else:
            encoding_strategy = EncodingStrategy.LABEL

        return PipelineConfig(
            source_type=SourceType.CSV,  # Default, will be overridden by pipeline
            cleaning_strategy=cleaning_strategy,
            scaling_method=scaling_method,
            encoding_strategy=encoding_strategy,
            feature_engineering=True,  # Enable for better anomaly detection
            validation_enabled=True,
            parallel_processing=True,
            memory_efficient=len(df) > 10000  # Enable for large datasets
        )

    async def _assess_data_quality(
        self,
        df: pd.DataFrame,
        dataset_id: str
    ) -> EnhancedDataQualityReport:
        """Assess data quality with enhanced metrics."""
        try:
            # Basic quality metrics
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            duplicate_ratio = df.duplicated().sum() / len(df)
            
            # Outlier detection for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            outlier_count = 0
            total_numeric_values = 0
            
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                total_numeric_values += len(df[col].dropna())
            
            outlier_ratio = outlier_count / total_numeric_values if total_numeric_values > 0 else 0
            
            # Sparsity calculation
            sparsity_ratio = (df == 0).sum().sum() / (len(df) * len(df.columns))

            # Feature types
            feature_types = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    feature_types[col] = 'numeric'
                elif df[col].dtype == 'object':
                    feature_types[col] = 'categorical'
                elif df[col].dtype == 'datetime64[ns]':
                    feature_types[col] = 'datetime'
                else:
                    feature_types[col] = 'other'

            # Generate issues and recommendations
            issues = []
            warnings = []
            recommendations = []
            
            if missing_ratio > 0.1:
                issues.append(f"High missing value ratio: {missing_ratio:.2%}")
                recommendations.append("Apply advanced missing value imputation")
            
            if duplicate_ratio > 0.05:
                issues.append(f"Duplicate rows detected: {duplicate_ratio:.2%}")
                recommendations.append("Remove duplicate records")
            
            if outlier_ratio > 0.1:
                warnings.append(f"High outlier ratio: {outlier_ratio:.2%}")
                recommendations.append("Consider outlier treatment")

            # Calculate quality score
            quality_score = max(0, 1 - (missing_ratio + duplicate_ratio + outlier_ratio))

            # Enhanced metrics if advanced features are available
            transformation_suggestions = []
            if self.enable_advanced_features:
                # Use data_transformation service for detailed analysis
                cleaning_report = self.cleaning_service.validate_data_quality(df)
                quality_score = cleaning_report.get("overall_score", quality_score) / 100
                
                transformation_suggestions.extend([
                    "Apply intelligent feature engineering",
                    "Use advanced outlier detection methods",
                    "Consider automated feature selection"
                ])

            return EnhancedDataQualityReport(
                dataset_id=dataset_id,
                assessment_time=datetime.now(),
                missing_values_ratio=missing_ratio,
                duplicate_rows_ratio=duplicate_ratio,
                outlier_ratio=outlier_ratio,
                sparsity_ratio=sparsity_ratio,
                n_samples=len(df),
                n_features=len(df.columns),
                feature_types=feature_types,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                transformation_suggestions=transformation_suggestions,
                quality_score=quality_score
            )

        except Exception as e:
            self.logger.error(f"Error in data quality assessment: {e}")
            # Return basic report on error
            return EnhancedDataQualityReport(
                dataset_id=dataset_id,
                assessment_time=datetime.now(),
                missing_values_ratio=0.0,
                duplicate_rows_ratio=0.0,
                outlier_ratio=0.0,
                sparsity_ratio=0.0,
                n_samples=len(df) if 'df' in locals() else 0,
                n_features=len(df.columns) if 'df' in locals() else 0,
                feature_types={},
                issues=[f"Error in assessment: {e}"],
                quality_score=0.0
            )

    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing fallback when advanced features are unavailable."""
        try:
            processed_df = df.copy()
            
            # Handle missing values
            numeric_cols = processed_df.select_dtypes(include=['number']).columns
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_cols:
                if processed_df[col].isnull().any():
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_cols:
                if processed_df[col].isnull().any():
                    mode_value = processed_df[col].mode()
                    if len(mode_value) > 0:
                        processed_df[col].fillna(mode_value.iloc[0], inplace=True)
            
            # Remove duplicates
            processed_df.drop_duplicates(inplace=True)
            
            self.logger.info("Applied basic preprocessing (fallback mode)")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in basic preprocessing: {e}")
            return df

    @require_feature("enhanced_preprocessing")
    async def optimize_for_algorithm(
        self,
        dataset: pd.DataFrame,
        algorithm_type: str,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Optimize preprocessing for specific anomaly detection algorithms.
        
        Args:
            dataset: Input dataset
            algorithm_type: Type of algorithm (isolation_forest, one_class_svm, etc.)
            target_column: Optional target column for supervised algorithms
            
        Returns:
            Optimized dataset for the specific algorithm
        """
        if not self.enable_advanced_features:
            return self._basic_preprocessing(dataset)
        
        # Algorithm-specific optimization configurations
        algorithm_configs = {
            "isolation_forest": PipelineConfig(
                source_type=SourceType.CSV,
                cleaning_strategy=CleaningStrategy.CONSERVATIVE,
                scaling_method=ScalingMethod.STANDARD,
                encoding_strategy=EncodingStrategy.ONEHOT,
                feature_engineering=False  # IF works well with raw features
            ),
            "one_class_svm": PipelineConfig(
                source_type=SourceType.CSV,
                cleaning_strategy=CleaningStrategy.AUTO,
                scaling_method=ScalingMethod.STANDARD,  # SVM requires scaling
                encoding_strategy=EncodingStrategy.LABEL,
                feature_engineering=True
            ),
            "local_outlier_factor": PipelineConfig(
                source_type=SourceType.CSV,
                cleaning_strategy=CleaningStrategy.STATISTICAL,
                scaling_method=ScalingMethod.ROBUST,  # Robust to outliers
                encoding_strategy=EncodingStrategy.ONEHOT,
                feature_engineering=True
            )
        }
        
        config = algorithm_configs.get(
            algorithm_type,
            self._create_anomaly_detection_config(dataset, await self._assess_data_quality(dataset, "temp"))
        )
        
        pipeline = DataPipelineUseCase(config)
        result = pipeline.execute(dataset, target=target_column)
        
        return result.data if result.success else self._basic_preprocessing(dataset)