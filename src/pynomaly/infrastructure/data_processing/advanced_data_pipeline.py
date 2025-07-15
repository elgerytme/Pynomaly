"""Advanced data processing pipeline with comprehensive preprocessing and validation."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError


class PreprocessingStep(Enum):
    """Available preprocessing steps."""

    # Data Cleaning
    REMOVE_DUPLICATES = "remove_duplicates"
    HANDLE_MISSING = "handle_missing"
    REMOVE_OUTLIERS = "remove_outliers"

    # Data Transformation
    SCALE_FEATURES = "scale_features"
    NORMALIZE_FEATURES = "normalize_features"
    ENCODE_CATEGORICAL = "encode_categorical"
    TRANSFORM_DISTRIBUTION = "transform_distribution"

    # Feature Engineering
    CREATE_POLYNOMIAL = "create_polynomial"
    CREATE_INTERACTIONS = "create_interactions"
    EXTRACT_DATETIME = "extract_datetime"
    BINNING = "binning"

    # Feature Selection
    REMOVE_LOW_VARIANCE = "remove_low_variance"
    SELECT_K_BEST = "select_k_best"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

    # Data Validation
    VALIDATE_TYPES = "validate_types"
    VALIDATE_RANGES = "validate_ranges"
    VALIDATE_DISTRIBUTIONS = "validate_distributions"


class ImputationStrategy(Enum):
    """Missing value imputation strategies."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    KNN = "knn"
    DROP = "drop"


class ScalingMethod(Enum):
    """Feature scaling methods."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    UNIT_VECTOR = "unit_vector"


class EncodingMethod(Enum):
    """Categorical encoding methods."""

    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    FREQUENCY = "frequency"


@dataclass
class ValidationRule:
    """Data validation rule."""

    column: str
    rule_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info
    message: str | None = None


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""

    # Basic cleaning
    remove_duplicates: bool = True
    handle_missing: bool = True
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    missing_threshold: float = 0.5  # Drop columns with >50% missing

    # Scaling and normalization
    apply_scaling: bool = True
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    scale_target: bool = False

    # Categorical encoding
    encode_categoricals: bool = True
    encoding_method: EncodingMethod = EncodingMethod.ONEHOT
    max_categories: int = 10

    # Feature selection
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    apply_feature_selection: bool = False
    max_features: int | None = None

    # Data validation
    validate_data: bool = True
    validation_rules: list[ValidationRule] = field(default_factory=list)
    strict_validation: bool = False

    # Performance options
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True

    # Output options
    preserve_index: bool = True
    add_metadata: bool = True


@dataclass
class ProcessingReport:
    """Report of data processing operations."""

    original_shape: tuple[int, int]
    final_shape: tuple[int, int]
    processing_time: float
    steps_performed: list[str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        """Number of rows removed during processing."""
        return self.original_shape[0] - self.final_shape[0]

    @property
    def features_removed(self) -> int:
        """Number of features removed during processing."""
        return self.original_shape[1] - self.final_shape[1]

    @property
    def success(self) -> bool:
        """Whether processing completed successfully."""
        return len(self.errors) == 0


class AdvancedDataPipeline:
    """Advanced data processing pipeline with comprehensive preprocessing capabilities."""

    def __init__(
        self,
        config: ProcessingConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the data pipeline.

        Args:
            config: Processing configuration
            logger: Optional logger instance
        """
        self.config = config or ProcessingConfig()
        self.logger = logger or self._setup_logger()

        # Fitted transformers for reuse
        self._fitted_transformers: dict[str, Any] = {}
        self._feature_names: list[str] | None = None
        self._original_columns: list[str] | None = None

        # Processing history
        self._processing_history: list[ProcessingReport] = []

    def process_dataset(
        self,
        dataset: Dataset,
        fit_transformers: bool = True,
        return_report: bool = True,
    ) -> Dataset | tuple[Dataset, ProcessingReport]:
        """Process a complete dataset through the pipeline.

        Args:
            dataset: Input dataset
            fit_transformers: Whether to fit transformers on this data
            return_report: Whether to return processing report

        Returns:
            Processed dataset, optionally with processing report
        """
        start_time = time.perf_counter()
        original_shape = dataset.data.shape

        self.logger.info(f"Starting pipeline processing for dataset: {dataset.name}")
        self.logger.info(f"Original shape: {original_shape}")

        # Initialize report
        report = ProcessingReport(
            original_shape=original_shape,
            final_shape=original_shape,  # Will be updated
            processing_time=0.0,
            steps_performed=[],
        )

        try:
            # Create working copy
            processed_data = dataset.data.copy()
            self._original_columns = list(processed_data.columns)

            # Apply processing steps
            processed_data = self._apply_pipeline_steps(
                processed_data, dataset, fit_transformers, report
            )

            # Update feature names
            self._feature_names = list(processed_data.columns)

            # Create processed dataset
            processed_dataset = Dataset(
                name=f"{dataset.name}_processed",
                data=processed_data,
                target_column=dataset.target_column,
                metadata={
                    **dataset.metadata,
                    "processing_config": self.config.__dict__,
                    "processing_timestamp": time.time(),
                    "original_shape": original_shape,
                    "processed_shape": processed_data.shape,
                },
            )

            # Finalize report
            report.final_shape = processed_data.shape
            report.processing_time = time.perf_counter() - start_time

            self.logger.info(f"Processing completed in {report.processing_time:.2f}s")
            self.logger.info(f"Final shape: {report.final_shape}")

            self._processing_history.append(report)

            if return_report:
                return processed_dataset, report
            return processed_dataset

        except Exception as e:
            report.errors.append(str(e))
            report.processing_time = time.perf_counter() - start_time
            self.logger.error(f"Pipeline processing failed: {e}")
            raise DataValidationError(
                f"Data processing failed: {e}",
                pipeline_step="unknown",
                original_shape=original_shape,
            ) from e

    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers.

        Args:
            data: New data to transform

        Returns:
            Transformed data
        """
        if not self._fitted_transformers:
            raise ValueError(
                "No fitted transformers available. Process training data first."
            )

        self.logger.info("Transforming new data using fitted transformers")

        # Apply transformations in order
        transformed_data = data.copy()

        # Handle missing values
        if "imputer" in self._fitted_transformers:
            imputer = self._fitted_transformers["imputer"]
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                transformed_data[numeric_cols] = imputer.transform(
                    transformed_data[numeric_cols]
                )

        # Scale features
        if "scaler" in self._fitted_transformers:
            scaler = self._fitted_transformers["scaler"]
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                transformed_data[numeric_cols] = scaler.transform(
                    transformed_data[numeric_cols]
                )

        # Encode categoricals
        if "encoders" in self._fitted_transformers:
            encoders = self._fitted_transformers["encoders"]
            for col, encoder in encoders.items():
                if col in transformed_data.columns:
                    if hasattr(encoder, "transform"):
                        transformed_data[col] = encoder.transform(transformed_data[col])

        # Feature selection
        if "feature_selector" in self._fitted_transformers:
            selector = self._fitted_transformers["feature_selector"]
            if hasattr(selector, "transform"):
                selected_features = selector.transform(transformed_data)
                # Get selected feature names
                if hasattr(selector, "get_support"):
                    mask = selector.get_support()
                    selected_columns = [
                        col
                        for col, selected in zip(
                            transformed_data.columns, mask, strict=False
                        )
                        if selected
                    ]
                    transformed_data = pd.DataFrame(
                        selected_features,
                        columns=selected_columns,
                        index=transformed_data.index,
                    )

        return transformed_data

    def validate_data(
        self, data: pd.DataFrame, rules: list[ValidationRule] | None = None
    ) -> tuple[bool, list[str], list[str]]:
        """Validate data against specified rules.

        Args:
            data: Data to validate
            rules: Custom validation rules

        Returns:
            Tuple of (is_valid, warnings, errors)
        """
        validation_rules = rules or self.config.validation_rules
        warnings_list = []
        errors_list = []

        # Basic validations
        if data.empty:
            errors_list.append("Dataset is empty")
            return False, warnings_list, errors_list

        # Check for all-null columns
        null_columns = data.columns[data.isnull().all()].tolist()
        if null_columns:
            warnings_list.append(f"Columns with all null values: {null_columns}")

        # Check for high missing value percentages
        missing_pct = data.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.8].index.tolist()
        if high_missing:
            warnings_list.append(f"Columns with >80% missing values: {high_missing}")

        # Check for duplicate rows
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            warnings_list.append(f"Found {n_duplicates} duplicate rows")

        # Apply custom validation rules
        for rule in validation_rules:
            try:
                self._apply_validation_rule(data, rule, warnings_list, errors_list)
            except Exception as e:
                errors_list.append(f"Validation rule failed for {rule.column}: {e}")

        is_valid = len(errors_list) == 0
        return is_valid, warnings_list, errors_list

    def get_processing_report(self) -> ProcessingReport | None:
        """Get the latest processing report."""
        return self._processing_history[-1] if self._processing_history else None

    def get_feature_info(self) -> dict[str, Any]:
        """Get information about processed features."""
        if not self._feature_names:
            return {}

        return {
            "original_features": self._original_columns,
            "processed_features": self._feature_names,
            "n_original": len(self._original_columns) if self._original_columns else 0,
            "n_processed": len(self._feature_names),
            "fitted_transformers": list(self._fitted_transformers.keys()),
        }

    def reset_transformers(self) -> None:
        """Reset all fitted transformers."""
        self._fitted_transformers.clear()
        self._feature_names = None
        self._original_columns = None
        self.logger.info("All transformers reset")

    def _apply_pipeline_steps(
        self,
        data: pd.DataFrame,
        dataset: Dataset,
        fit_transformers: bool,
        report: ProcessingReport,
    ) -> pd.DataFrame:
        """Apply all pipeline steps to the data."""

        # 1. Data Validation
        if self.config.validate_data:
            self._step_validate_data(data, report)

        # 2. Remove Duplicates
        if self.config.remove_duplicates:
            data = self._step_remove_duplicates(data, report)

        # 3. Handle Missing Values
        if self.config.handle_missing:
            data = self._step_handle_missing(data, fit_transformers, report)

        # 4. Data Type Optimization
        data = self._step_optimize_dtypes(data, report)

        # 5. Encode Categorical Variables
        if self.config.encode_categoricals:
            data = self._step_encode_categoricals(data, fit_transformers, report)

        # 6. Feature Scaling
        if self.config.apply_scaling:
            data = self._step_scale_features(data, fit_transformers, report)

        # 7. Feature Selection
        if self.config.remove_low_variance:
            data = self._step_remove_low_variance(data, fit_transformers, report)

        if self.config.apply_feature_selection and self.config.max_features:
            data = self._step_feature_selection(data, dataset, fit_transformers, report)

        return data

    def _step_validate_data(self, data: pd.DataFrame, report: ProcessingReport) -> None:
        """Validate data step."""
        self.logger.info("Validating data...")

        is_valid, warnings_list, errors_list = self.validate_data(data)

        report.warnings.extend(warnings_list)
        report.errors.extend(errors_list)
        report.steps_performed.append("validate_data")

        if not is_valid and self.config.strict_validation:
            raise DataValidationError(
                f"Data validation failed: {errors_list}", pipeline_step="validate_data"
            )

    def _step_remove_duplicates(
        self, data: pd.DataFrame, report: ProcessingReport
    ) -> pd.DataFrame:
        """Remove duplicate rows."""
        self.logger.info("Removing duplicate rows...")

        initial_rows = len(data)
        data = data.drop_duplicates()
        removed_rows = initial_rows - len(data)

        if removed_rows > 0:
            self.logger.info(f"Removed {removed_rows} duplicate rows")
            report.warnings.append(f"Removed {removed_rows} duplicate rows")

        report.steps_performed.append("remove_duplicates")
        return data

    def _step_handle_missing(
        self, data: pd.DataFrame, fit_transformers: bool, report: ProcessingReport
    ) -> pd.DataFrame:
        """Handle missing values."""
        self.logger.info("Handling missing values...")

        # Drop columns with too many missing values
        missing_pct = data.isnull().mean()
        cols_to_drop = missing_pct[
            missing_pct > self.config.missing_threshold
        ].index.tolist()

        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            self.logger.info(
                f"Dropped columns with >{self.config.missing_threshold * 100}% missing: {cols_to_drop}"
            )
            report.warnings.append(f"Dropped high-missing columns: {cols_to_drop}")

        # Impute remaining missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if fit_transformers and len(numeric_cols) > 0:
            if self.config.imputation_strategy == ImputationStrategy.KNN:
                imputer = KNNImputer(n_neighbors=5)
            else:
                strategy = self.config.imputation_strategy.value
                if strategy in ["mean", "median", "most_frequent"]:
                    imputer = SimpleImputer(strategy=strategy)
                else:
                    imputer = SimpleImputer(strategy="median")  # fallback

            imputer.fit(data[numeric_cols])
            self._fitted_transformers["imputer"] = imputer

        if "imputer" in self._fitted_transformers and len(numeric_cols) > 0:
            data[numeric_cols] = self._fitted_transformers["imputer"].transform(
                data[numeric_cols]
            )

        # Handle categorical missing values
        for col in categorical_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna("Unknown")

        report.steps_performed.append("handle_missing")
        return data

    def _step_optimize_dtypes(
        self, data: pd.DataFrame, report: ProcessingReport
    ) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        if not self.config.memory_efficient:
            return data

        self.logger.info("Optimizing data types...")

        # Optimize numeric types
        for col in data.select_dtypes(include=["int64"]).columns:
            col_min = data[col].min()
            col_max = data[col].max()

            if col_min >= 0:
                if col_max < 255:
                    data[col] = data[col].astype(np.uint8)
                elif col_max < 65535:
                    data[col] = data[col].astype(np.uint16)
                elif col_max < 4294967295:
                    data[col] = data[col].astype(np.uint32)
            else:
                if col_min >= -128 and col_max <= 127:
                    data[col] = data[col].astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    data[col] = data[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    data[col] = data[col].astype(np.int32)

        # Optimize float types
        for col in data.select_dtypes(include=["float64"]).columns:
            data[col] = pd.to_numeric(data[col], downcast="float")

        # Convert object columns to category if beneficial
        for col in data.select_dtypes(include=["object"]).columns:
            num_unique = data[col].nunique()
            num_total = len(data[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                data[col] = data[col].astype("category")

        report.steps_performed.append("optimize_dtypes")
        return data

    def _step_encode_categoricals(
        self, data: pd.DataFrame, fit_transformers: bool, report: ProcessingReport
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) == 0:
            return data

        self.logger.info(f"Encoding {len(categorical_cols)} categorical columns...")

        if fit_transformers:
            self._fitted_transformers["encoders"] = {}

        for col in categorical_cols:
            if col == data.index.name:  # Skip index column
                continue

            # Skip if too many categories
            if data[col].nunique() > self.config.max_categories:
                self.logger.warning(
                    f"Skipping {col}: too many categories ({data[col].nunique()})"
                )
                continue

            if fit_transformers:
                if self.config.encoding_method == EncodingMethod.LABEL:
                    encoder = LabelEncoder()
                    data[col] = encoder.fit_transform(data[col].astype(str))
                    self._fitted_transformers["encoders"][col] = encoder

                elif self.config.encoding_method == EncodingMethod.ONEHOT:
                    encoder = OneHotEncoder(
                        sparse_output=False, handle_unknown="ignore"
                    )
                    encoded = encoder.fit_transform(data[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                    # Replace original column with encoded columns
                    encoded_df = pd.DataFrame(
                        encoded, columns=feature_names, index=data.index
                    )
                    data = data.drop(columns=[col])
                    data = pd.concat([data, encoded_df], axis=1)

                    self._fitted_transformers["encoders"][col] = encoder

                elif self.config.encoding_method == EncodingMethod.ORDINAL:
                    encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    data[col] = encoder.fit_transform(data[[col]]).ravel()
                    self._fitted_transformers["encoders"][col] = encoder

            else:
                # Transform using fitted encoders
                if col in self._fitted_transformers.get("encoders", {}):
                    encoder = self._fitted_transformers["encoders"][col]
                    if hasattr(encoder, "transform"):
                        data[col] = encoder.transform(data[col].astype(str))

        report.steps_performed.append("encode_categoricals")
        return data

    def _step_scale_features(
        self, data: pd.DataFrame, fit_transformers: bool, report: ProcessingReport
    ) -> pd.DataFrame:
        """Scale numerical features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return data

        self.logger.info(f"Scaling {len(numeric_cols)} numerical features...")

        if fit_transformers:
            if self.config.scaling_method == ScalingMethod.STANDARD:
                scaler = StandardScaler()
            elif self.config.scaling_method == ScalingMethod.MINMAX:
                scaler = MinMaxScaler()
            elif self.config.scaling_method == ScalingMethod.ROBUST:
                scaler = RobustScaler()
            elif self.config.scaling_method == ScalingMethod.QUANTILE:
                scaler = QuantileTransformer()
            elif self.config.scaling_method == ScalingMethod.POWER:
                scaler = PowerTransformer()
            else:
                scaler = RobustScaler()  # Default

            scaler.fit(data[numeric_cols])
            self._fitted_transformers["scaler"] = scaler

        if "scaler" in self._fitted_transformers:
            data[numeric_cols] = self._fitted_transformers["scaler"].transform(
                data[numeric_cols]
            )

        report.steps_performed.append("scale_features")
        return data

    def _step_remove_low_variance(
        self, data: pd.DataFrame, fit_transformers: bool, report: ProcessingReport
    ) -> pd.DataFrame:
        """Remove low variance features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return data

        self.logger.info("Removing low variance features...")

        if fit_transformers:
            variance_selector = VarianceThreshold(
                threshold=self.config.variance_threshold
            )
            variance_selector.fit(data[numeric_cols])
            self._fitted_transformers["variance_selector"] = variance_selector

        if "variance_selector" in self._fitted_transformers:
            selector = self._fitted_transformers["variance_selector"]
            selected_features = selector.transform(data[numeric_cols])

            # Get selected column names
            mask = selector.get_support()
            selected_columns = numeric_cols[mask]

            # Replace numeric columns with selected ones
            data = data.drop(columns=numeric_cols)
            selected_df = pd.DataFrame(
                selected_features, columns=selected_columns, index=data.index
            )
            data = pd.concat([data, selected_df], axis=1)

            removed_features = len(numeric_cols) - len(selected_columns)
            if removed_features > 0:
                self.logger.info(f"Removed {removed_features} low variance features")
                report.warnings.append(
                    f"Removed {removed_features} low variance features"
                )

        report.steps_performed.append("remove_low_variance")
        return data

    def _step_feature_selection(
        self,
        data: pd.DataFrame,
        dataset: Dataset,
        fit_transformers: bool,
        report: ProcessingReport,
    ) -> pd.DataFrame:
        """Apply feature selection."""
        if not dataset.has_target or self.config.max_features is None:
            return data

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= self.config.max_features:
            return data

        self.logger.info(f"Selecting top {self.config.max_features} features...")

        target = dataset.target
        if target is None:
            return data

        if fit_transformers:
            selector = SelectKBest(score_func=f_classif, k=self.config.max_features)
            selector.fit(data[numeric_cols], target)
            self._fitted_transformers["feature_selector"] = selector

        if "feature_selector" in self._fitted_transformers:
            selector = self._fitted_transformers["feature_selector"]
            selected_features = selector.transform(data[numeric_cols])

            # Get selected column names
            mask = selector.get_support()
            selected_columns = numeric_cols[mask]

            # Replace numeric columns with selected ones
            data = data.drop(columns=numeric_cols)
            selected_df = pd.DataFrame(
                selected_features, columns=selected_columns, index=data.index
            )
            data = pd.concat([data, selected_df], axis=1)

            removed_features = len(numeric_cols) - len(selected_columns)
            self.logger.info(
                f"Selected {len(selected_columns)} features, removed {removed_features}"
            )
            report.warnings.append(
                f"Feature selection: kept {len(selected_columns)}, removed {removed_features}"
            )

        report.steps_performed.append("feature_selection")
        return data

    def _apply_validation_rule(
        self,
        data: pd.DataFrame,
        rule: ValidationRule,
        warnings_list: list[str],
        errors_list: list[str],
    ) -> None:
        """Apply a single validation rule."""
        if rule.column not in data.columns:
            if rule.severity == "error":
                errors_list.append(f"Column '{rule.column}' not found")
            else:
                warnings_list.append(f"Column '{rule.column}' not found")
            return

        column_data = data[rule.column]
        message = rule.message or f"Validation failed for column '{rule.column}'"

        if rule.rule_type == "range":
            min_val = rule.parameters.get("min")
            max_val = rule.parameters.get("max")

            if min_val is not None and (column_data < min_val).any():
                if rule.severity == "error":
                    errors_list.append(f"{message}: values below {min_val}")
                else:
                    warnings_list.append(f"{message}: values below {min_val}")

            if max_val is not None and (column_data > max_val).any():
                if rule.severity == "error":
                    errors_list.append(f"{message}: values above {max_val}")
                else:
                    warnings_list.append(f"{message}: values above {max_val}")

        elif rule.rule_type == "type":
            expected_type = rule.parameters.get("dtype")
            if expected_type and column_data.dtype != expected_type:
                if rule.severity == "error":
                    errors_list.append(
                        f"{message}: expected {expected_type}, got {column_data.dtype}"
                    )
                else:
                    warnings_list.append(
                        f"{message}: expected {expected_type}, got {column_data.dtype}"
                    )

        elif rule.rule_type == "null_percentage":
            max_null_pct = rule.parameters.get("max_percentage", 0.5)
            null_pct = column_data.isnull().mean()

            if null_pct > max_null_pct:
                if rule.severity == "error":
                    errors_list.append(
                        f"{message}: {null_pct:.2%} null values (max allowed: {max_null_pct:.2%})"
                    )
                else:
                    warnings_list.append(
                        f"{message}: {null_pct:.2%} null values (max allowed: {max_null_pct:.2%})"
                    )

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the pipeline."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger


# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
