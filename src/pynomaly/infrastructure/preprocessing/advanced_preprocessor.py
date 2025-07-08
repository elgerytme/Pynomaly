"""Advanced data preprocessing for anomaly detection."""

import asyncio
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import (
        RFE,
        RFECV,
        SelectKBest,
        SelectPercentile,
        VarianceThreshold,
        chi2,
        f_regression,
        mutual_info_regression,
    )
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        LabelEncoder,
        MaxAbsScaler,
        MinMaxScaler,
        OneHotEncoder,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import category_encoders as ce

    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

from ...domain.entities.dataset import Dataset
from ...infrastructure.config.settings import Settings

# Optional monitoring import
try:
    from ..monitoring.distributed_tracing import trace_operation

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def trace_operation(name):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


class PreprocessingStep(Enum):
    """Available preprocessing steps."""

    # Data cleaning
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_DETECTION = "outlier_detection"
    DUPLICATE_REMOVAL = "duplicate_removal"
    DATA_TYPE_CONVERSION = "data_type_conversion"

    # Feature engineering
    FEATURE_SCALING = "feature_scaling"
    FEATURE_ENCODING = "feature_encoding"
    FEATURE_TRANSFORMATION = "feature_transformation"
    FEATURE_CREATION = "feature_creation"

    # Feature selection
    VARIANCE_FILTERING = "variance_filtering"
    CORRELATION_FILTERING = "correlation_filtering"
    STATISTICAL_SELECTION = "statistical_selection"
    MODEL_BASED_SELECTION = "model_based_selection"

    # Dimensionality reduction
    PCA_REDUCTION = "pca_reduction"
    ICA_REDUCTION = "ica_reduction"
    SVD_REDUCTION = "svd_reduction"

    # Advanced transformations
    POWER_TRANSFORMATION = "power_transformation"
    QUANTILE_TRANSFORMATION = "quantile_transformation"
    CUSTOM_TRANSFORMATION = "custom_transformation"


class ImputationMethod(Enum):
    """Imputation methods for missing values."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"


class ScalingMethod(Enum):
    """Scaling methods for numerical features."""

    STANDARD = "standard"
    ROBUST = "robust"
    MIN_MAX = "min_max"
    MAX_ABS = "max_abs"
    UNIT_VECTOR = "unit_vector"


class EncodingMethod(Enum):
    """Encoding methods for categorical features."""

    ONE_HOT = "one_hot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    HASH = "hash"
    LEAVE_ONE_OUT = "leave_one_out"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    # Steps to apply (in order)
    steps: List[PreprocessingStep] = field(default_factory=list)

    # Missing value handling
    imputation_method: ImputationMethod = ImputationMethod.MEDIAN
    imputation_constant: Optional[Any] = None

    # Outlier detection and handling
    outlier_detection_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    outlier_action: str = "cap"  # "remove", "cap", "transform"

    # Feature scaling
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scaling_features: Optional[List[str]] = None

    # Feature encoding
    encoding_method: EncodingMethod = EncodingMethod.ONE_HOT
    encoding_features: Optional[List[str]] = None
    max_categories: int = 20

    # Feature selection
    feature_selection_method: str = (
        "variance"  # "variance", "correlation", "mutual_info", "rfe"
    )
    feature_selection_threshold: float = 0.01
    max_features: Optional[int] = None

    # Dimensionality reduction
    reduction_method: str = "pca"  # "pca", "ica", "svd"
    n_components: Optional[int] = None
    explained_variance_threshold: float = 0.95

    # Transformation
    power_transform_method: str = "yeo-johnson"  # "box-cox", "yeo-johnson"
    quantile_transform_output: str = "uniform"  # "uniform", "normal"

    # Data validation
    enable_validation: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    # Performance optimization
    chunk_size: Optional[int] = None
    parallel_processing: bool = True
    memory_efficient: bool = True

    # Output configuration
    preserve_original_data: bool = False
    add_preprocessing_metadata: bool = True


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations."""

    original_dataset: Dataset
    processed_dataset: Dataset
    preprocessing_metadata: Dict[str, Any]

    # Transformations applied
    transformations: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics and metrics
    original_shape: Tuple[int, int] = (0, 0)
    processed_shape: Tuple[int, int] = (0, 0)
    features_removed: List[str] = field(default_factory=list)
    features_added: List[str] = field(default_factory=list)
    missing_values_handled: int = 0
    outliers_detected: int = 0

    # Quality metrics
    data_quality_score: float = 0.0
    preprocessing_time: float = 0.0

    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of transformations applied."""
        return {
            "total_transformations": len(self.transformations),
            "original_shape": self.original_shape,
            "processed_shape": self.processed_shape,
            "features_removed": len(self.features_removed),
            "features_added": len(self.features_added),
            "missing_values_handled": self.missing_values_handled,
            "outliers_detected": self.outliers_detected,
            "data_quality_score": self.data_quality_score,
            "preprocessing_time": self.preprocessing_time,
        }


class AdvancedPreprocessor:
    """Advanced data preprocessing for anomaly detection."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize advanced preprocessor."""
        self.config = config or PreprocessingConfig()

        # Fitted transformers (for consistency across datasets)
        self.fitted_transformers: Dict[str, Any] = {}

        # Feature information
        self.feature_info: Dict[str, Any] = {}

        # Preprocessing pipeline
        self.pipeline: Optional[Pipeline] = None

        # Statistics
        self.preprocessing_history: List[Dict[str, Any]] = []

        logger.info("Advanced preprocessor initialized")

    @trace_operation("advanced_preprocessing")
    async def preprocess_dataset(
        self, dataset: Dataset, config: Optional[PreprocessingConfig] = None
    ) -> PreprocessingResult:
        """Preprocess a dataset with advanced techniques."""

        preprocessing_config = config or self.config
        start_time = datetime.now()

        # Initialize result
        result = PreprocessingResult(
            original_dataset=dataset,
            processed_dataset=dataset,
            preprocessing_metadata={},
            original_shape=dataset.data.shape if hasattr(dataset, "data") else (0, 0),
        )

        try:
            # Get data
            if not hasattr(dataset, "data") or dataset.data is None:
                raise ValueError("Dataset has no data to preprocess")

            data = dataset.data.copy()
            result.original_shape = data.shape

            # Apply preprocessing steps
            for step in preprocessing_config.steps:
                logger.debug(f"Applying preprocessing step: {step.value}")

                if step == PreprocessingStep.MISSING_VALUE_IMPUTATION:
                    data, step_metadata = await self._handle_missing_values(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.OUTLIER_DETECTION:
                    data, step_metadata = await self._handle_outliers(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.DUPLICATE_REMOVAL:
                    data, step_metadata = await self._remove_duplicates(data)

                elif step == PreprocessingStep.DATA_TYPE_CONVERSION:
                    data, step_metadata = await self._convert_data_types(data)

                elif step == PreprocessingStep.FEATURE_SCALING:
                    data, step_metadata = await self._scale_features(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.FEATURE_ENCODING:
                    data, step_metadata = await self._encode_features(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.FEATURE_TRANSFORMATION:
                    data, step_metadata = await self._transform_features(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.FEATURE_CREATION:
                    data, step_metadata = await self._create_features(data)

                elif step == PreprocessingStep.VARIANCE_FILTERING:
                    data, step_metadata = await self._filter_by_variance(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.CORRELATION_FILTERING:
                    data, step_metadata = await self._filter_by_correlation(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.STATISTICAL_SELECTION:
                    data, step_metadata = await self._select_features_statistical(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.MODEL_BASED_SELECTION:
                    data, step_metadata = await self._select_features_model_based(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.PCA_REDUCTION:
                    data, step_metadata = await self._apply_pca(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.ICA_REDUCTION:
                    data, step_metadata = await self._apply_ica(
                        data, preprocessing_config
                    )

                elif step == PreprocessingStep.SVD_REDUCTION:
                    data, step_metadata = await self._apply_svd(
                        data, preprocessing_config
                    )

                else:
                    logger.warning(f"Unknown preprocessing step: {step}")
                    continue

                # Record transformation
                transformation_record = {
                    "step": step.value,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": step_metadata,
                    "shape_before": (
                        result.processed_shape
                        if result.processed_shape != (0, 0)
                        else result.original_shape
                    ),
                    "shape_after": data.shape,
                }
                result.transformations.append(transformation_record)

                # Update result metrics
                if "missing_values_handled" in step_metadata:
                    result.missing_values_handled += step_metadata[
                        "missing_values_handled"
                    ]
                if "outliers_detected" in step_metadata:
                    result.outliers_detected += step_metadata["outliers_detected"]
                if "features_removed" in step_metadata:
                    result.features_removed.extend(step_metadata["features_removed"])
                if "features_added" in step_metadata:
                    result.features_added.extend(step_metadata["features_added"])

            # Final data validation
            if preprocessing_config.enable_validation:
                validation_results = await self._validate_processed_data(
                    data, preprocessing_config
                )
                result.preprocessing_metadata["validation"] = validation_results

            # Calculate data quality score
            result.data_quality_score = await self._calculate_data_quality_score(data)

            # Create processed dataset
            processed_dataset = Dataset(
                id=f"{dataset.id}_processed",
                name=f"{dataset.name}_processed",
                description=f"Preprocessed version of {dataset.name}",
                data=data,
                metadata={
                    **dataset.metadata,
                    "preprocessing_applied": True,
                    "preprocessing_config": preprocessing_config.__dict__,
                    "transformations_applied": [
                        t["step"] for t in result.transformations
                    ],
                    "original_shape": result.original_shape,
                    "processed_shape": data.shape,
                    "data_quality_score": result.data_quality_score,
                },
            )

            result.processed_dataset = processed_dataset
            result.processed_shape = data.shape
            result.preprocessing_time = (datetime.now() - start_time).total_seconds()

            # Store preprocessing history
            self.preprocessing_history.append(
                {
                    "dataset_id": dataset.id,
                    "timestamp": start_time.isoformat(),
                    "config": preprocessing_config.__dict__,
                    "summary": result.get_transformation_summary(),
                }
            )

            logger.info(
                f"Preprocessing completed: {result.original_shape} -> {result.processed_shape}"
            )
            return result

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    async def _handle_missing_values(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in the dataset."""

        missing_before = data.isnull().sum().sum()
        if missing_before == 0:
            return data, {"missing_values_handled": 0, "method": "none_needed"}

        processed_data = data.copy()

        if config.imputation_method == ImputationMethod.MEAN:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                processed_data[numeric_cols].mean()
            )

        elif config.imputation_method == ImputationMethod.MEDIAN:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                processed_data[numeric_cols].median()
            )

        elif config.imputation_method == ImputationMethod.MODE:
            for col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(
                    processed_data[col].mode().iloc[0]
                    if not processed_data[col].mode().empty
                    else 0
                )

        elif config.imputation_method == ImputationMethod.CONSTANT:
            processed_data = processed_data.fillna(config.imputation_constant or 0)

        elif config.imputation_method == ImputationMethod.FORWARD_FILL:
            processed_data = processed_data.ffill()

        elif config.imputation_method == ImputationMethod.BACKWARD_FILL:
            processed_data = processed_data.bfill()

        elif config.imputation_method == ImputationMethod.INTERPOLATE:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].interpolate()

        elif config.imputation_method == ImputationMethod.KNN and SKLEARN_AVAILABLE:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                processed_data[numeric_cols] = imputer.fit_transform(
                    processed_data[numeric_cols]
                )

        elif (
            config.imputation_method == ImputationMethod.ITERATIVE and SKLEARN_AVAILABLE
        ):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = IterativeImputer(random_state=42)
                processed_data[numeric_cols] = imputer.fit_transform(
                    processed_data[numeric_cols]
                )

        # Handle remaining missing values in categorical columns
        categorical_cols = processed_data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if processed_data[col].isnull().any():
                processed_data[col] = processed_data[col].fillna("MISSING")

        missing_after = processed_data.isnull().sum().sum()
        missing_handled = missing_before - missing_after

        return processed_data, {
            "missing_values_handled": int(missing_handled),
            "method": config.imputation_method.value,
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
        }

    async def _handle_outliers(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers in numerical columns."""

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data, {"outliers_detected": 0, "method": "no_numeric_columns"}

        processed_data = data.copy()
        total_outliers = 0
        outlier_details = {}

        for col in numeric_cols:
            col_data = processed_data[col]

            if config.outlier_detection_method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)

            elif config.outlier_detection_method == "zscore":
                z_scores = np.abs(stats.zscore(col_data.dropna()))
                outlier_mask = pd.Series(
                    z_scores > config.outlier_threshold, index=col_data.dropna().index
                )
                outlier_mask = outlier_mask.reindex(col_data.index, fill_value=False)

            elif (
                config.outlier_detection_method == "isolation_forest"
                and SKLEARN_AVAILABLE
            ):
                from sklearn.ensemble import IsolationForest

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(
                    col_data.dropna().values.reshape(-1, 1)
                )
                outlier_mask = pd.Series(
                    outlier_pred == -1, index=col_data.dropna().index
                )
                outlier_mask = outlier_mask.reindex(col_data.index, fill_value=False)

            else:
                continue

            outliers_count = outlier_mask.sum()
            total_outliers += outliers_count
            outlier_details[col] = int(outliers_count)

            if outliers_count > 0:
                if config.outlier_action == "remove":
                    processed_data = processed_data[~outlier_mask]
                elif config.outlier_action == "cap":
                    if config.outlier_detection_method == "iqr":
                        processed_data.loc[
                            outlier_mask & (col_data < lower_bound), col
                        ] = lower_bound
                        processed_data.loc[
                            outlier_mask & (col_data > upper_bound), col
                        ] = upper_bound
                    elif config.outlier_detection_method == "zscore":
                        median_val = col_data.median()
                        processed_data.loc[outlier_mask, col] = median_val
                elif config.outlier_action == "transform":
                    # Apply log transformation to reduce outlier impact
                    if (col_data > 0).all():
                        processed_data[col] = np.log1p(col_data)

        return processed_data, {
            "outliers_detected": int(total_outliers),
            "method": config.outlier_detection_method,
            "action": config.outlier_action,
            "outliers_by_column": outlier_details,
        }

    async def _remove_duplicates(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows from the dataset."""

        duplicates_before = data.duplicated().sum()
        if duplicates_before == 0:
            return data, {"duplicates_removed": 0}

        processed_data = data.drop_duplicates()
        duplicates_removed = duplicates_before

        return processed_data, {
            "duplicates_removed": int(duplicates_removed),
            "rows_before": len(data),
            "rows_after": len(processed_data),
        }

    async def _convert_data_types(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize data types for memory efficiency."""

        processed_data = data.copy()
        conversions = {}

        # Convert integer columns to appropriate size
        for col in processed_data.select_dtypes(include=["int64"]).columns:
            col_min = processed_data[col].min()
            col_max = processed_data[col].max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                processed_data[col] = processed_data[col].astype(np.int8)
                conversions[col] = "int8"
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                processed_data[col] = processed_data[col].astype(np.int16)
                conversions[col] = "int16"
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                processed_data[col] = processed_data[col].astype(np.int32)
                conversions[col] = "int32"

        # Convert float columns to appropriate size
        for col in processed_data.select_dtypes(include=["float64"]).columns:
            col_min = processed_data[col].min()
            col_max = processed_data[col].max()

            if (
                np.finfo(np.float32).min
                <= col_min
                <= col_max
                <= np.finfo(np.float32).max
            ):
                processed_data[col] = processed_data[col].astype(np.float32)
                conversions[col] = "float32"

        # Convert object columns to category where appropriate
        for col in processed_data.select_dtypes(include=["object"]).columns:
            num_unique = processed_data[col].nunique()
            num_total = len(processed_data[col])

            # Convert to category if less than 50% unique values
            if num_unique / num_total < 0.5:
                processed_data[col] = processed_data[col].astype("category")
                conversions[col] = "category"

        return processed_data, {
            "conversions_applied": conversions,
            "memory_reduction": "optimized",
        }

    async def _scale_features(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data, {"features_scaled": 0}

        # Select features to scale
        features_to_scale = config.scaling_features or numeric_cols
        features_to_scale = [col for col in features_to_scale if col in numeric_cols]

        if not features_to_scale:
            return data, {"features_scaled": 0}

        processed_data = data.copy()

        # Choose scaler
        if config.scaling_method == ScalingMethod.STANDARD:
            scaler = StandardScaler()
        elif config.scaling_method == ScalingMethod.ROBUST:
            scaler = RobustScaler()
        elif config.scaling_method == ScalingMethod.MIN_MAX:
            scaler = MinMaxScaler()
        elif config.scaling_method == ScalingMethod.MAX_ABS:
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()

        # Fit and transform
        processed_data[features_to_scale] = scaler.fit_transform(
            processed_data[features_to_scale]
        )

        # Store scaler for later use
        self.fitted_transformers[f"scaler_{config.scaling_method.value}"] = scaler

        return processed_data, {
            "features_scaled": len(features_to_scale),
            "scaling_method": config.scaling_method.value,
            "scaled_features": features_to_scale,
        }

    async def _encode_features(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features."""

        categorical_cols = data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) == 0:
            return data, {"features_encoded": 0}

        # Select features to encode
        features_to_encode = config.encoding_features or categorical_cols
        features_to_encode = [
            col for col in features_to_encode if col in categorical_cols
        ]

        if not features_to_encode:
            return data, {"features_encoded": 0}

        processed_data = data.copy()
        encoded_features = []
        new_features = []

        for col in features_to_encode:
            # Check cardinality
            num_unique = processed_data[col].nunique()
            if num_unique > config.max_categories:
                logger.warning(
                    f"Column {col} has {num_unique} categories, skipping encoding"
                )
                continue

            if config.encoding_method == EncodingMethod.ONE_HOT and SKLEARN_AVAILABLE:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(processed_data[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                # Add encoded features
                for i, feature_name in enumerate(feature_names):
                    processed_data[feature_name] = encoded[:, i]
                    new_features.append(feature_name)

                # Remove original column
                processed_data = processed_data.drop(columns=[col])
                encoded_features.append(col)

            elif config.encoding_method == EncodingMethod.LABEL and SKLEARN_AVAILABLE:
                # Label encoding
                encoder = LabelEncoder()
                processed_data[col] = encoder.fit_transform(
                    processed_data[col].astype(str)
                )
                encoded_features.append(col)

            elif (
                config.encoding_method == EncodingMethod.TARGET
                and CATEGORY_ENCODERS_AVAILABLE
            ):
                # Target encoding (requires target variable - skip for now)
                logger.warning(
                    f"Target encoding requires target variable, skipping {col}"
                )
                continue

            else:
                # Default to label encoding
                processed_data[col] = pd.Categorical(processed_data[col]).codes
                encoded_features.append(col)

        return processed_data, {
            "features_encoded": len(encoded_features),
            "encoding_method": config.encoding_method.value,
            "encoded_features": encoded_features,
            "features_added": new_features,
        }

    async def _transform_features(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply advanced transformations to features."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data, {"features_transformed": 0}

        processed_data = data.copy()
        transformed_features = []

        # Power transformation
        if config.power_transform_method:
            try:
                power_transformer = PowerTransformer(
                    method=config.power_transform_method
                )
                processed_data[numeric_cols] = power_transformer.fit_transform(
                    processed_data[numeric_cols]
                )
                transformed_features.extend(numeric_cols)
                self.fitted_transformers["power_transformer"] = power_transformer
            except Exception as e:
                logger.warning(f"Power transformation failed: {e}")

        return processed_data, {
            "features_transformed": len(transformed_features),
            "transformation_method": config.power_transform_method,
            "transformed_features": list(transformed_features),
        }

    async def _create_features(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create new features through feature engineering."""

        processed_data = data.copy()
        created_features = []

        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            # Create polynomial features for first two numeric columns
            col1, col2 = numeric_cols[0], numeric_cols[1]

            # Interaction feature
            interaction_name = f"{col1}_{col2}_interaction"
            processed_data[interaction_name] = (
                processed_data[col1] * processed_data[col2]
            )
            created_features.append(interaction_name)

            # Ratio feature (if no zeros)
            if not (processed_data[col2] == 0).any():
                ratio_name = f"{col1}_{col2}_ratio"
                processed_data[ratio_name] = processed_data[col1] / processed_data[col2]
                created_features.append(ratio_name)

        # Create statistical features
        if len(numeric_cols) >= 3:
            # Row-wise statistics
            processed_data["row_mean"] = processed_data[numeric_cols].mean(axis=1)
            processed_data["row_std"] = processed_data[numeric_cols].std(axis=1)
            processed_data["row_median"] = processed_data[numeric_cols].median(axis=1)
            created_features.extend(["row_mean", "row_std", "row_median"])

        return processed_data, {
            "features_created": len(created_features),
            "created_features": created_features,
        }

    async def _filter_by_variance(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove features with low variance."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data, {"features_removed": 0}

        # Apply variance threshold
        selector = VarianceThreshold(threshold=config.feature_selection_threshold)
        selected_data = selector.fit_transform(data[numeric_cols])

        # Get selected feature names
        selected_features = numeric_cols[selector.get_support()]
        removed_features = numeric_cols[~selector.get_support()]

        # Create processed data
        processed_data = data.copy()
        processed_data = processed_data.drop(columns=removed_features)

        return processed_data, {
            "features_removed": len(removed_features),
            "removed_features": list(removed_features),
            "selection_method": "variance_threshold",
            "threshold": config.feature_selection_threshold,
        }

    async def _filter_by_correlation(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove highly correlated features."""

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return data, {"features_removed": 0}

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr().abs()

        # Find highly correlated feature pairs
        threshold = 0.95  # High correlation threshold
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to remove
        to_remove = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        processed_data = data.drop(columns=to_remove)

        return processed_data, {
            "features_removed": len(to_remove),
            "removed_features": to_remove,
            "selection_method": "correlation_filtering",
            "threshold": threshold,
        }

    async def _select_features_statistical(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select features using statistical methods."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return data, {"features_removed": 0}

        # For unsupervised feature selection, use variance as proxy
        # In real scenarios, this would use target variable

        # Select top K features by variance
        k = config.max_features or min(10, len(numeric_cols))
        selector = SelectKBest(score_func=f_regression, k=k)

        # Create dummy target for demonstration (in practice, use real target)
        dummy_target = data[numeric_cols].iloc[:, 0]  # Use first column as dummy target
        selected_data = selector.fit_transform(data[numeric_cols], dummy_target)

        # Get selected features
        selected_features = numeric_cols[selector.get_support()]
        removed_features = numeric_cols[~selector.get_support()]

        # Create processed data
        processed_data = data.copy()
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        processed_data = pd.concat(
            [
                data[non_numeric_cols],
                pd.DataFrame(
                    selected_data, columns=selected_features, index=data.index
                ),
            ],
            axis=1,
        )

        return processed_data, {
            "features_removed": len(removed_features),
            "removed_features": list(removed_features),
            "selection_method": "statistical_selection",
            "k_features": k,
        }

    async def _select_features_model_based(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select features using model-based methods."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 2:
            return data, {"features_removed": 0}

        # Use Random Forest for feature importance
        # Create dummy target (first principal component)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        dummy_target = pca.fit_transform(data[numeric_cols]).ravel()

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(data[numeric_cols], dummy_target)

        # Get feature importances
        importances = rf.feature_importances_

        # Select features above threshold
        threshold = np.percentile(importances, 50)  # Top 50%
        selected_mask = importances >= threshold

        selected_features = numeric_cols[selected_mask]
        removed_features = numeric_cols[~selected_mask]

        # Create processed data
        processed_data = data.copy()
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        processed_data = pd.concat(
            [data[non_numeric_cols], data[selected_features]], axis=1
        )

        return processed_data, {
            "features_removed": len(removed_features),
            "removed_features": list(removed_features),
            "selection_method": "random_forest_importance",
            "importance_threshold": threshold,
        }

    async def _apply_pca(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Principal Component Analysis."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return data, {"components_created": 0}

        # Determine number of components
        if config.n_components:
            n_components = min(config.n_components, len(numeric_cols))
        else:
            # Find components needed for explained variance threshold
            pca_temp = PCA()
            pca_temp.fit(data[numeric_cols])
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= config.explained_variance_threshold) + 1

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data[numeric_cols])

        # Create component names
        component_names = [f"PC{i+1}" for i in range(n_components)]

        # Create processed data
        processed_data = data.select_dtypes(exclude=[np.number]).copy()
        for i, name in enumerate(component_names):
            processed_data[name] = pca_data[:, i]

        # Store PCA transformer
        self.fitted_transformers["pca"] = pca

        return processed_data, {
            "components_created": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "original_features": len(numeric_cols),
            "features_removed": list(numeric_cols),
        }

    async def _apply_ica(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Independent Component Analysis."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return data, {"components_created": 0}

        # Determine number of components
        n_components = config.n_components or min(len(numeric_cols), 10)

        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42)
        ica_data = ica.fit_transform(data[numeric_cols])

        # Create component names
        component_names = [f"IC{i+1}" for i in range(n_components)]

        # Create processed data
        processed_data = data.select_dtypes(exclude=[np.number]).copy()
        for i, name in enumerate(component_names):
            processed_data[name] = ica_data[:, i]

        # Store ICA transformer
        self.fitted_transformers["ica"] = ica

        return processed_data, {
            "components_created": n_components,
            "original_features": len(numeric_cols),
            "features_removed": list(numeric_cols),
        }

    async def _apply_svd(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply Singular Value Decomposition."""

        if not SKLEARN_AVAILABLE:
            return data, {"error": "scikit-learn not available"}

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return data, {"components_created": 0}

        # Determine number of components
        n_components = config.n_components or min(len(numeric_cols), 10)

        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_data = svd.fit_transform(data[numeric_cols])

        # Create component names
        component_names = [f"SVD{i+1}" for i in range(n_components)]

        # Create processed data
        processed_data = data.select_dtypes(exclude=[np.number]).copy()
        for i, name in enumerate(component_names):
            processed_data[name] = svd_data[:, i]

        # Store SVD transformer
        self.fitted_transformers["svd"] = svd

        return processed_data, {
            "components_created": n_components,
            "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(np.sum(svd.explained_variance_ratio_)),
            "original_features": len(numeric_cols),
            "features_removed": list(numeric_cols),
        }

    async def _validate_processed_data(
        self, data: pd.DataFrame, config: PreprocessingConfig
    ) -> Dict[str, Any]:
        """Validate the processed data quality."""

        validation_results = {
            "passed": True,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check for missing values
        missing_count = data.isnull().sum().sum()
        validation_results["checks"]["missing_values"] = {
            "count": int(missing_count),
            "passed": missing_count == 0,
        }

        if missing_count > 0:
            validation_results["warnings"].append(
                f"Found {missing_count} missing values"
            )

        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(data[numeric_cols]).sum().sum()
            validation_results["checks"]["infinite_values"] = {
                "count": int(inf_count),
                "passed": inf_count == 0,
            }

            if inf_count > 0:
                validation_results["errors"].append(
                    f"Found {inf_count} infinite values"
                )
                validation_results["passed"] = False

        # Check data shape
        validation_results["checks"]["data_shape"] = {
            "rows": len(data),
            "columns": len(data.columns),
            "passed": len(data) > 0 and len(data.columns) > 0,
        }

        if len(data) == 0:
            validation_results["errors"].append("Dataset is empty")
            validation_results["passed"] = False

        # Check feature variance
        if len(numeric_cols) > 0:
            low_variance_cols = []
            for col in numeric_cols:
                if data[col].var() < 1e-10:
                    low_variance_cols.append(col)

            validation_results["checks"]["feature_variance"] = {
                "low_variance_features": low_variance_cols,
                "count": len(low_variance_cols),
                "passed": len(low_variance_cols) == 0,
            }

            if low_variance_cols:
                validation_results["warnings"].append(
                    f"Found {len(low_variance_cols)} low variance features"
                )

        return validation_results

    async def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate an overall data quality score."""

        score = 1.0

        # Penalize missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3

        # Penalize infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_ratio = np.isinf(data[numeric_cols]).sum().sum() / (
                len(data) * len(numeric_cols)
            )
            score -= inf_ratio * 0.5

        # Penalize low variance features
        if len(numeric_cols) > 0:
            low_variance_count = sum(
                1 for col in numeric_cols if data[col].var() < 1e-10
            )
            low_variance_ratio = low_variance_count / len(numeric_cols)
            score -= low_variance_ratio * 0.2

        return max(0.0, score)

    async def get_preprocessing_recommendations(
        self, dataset: Dataset
    ) -> Dict[str, Any]:
        """Get preprocessing recommendations for a dataset."""

        if not hasattr(dataset, "data") or dataset.data is None:
            return {"error": "Dataset has no data"}

        data = dataset.data
        recommendations = {
            "recommended_steps": [],
            "step_configs": {},
            "reasoning": [],
            "data_analysis": {},
        }

        # Analyze data characteristics
        data_analysis = {
            "shape": data.shape,
            "missing_values": data.isnull().sum().sum(),
            "duplicate_rows": data.duplicated().sum(),
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(
                data.select_dtypes(include=["object", "category"]).columns
            ),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        recommendations["data_analysis"] = data_analysis

        # Missing value recommendations
        if data_analysis["missing_values"] > 0:
            missing_ratio = data_analysis["missing_values"] / (
                data.shape[0] * data.shape[1]
            )
            recommendations["recommended_steps"].append(
                PreprocessingStep.MISSING_VALUE_IMPUTATION
            )

            if missing_ratio > 0.3:
                recommendations["step_configs"][
                    "imputation_method"
                ] = ImputationMethod.KNN
                recommendations["reasoning"].append(
                    "High missing value ratio - recommend KNN imputation"
                )
            else:
                recommendations["step_configs"][
                    "imputation_method"
                ] = ImputationMethod.MEDIAN
                recommendations["reasoning"].append(
                    "Moderate missing values - recommend median imputation"
                )

        # Duplicate removal
        if data_analysis["duplicate_rows"] > 0:
            recommendations["recommended_steps"].append(
                PreprocessingStep.DUPLICATE_REMOVAL
            )
            recommendations["reasoning"].append(
                f"Found {data_analysis['duplicate_rows']} duplicate rows"
            )

        # Data type optimization
        if data_analysis["memory_usage_mb"] > 100:
            recommendations["recommended_steps"].append(
                PreprocessingStep.DATA_TYPE_CONVERSION
            )
            recommendations["reasoning"].append(
                "Large memory usage - recommend data type optimization"
            )

        # Feature scaling for numeric features
        if data_analysis["numeric_columns"] > 0:
            recommendations["recommended_steps"].append(
                PreprocessingStep.FEATURE_SCALING
            )
            recommendations["step_configs"]["scaling_method"] = ScalingMethod.ROBUST
            recommendations["reasoning"].append(
                "Numeric features present - recommend robust scaling"
            )

        # Feature encoding for categorical features
        if data_analysis["categorical_columns"] > 0:
            recommendations["recommended_steps"].append(
                PreprocessingStep.FEATURE_ENCODING
            )
            recommendations["step_configs"]["encoding_method"] = EncodingMethod.ONE_HOT
            recommendations["reasoning"].append(
                "Categorical features present - recommend one-hot encoding"
            )

        # Outlier detection for numeric features
        if data_analysis["numeric_columns"] > 0:
            recommendations["recommended_steps"].append(
                PreprocessingStep.OUTLIER_DETECTION
            )
            recommendations["step_configs"]["outlier_detection_method"] = "iqr"
            recommendations["reasoning"].append(
                "Numeric features present - recommend outlier detection"
            )

        # Feature selection for high-dimensional data
        total_features = (
            data_analysis["numeric_columns"] + data_analysis["categorical_columns"]
        )
        if total_features > 50:
            recommendations["recommended_steps"].append(
                PreprocessingStep.VARIANCE_FILTERING
            )
            recommendations["recommended_steps"].append(
                PreprocessingStep.CORRELATION_FILTERING
            )
            recommendations["reasoning"].append(
                "High-dimensional data - recommend feature selection"
            )

        # Dimensionality reduction for very high-dimensional data
        if data_analysis["numeric_columns"] > 100:
            recommendations["recommended_steps"].append(PreprocessingStep.PCA_REDUCTION)
            recommendations["step_configs"]["explained_variance_threshold"] = 0.95
            recommendations["reasoning"].append(
                "Very high-dimensional numeric data - recommend PCA"
            )

        return recommendations

    def get_preprocessing_history(self) -> List[Dict[str, Any]]:
        """Get history of preprocessing operations."""
        return self.preprocessing_history.copy()

    def get_fitted_transformers(self) -> Dict[str, Any]:
        """Get fitted transformers for reuse."""
        return self.fitted_transformers.copy()

    async def save_preprocessing_pipeline(self, filepath: str) -> None:
        """Save the preprocessing pipeline for reuse."""
        import pickle

        pipeline_data = {
            "config": self.config.__dict__,
            "fitted_transformers": self.fitted_transformers,
            "feature_info": self.feature_info,
            "preprocessing_history": self.preprocessing_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(pipeline_data, f)

        logger.info(f"Preprocessing pipeline saved to {filepath}")

    async def load_preprocessing_pipeline(self, filepath: str) -> None:
        """Load a saved preprocessing pipeline."""
        import pickle

        with open(filepath, "rb") as f:
            pipeline_data = pickle.load(f)

        self.config = PreprocessingConfig(**pipeline_data["config"])
        self.fitted_transformers = pipeline_data["fitted_transformers"]
        self.feature_info = pipeline_data["feature_info"]
        self.preprocessing_history = pipeline_data["preprocessing_history"]

        logger.info(f"Preprocessing pipeline loaded from {filepath}")


# Factory function
def create_advanced_preprocessor(
    config: Optional[PreprocessingConfig] = None,
) -> AdvancedPreprocessor:
    """Create an advanced preprocessor with the given configuration."""
    return AdvancedPreprocessor(config)
