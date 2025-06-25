#!/usr/bin/env python3
"""
Financial Data Preprocessing Pipeline Template

This template provides a comprehensive preprocessing pipeline specifically designed
for financial data including transactions, trading data, and regulatory reporting.
"""

import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Data processing imports
import logging

# Statistical imports
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for financial datasets.

    Features:
    - Transaction data cleaning and normalization
    - Time-based feature engineering
    - Risk assessment calculations
    - Regulatory compliance preprocessing
    - AML/KYC data preparation
    - Fraud detection preprocessing
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        preserve_original: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the financial data preprocessor.

        Args:
            config: Configuration dictionary for preprocessing steps
            preserve_original: Whether to preserve original column values
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.preserve_original = preserve_original
        self.verbose = verbose

        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}

        # Metadata tracking
        self.preprocessing_steps = []
        self.data_profile = {}
        self.feature_mappings = {}

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for financial data preprocessing."""
        return {
            "missing_values": {
                "strategy": "knn",  # 'mean', 'median', 'mode', 'knn', 'forward_fill'
                "threshold": 0.5,  # Drop columns with >50% missing
                "knn_neighbors": 5,
            },
            "outliers": {
                "method": "iqr",  # 'iqr', 'zscore', 'isolation_forest'
                "threshold": 3.0,
                "action": "cap",  # 'remove', 'cap', 'transform'
            },
            "scaling": {
                "method": "robust",  # 'standard', 'minmax', 'robust'
                "feature_range": (0, 1),
            },
            "encoding": {
                "categorical_threshold": 10,  # Max unique values for one-hot
                "high_cardinality_method": "target",  # 'target', 'frequency', 'binary'
                "handle_unknown": "ignore",
            },
            "feature_engineering": {
                "time_features": True,
                "transaction_features": True,
                "risk_features": True,
                "aggregation_features": True,
            },
            "feature_selection": {
                "variance_threshold": 0.01,
                "correlation_threshold": 0.95,
                "k_best": None,  # None for auto, int for specific number
            },
            "validation": {
                "amount_ranges": True,
                "date_consistency": True,
                "currency_validation": True,
                "account_validation": True,
            },
        }

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply comprehensive preprocessing pipeline to financial data.

        Args:
            data: Input DataFrame
            target_column: Name of target column (for supervised preprocessing)

        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        logger.info("Starting financial data preprocessing pipeline")

        # Create copy to avoid modifying original
        df = data.copy()
        original_shape = df.shape

        # 1. Data Validation and Quality Assessment
        self._log_step("Data Validation and Quality Assessment")
        validation_results = self._validate_financial_data(df)

        # 2. Handle Missing Values
        self._log_step("Missing Value Treatment")
        df = self._handle_missing_values(df, target_column)

        # 3. Outlier Detection and Treatment
        self._log_step("Outlier Detection and Treatment")
        df = self._handle_outliers(df, target_column)

        # 4. Data Type Optimization
        self._log_step("Data Type Optimization")
        df = self._optimize_data_types(df)

        # 5. Financial Feature Engineering
        self._log_step("Financial Feature Engineering")
        df = self._engineer_financial_features(df)

        # 6. Categorical Encoding
        self._log_step("Categorical Variable Encoding")
        df = self._encode_categorical_variables(df, target_column)

        # 7. Feature Scaling
        self._log_step("Feature Scaling")
        df = self._scale_features(df, target_column)

        # 8. Feature Selection
        self._log_step("Feature Selection")
        df = self._select_features(df, target_column)

        # 9. Final Validation
        self._log_step("Final Validation")
        final_validation = self._final_validation(df, original_shape)

        # Prepare metadata
        metadata = {
            "preprocessing_steps": self.preprocessing_steps,
            "data_profile": self.data_profile,
            "feature_mappings": self.feature_mappings,
            "validation_results": validation_results,
            "final_validation": final_validation,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "config": self.config,
        }

        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        return df, metadata

    def _validate_financial_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate financial data for common issues and compliance."""
        validation_results = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "missing_data_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "duplicate_records": df.duplicated().sum(),
            "issues": [],
        }

        # Check for financial data specific issues
        for col in df.columns:
            if (
                "amount" in col.lower()
                or "value" in col.lower()
                or "balance" in col.lower()
            ):
                # Check for negative amounts where not expected
                if df[col].dtype in ["int64", "float64"]:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validation_results["issues"].append(
                            {
                                "type": "negative_amounts",
                                "column": col,
                                "count": negative_count,
                                "severity": "medium",
                            }
                        )

                    # Check for extreme values
                    if df[col].max() > df[col].quantile(0.99) * 100:
                        validation_results["issues"].append(
                            {
                                "type": "extreme_amounts",
                                "column": col,
                                "max_value": df[col].max(),
                                "severity": "high",
                            }
                        )

        # Check date consistency
        date_columns = df.select_dtypes(include=["datetime64"]).columns
        for col in date_columns:
            future_dates = (df[col] > datetime.now()).sum()
            if future_dates > 0:
                validation_results["issues"].append(
                    {
                        "type": "future_dates",
                        "column": col,
                        "count": future_dates,
                        "severity": "high",
                    }
                )

        self.data_profile["validation"] = validation_results
        return validation_results

    def _handle_missing_values(
        self, df: pd.DataFrame, target_column: str = None
    ) -> pd.DataFrame:
        """Handle missing values using financial data appropriate strategies."""
        strategy = self.config["missing_values"]["strategy"]
        threshold = self.config["missing_values"]["threshold"]

        # Drop columns with too many missing values
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.preprocessing_steps.append(
                {
                    "step": "drop_high_missing_columns",
                    "columns_dropped": cols_to_drop,
                    "threshold": threshold,
                }
            )

        # Handle remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        if target_column and target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)

        # Numeric columns
        if len(numeric_columns) > 0:
            if strategy == "knn":
                imputer = KNNImputer(
                    n_neighbors=self.config["missing_values"]["knn_neighbors"]
                )
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
                self.imputers["numeric"] = imputer
            elif strategy in ["mean", "median"]:
                imputer = SimpleImputer(strategy=strategy)
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
                self.imputers["numeric"] = imputer

        # Categorical columns
        if len(categorical_columns) > 0:
            imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
            self.imputers["categorical"] = imputer

        self.preprocessing_steps.append(
            {
                "step": "missing_value_imputation",
                "strategy": strategy,
                "numeric_columns": list(numeric_columns),
                "categorical_columns": list(categorical_columns),
            }
        )

        return df

    def _handle_outliers(
        self, df: pd.DataFrame, target_column: str = None
    ) -> pd.DataFrame:
        """Handle outliers in financial data with appropriate methods."""
        method = self.config["outliers"]["method"]
        threshold = self.config["outliers"]["threshold"]
        action = self.config["outliers"]["action"]

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)

        outlier_info = {}

        for col in numeric_columns:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs(zscore(df[col]))
                outliers = z_scores > threshold

            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_info[col] = outlier_count

                if action == "cap":
                    if method == "iqr":
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        df.loc[df[col] > upper_bound, col] = upper_bound
                    elif method == "zscore":
                        lower_cap = df[col].quantile(0.01)
                        upper_cap = df[col].quantile(0.99)
                        df.loc[df[col] < lower_cap, col] = lower_cap
                        df.loc[df[col] > upper_cap, col] = upper_cap
                elif action == "remove":
                    df = df[~outliers]

        self.preprocessing_steps.append(
            {
                "step": "outlier_treatment",
                "method": method,
                "action": action,
                "outliers_found": outlier_info,
            }
        )

        return df

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency and performance."""
        optimization_info = {}

        for col in df.columns:
            original_dtype = df[col].dtype

            if df[col].dtype == "object":
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    optimization_info[col] = f"{original_dtype} -> {df[col].dtype}"
                except:
                    # Try to convert to category if low cardinality
                    if df[col].nunique() < len(df) * 0.1:
                        df[col] = df[col].astype("category")
                        optimization_info[col] = f"{original_dtype} -> category"

            elif df[col].dtype in ["int64", "float64"]:
                # Downcast numeric types
                if df[col].dtype == "int64":
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                else:
                    df[col] = pd.to_numeric(df[col], downcast="float")

                if df[col].dtype != original_dtype:
                    optimization_info[col] = f"{original_dtype} -> {df[col].dtype}"

        self.preprocessing_steps.append(
            {"step": "data_type_optimization", "optimizations": optimization_info}
        )

        return df

    def _engineer_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer financial-specific features."""
        if not self.config["feature_engineering"]["time_features"]:
            return df

        new_features = []

        # Time-based features
        for col in df.columns:
            if df[col].dtype.name.startswith("datetime"):
                base_name = col.replace("_date", "").replace("_time", "")

                # Extract time components
                df[f"{base_name}_year"] = df[col].dt.year
                df[f"{base_name}_month"] = df[col].dt.month
                df[f"{base_name}_day"] = df[col].dt.day
                df[f"{base_name}_hour"] = df[col].dt.hour
                df[f"{base_name}_dayofweek"] = df[col].dt.dayofweek
                df[f"{base_name}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)

                new_features.extend(
                    [
                        f"{base_name}_year",
                        f"{base_name}_month",
                        f"{base_name}_day",
                        f"{base_name}_hour",
                        f"{base_name}_dayofweek",
                        f"{base_name}_is_weekend",
                    ]
                )

        # Amount-based features
        amount_columns = [col for col in df.columns if "amount" in col.lower()]
        for col in amount_columns:
            if df[col].dtype in ["int64", "float64"]:
                # Log transformation for right-skewed financial data
                df[f"{col}_log"] = np.log1p(df[col])

                # Amount ranges
                df[f"{col}_range"] = pd.cut(
                    df[col],
                    bins=5,
                    labels=["very_low", "low", "medium", "high", "very_high"],
                )

                new_features.extend([f"{col}_log", f"{col}_range"])

        # Transaction frequency features (if applicable)
        if "account_id" in df.columns and len(amount_columns) > 0:
            for amount_col in amount_columns:
                # Daily transaction statistics per account
                df["transaction_count"] = df.groupby("account_id")[
                    amount_col
                ].transform("count")
                df["avg_transaction_amount"] = df.groupby("account_id")[
                    amount_col
                ].transform("mean")
                df["total_transaction_amount"] = df.groupby("account_id")[
                    amount_col
                ].transform("sum")

                new_features.extend(
                    [
                        "transaction_count",
                        "avg_transaction_amount",
                        "total_transaction_amount",
                    ]
                )

        self.preprocessing_steps.append(
            {"step": "financial_feature_engineering", "new_features": new_features}
        )

        return df

    def _encode_categorical_variables(
        self, df: pd.DataFrame, target_column: str = None
    ) -> pd.DataFrame:
        """Encode categorical variables using appropriate methods."""
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        if target_column and target_column in categorical_columns:
            categorical_columns = categorical_columns.drop(target_column)

        encoding_info = {}

        for col in categorical_columns:
            unique_count = df[col].nunique()

            if unique_count <= self.config["encoding"]["categorical_threshold"]:
                # One-hot encoding for low cardinality
                encoder = OneHotEncoder(
                    drop="first", sparse=False, handle_unknown="ignore"
                )
                encoded_cols = encoder.fit_transform(df[[col]])

                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(
                    encoded_cols, columns=feature_names, index=df.index
                )

                # Add to dataframe and remove original
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col])

                self.encoders[col] = encoder
                encoding_info[col] = f"one_hot_{len(feature_names)}_features"

            else:
                # Label encoding for high cardinality
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))

                self.encoders[col] = encoder
                encoding_info[col] = "label_encoding"

        self.preprocessing_steps.append(
            {"step": "categorical_encoding", "encoding_methods": encoding_info}
        )

        return df

    def _scale_features(
        self, df: pd.DataFrame, target_column: str = None
    ) -> pd.DataFrame:
        """Scale numerical features using appropriate method."""
        method = self.config["scaling"]["method"]

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)

        if len(numeric_columns) > 0:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler(
                    feature_range=self.config["scaling"]["feature_range"]
                )
            elif method == "robust":
                scaler = RobustScaler()

            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            self.scalers["numeric"] = scaler

            self.preprocessing_steps.append(
                {
                    "step": "feature_scaling",
                    "method": method,
                    "columns_scaled": list(numeric_columns),
                }
            )

        return df

    def _select_features(
        self, df: pd.DataFrame, target_column: str = None
    ) -> pd.DataFrame:
        """Select most relevant features for anomaly detection."""
        original_features = len(df.columns)

        # Variance threshold
        variance_threshold = self.config["feature_selection"]["variance_threshold"]
        if variance_threshold > 0:
            selector = VarianceThreshold(threshold=variance_threshold)
            feature_mask = selector.fit_transform(df.select_dtypes(include=[np.number]))

            # Get selected columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            selected_numeric_cols = numeric_cols[selector.get_support()]

            # Keep non-numeric columns and selected numeric columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            df = df[list(non_numeric_cols) + list(selected_numeric_cols)]

            self.feature_selectors["variance"] = selector

        # Correlation threshold
        correlation_threshold = self.config["feature_selection"][
            "correlation_threshold"
        ]
        if correlation_threshold < 1.0:
            # Remove highly correlated features
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                to_drop = [
                    column
                    for column in upper_triangle.columns
                    if any(upper_triangle[column] > correlation_threshold)
                ]

                df = df.drop(columns=to_drop)

                if to_drop:
                    self.preprocessing_steps.append(
                        {
                            "step": "correlation_feature_removal",
                            "threshold": correlation_threshold,
                            "removed_features": to_drop,
                        }
                    )

        self.preprocessing_steps.append(
            {
                "step": "feature_selection_summary",
                "original_features": original_features,
                "final_features": len(df.columns),
                "features_removed": original_features - len(df.columns),
            }
        )

        return df

    def _final_validation(
        self, df: pd.DataFrame, original_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Perform final validation of processed data."""
        validation_results = {
            "shape_change": f"{original_shape} -> {df.shape}",
            "missing_values": df.isnull().sum().sum(),
            "infinite_values": np.isinf(df.select_dtypes(include=[np.number]))
            .sum()
            .sum(),
            "data_types": dict(df.dtypes.astype(str)),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "processing_success": True,
        }

        # Check for any remaining issues
        issues = []
        if validation_results["missing_values"] > 0:
            issues.append("Missing values still present")
        if validation_results["infinite_values"] > 0:
            issues.append("Infinite values detected")

        validation_results["issues"] = issues
        validation_results["processing_success"] = len(issues) == 0

        return validation_results

    def _log_step(self, step_name: str):
        """Log preprocessing step."""
        if self.verbose:
            logger.info(f"Executing: {step_name}")

    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline configuration and fitted components."""
        pipeline_data = {
            "config": self.config,
            "preprocessing_steps": self.preprocessing_steps,
            "feature_mappings": self.feature_mappings,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load a saved preprocessing pipeline configuration."""
        with open(filepath) as f:
            pipeline_data = json.load(f)

        self.config = pipeline_data["config"]
        self.preprocessing_steps = pipeline_data.get("preprocessing_steps", [])
        self.feature_mappings = pipeline_data.get("feature_mappings", {})

        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Example usage of the Financial Data Preprocessor."""
    # Create sample financial data
    np.random.seed(42)
    n_samples = 10000

    # Generate synthetic transaction data
    data = {
        "transaction_id": range(n_samples),
        "account_id": np.random.choice(range(1000), n_samples),
        "amount": np.random.lognormal(3, 1, n_samples),
        "transaction_type": np.random.choice(
            ["debit", "credit", "transfer"], n_samples
        ),
        "merchant_category": np.random.choice(
            ["grocery", "gas", "restaurant", "retail", "other"], n_samples
        ),
        "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="H"),
        "balance_after": np.random.normal(5000, 2000, n_samples),
        "currency": np.random.choice(
            ["USD", "EUR", "GBP"], n_samples, p=[0.7, 0.2, 0.1]
        ),
    }

    # Add some missing values and outliers
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 500), "amount"] = np.nan
    df.loc[np.random.choice(df.index, 100), "amount"] = np.random.uniform(
        50000, 100000, 100
    )

    print("Original Data Shape:", df.shape)
    print("\nOriginal Data Info:")
    print(df.info())

    # Initialize preprocessor with custom config
    config = {
        "missing_values": {"strategy": "knn", "threshold": 0.6, "knn_neighbors": 5},
        "outliers": {"method": "iqr", "action": "cap"},
        "scaling": {"method": "robust"},
        "feature_engineering": {"time_features": True, "transaction_features": True},
    }

    preprocessor = FinancialDataPreprocessor(config=config, verbose=True)

    # Apply preprocessing
    processed_df, metadata = preprocessor.preprocess(df)

    print(f"\nProcessed Data Shape: {processed_df.shape}")
    print("\nPreprocessing Steps Applied:")
    for i, step in enumerate(metadata["preprocessing_steps"], 1):
        print(f"{i}. {step['step']}")

    print("\nValidation Results:")
    print(f"- Processing Success: {metadata['final_validation']['processing_success']}")
    print(f"- Missing Values: {metadata['final_validation']['missing_values']}")
    print(f"- Memory Usage: {metadata['final_validation']['memory_usage_mb']:.2f} MB")

    # Save pipeline for reuse
    preprocessor.save_pipeline("financial_preprocessing_pipeline.json")

    print("\nFinancial preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
