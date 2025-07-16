"""Domain service for feature validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataTypeError, FeatureMismatchError


class FeatureValidator:
    """Domain service for validating dataset features.

    This service ensures data compatibility between training
    and inference, and validates feature requirements.
    """

    @staticmethod
    def validate_compatibility(
        reference_dataset: Dataset, target_dataset: Dataset, strict: bool = True
    ) -> None:
        """Validate that two datasets are compatible.

        Args:
            reference_dataset: Reference dataset (e.g., training data)
            target_dataset: Dataset to validate against reference
            strict: If True, require exact feature match

        Raises:
            FeatureMismatchError: If features don't match
            DataTypeError: If data types are incompatible
        """
        ref_features = set(reference_dataset.feature_names or [])
        target_features = set(target_dataset.feature_names or [])

        if reference_dataset.target_column:
            ref_features.discard(reference_dataset.target_column)
        if target_dataset.target_column:
            target_features.discard(target_dataset.target_column)

        # Check feature names
        missing_features = ref_features - target_features
        extra_features = target_features - ref_features

        if strict and (missing_features or extra_features):
            raise FeatureMismatchError(
                "Feature mismatch between datasets",
                expected_features=sorted(ref_features),
                actual_features=sorted(target_features),
                missing_features=sorted(missing_features) if missing_features else None,
                extra_features=sorted(extra_features) if extra_features else None,
            )
        elif missing_features:
            # In non-strict mode, only missing features are problematic
            raise FeatureMismatchError(
                f"Target dataset missing required features: {missing_features}",
                missing_features=sorted(missing_features),
            )

        # Check data types for common features
        common_features = ref_features & target_features
        FeatureValidator._validate_dtypes(
            reference_dataset, target_dataset, common_features
        )

    @staticmethod
    def _validate_dtypes(
        ref_dataset: Dataset, target_dataset: Dataset, features: set[str]
    ) -> None:
        """Validate data types match between datasets."""
        for feature in features:
            ref_dtype = ref_dataset.data[feature].dtype
            target_dtype = target_dataset.data[feature].dtype

            # Check if types are compatible (not necessarily identical)
            if not FeatureValidator._are_dtypes_compatible(ref_dtype, target_dtype):
                raise DataTypeError(
                    f"Incompatible data types for feature '{feature}'",
                    feature=feature,
                    expected_type=str(ref_dtype),
                    actual_type=str(target_dtype),
                )

    @staticmethod
    def _are_dtypes_compatible(dtype1: np.dtype, dtype2: np.dtype) -> bool:
        """Check if two data types are compatible."""
        # Numeric types are generally compatible
        numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

        if str(dtype1) in numeric_types and str(dtype2) in numeric_types:
            return True

        # Otherwise require exact match
        return dtype1 == dtype2

    @staticmethod
    def validate_numeric_features(
        dataset: Dataset, features: list[str] | None = None
    ) -> list[str]:
        """Validate that features are numeric.

        Args:
            dataset: Dataset to validate
            features: Specific features to check (None = all features)

        Returns:
            List of valid numeric features

        Raises:
            DataTypeError: If any specified feature is not numeric
        """
        if features is None:
            features = dataset.get_numeric_features()

        numeric_features = []

        for feature in features:
            if feature not in dataset.data.columns:
                raise FeatureMismatchError(
                    f"Feature '{feature}' not found in dataset",
                    missing_features=[feature],
                )

            if not pd.api.types.is_numeric_dtype(dataset.data[feature]):
                raise DataTypeError(
                    f"Feature '{feature}' is not numeric",
                    feature=feature,
                    actual_type=str(dataset.data[feature].dtype),
                )

            numeric_features.append(feature)

        return numeric_features

    @staticmethod
    def check_data_quality(
        dataset: Dataset,
        max_missing_ratio: float = 0.1,
        max_constant_ratio: float = 0.95,
    ) -> dict[str, Any]:
        """Check data quality issues.

        Args:
            dataset: Dataset to check
            max_missing_ratio: Maximum allowed ratio of missing values
            max_constant_ratio: Maximum allowed ratio of constant values

        Returns:
            Dictionary with quality check results
        """
        results = {
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 1.0,  # 1.0 is perfect
        }

        # Check missing values
        missing_counts = dataset.data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                ratio = count / dataset.n_samples
                results["missing_values"][col] = {
                    "count": int(count),
                    "ratio": float(ratio),
                }

                if ratio > max_missing_ratio:
                    results["quality_score"] *= 1 - ratio

        # Check constant features
        for col in dataset.get_numeric_features():
            unique_ratio = dataset.data[col].nunique() / dataset.n_samples

            if unique_ratio < (1 - max_constant_ratio):
                results["constant_features"].append(col)
                results["quality_score"] *= 0.9
            elif unique_ratio < 0.1:  # Low variance
                results["low_variance_features"].append(col)
                results["quality_score"] *= 0.95

        # Check infinite values
        for col in dataset.get_numeric_features():
            inf_count = np.isinf(dataset.data[col]).sum()
            if inf_count > 0:
                results["infinite_values"][col] = int(inf_count)
                results["quality_score"] *= 0.8

        # Check duplicate rows
        duplicate_count = dataset.data.duplicated().sum()
        results["duplicate_rows"] = int(duplicate_count)
        if duplicate_count > 0:
            dup_ratio = duplicate_count / dataset.n_samples
            results["quality_score"] *= 1 - dup_ratio * 0.5

        # Ensure quality score doesn't go below 0
        results["quality_score"] = max(0.0, results["quality_score"])

        return results

    @staticmethod
    def suggest_preprocessing(quality_report: dict[str, Any]) -> list[str]:
        """Suggest preprocessing steps based on quality report.

        Args:
            quality_report: Output from check_data_quality

        Returns:
            List of suggested preprocessing steps
        """
        suggestions = []

        if quality_report["missing_values"]:
            suggestions.append("Handle missing values: Consider imputation or removal")

        if quality_report["constant_features"]:
            features = ", ".join(quality_report["constant_features"])
            suggestions.append(f"Remove constant features: {features}")

        if quality_report["low_variance_features"]:
            suggestions.append(
                "Consider removing low variance features or applying variance threshold"
            )

        if quality_report["infinite_values"]:
            suggestions.append(
                "Handle infinite values: Replace with large finite values or remove"
            )

        if quality_report["duplicate_rows"] > 0:
            suggestions.append(
                f"Remove {quality_report['duplicate_rows']} duplicate rows"
            )

        if quality_report["quality_score"] < 0.8:
            suggestions.append(
                "Data quality is low - careful preprocessing recommended"
            )

        return suggestions
