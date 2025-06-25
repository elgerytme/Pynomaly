"""Preprocessing pipeline for chaining multiple preprocessing steps."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any

from pynomaly.domain.entities import Dataset

from .data_cleaner import DataCleaner, MissingValueStrategy, OutlierStrategy
from .data_transformer import (
    DataTransformer,
    EncodingStrategy,
    FeatureSelectionStrategy,
    ScalingStrategy,
)


@dataclass
class PreprocessingStep:
    """Represents a single preprocessing step."""

    name: str
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str | None = None


class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessing operations."""

    def __init__(self, name: str = "preprocessing_pipeline"):
        """Initialize the preprocessing pipeline.

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.steps: list[PreprocessingStep] = []
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self._fitted = False
        self._fit_dataset_id: str | None = None

    def add_step(
        self,
        name: str,
        operation: str,
        parameters: dict[str, Any] | None = None,
        enabled: bool = True,
        description: str | None = None,
    ) -> PreprocessingPipeline:
        """Add a preprocessing step to the pipeline.

        Args:
            name: Name of the step
            operation: Operation to perform
            parameters: Parameters for the operation
            enabled: Whether the step is enabled
            description: Optional description

        Returns:
            Self for method chaining
        """
        step = PreprocessingStep(
            name=name,
            operation=operation,
            parameters=parameters or {},
            enabled=enabled,
            description=description,
        )
        self.steps.append(step)
        return self

    def add_missing_value_handling(
        self,
        strategy: MissingValueStrategy | str = MissingValueStrategy.FILL_MEDIAN,
        threshold: float = 0.5,
        fill_value: Any = None,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add missing value handling step."""
        return self.add_step(
            name="handle_missing_values",
            operation="missing_values",
            parameters={
                "strategy": strategy,
                "threshold": threshold,
                "fill_value": fill_value,
                "columns": columns,
            },
            enabled=enabled,
            description=f"Handle missing values using {strategy}",
        )

    def add_duplicate_removal(
        self,
        subset: list[str] | None = None,
        keep: str = "first",
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add duplicate removal step."""
        return self.add_step(
            name="remove_duplicates",
            operation="duplicates",
            parameters={"subset": subset, "keep": keep},
            enabled=enabled,
            description="Remove duplicate rows",
        )

    def add_outlier_handling(
        self,
        strategy: OutlierStrategy | str = OutlierStrategy.CLIP,
        method: str = "iqr",
        threshold: float = 1.5,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add outlier handling step."""
        return self.add_step(
            name="handle_outliers",
            operation="outliers",
            parameters={
                "strategy": strategy,
                "method": method,
                "threshold": threshold,
                "columns": columns,
            },
            enabled=enabled,
            description=f"Handle outliers using {strategy}",
        )

    def add_zero_value_handling(
        self,
        strategy: str = "keep",
        replacement_value: float | None = None,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add zero value handling step."""
        return self.add_step(
            name="handle_zeros",
            operation="zeros",
            parameters={
                "strategy": strategy,
                "replacement_value": replacement_value,
                "columns": columns,
            },
            enabled=enabled,
            description=f"Handle zero values using {strategy}",
        )

    def add_infinite_value_handling(
        self,
        strategy: str = "replace",
        replacement_value: float | None = None,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add infinite value handling step."""
        return self.add_step(
            name="handle_infinites",
            operation="infinites",
            parameters={
                "strategy": strategy,
                "replacement_value": replacement_value,
                "columns": columns,
            },
            enabled=enabled,
            description=f"Handle infinite values using {strategy}",
        )

    def add_scaling(
        self,
        strategy: ScalingStrategy | str = ScalingStrategy.STANDARD,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add feature scaling step."""
        return self.add_step(
            name="scale_features",
            operation="scaling",
            parameters={"strategy": strategy, "columns": columns},
            enabled=enabled,
            description=f"Scale features using {strategy}",
        )

    def add_categorical_encoding(
        self,
        strategy: EncodingStrategy | str = EncodingStrategy.ONEHOT,
        columns: list[str] | None = None,
        drop_first: bool = True,
        handle_unknown: str = "ignore",
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add categorical encoding step."""
        return self.add_step(
            name="encode_categorical",
            operation="encoding",
            parameters={
                "strategy": strategy,
                "columns": columns,
                "drop_first": drop_first,
                "handle_unknown": handle_unknown,
            },
            enabled=enabled,
            description=f"Encode categorical features using {strategy}",
        )

    def add_feature_selection(
        self,
        strategy: FeatureSelectionStrategy | str = FeatureSelectionStrategy.VARIANCE_THRESHOLD,
        k: int = 10,
        threshold: float = 0.01,
        columns: list[str] | None = None,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add feature selection step."""
        return self.add_step(
            name="select_features",
            operation="feature_selection",
            parameters={
                "strategy": strategy,
                "k": k,
                "threshold": threshold,
                "columns": columns,
            },
            enabled=enabled,
            description=f"Select features using {strategy}",
        )

    def add_polynomial_features(
        self,
        degree: int = 2,
        columns: list[str] | None = None,
        interaction_only: bool = False,
        include_bias: bool = False,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add polynomial feature generation step."""
        return self.add_step(
            name="polynomial_features",
            operation="polynomial",
            parameters={
                "degree": degree,
                "columns": columns,
                "interaction_only": interaction_only,
                "include_bias": include_bias,
            },
            enabled=enabled,
            description=f"Create polynomial features of degree {degree}",
        )

    def add_type_conversion(
        self,
        type_mapping: dict[str, str] | None = None,
        infer_types: bool = True,
        optimize_memory: bool = True,
        enabled: bool = True,
    ) -> PreprocessingPipeline:
        """Add data type conversion step."""
        return self.add_step(
            name="convert_types",
            operation="type_conversion",
            parameters={
                "type_mapping": type_mapping,
                "infer_types": infer_types,
                "optimize_memory": optimize_memory,
            },
            enabled=enabled,
            description="Convert and optimize data types",
        )

    def fit(self, dataset: Dataset) -> PreprocessingPipeline:
        """Fit the pipeline on a training dataset.

        Args:
            dataset: Training dataset

        Returns:
            Self for method chaining
        """
        current_dataset = dataset

        for step in self.steps:
            if not step.enabled:
                continue

            try:
                # Apply step with fit=True to learn parameters
                current_dataset = self._apply_step(step, current_dataset, fit=True)

            except Exception as e:
                warnings.warn(f"Failed to fit step '{step.name}': {e}")
                continue

        self._fitted = True
        self._fit_dataset_id = str(dataset.id)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Apply the fitted pipeline to transform a dataset.

        Args:
            dataset: Dataset to transform

        Returns:
            Transformed dataset
        """
        if not self._fitted:
            warnings.warn("Pipeline not fitted. Use fit_transform() for training data.")
            return self.fit_transform(dataset)

        current_dataset = dataset
        step_results = []

        for step in self.steps:
            if not step.enabled:
                continue

            try:
                # Apply step with fit=False to use learned parameters
                before_shape = current_dataset.shape
                current_dataset = self._apply_step(step, current_dataset, fit=False)
                after_shape = current_dataset.shape

                step_results.append(
                    {
                        "step": step.name,
                        "operation": step.operation,
                        "before_shape": before_shape,
                        "after_shape": after_shape,
                        "parameters": step.parameters,
                    }
                )

            except Exception as e:
                warnings.warn(f"Failed to apply step '{step.name}': {e}")
                continue

        # Update metadata
        current_dataset.metadata.update(
            {
                "preprocessing_pipeline": self.name,
                "pipeline_steps": step_results,
                "original_dataset_id": str(dataset.id),
                "fit_dataset_id": self._fit_dataset_id,
            }
        )

        return current_dataset

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit the pipeline and transform the dataset in one step.

        Args:
            dataset: Dataset to fit and transform

        Returns:
            Transformed dataset
        """
        return self.fit(dataset).transform(dataset)

    def _apply_step(
        self, step: PreprocessingStep, dataset: Dataset, fit: bool = True
    ) -> Dataset:
        """Apply a single preprocessing step.

        Args:
            step: Preprocessing step to apply
            dataset: Input dataset
            fit: Whether to fit parameters

        Returns:
            Transformed dataset
        """
        operation = step.operation
        params = step.parameters.copy()

        # Add fit parameter where applicable
        if operation in ["scaling", "encoding"]:
            params["fit"] = fit

        if operation == "missing_values":
            return self.cleaner.handle_missing_values(dataset, **params)
        elif operation == "duplicates":
            return self.cleaner.remove_duplicates(dataset, **params)
        elif operation == "outliers":
            return self.cleaner.handle_outliers(dataset, **params)
        elif operation == "zeros":
            return self.cleaner.handle_zero_values(dataset, **params)
        elif operation == "infinites":
            return self.cleaner.handle_infinite_values(dataset, **params)
        elif operation == "scaling":
            return self.transformer.scale_features(dataset, **params)
        elif operation == "encoding":
            return self.transformer.encode_categorical_features(dataset, **params)
        elif operation == "feature_selection":
            return self.transformer.select_features(dataset, **params)
        elif operation == "polynomial":
            return self.transformer.create_polynomial_features(dataset, **params)
        elif operation == "type_conversion":
            return self.transformer.convert_data_types(dataset, **params)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def remove_step(self, name: str) -> PreprocessingPipeline:
        """Remove a step from the pipeline.

        Args:
            name: Name of the step to remove

        Returns:
            Self for method chaining
        """
        self.steps = [step for step in self.steps if step.name != name]
        self._fitted = False  # Need to refit after removing steps
        return self

    def enable_step(self, name: str) -> PreprocessingPipeline:
        """Enable a step in the pipeline.

        Args:
            name: Name of the step to enable

        Returns:
            Self for method chaining
        """
        for step in self.steps:
            if step.name == name:
                step.enabled = True
                break
        return self

    def disable_step(self, name: str) -> PreprocessingPipeline:
        """Disable a step in the pipeline.

        Args:
            name: Name of the step to disable

        Returns:
            Self for method chaining
        """
        for step in self.steps:
            if step.name == name:
                step.enabled = False
                break
        return self

    def get_step_info(self) -> list[dict[str, Any]]:
        """Get information about all steps in the pipeline.

        Returns:
            List of step information
        """
        return [
            {
                "name": step.name,
                "operation": step.operation,
                "enabled": step.enabled,
                "description": step.description,
                "parameters": step.parameters,
            }
            for step in self.steps
        ]

    def clone(self) -> PreprocessingPipeline:
        """Create a copy of the pipeline.

        Returns:
            Cloned pipeline
        """
        new_pipeline = PreprocessingPipeline(name=f"{self.name}_clone")

        for step in self.steps:
            new_pipeline.add_step(
                name=step.name,
                operation=step.operation,
                parameters=step.parameters.copy(),
                enabled=step.enabled,
                description=step.description,
            )

        return new_pipeline

    def save_config(self, file_path: str) -> None:
        """Save pipeline configuration to a JSON file.

        Args:
            file_path: Path to save the configuration
        """
        config = {"name": self.name, "steps": []}

        for step in self.steps:
            step_config = {
                "name": step.name,
                "operation": step.operation,
                "parameters": step.parameters,
                "enabled": step.enabled,
                "description": step.description,
            }
            config["steps"].append(step_config)

        with open(file_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    @classmethod
    def load_config(cls, file_path: str) -> PreprocessingPipeline:
        """Load pipeline configuration from a JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            Loaded pipeline
        """
        with open(file_path) as f:
            config = json.load(f)

        pipeline = cls(name=config["name"])

        for step_config in config["steps"]:
            pipeline.add_step(
                name=step_config["name"],
                operation=step_config["operation"],
                parameters=step_config["parameters"],
                enabled=step_config["enabled"],
                description=step_config.get("description"),
            )

        return pipeline

    @classmethod
    def create_basic_pipeline(cls) -> PreprocessingPipeline:
        """Create a basic preprocessing pipeline with common steps.

        Returns:
            Basic preprocessing pipeline
        """
        pipeline = cls("basic_preprocessing")

        pipeline.add_missing_value_handling(
            strategy=MissingValueStrategy.FILL_MEDIAN,
            description="Handle missing values with median imputation",
        )

        pipeline.add_duplicate_removal(description="Remove duplicate rows")

        pipeline.add_infinite_value_handling(
            strategy="replace", description="Replace infinite values"
        )

        pipeline.add_outlier_handling(
            strategy=OutlierStrategy.CLIP,
            description="Clip outliers to reasonable bounds",
        )

        pipeline.add_categorical_encoding(
            strategy=EncodingStrategy.ONEHOT,
            description="One-hot encode categorical features",
        )

        pipeline.add_scaling(
            strategy=ScalingStrategy.STANDARD,
            description="Standard scale numeric features",
        )

        return pipeline

    @classmethod
    def create_anomaly_detection_pipeline(cls) -> PreprocessingPipeline:
        """Create a pipeline optimized for anomaly detection.

        Returns:
            Anomaly detection preprocessing pipeline
        """
        pipeline = cls("anomaly_detection_preprocessing")

        # More conservative missing value handling for anomaly detection
        pipeline.add_missing_value_handling(
            strategy=MissingValueStrategy.DROP_ROWS,
            threshold=0.1,  # Drop rows with any missing values
            description="Drop rows with missing values (anomaly detection)",
        )

        # Keep duplicates as they might be anomalous patterns
        # pipeline.add_duplicate_removal(enabled=False)

        # Handle infinite values
        pipeline.add_infinite_value_handling(
            strategy="replace", description="Replace infinite values"
        )

        # More conservative outlier handling - don't remove potential anomalies
        pipeline.add_outlier_handling(
            strategy=OutlierStrategy.WINSORIZE,
            method="iqr",
            threshold=3.0,  # More conservative threshold
            description="Winsorize extreme outliers only",
        )

        # Handle zero values carefully
        pipeline.add_zero_value_handling(
            strategy="keep",  # Keep zeros as they might be meaningful
            description="Keep zero values as they may be anomalous",
        )

        # Encode categorical features
        pipeline.add_categorical_encoding(
            strategy=EncodingStrategy.LABEL,  # Simpler for anomaly detection
            description="Label encode categorical features",
        )

        # Robust scaling for anomaly detection
        pipeline.add_scaling(
            strategy=ScalingStrategy.ROBUST,  # Less sensitive to outliers
            description="Robust scale features for anomaly detection",
        )

        # Optional feature selection with high threshold
        pipeline.add_feature_selection(
            strategy=FeatureSelectionStrategy.VARIANCE_THRESHOLD,
            threshold=0.001,  # Very low threshold to keep most features
            enabled=False,  # Disabled by default
            description="Remove only constant features",
        )

        return pipeline

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        enabled_steps = [step.name for step in self.steps if step.enabled]
        return f"PreprocessingPipeline(name='{self.name}', steps={len(self.steps)}, enabled={len(enabled_steps)})"
