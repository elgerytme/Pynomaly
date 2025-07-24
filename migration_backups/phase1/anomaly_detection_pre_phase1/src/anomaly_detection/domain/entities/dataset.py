"""Dataset entity for managing and validating data used in anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Any, Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum


class DatasetType(Enum):
    """Types of datasets for anomaly detection."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    INFERENCE = "inference"


class DataFormat(Enum):
    """Supported data formats."""
    NUMPY = "numpy"
    PANDAS = "pandas"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


@dataclass
class DatasetMetadata:
    """Metadata about a dataset."""
    name: str
    description: str = ""
    source: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    feature_names: Optional[List[str]] = None
    target_column: Optional[str] = None
    contamination_rate: Optional[float] = None


@dataclass
class Dataset:
    """Container for dataset used in anomaly detection.
    
    Manages data, metadata, validation, and provides utilities
    for preprocessing and analysis.
    """
    
    data: Union[npt.NDArray[np.floating], pd.DataFrame]
    dataset_type: DatasetType = DatasetType.INFERENCE
    metadata: Optional[DatasetMetadata] = None
    labels: Optional[npt.NDArray[np.integer]] = None
    feature_names: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Initialize derived fields after object creation."""
        if self.metadata is None:
            self.metadata = DatasetMetadata(name="unknown_dataset")
            
        # Extract feature names if not provided
        if self.feature_names is None:
            if isinstance(self.data, pd.DataFrame):
                self.feature_names = list(self.data.columns)
            else:
                n_features = self.data.shape[1] if len(self.data.shape) > 1 else 1
                self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Update metadata feature names
        if self.metadata.feature_names is None:
            self.metadata.feature_names = self.feature_names
        
        # Calculate contamination rate if labels are provided
        if self.labels is not None and self.metadata.contamination_rate is None:
            anomaly_count = np.sum(self.labels == -1)
            total_count = len(self.labels)
            self.metadata.contamination_rate = anomaly_count / total_count if total_count > 0 else 0.0
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of the data."""
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        """Get number of samples in dataset."""
        return len(self.data)
    
    @property
    def n_features(self) -> int:
        """Get number of features in dataset."""
        return self.data.shape[1] if len(self.data.shape) > 1 else 1
    
    @property
    def has_labels(self) -> bool:
        """Check if dataset has ground truth labels."""
        return self.labels is not None
    
    def to_numpy(self) -> npt.NDArray[np.floating]:
        """Convert data to NumPy array."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.values.astype(np.float64)
        return self.data.astype(np.float64)
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert data to Pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(self.n_features)]
        return pd.DataFrame(self.data, columns=feature_names)
    
    def validate(self) -> List[str]:
        """Validate dataset and return list of issues found."""
        issues = []
        
        # Check for empty data
        if self.n_samples == 0:
            issues.append("Dataset is empty")
            return issues
        
        # Check for missing values
        if isinstance(self.data, pd.DataFrame):
            if self.data.isnull().any().any():
                null_cols = self.data.columns[self.data.isnull().any()].tolist()
                issues.append(f"Missing values found in columns: {null_cols}")
        else:
            if np.isnan(self.data).any():
                issues.append("Missing values (NaN) found in data")
        
        # Check for infinite values
        if isinstance(self.data, pd.DataFrame):
            if np.isinf(self.data.select_dtypes(include=[np.number])).any().any():
                issues.append("Infinite values found in data")
        else:
            if np.isinf(self.data).any():
                issues.append("Infinite values found in data")
        
        # Check data types
        if isinstance(self.data, pd.DataFrame):
            non_numeric = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                issues.append(f"Non-numeric columns found: {non_numeric}")
        
        # Check label consistency
        if self.has_labels:
            unique_labels = np.unique(self.labels)
            expected_labels = {-1, 1}  # Anomaly detection labels
            if not set(unique_labels).issubset(expected_labels):
                issues.append(f"Invalid labels found. Expected -1 (anomaly) and 1 (normal), got: {unique_labels}")
            
            if len(self.labels) != self.n_samples:
                issues.append(f"Label count ({len(self.labels)}) doesn't match sample count ({self.n_samples})")
        
        # Check feature names consistency
        if len(self.feature_names) != self.n_features:
            issues.append(f"Feature names count ({len(self.feature_names)}) doesn't match feature count ({self.n_features})")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        df = self.to_pandas()
        
        stats = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "data_types": df.dtypes.to_dict() if isinstance(df.dtypes, pd.Series) else str(df.dtypes),
            "missing_values": df.isnull().sum().to_dict() if hasattr(df.isnull().sum(), 'to_dict') else int(df.isnull().sum()),
            "has_labels": self.has_labels
        }
        
        if self.has_labels:
            stats.update({
                "normal_count": int(np.sum(self.labels == 1)),
                "anomaly_count": int(np.sum(self.labels == -1)),
                "contamination_rate": self.metadata.contamination_rate or 0.0
            })
        
        # Add numeric statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats["numeric_stats"] = {
                "mean": numeric_df.mean().to_dict(),
                "std": numeric_df.std().to_dict(),
                "min": numeric_df.min().to_dict(),
                "max": numeric_df.max().to_dict()
            }
        
        return stats
    
    def sample(self, n: int, random_state: Optional[int] = None) -> Dataset:
        """Create a random sample of the dataset."""
        if n >= self.n_samples:
            return self
        
        np.random.seed(random_state)
        indices = np.random.choice(self.n_samples, size=n, replace=False)
        
        if isinstance(self.data, pd.DataFrame):
            sampled_data = self.data.iloc[indices].copy()
        else:
            sampled_data = self.data[indices]
        
        sampled_labels = self.labels[indices] if self.has_labels else None
        
        sampled_metadata = DatasetMetadata(
            name=f"{self.metadata.name}_sample_{n}",
            description=f"Sample of {n} records from {self.metadata.name}",
            source=self.metadata.source,
            feature_names=self.metadata.feature_names,
            target_column=self.metadata.target_column
        )
        
        return Dataset(
            data=sampled_data,
            dataset_type=self.dataset_type,
            metadata=sampled_metadata,
            labels=sampled_labels,
            feature_names=self.feature_names
        )
    
    @classmethod
    def from_csv(cls, file_path: Union[str, Path], **kwargs) -> Dataset:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path, **kwargs)
        
        metadata = DatasetMetadata(
            name=Path(file_path).stem,
            source=str(file_path),
            description=f"Dataset loaded from {file_path}"
        )
        
        return cls(
            data=df,
            metadata=metadata,
            dataset_type=DatasetType.INFERENCE
        )
    
    @classmethod
    def from_numpy(cls, data: npt.NDArray[np.floating], 
                   labels: Optional[npt.NDArray[np.integer]] = None,
                   feature_names: Optional[List[str]] = None,
                   name: str = "numpy_dataset") -> Dataset:
        """Create dataset from NumPy arrays."""
        metadata = DatasetMetadata(
            name=name,
            description="Dataset created from NumPy arrays"
        )
        
        return cls(
            data=data,
            labels=labels,
            feature_names=feature_names,
            metadata=metadata,
            dataset_type=DatasetType.INFERENCE
        )
    
    def __len__(self) -> int:
        """Get number of samples."""
        return self.n_samples
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Dataset(name='{self.metadata.name}', shape={self.shape}, "
            f"type={self.dataset_type.value}, has_labels={self.has_labels})"
        )