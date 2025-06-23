"""Dataset entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from pynomaly.domain.exceptions import InvalidDataError


@dataclass
class Dataset:
    """Entity representing a dataset for anomaly detection.
    
    Attributes:
        id: Unique identifier for the dataset
        name: Name of the dataset
        data: The actual data (DataFrame or array)
        feature_names: Names of features in the dataset
        metadata: Additional metadata about the dataset
        created_at: When the dataset was created
        description: Optional description of the dataset
        target_column: Optional target/label column name
    """
    
    name: str
    data: Union[pd.DataFrame, np.ndarray]
    id: UUID = field(default_factory=uuid4)
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    target_column: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and process dataset after initialization."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        
        # Convert to DataFrame if numpy array
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)
            
            if self.feature_names:
                if len(self.feature_names) != self.data.shape[1]:
                    raise ValueError(
                        f"Number of feature names ({len(self.feature_names)}) "
                        f"doesn't match data dimensions ({self.data.shape[1]})"
                    )
                self.data = pd.DataFrame(self.data, columns=self.feature_names)
            else:
                # Generate default feature names
                self.feature_names = [f"feature_{i}" for i in range(self.data.shape[1])]
                self.data = pd.DataFrame(self.data, columns=self.feature_names)
        
        elif isinstance(self.data, pd.DataFrame):
            if self.data.empty:
                raise InvalidDataError("Dataset cannot be empty")
            if self.feature_names is None:
                self.feature_names = list(self.data.columns)
        else:
            raise TypeError(
                f"Data must be pandas DataFrame or numpy array, got {type(self.data)}"
            )
        
        # Validate target column if specified
        if self.target_column and self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )
    
    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of the dataset."""
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return self.shape[0]
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self.shape[1] - (1 if self.target_column else 0)
    
    @property
    def features(self) -> pd.DataFrame:
        """Get feature data (excluding target if present)."""
        if self.target_column:
            return self.data.drop(columns=[self.target_column])
        return self.data
    
    @property
    def target(self) -> Optional[pd.Series]:
        """Get target data if available."""
        if self.target_column:
            return self.data[self.target_column]
        return None
    
    @property
    def has_target(self) -> bool:
        """Check if dataset has target labels."""
        return self.target_column is not None
    
    @property
    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return int(self.data.memory_usage(deep=True).sum())
    
    @property
    def dtypes(self) -> pd.Series:
        """Get data types of columns."""
        return self.data.dtypes
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature names."""
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        return [col for col in self.feature_names or [] 
                if self.data[col].dtype in numeric_dtypes 
                and col != self.target_column]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        return [col for col in self.feature_names or []
                if self.data[col].dtype == 'object'
                and col != self.target_column]
    
    def sample(self, n: int, random_state: Optional[int] = None) -> Dataset:
        """Create a new dataset with random sample of rows."""
        if n > self.n_samples:
            raise ValueError(
                f"Cannot sample {n} rows from dataset with {self.n_samples} rows"
            )
        
        sampled_data = self.data.sample(n=n, random_state=random_state)
        
        return Dataset(
            name=f"{self.name}_sample_{n}",
            data=sampled_data,
            feature_names=self.feature_names,
            metadata={**self.metadata, "parent_dataset_id": str(self.id)},
            description=f"Sample of {n} rows from {self.name}",
            target_column=self.target_column
        )
    
    def split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> tuple[Dataset, Dataset]:
        """Split dataset into train and test sets."""
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        # Shuffle and split
        shuffled = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1 - test_size))
        
        train_data = shuffled.iloc[:split_idx]
        test_data = shuffled.iloc[split_idx:]
        
        train_dataset = Dataset(
            name=f"{self.name}_train",
            data=train_data,
            feature_names=self.feature_names,
            metadata={**self.metadata, "split": "train", "parent_dataset_id": str(self.id)},
            target_column=self.target_column
        )
        
        test_dataset = Dataset(
            name=f"{self.name}_test",
            data=test_data,
            feature_names=self.feature_names,
            metadata={**self.metadata, "split": "test", "parent_dataset_id": str(self.id)},
            target_column=self.target_column
        )
        
        return train_dataset, test_dataset
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the dataset."""
        self.metadata[key] = value
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        return {
            "id": str(self.id),
            "name": self.name,
            "shape": self.shape,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "memory_usage_mb": self.memory_usage / 1024 / 1024,
            "has_target": self.has_target,
            "numeric_features": len(self.get_numeric_features()),
            "categorical_features": len(self.get_categorical_features()),
            "created_at": self.created_at.isoformat(),
            "description": self.description,
        }
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Dataset(id={self.id}, name='{self.name}', "
            f"shape={self.shape}, has_target={self.has_target})"
        )