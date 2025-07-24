"""Feature store service for consistent feature management and data access."""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

try:
    from data.processing.domain.entities.dataset_entity import DatasetEntity
except ImportError:
    from anomaly_detection.domain.entities.dataset_entity import DatasetEntity


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    EMBEDDING = "embedding"


class AggregationType(Enum):
    """Feature aggregation types."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    STDDEV = "stddev"
    PERCENTILE = "percentile"


@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    feature_id: str
    name: str
    description: str
    feature_type: FeatureType
    source_table: str
    source_column: str
    transformation_logic: Optional[str] = None
    aggregation_type: Optional[AggregationType] = None
    aggregation_window: Optional[str] = None  # e.g., "1h", "1d", "7d"
    default_value: Optional[Any] = None
    validation_rules: List[str] = None
    tags: Dict[str, str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []
        if self.tags is None:
            self.tags = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "transformation_logic": self.transformation_logic,
            "aggregation_type": self.aggregation_type.value if self.aggregation_type else None,
            "aggregation_window": self.aggregation_window,
            "default_value": self.default_value,
            "validation_rules": self.validation_rules,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class FeatureGroup:
    """Group of related features."""
    group_id: str
    name: str
    description: str
    features: List[str]  # List of feature IDs
    entity_keys: List[str]  # Keys used to join features (e.g., user_id, timestamp)
    serving_endpoint: Optional[str] = None
    batch_source: Optional[str] = None
    streaming_source: Optional[str] = None
    refresh_interval: Optional[str] = None  # e.g., "1h", "1d"
    tags: Dict[str, str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "entity_keys": self.entity_keys,
            "serving_endpoint": self.serving_endpoint,
            "batch_source": self.batch_source,
            "streaming_source": self.streaming_source,
            "refresh_interval": self.refresh_interval,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }


@dataclass  
class FeatureVector:
    """Feature vector for a specific entity and timestamp."""
    entity_id: str
    timestamp: datetime
    features: Dict[str, Any]
    feature_group_id: str
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "entity_id": self.entity_id,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "feature_group_id": self.feature_group_id,
            "version": self.version
        }


class FeatureStore(ABC):
    """Abstract base class for feature store implementations."""
    
    @abstractmethod
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition."""
        pass
    
    @abstractmethod
    def register_feature_group(self, feature_group: FeatureGroup) -> bool:
        """Register a new feature group."""
        pass
    
    @abstractmethod
    def get_features_online(self, 
                           entity_ids: List[str],
                           feature_names: List[str],
                           timestamp: Optional[datetime] = None) -> List[FeatureVector]:
        """Get features for online serving."""
        pass
    
    @abstractmethod
    def get_features_batch(self,
                          entity_ids: List[str],
                          feature_names: List[str],
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Get features for batch processing."""
        pass
    
    @abstractmethod
    def ingest_features(self, feature_vectors: List[FeatureVector]) -> bool:
        """Ingest feature vectors into the store."""
        pass


class InMemoryFeatureStore(FeatureStore):
    """In-memory implementation of feature store for development/testing."""
    
    def __init__(self):
        """Initialize in-memory feature store."""
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self._features: Dict[str, FeatureDefinition] = {}
        self._feature_groups: Dict[str, FeatureGroup] = {}
        self._feature_vectors: Dict[str, List[FeatureVector]] = {}  # entity_id -> vectors
    
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition."""
        try:
            self._features[feature_def.feature_id] = feature_def
            self.logger.info(f"Registered feature: {feature_def.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register feature {feature_def.name}: {e}")
            return False
    
    def register_feature_group(self, feature_group: FeatureGroup) -> bool:
        """Register a new feature group."""
        try:
            # Validate that all features in the group exist
            for feature_id in feature_group.features:
                if feature_id not in self._features:
                    raise ValueError(f"Feature {feature_id} not found")
            
            self._feature_groups[feature_group.group_id] = feature_group
            self.logger.info(f"Registered feature group: {feature_group.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register feature group {feature_group.name}: {e}")
            return False
    
    def get_features_online(self,
                           entity_ids: List[str],
                           feature_names: List[str],
                           timestamp: Optional[datetime] = None) -> List[FeatureVector]:
        """Get features for online serving."""
        result_vectors = []
        
        for entity_id in entity_ids:
            if entity_id not in self._feature_vectors:
                continue
            
            # Get latest vector for entity (or at specific timestamp)
            entity_vectors = self._feature_vectors[entity_id]
            
            if timestamp:
                # Find vector closest to timestamp
                closest_vector = min(
                    entity_vectors,
                    key=lambda v: abs((v.timestamp - timestamp).total_seconds())
                )
            else:
                # Get latest vector
                closest_vector = max(entity_vectors, key=lambda v: v.timestamp)
            
            # Filter features
            filtered_features = {
                name: closest_vector.features.get(name)
                for name in feature_names
                if name in closest_vector.features
            }
            
            result_vector = FeatureVector(
                entity_id=entity_id,
                timestamp=closest_vector.timestamp,
                features=filtered_features,
                feature_group_id=closest_vector.feature_group_id,
                version=closest_vector.version
            )
            
            result_vectors.append(result_vector)
        
        return result_vectors
    
    def get_features_batch(self,
                          entity_ids: List[str],
                          feature_names: List[str],
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Get features for batch processing."""
        rows = []
        
        for entity_id in entity_ids:
            if entity_id not in self._feature_vectors:
                continue
            
            entity_vectors = self._feature_vectors[entity_id]
            
            # Filter by time range
            time_filtered_vectors = [
                v for v in entity_vectors
                if start_time <= v.timestamp <= end_time
            ]
            
            for vector in time_filtered_vectors:
                row = {
                    "entity_id": entity_id,
                    "timestamp": vector.timestamp
                }
                
                # Add requested features
                for feature_name in feature_names:
                    row[feature_name] = vector.features.get(feature_name)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def ingest_features(self, feature_vectors: List[FeatureVector]) -> bool:
        """Ingest feature vectors into the store."""
        try:
            for vector in feature_vectors:
                entity_id = vector.entity_id
                
                if entity_id not in self._feature_vectors:
                    self._feature_vectors[entity_id] = []
                
                self._feature_vectors[entity_id].append(vector)
            
            self.logger.info(f"Ingested {len(feature_vectors)} feature vectors")
            return True
        except Exception as e:
            self.logger.error(f"Failed to ingest feature vectors: {e}")
            return False


class FeatureStoreService:
    """Service for feature store operations."""
    
    def __init__(self, feature_store: FeatureStore):
        """Initialize feature store service.
        
        Args:
            feature_store: Feature store implementation
        """
        self.feature_store = feature_store
        self.logger = logging.getLogger(__name__)
        
        # Feature transformations
        self._transformations: Dict[str, Callable] = {}
        
        # Feature validators
        self._validators: Dict[str, List[Callable]] = {}
    
    def register_transformation(self, name: str, transformation_func: Callable):
        """Register a feature transformation function.
        
        Args:
            name: Name of the transformation
            transformation_func: Function that transforms input data
        """
        self._transformations[name] = transformation_func
        self.logger.info(f"Registered transformation: {name}")
    
    def register_validator(self, feature_name: str, validator_func: Callable):
        """Register a feature validation function.
        
        Args:
            feature_name: Name of the feature
            validator_func: Function that validates feature values
        """
        if feature_name not in self._validators:
            self._validators[feature_name] = []
        
        self._validators[feature_name].append(validator_func)
        self.logger.info(f"Registered validator for feature: {feature_name}")
    
    def create_feature_definition(self,
                                 name: str,
                                 description: str,
                                 feature_type: FeatureType,
                                 source_table: str,
                                 source_column: str,
                                 **kwargs) -> FeatureDefinition:
        """Create a new feature definition.
        
        Args:
            name: Feature name
            description: Feature description
            feature_type: Type of feature
            source_table: Source table name
            source_column: Source column name
            **kwargs: Additional arguments
            
        Returns:
            FeatureDefinition object
        """
        feature_id = str(uuid.uuid4())
        
        feature_def = FeatureDefinition(
            feature_id=feature_id,
            name=name,
            description=description,
            feature_type=feature_type,
            source_table=source_table,
            source_column=source_column,
            **kwargs
        )
        
        return feature_def
    
    def create_feature_group(self,
                            name: str,
                            description: str,
                            feature_definitions: List[FeatureDefinition],
                            entity_keys: List[str],
                            **kwargs) -> FeatureGroup:
        """Create a new feature group.
        
        Args:
            name: Group name
            description: Group description
            feature_definitions: List of feature definitions
            entity_keys: Entity keys for joining
            **kwargs: Additional arguments
            
        Returns:
            FeatureGroup object
        """
        group_id = str(uuid.uuid4())
        
        # Register features first
        feature_ids = []
        for feature_def in feature_definitions:
            self.feature_store.register_feature(feature_def)
            feature_ids.append(feature_def.feature_id)
        
        feature_group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            features=feature_ids,
            entity_keys=entity_keys,
            **kwargs
        )
        
        return feature_group
    
    def compute_features_from_dataset(self,
                                    dataset: DatasetEntity,
                                    feature_definitions: List[FeatureDefinition],
                                    entity_column: str = "entity_id",
                                    timestamp_column: Optional[str] = None) -> List[FeatureVector]:
        """Compute features from a dataset.
        
        Args:
            dataset: Source dataset
            feature_definitions: Features to compute
            entity_column: Column containing entity IDs
            timestamp_column: Optional timestamp column
            
        Returns:
            List of computed feature vectors
        """
        # Convert dataset to DataFrame for processing
        if hasattr(dataset.data, 'to_dataframe'):
            df = dataset.data.to_dataframe()
        else:
            df = pd.DataFrame(dataset.data)
        
        feature_vectors = []
        
        # Group by entity
        for entity_id, entity_df in df.groupby(entity_column):
            if timestamp_column:
                # Process each timestamp
                for _, row in entity_df.iterrows():
                    timestamp = pd.to_datetime(row[timestamp_column])
                    features = self._compute_features_for_row(row, feature_definitions)
                    
                    vector = FeatureVector(
                        entity_id=str(entity_id),
                        timestamp=timestamp,
                        features=features,
                        feature_group_id="computed"
                    )
                    feature_vectors.append(vector)
            else:
                # Use current time and aggregate entity data
                timestamp = datetime.now()
                features = self._compute_features_for_entity(entity_df, feature_definitions)
                
                vector = FeatureVector(
                    entity_id=str(entity_id),
                    timestamp=timestamp,
                    features=features,
                    feature_group_id="computed"
                )
                feature_vectors.append(vector)
        
        return feature_vectors
    
    def _compute_features_for_row(self, 
                                 row: pd.Series,
                                 feature_definitions: List[FeatureDefinition]) -> Dict[str, Any]:
        """Compute features for a single row.
        
        Args:
            row: Data row
            feature_definitions: Features to compute
            
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        for feature_def in feature_definitions:
            try:
                # Get base value
                if feature_def.source_column in row:
                    value = row[feature_def.source_column]
                else:
                    value = feature_def.default_value
                
                # Apply transformation if specified
                if feature_def.transformation_logic and feature_def.transformation_logic in self._transformations:
                    transformation_func = self._transformations[feature_def.transformation_logic]
                    value = transformation_func(value)
                
                # Validate feature value
                if feature_def.name in self._validators:
                    for validator in self._validators[feature_def.name]:
                        if not validator(value):
                            self.logger.warning(f"Feature {feature_def.name} failed validation")
                            value = feature_def.default_value
                            break
                
                features[feature_def.name] = value
                
            except Exception as e:
                self.logger.error(f"Error computing feature {feature_def.name}: {e}")
                features[feature_def.name] = feature_def.default_value
        
        return features
    
    def _compute_features_for_entity(self,
                                   entity_df: pd.DataFrame,
                                   feature_definitions: List[FeatureDefinition]) -> Dict[str, Any]:
        """Compute aggregated features for an entity.
        
        Args:
            entity_df: DataFrame for a single entity
            feature_definitions: Features to compute
            
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        for feature_def in feature_definitions:
            try:
                if feature_def.source_column not in entity_df.columns:
                    features[feature_def.name] = feature_def.default_value
                    continue
                
                column_data = entity_df[feature_def.source_column]
                
                # Apply aggregation if specified
                if feature_def.aggregation_type:
                    if feature_def.aggregation_type == AggregationType.SUM:
                        value = column_data.sum()
                    elif feature_def.aggregation_type == AggregationType.MEAN:
                        value = column_data.mean()
                    elif feature_def.aggregation_type == AggregationType.MEDIAN:
                        value = column_data.median()
                    elif feature_def.aggregation_type == AggregationType.MIN:
                        value = column_data.min()
                    elif feature_def.aggregation_type == AggregationType.MAX:
                        value = column_data.max()
                    elif feature_def.aggregation_type == AggregationType.COUNT:
                        value = column_data.count()
                    elif feature_def.aggregation_type == AggregationType.COUNT_DISTINCT:
                        value = column_data.nunique()
                    elif feature_def.aggregation_type == AggregationType.STDDEV:
                        value = column_data.std()
                    else:
                        value = column_data.iloc[0] if len(column_data) > 0 else feature_def.default_value
                else:
                    # Use first value if no aggregation
                    value = column_data.iloc[0] if len(column_data) > 0 else feature_def.default_value
                
                # Apply transformation if specified
                if feature_def.transformation_logic and feature_def.transformation_logic in self._transformations:
                    transformation_func = self._transformations[feature_def.transformation_logic]
                    value = transformation_func(value)
                
                features[feature_def.name] = value
                
            except Exception as e:
                self.logger.error(f"Error computing feature {feature_def.name}: {e}")
                features[feature_def.name] = feature_def.default_value
        
        return features
    
    def get_feature_serving_data(self,
                               entity_ids: List[str],
                               feature_group_name: str,
                               timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Get feature data for model serving.
        
        Args:
            entity_ids: List of entity IDs
            feature_group_name: Name of feature group
            timestamp: Optional timestamp for point-in-time lookup
            
        Returns:
            DataFrame with features for serving
        """
        # Find feature group
        feature_group = None
        for group in self.feature_store._feature_groups.values():
            if group.name == feature_group_name:
                feature_group = group
                break
        
        if not feature_group:
            raise ValueError(f"Feature group '{feature_group_name}' not found")
        
        # Get feature names
        feature_names = []
        for feature_id in feature_group.features:
            if feature_id in self.feature_store._features:
                feature_names.append(self.feature_store._features[feature_id].name)
        
        # Get feature vectors
        feature_vectors = self.feature_store.get_features_online(
            entity_ids=entity_ids,
            feature_names=feature_names,
            timestamp=timestamp
        )
        
        # Convert to DataFrame
        rows = []
        for vector in feature_vectors:
            row = {"entity_id": vector.entity_id, "timestamp": vector.timestamp}
            row.update(vector.features)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_training_dataset(self,
                              entity_ids: List[str],
                              feature_group_names: List[str],
                              start_time: datetime,
                              end_time: datetime,
                              target_column: Optional[str] = None) -> pd.DataFrame:
        """Create a training dataset with features.
        
        Args:
            entity_ids: List of entity IDs
            feature_group_names: Names of feature groups to include
            start_time: Start time for data
            end_time: End time for data
            target_column: Optional target column name
            
        Returns:
            Training dataset DataFrame
        """
        all_features = []
        
        # Collect features from all groups
        for group_name in feature_group_names:
            feature_group = None
            for group in self.feature_store._feature_groups.values():
                if group.name == group_name:
                    feature_group = group
                    break
            
            if not feature_group:
                self.logger.warning(f"Feature group '{group_name}' not found")
                continue
            
            # Get feature names
            feature_names = []
            for feature_id in feature_group.features:
                if feature_id in self.feature_store._features:
                    feature_names.append(self.feature_store._features[feature_id].name)
            
            all_features.extend(feature_names)
        
        # Remove duplicates
        all_features = list(set(all_features))
        
        # Get batch features
        df = self.feature_store.get_features_batch(
            entity_ids=entity_ids,
            feature_names=all_features,
            start_time=start_time,
            end_time=end_time
        )
        
        return df
    
    def validate_feature_quality(self, feature_vectors: List[FeatureVector]) -> Dict[str, Any]:
        """Validate the quality of feature vectors.
        
        Args:
            feature_vectors: List of feature vectors to validate
            
        Returns:
            Quality metrics and issues
        """
        if not feature_vectors:
            return {"error": "No feature vectors provided"}
        
        quality_report = {
            "total_vectors": len(feature_vectors),
            "feature_coverage": {},
            "missing_values": {},
            "data_types": {},
            "validation_errors": []
        }
        
        # Collect all feature names
        all_features = set()
        for vector in feature_vectors:
            all_features.update(vector.features.keys())
        
        # Analyze each feature
        for feature_name in all_features:
            values = [vector.features.get(feature_name) for vector in feature_vectors]
            non_null_values = [v for v in values if v is not None]
            
            # Coverage
            coverage = len(non_null_values) / len(values) if values else 0
            quality_report["feature_coverage"][feature_name] = coverage
            
            # Missing values
            missing_count = len(values) - len(non_null_values)
            quality_report["missing_values"][feature_name] = missing_count
            
            # Data types
            if non_null_values:
                data_types = set(type(v).__name__ for v in non_null_values)
                quality_report["data_types"][feature_name] = list(data_types)
        
        return quality_report