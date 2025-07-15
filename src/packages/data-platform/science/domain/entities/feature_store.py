"""Feature Store entity for feature management and lineage."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class FeatureType(str, Enum):
    """Types of features."""
    
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    TEXT = "text"
    DATETIME = "datetime"
    GEOSPATIAL = "geospatial"
    IMAGE = "image"
    AUDIO = "audio"
    EMBEDDED = "embedded"
    DERIVED = "derived"
    AGGREGATED = "aggregated"
    TRANSFORMED = "transformed"


class FeatureStatus(str, Enum):
    """Feature lifecycle status."""
    
    DRAFT = "draft"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"
    EXPERIMENTAL = "experimental"
    PRODUCTION = "production"


class ComputeMode(str, Enum):
    """Feature computation modes."""
    
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"
    PRECOMPUTED = "precomputed"
    HYBRID = "hybrid"


class FeatureStore(BaseEntity):
    """Entity representing a feature store for ML feature management.
    
    This entity manages the complete lifecycle of ML features including
    definition, computation, storage, serving, and lineage tracking.
    
    Attributes:
        name: Human-readable name for the feature store
        description: Detailed description of the feature store
        namespace: Logical grouping namespace
        features: Dictionary of features with metadata
        feature_groups: Groups of related features
        schemas: Feature schema definitions
        transformations: Feature transformation definitions
        serving_config: Configuration for feature serving
        storage_config: Feature storage configuration
        compute_config: Feature computation configuration
        monitoring_config: Feature monitoring and alerting
        access_control: Feature access control settings
        lineage_info: Feature lineage and dependency tracking
        data_sources: Source datasets and tables
        refresh_schedule: Feature refresh scheduling
        validation_rules: Feature validation constraints
        quality_metrics: Feature quality measurements
        usage_statistics: Feature usage analytics
        cost_tracking: Feature computation cost tracking
        version_history: Feature versioning information
        tags: Searchable tags for organization
        owner: Owner of the feature store
        collaborators: Users with access to the feature store
        created_by: User who created the feature store
        last_updated_by: User who last updated the feature store
        approval_required: Whether changes require approval
        compliance_info: Compliance and governance metadata
    """
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    namespace: str = Field(..., min_length=1, max_length=100)
    
    # Feature definitions
    features: dict[str, dict[str, Any]] = Field(default_factory=dict)
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    transformations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    
    # Infrastructure configuration
    serving_config: dict[str, Any] = Field(default_factory=dict)
    storage_config: dict[str, Any] = Field(default_factory=dict)
    compute_config: dict[str, Any] = Field(default_factory=dict)
    monitoring_config: dict[str, Any] = Field(default_factory=dict)
    
    # Data governance
    access_control: dict[str, Any] = Field(default_factory=dict)
    lineage_info: dict[str, Any] = Field(default_factory=dict)
    data_sources: list[dict[str, Any]] = Field(default_factory=list)
    validation_rules: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    
    # Operations
    refresh_schedule: dict[str, Any] = Field(default_factory=dict)
    quality_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    usage_statistics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cost_tracking: dict[str, dict[str, float]] = Field(default_factory=dict)
    
    # Versioning and collaboration
    version_history: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    owner: Optional[str] = None
    collaborators: list[str] = Field(default_factory=list)
    created_by: Optional[str] = None
    last_updated_by: Optional[str] = None
    
    # Governance
    approval_required: bool = Field(default=False)
    compliance_info: dict[str, Any] = Field(default_factory=dict)
    
    @validator('features')
    def validate_features(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Validate feature definitions."""
        for feature_name, feature_def in v.items():
            if not feature_name.strip():
                raise ValueError("Feature names cannot be empty")
                
            required_fields = ["type", "description"]
            for field in required_fields:
                if field not in feature_def:
                    raise ValueError(f"Feature '{feature_name}' missing required field: {field}")
                    
            # Validate feature type
            try:
                FeatureType(feature_def["type"])
            except ValueError:
                raise ValueError(f"Invalid feature type for '{feature_name}': {feature_def['type']}")
                
        return v
    
    @validator('feature_groups')
    def validate_feature_groups(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        """Validate feature groups."""
        for group_name, feature_list in v.items():
            if not group_name.strip():
                raise ValueError("Feature group names cannot be empty")
                
            if not feature_list:
                raise ValueError(f"Feature group '{group_name}' cannot be empty")
                
            # Check for duplicates
            if len(feature_list) != len(set(feature_list)):
                raise ValueError(f"Feature group '{group_name}' contains duplicate features")
                
        return v
    
    @validator('tags')
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags list."""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    def add_feature(self, name: str, feature_type: FeatureType, 
                   description: str, configuration: dict[str, Any]) -> None:
        """Add a new feature to the store."""
        if name in self.features:
            raise ValueError(f"Feature '{name}' already exists")
            
        feature_def = {
            "type": feature_type.value,
            "description": description,
            "status": FeatureStatus.DRAFT.value,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            **configuration
        }
        
        self.features[name] = feature_def
        self._add_to_version_history("add_feature", {"feature_name": name})
        self.mark_as_updated()
    
    def update_feature(self, name: str, updates: dict[str, Any], 
                      updated_by: str) -> None:
        """Update an existing feature."""
        if name not in self.features:
            raise ValueError(f"Feature '{name}' does not exist")
            
        self.features[name].update(updates)
        self.features[name]["updated_at"] = datetime.utcnow().isoformat()
        self.features[name]["updated_by"] = updated_by
        
        # Increment version if this is a significant update
        if any(key in updates for key in ["type", "transformation", "source"]):
            self._increment_feature_version(name)
            
        self.last_updated_by = updated_by
        self._add_to_version_history("update_feature", {
            "feature_name": name,
            "updates": list(updates.keys())
        })
        self.mark_as_updated()
    
    def remove_feature(self, name: str, removed_by: str) -> None:
        """Remove a feature from the store."""
        if name not in self.features:
            raise ValueError(f"Feature '{name}' does not exist")
            
        # Move to archived status first, then remove
        self.features[name]["status"] = FeatureStatus.ARCHIVED.value
        self.features[name]["archived_at"] = datetime.utcnow().isoformat()
        self.features[name]["archived_by"] = removed_by
        
        # Remove from feature groups
        for group_name, feature_list in self.feature_groups.items():
            if name in feature_list:
                feature_list.remove(name)
                
        del self.features[name]
        
        self.last_updated_by = removed_by
        self._add_to_version_history("remove_feature", {"feature_name": name})
        self.mark_as_updated()
    
    def create_feature_group(self, group_name: str, feature_names: list[str],
                           description: Optional[str] = None) -> None:
        """Create a new feature group."""
        if group_name in self.feature_groups:
            raise ValueError(f"Feature group '{group_name}' already exists")
            
        # Validate all features exist
        missing_features = set(feature_names) - set(self.features.keys())
        if missing_features:
            raise ValueError(f"Unknown features: {missing_features}")
            
        self.feature_groups[group_name] = feature_names
        
        if description:
            self.metadata[f"group_{group_name}_description"] = description
            
        self._add_to_version_history("create_group", {
            "group_name": group_name,
            "feature_count": len(feature_names)
        })
        self.mark_as_updated()
    
    def add_feature_to_group(self, group_name: str, feature_name: str) -> None:
        """Add a feature to an existing group."""
        if group_name not in self.feature_groups:
            raise ValueError(f"Feature group '{group_name}' does not exist")
            
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        if feature_name not in self.feature_groups[group_name]:
            self.feature_groups[group_name].append(feature_name)
            self.mark_as_updated()
    
    def remove_feature_from_group(self, group_name: str, feature_name: str) -> None:
        """Remove a feature from a group."""
        if group_name not in self.feature_groups:
            raise ValueError(f"Feature group '{group_name}' does not exist")
            
        if feature_name in self.feature_groups[group_name]:
            self.feature_groups[group_name].remove(feature_name)
            self.mark_as_updated()
    
    def set_feature_status(self, feature_name: str, status: FeatureStatus,
                          updated_by: str) -> None:
        """Update feature status."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        old_status = self.features[feature_name].get("status")
        self.features[feature_name]["status"] = status.value
        self.features[feature_name]["status_updated_at"] = datetime.utcnow().isoformat()
        self.features[feature_name]["status_updated_by"] = updated_by
        
        self._add_to_version_history("status_change", {
            "feature_name": feature_name,
            "old_status": old_status,
            "new_status": status.value
        })
        self.mark_as_updated()
    
    def add_validation_rule(self, feature_name: str, rule: dict[str, Any]) -> None:
        """Add validation rule for a feature."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        if feature_name not in self.validation_rules:
            self.validation_rules[feature_name] = []
            
        rule["created_at"] = datetime.utcnow().isoformat()
        self.validation_rules[feature_name].append(rule)
        self.mark_as_updated()
    
    def record_quality_metric(self, feature_name: str, metric_name: str,
                            value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a quality metric for a feature."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        if feature_name not in self.quality_metrics:
            self.quality_metrics[feature_name] = {}
            
        timestamp_str = (timestamp or datetime.utcnow()).isoformat()
        self.quality_metrics[feature_name][metric_name] = {
            "value": value,
            "recorded_at": timestamp_str
        }
        self.mark_as_updated()
    
    def record_usage(self, feature_name: str, usage_type: str,
                    metadata: Optional[dict[str, Any]] = None) -> None:
        """Record feature usage statistics."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        if feature_name not in self.usage_statistics:
            self.usage_statistics[feature_name] = {
                "total_requests": 0,
                "usage_by_type": {},
                "first_used": datetime.utcnow().isoformat(),
                "last_used": datetime.utcnow().isoformat()
            }
            
        stats = self.usage_statistics[feature_name]
        stats["total_requests"] += 1
        stats["last_used"] = datetime.utcnow().isoformat()
        
        if usage_type not in stats["usage_by_type"]:
            stats["usage_by_type"][usage_type] = 0
        stats["usage_by_type"][usage_type] += 1
        
        if metadata:
            stats[f"last_{usage_type}_metadata"] = metadata
            
        self.mark_as_updated()
    
    def add_lineage_dependency(self, feature_name: str, 
                             dependency: dict[str, Any]) -> None:
        """Add lineage dependency for a feature."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        if "dependencies" not in self.lineage_info:
            self.lineage_info["dependencies"] = {}
            
        if feature_name not in self.lineage_info["dependencies"]:
            self.lineage_info["dependencies"][feature_name] = []
            
        dependency["added_at"] = datetime.utcnow().isoformat()
        self.lineage_info["dependencies"][feature_name].append(dependency)
        self.mark_as_updated()
    
    def get_feature_dependencies(self, feature_name: str) -> list[dict[str, Any]]:
        """Get dependencies for a feature."""
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' does not exist")
            
        return self.lineage_info.get("dependencies", {}).get(feature_name, [])
    
    def get_features_by_status(self, status: FeatureStatus) -> list[str]:
        """Get features by status."""
        return [
            name for name, feature_def in self.features.items()
            if feature_def.get("status") == status.value
        ]
    
    def get_features_by_type(self, feature_type: FeatureType) -> list[str]:
        """Get features by type."""
        return [
            name for name, feature_def in self.features.items()
            if feature_def.get("type") == feature_type.value
        ]
    
    def _increment_feature_version(self, feature_name: str) -> None:
        """Increment feature version."""
        current_version = self.features[feature_name].get("version", "1.0.0")
        try:
            major, minor, patch = map(int, current_version.split("."))
            new_version = f"{major}.{minor}.{patch + 1}"
            self.features[feature_name]["version"] = new_version
        except ValueError:
            self.features[feature_name]["version"] = "1.0.1"
    
    def _add_to_version_history(self, action: str, details: dict[str, Any]) -> None:
        """Add entry to version history."""
        entry = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "user": self.last_updated_by,
            "details": details
        }
        self.version_history.append(entry)
        
        # Keep only last 100 entries
        if len(self.version_history) > 100:
            self.version_history = self.version_history[-100:]
    
    def get_store_summary(self) -> dict[str, Any]:
        """Get feature store summary."""
        total_features = len(self.features)
        active_features = len(self.get_features_by_status(FeatureStatus.ACTIVE))
        production_features = len(self.get_features_by_status(FeatureStatus.PRODUCTION))
        
        feature_types = {}
        for feature_def in self.features.values():
            ftype = feature_def.get("type", "unknown")
            feature_types[ftype] = feature_types.get(ftype, 0) + 1
        
        return {
            "store_id": str(self.id),
            "name": self.name,
            "namespace": self.namespace,
            "total_features": total_features,
            "active_features": active_features,
            "production_features": production_features,
            "feature_groups": len(self.feature_groups),
            "feature_types": feature_types,
            "data_sources": len(self.data_sources),
            "owner": self.owner,
            "collaborators_count": len(self.collaborators),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.updated_at.isoformat(),
            "version_entries": len(self.version_history),
        }
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Feature groups can only contain existing features
        for group_name, feature_list in self.feature_groups.items():
            missing_features = set(feature_list) - set(self.features.keys())
            if missing_features:
                raise ValueError(f"Group '{group_name}' references non-existent features: {missing_features}")
        
        # Business rule: Production features must have validation rules
        production_features = self.get_features_by_status(FeatureStatus.PRODUCTION)
        for feature_name in production_features:
            if feature_name not in self.validation_rules:
                raise ValueError(f"Production feature '{feature_name}' must have validation rules")
        
        # Business rule: Feature store must have an owner
        if not self.owner and len(self.features) > 0:
            raise ValueError("Feature store with features must have an owner")