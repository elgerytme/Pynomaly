"""Data asset entity."""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from packages.core.domain.abstractions.base_entity import BaseEntity
from ..value_objects.data_classification import DataClassification
from ..value_objects.data_schema import DataSchema


class AssetType(str, Enum):
    """Types of data assets."""
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    DASHBOARD = "dashboard"
    API_ENDPOINT = "api_endpoint"
    STREAM = "stream"
    COLLECTION = "collection"


class AssetStatus(str, Enum):
    """Data asset status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"


class DataAsset(BaseEntity):
    """Represents a logical data asset that can contain multiple datasets."""
    
    asset_id: UUID = Field(default_factory=uuid4, description="Unique asset identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Asset name")
    asset_type: AssetType = Field(..., description="Type of data asset")
    description: Optional[str] = Field(None, description="Asset description")
    business_purpose: Optional[str] = Field(None, description="Business purpose and value")
    status: AssetStatus = Field(default=AssetStatus.ACTIVE, description="Asset status")
    classification: Optional[DataClassification] = Field(None, description="Data classification")
    schema: Optional[DataSchema] = Field(None, description="Asset schema definition")
    owner: str = Field(..., description="Asset owner or steward")
    steward: Optional[str] = Field(None, description="Data steward")
    business_contact: Optional[str] = Field(None, description="Business contact")
    technical_contact: Optional[str] = Field(None, description="Technical contact")
    domain: Optional[str] = Field(None, description="Business domain")
    subject_area: Optional[str] = Field(None, description="Subject area")
    source_system: Optional[str] = Field(None, description="Source system name")
    lineage_upstream: List[UUID] = Field(default_factory=list, description="Upstream asset dependencies")
    lineage_downstream: List[UUID] = Field(default_factory=list, description="Downstream asset consumers")
    related_assets: List[UUID] = Field(default_factory=list, description="Related asset references")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall quality score")
    usage_frequency: int = Field(default=0, ge=0, description="Usage frequency counter")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    last_modified_at: Optional[datetime] = Field(None, description="Last modification timestamp")
    version: str = Field(default="1.0.0", description="Asset version")
    change_log: List[Dict[str, Any]] = Field(default_factory=list, description="Change history")
    quality_rules: List[str] = Field(default_factory=list, description="Data quality rules")
    business_rules: List[str] = Field(default_factory=list, description="Business rules")
    access_patterns: List[str] = Field(default_factory=list, description="Common access patterns")
    sla_requirements: Dict[str, Any] = Field(default_factory=dict, description="SLA requirements")
    compliance_notes: List[str] = Field(default_factory=list, description="Compliance notes")
    documentation_urls: List[str] = Field(default_factory=list, description="Documentation links")
    sample_queries: List[str] = Field(default_factory=list, description="Sample usage queries")
    tags: List[str] = Field(default_factory=list, description="Asset tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate asset name format."""
        if not v.strip():
            raise ValueError("Asset name cannot be empty")
        return v.strip()
    
    @validator('owner')
    def validate_owner(cls, v: str) -> str:
        """Validate owner is not empty."""
        if not v.strip():
            raise ValueError("Asset owner cannot be empty")
        return v.strip()
    
    @validator('quality_score')
    def validate_quality_score(cls, v: float) -> float:
        """Validate quality score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        return v
    
    def activate(self) -> None:
        """Activate the asset."""
        self.status = AssetStatus.ACTIVE
        self.updated_at = datetime.utcnow()
        self._log_change("ACTIVATED", "Asset activated")
    
    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate the asset."""
        self.status = AssetStatus.INACTIVE
        self.updated_at = datetime.utcnow()
        self._log_change("DEACTIVATED", reason or "Asset deactivated")
    
    def deprecate(self, replacement_asset_id: Optional[UUID] = None, reason: Optional[str] = None) -> None:
        """Deprecate the asset."""
        self.status = AssetStatus.DEPRECATED
        self.updated_at = datetime.utcnow()
        
        change_details = {"reason": reason or "Asset deprecated"}
        if replacement_asset_id:
            change_details["replacement_asset_id"] = str(replacement_asset_id)
        
        self._log_change("DEPRECATED", "Asset deprecated", change_details)
    
    def archive(self, reason: Optional[str] = None) -> None:
        """Archive the asset."""
        self.status = AssetStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
        self._log_change("ARCHIVED", reason or "Asset archived")
    
    def update_schema(self, new_schema: DataSchema) -> None:
        """Update asset schema and log the change."""
        old_version = self.schema.version if self.schema else None
        self.schema = new_schema
        self.last_modified_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        self._log_change("SCHEMA_UPDATED", f"Schema updated from {old_version} to {new_schema.version}")
    
    def update_classification(self, new_classification: DataClassification) -> None:
        """Update asset classification."""
        old_sensitivity = self.classification.sensitivity_level if self.classification else None
        self.classification = new_classification
        self.updated_at = datetime.utcnow()
        
        self._log_change("CLASSIFICATION_UPDATED", 
                        f"Classification updated from {old_sensitivity} to {new_classification.sensitivity_level}")
    
    def record_access(self, user: Optional[str] = None) -> None:
        """Record an access to this asset."""
        self.last_accessed_at = datetime.utcnow()
        self.usage_frequency += 1
        self.updated_at = datetime.utcnow()
        
        access_details = {"user": user} if user else {}
        self._log_change("ACCESSED", "Asset accessed", access_details)
    
    def update_quality_score(self, score: float, reason: Optional[str] = None) -> None:
        """Update the asset quality score."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        
        old_score = self.quality_score
        self.quality_score = score
        self.updated_at = datetime.utcnow()
        
        self._log_change("QUALITY_UPDATED", 
                        f"Quality score updated from {old_score:.2f} to {score:.2f}",
                        {"reason": reason})
    
    def add_upstream_dependency(self, upstream_asset_id: UUID) -> None:
        """Add upstream dependency."""
        if upstream_asset_id not in self.lineage_upstream:
            self.lineage_upstream.append(upstream_asset_id)
            self.updated_at = datetime.utcnow()
            self._log_change("DEPENDENCY_ADDED", f"Added upstream dependency: {upstream_asset_id}")
    
    def remove_upstream_dependency(self, upstream_asset_id: UUID) -> None:
        """Remove upstream dependency."""
        if upstream_asset_id in self.lineage_upstream:
            self.lineage_upstream.remove(upstream_asset_id)
            self.updated_at = datetime.utcnow()
            self._log_change("DEPENDENCY_REMOVED", f"Removed upstream dependency: {upstream_asset_id}")
    
    def add_downstream_consumer(self, downstream_asset_id: UUID) -> None:
        """Add downstream consumer."""
        if downstream_asset_id not in self.lineage_downstream:
            self.lineage_downstream.append(downstream_asset_id)
            self.updated_at = datetime.utcnow()
            self._log_change("CONSUMER_ADDED", f"Added downstream consumer: {downstream_asset_id}")
    
    def remove_downstream_consumer(self, downstream_asset_id: UUID) -> None:
        """Remove downstream consumer."""
        if downstream_asset_id in self.lineage_downstream:
            self.lineage_downstream.remove(downstream_asset_id)
            self.updated_at = datetime.utcnow()
            self._log_change("CONSUMER_REMOVED", f"Removed downstream consumer: {downstream_asset_id}")
    
    def increment_version(self, change_description: Optional[str] = None) -> None:
        """Increment asset version."""
        version_parts = self.version.split('.')
        if len(version_parts) >= 3:
            patch = int(version_parts[2]) + 1
            self.version = f"{version_parts[0]}.{version_parts[1]}.{patch}"
        else:
            self.version = f"{self.version}.1"
        
        self.updated_at = datetime.utcnow()
        self._log_change("VERSION_INCREMENTED", 
                        f"Version incremented to {self.version}",
                        {"change_description": change_description})
    
    def _log_change(self, change_type: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a change to the asset."""
        change_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': change_type,
            'description': description,
            'details': details or {}
        }
        
        self.change_log.append(change_entry)
        
        # Keep only last 200 changes
        if len(self.change_log) > 200:
            self.change_log = self.change_log[-200:]
    
    def is_active(self) -> bool:
        """Check if asset is currently active."""
        return self.status == AssetStatus.ACTIVE
    
    def is_deprecated(self) -> bool:
        """Check if asset is deprecated."""
        return self.status == AssetStatus.DEPRECATED
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if asset meets high quality threshold."""
        return self.quality_score >= threshold
    
    def is_frequently_used(self, threshold: int = 100) -> bool:
        """Check if asset is frequently accessed."""
        return self.usage_frequency >= threshold
    
    def has_lineage(self) -> bool:
        """Check if asset has lineage information."""
        return bool(self.lineage_upstream) or bool(self.lineage_downstream)
    
    def requires_governance(self) -> bool:
        """Check if asset requires special governance."""
        return (
            self.classification and self.classification.requires_access_controls() or
            bool(self.compliance_notes) or
            self.is_frequently_used()
        )
    
    def get_impact_score(self) -> float:
        """Calculate impact score based on usage and dependencies."""
        base_score = min(1.0, self.usage_frequency / 1000.0)
        dependency_score = min(1.0, len(self.lineage_downstream) / 10.0)
        return (base_score + dependency_score) / 2.0