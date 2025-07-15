"""
Data Catalog Domain Entities

Defines the domain model for data catalog management, including data assets,
schemas, metadata, and usage tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DataAssetType(str, Enum):
    """Types of data assets in the catalog."""
    
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    STREAM = "stream"
    MODEL = "model"
    FEATURE = "feature"
    DATASET = "dataset"
    REPORT = "report"
    API = "api"


class DataFormat(str, Enum):
    """Data formats supported in the catalog."""
    
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    XML = "xml"
    EXCEL = "excel"
    SQL = "sql"
    BINARY = "binary"
    TEXT = "text"


class AccessLevel(str, Enum):
    """Access levels for data assets."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"


class DataClassification(str, Enum):
    """Data classification levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PII = "pii"
    PHI = "phi"
    FINANCIAL = "financial"
    CONFIDENTIAL = "confidential"


class DataQuality(str, Enum):
    """Data quality levels."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class ColumnMetadata:
    """Metadata for a data column/field."""
    
    name: str
    data_type: str
    nullable: bool = True
    description: Optional[str] = None
    constraints: List[str] = None
    tags: Set[str] = None
    
    # Statistical metadata
    min_value: Optional[Union[int, float, str]] = None
    max_value: Optional[Union[int, float, str]] = None
    avg_value: Optional[float] = None
    null_count: Optional[int] = None
    unique_count: Optional[int] = None
    
    # Quality metadata
    quality_score: Optional[float] = None
    completeness: Optional[float] = None
    validity: Optional[float] = None
    
    # Classification
    classification: Optional[DataClassification] = None
    is_pii: bool = False
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.tags is None:
            self.tags = set()


class DataSchema(BaseModel):
    """Represents the schema of a data asset."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Schema name")
    version: str = Field(default="1.0", description="Schema version")
    description: Optional[str] = None
    
    columns: List[ColumnMetadata] = Field(default_factory=list)
    
    # Schema metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    # Compatibility
    compatible_formats: List[DataFormat] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
    
    def add_column(self, column: ColumnMetadata) -> None:
        """Add a column to the schema."""
        self.columns.append(column)
        self.updated_at = datetime.utcnow()
    
    def get_column(self, name: str) -> Optional[ColumnMetadata]:
        """Get a column by name."""
        for column in self.columns:
            if column.name == name:
                return column
        return None
    
    def get_pii_columns(self) -> List[ColumnMetadata]:
        """Get all PII columns."""
        return [col for col in self.columns if col.is_pii]
    
    def get_columns_by_classification(self, classification: DataClassification) -> List[ColumnMetadata]:
        """Get columns by classification."""
        return [col for col in self.columns if col.classification == classification]


class DataUsage(BaseModel):
    """Tracks usage of a data asset."""
    
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID = Field(..., description="ID of the data asset")
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    usage_type: str = Field(..., description="Type of usage (read, write, query, etc.)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Usage details
    query: Optional[str] = None
    rows_accessed: Optional[int] = None
    columns_accessed: List[str] = Field(default_factory=list)
    duration_ms: Optional[int] = None
    
    # Context
    application: Optional[str] = None
    purpose: Optional[str] = None
    ip_address: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DataCatalogEntry(BaseModel):
    """Represents an entry in the data catalog."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the data asset")
    type: DataAssetType = Field(..., description="Type of data asset")
    description: Optional[str] = None
    
    # Location and access
    location: str = Field(..., description="Location/path of the data asset")
    format: DataFormat = Field(..., description="Data format")
    access_level: AccessLevel = Field(default=AccessLevel.INTERNAL)
    
    # Ownership and governance
    owner: Optional[str] = None
    steward: Optional[str] = None
    domain: Optional[str] = None
    project: Optional[str] = None
    
    # Schema and structure
    schema_: Optional[DataSchema] = Field(None, alias="schema")
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    
    # Quality and lineage
    quality: DataQuality = Field(default=DataQuality.UNKNOWN)
    quality_score: Optional[float] = None
    lineage_upstream: Set[UUID] = Field(default_factory=set)
    lineage_downstream: Set[UUID] = Field(default_factory=set)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    
    # Tags and classification
    tags: Set[str] = Field(default_factory=set)
    classification: DataClassification = Field(default=DataClassification.INTERNAL)
    
    # Business metadata
    business_terms: Set[str] = Field(default_factory=set)
    related_assets: Set[UUID] = Field(default_factory=set)
    
    # Usage tracking
    usage_stats: Dict[str, Any] = Field(default_factory=dict)
    
    # Custom properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the asset."""
        self.tags.add(tag)
        self.updated_at = datetime.utcnow()
    
    def add_business_term(self, term: str) -> None:
        """Add a business term to the asset."""
        self.business_terms.add(term)
        self.updated_at = datetime.utcnow()
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a custom property."""
        self.properties[key] = value
        self.updated_at = datetime.utcnow()
    
    def record_access(self, user: str = None, usage_type: str = "read") -> None:
        """Record access to this asset."""
        self.last_accessed = datetime.utcnow()
        
        # Update usage statistics
        if "access_count" not in self.usage_stats:
            self.usage_stats["access_count"] = 0
        self.usage_stats["access_count"] += 1
        
        if "last_user" not in self.usage_stats:
            self.usage_stats["last_user"] = user
        
        self.updated_at = datetime.utcnow()
    
    def update_quality_score(self, score: float, quality_level: DataQuality = None) -> None:
        """Update quality score and level."""
        self.quality_score = score
        
        if quality_level:
            self.quality = quality_level
        else:
            # Auto-determine quality level based on score
            if score >= 0.9:
                self.quality = DataQuality.EXCELLENT
            elif score >= 0.8:
                self.quality = DataQuality.GOOD
            elif score >= 0.6:
                self.quality = DataQuality.FAIR
            else:
                self.quality = DataQuality.POOR
        
        self.updated_at = datetime.utcnow()
    
    def add_lineage_relationship(self, upstream_id: UUID = None, downstream_id: UUID = None) -> None:
        """Add lineage relationships."""
        if upstream_id:
            self.lineage_upstream.add(upstream_id)
        if downstream_id:
            self.lineage_downstream.add(downstream_id)
        
        self.updated_at = datetime.utcnow()
    
    def get_popularity_score(self) -> float:
        """Calculate popularity score based on usage."""
        access_count = self.usage_stats.get("access_count", 0)
        days_since_creation = (datetime.utcnow() - self.created_at).days
        
        if days_since_creation == 0:
            return access_count
        
        # Popularity = accesses per day
        return access_count / max(days_since_creation, 1)
    
    def get_freshness_score(self) -> float:
        """Calculate freshness score based on last modification."""
        if not self.last_modified:
            return 0.0
        
        days_since_modified = (datetime.utcnow() - self.last_modified).days
        
        # Freshness decreases exponentially with age
        # 1.0 for same day, 0.5 for 7 days, 0.1 for 30 days
        import math
        return math.exp(-days_since_modified / 10.0)
    
    def get_relevance_score(self, query_terms: List[str]) -> float:
        """Calculate relevance score for search query."""
        if not query_terms:
            return 0.0
        
        # Combine text from name, description, tags, and business terms
        searchable_text = " ".join([
            self.name.lower(),
            self.description.lower() if self.description else "",
            " ".join(self.tags).lower(),
            " ".join(self.business_terms).lower()
        ])
        
        # Calculate relevance based on term matches
        matches = 0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term.lower() in searchable_text:
                matches += 1
        
        base_relevance = matches / total_terms if total_terms > 0 else 0.0
        
        # Boost for exact name matches
        name_boost = 1.0
        for term in query_terms:
            if term.lower() in self.name.lower():
                name_boost += 0.5
        
        return min(1.0, base_relevance * name_boost)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "location": self.location,
            "format": self.format,
            "owner": self.owner,
            "quality": self.quality,
            "quality_score": self.quality_score,
            "classification": self.classification,
            "tags": list(self.tags),
            "business_terms": list(self.business_terms),
            "popularity_score": self.get_popularity_score(),
            "freshness_score": self.get_freshness_score(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }